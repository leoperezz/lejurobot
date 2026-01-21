#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any

import numpy as np

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    load_training_state,
    save_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)

from lejurobot.policies.factory import make_lejurobot_policy, make_lejurobot_pre_post_processors
from lejurobot.configs.train import TrainPipelineConfigLejuRobot
from lejurobot.datasets.factory import make_dataset_lejurobot
from lejurobot.datasets.utils import split_train_eval_episodes
from lejurobot.logger import logger


@dataclass
class EvalBatchStats:
    """Statistics computed from evaluation batch losses."""
    losses: list = field(default_factory=list)
    
    @property
    def mean(self) -> float:
        return float(np.mean(self.losses)) if self.losses else float('inf')
    
    @property
    def std(self) -> float:
        return float(np.std(self.losses)) if len(self.losses) > 1 else 0.0
    
    @property
    def min(self) -> float:
        return float(np.min(self.losses)) if self.losses else float('inf')
    
    @property
    def max(self) -> float:
        return float(np.max(self.losses)) if self.losses else float('inf')
    
    @property
    def percentile_90(self) -> float:
        return float(np.percentile(self.losses, 90)) if self.losses else float('inf')
    
    @property
    def percentile_95(self) -> float:
        return float(np.percentile(self.losses, 95)) if self.losses else float('inf')
    
    @property
    def median(self) -> float:
        return float(np.median(self.losses)) if self.losses else float('inf')
    
    def score_mean(self) -> float:
        """Strategy 1: Simple mean (lower is better)."""
        return self.mean
    
    def score_stable(self, lambda_penalty: float = 1.0) -> float:
        """Strategy 2: Mean + lambda * std (penalizes instability)."""
        return self.mean + lambda_penalty * self.std
    
    def score_robust(self) -> float:
        """Strategy 3: Percentile 90 (robust to outliers)."""
        return self.percentile_90
    
    def score_minimax(self) -> float:
        """Strategy 4: Max loss (minimax - minimize worst case)."""
        return self.max


@dataclass  
class CheckpointStrategy:
    """Tracks best scores for a specific checkpoint strategy."""
    name: str
    best_score: float = float('inf')
    best_step: int = -1
    
    def is_better(self, new_score: float) -> bool:
        """Returns True if new_score is better than current best (lower is better)."""
        return new_score < self.best_score
    
    def update(self, new_score: float, step: int) -> bool:
        """Update best score if new_score is better. Returns True if updated."""
        if self.is_better(new_score):
            self.best_score = new_score
            self.best_step = step
            return True
        return False

def load_dataset(cfg: TrainPipelineConfigLejuRobot, episodes: list[int],is_main_process: bool = True, accelerator: Accelerator | None = None):
    """
    Load the dataset for training and evaluation.
    """
    # Dataset loading synchronization: main process downloads first to avoid race conditions
    cfg.dataset.episodes = episodes

    if is_main_process:
        logger.info("Creating dataset")
        dataset = make_dataset_lejurobot(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset_lejurobot(cfg)
    
    return dataset

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def evaluate_policy(
    eval_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    eval_dataloader: torch.utils.data.DataLoader,
    preprocessor: Any,
    accelerator: Accelerator,
    num_eval_batches: int = 50,
) -> tuple[MetricsTracker, dict, EvalBatchStats]:
    """
    Evaluates the policy on the evaluation dataset.

    This function runs the policy in eval mode on the evaluation dataset and computes metrics.

    Args:
        eval_metrics: A MetricsTracker instance to record evaluation statistics.
        policy: The policy model to evaluate.
        eval_dataloader: DataLoader for the evaluation dataset.
        preprocessor: Preprocessor to apply to batches.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        num_eval_batches: Number of batches to evaluate on.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with evaluation statistics.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
        - EvalBatchStats with per-batch loss statistics.
    """
    start_time = time.perf_counter()
    policy.eval()

    batch_losses = []
    output_dicts = []
    
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if i >= num_eval_batches:
                break
            
            batch = preprocessor(batch)
            
            # Let accelerator handle mixed precision
            with accelerator.autocast():
                loss, output_dict = policy.forward(batch)
            
            batch_losses.append(loss.item())
            output_dicts.append(output_dict)
    
    # Create batch statistics
    batch_stats = EvalBatchStats(losses=batch_losses)
    
    # Aggregate output_dicts if needed
    aggregated_output = {}
    if output_dicts:
        for key in output_dicts[0].keys():
            values = [d[key] for d in output_dicts if key in d]
            if values and isinstance(values[0], (int, float)):
                aggregated_output[f"eval_{key}"] = sum(values) / len(values)
    
    eval_metrics.loss = batch_stats.mean
    eval_metrics.update_s = time.perf_counter() - start_time
    
    return eval_metrics, aggregated_output, batch_stats


@parser.wrap()
def train(cfg: TrainPipelineConfigLejuRobot, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfigLejuRobot` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logger.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logger.info("Creating dataset")
        dataset = make_dataset_lejurobot(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset_lejurobot(cfg)

    episodes = list(range(dataset.meta.total_episodes))

    episodes = episodes[1700:]

    train_episodes, eval_episodes = split_train_eval_episodes(episodes, split_ratio=cfg.split_ratio, seed=42)
    
    #train_episodes = episodes[522:len(episodes)-1]
    #eval_episodes = [22, 413, 135, 309, 176, 433, 183, 49, 194, 510, 52, 172, 390, 110, 79, 494, 174, 216, 357, 81, 412, 388, 3, 445, 414, 516, 301, 229, 112, 214, 279, 359, 332, 366, 101, 287, 13, 308, 258, 119, 111, 47, 30, 32, 432, 89, 104, 142, 228, 250, 281, 25, 114]

    del dataset
    
    #TODO: This is a hack to avoid memory issues. We need to find a better way to do this.
    logger.info(f"Loading train dataset with {train_episodes} episodes")
    logger.info(f"Loading eval dataset with {eval_episodes} episodes")
    dataset = load_dataset(cfg, train_episodes, is_main_process, accelerator)
    eval_dataset = load_dataset(cfg, eval_episodes, is_main_process, accelerator)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.

    if is_main_process:
        logger.info("Creating policy")
    policy = make_lejurobot_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_lejurobot_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logger.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logger.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logger.info(f"{cfg.env.task=}")
        logger.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logger.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logger.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logger.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logger.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logger.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        logger.info(f"Dropping {cfg.policy.drop_n_last_frames} last frames")
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        eval_sampler = EpisodeAwareSampler(
            eval_dataset.meta.episodes["dataset_from_index"],
            eval_dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=False,  # No shuffle for eval
        )
    else:
        logger.info("Not dropping any frames")
        shuffle = True
        sampler = None
        eval_sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Create eval dataloader
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,  # No shuffle for eval
        sampler=eval_sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler, eval_dataloader
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Create eval metrics (similar to train metrics but without grad_norm and lr)
    eval_metrics = {
        "loss": AverageMeter("eval_loss", ":.3f"),
        "update_s": AverageMeter("eval_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    eval_tracker = MetricsTracker(
        effective_batch_size,
        eval_dataset.num_frames,
        eval_dataset.num_episodes,
        eval_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    # Track best checkpoints for 4 different strategies
    checkpoint_strategies = {
        "best_mean": CheckpointStrategy(name="best_mean"),        # Strategy 1: Simple mean
        "best_stable": CheckpointStrategy(name="best_stable"),    # Strategy 2: Mean + λ*σ  
        "best_robust": CheckpointStrategy(name="best_robust"),    # Strategy 3: Percentile 90
        "best_minimax": CheckpointStrategy(name="best_minimax"),  # Strategy 4: Minimax (min of max)
    }
    STABILITY_LAMBDA = 1.0  # λ parameter for stable strategy

    if is_main_process:
        logger.info("Start offline training on a fixed dataset")
        logger.info(f"Train episodes: {len(train_episodes)}, Eval episodes: {len(eval_episodes)}")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # Log to WandB at every step (if wandb is enabled)
        if wandb_logger and is_main_process:
            wandb_log_dict = {
                "train/loss": train_tracker.loss.val,  # Current step loss
                "train/grad_norm": train_tracker.grad_norm.val,
                "train/lr": train_tracker.lr.val,
                "train/update_s": train_tracker.update_s.val,
                "train/dataloading_s": train_tracker.dataloading_s.val,
            }
            if output_dict:
                # Add any additional metrics from the policy output
                for key, value in output_dict.items():
                    if isinstance(value, (int, float)):
                        wandb_log_dict[f"train/{key}"] = value
            wandb_logger.log_dict(wandb_log_dict, step)

        # Log to terminal only at log_freq intervals
        if is_log_step:
            logger.info(train_tracker)
            train_tracker.reset_averages()

        # Evaluation step
        if is_eval_step:
            if is_main_process:
                logger.info(f"\n{'='*80}")
                logger.info(f"Evaluating policy at step {step}")
                logger.info(f"{'='*80}")
            
            eval_tracker, eval_output_dict, batch_stats = evaluate_policy(
                eval_tracker,
                policy,
                eval_dataloader,
                preprocessor,
                accelerator,
                num_eval_batches=50,
            )
            
            if is_main_process:
                # Log detailed batch statistics to terminal
                logger.info(f"\n{'-'*80}")
                logger.info(f"EVALUATION RESULTS AT STEP {step}:")
                logger.info(f"{'-'*80}")
                logger.info(f"  📊 BATCH STATISTICS ({len(batch_stats.losses)} batches):")
                logger.info(f"     Mean:          {batch_stats.mean:.6f}")
                logger.info(f"     Std:           {batch_stats.std:.6f}")
                logger.info(f"     Min:           {batch_stats.min:.6f}")
                logger.info(f"     Max:           {batch_stats.max:.6f}")
                logger.info(f"     Median:        {batch_stats.median:.6f}")
                logger.info(f"     Percentile 90: {batch_stats.percentile_90:.6f}")
                logger.info(f"     Percentile 95: {batch_stats.percentile_95:.6f}")
                logger.info(f"  ⏱️  Eval Time:      {eval_tracker.update_s.avg:.3f}s")
                
                # Calculate scores for each strategy
                score_mean = batch_stats.score_mean()
                score_stable = batch_stats.score_stable(STABILITY_LAMBDA)
                score_robust = batch_stats.score_robust()
                score_minimax = batch_stats.score_minimax()
                
                logger.info(f"\n  🎯 CHECKPOINT SCORES:")
                logger.info(f"     Score Mean (μ):           {score_mean:.6f}")
                logger.info(f"     Score Stable (μ + {STABILITY_LAMBDA}σ):   {score_stable:.6f}")
                logger.info(f"     Score Robust (P90):       {score_robust:.6f}")
                logger.info(f"     Score Minimax (max):      {score_minimax:.6f}")
                
                # Log to WandB with proper prefixes
                if wandb_logger:
                    eval_wandb_log_dict = {
                        "eval/loss_mean": batch_stats.mean,
                        "eval/loss_std": batch_stats.std,
                        "eval/loss_min": batch_stats.min,
                        "eval/loss_max": batch_stats.max,
                        "eval/loss_p90": batch_stats.percentile_90,
                        "eval/score_mean": score_mean,
                        "eval/score_stable": score_stable,
                        "eval/score_robust": score_robust,
                        "eval/score_minimax": score_minimax,
                        "eval/time_s": eval_tracker.update_s.avg,
                    }
                    
                    # Add any additional metrics from policy output
                    if eval_output_dict:
                        for key, value in eval_output_dict.items():
                            clean_key = key.replace("eval_", "")
                            if isinstance(value, (int, float)):
                                eval_wandb_log_dict[f"eval/{clean_key}"] = value
                    
                    wandb_logger.log_dict(eval_wandb_log_dict, step)
                
                # Check and save checkpoints for each strategy
                scores = {
                    "best_mean": score_mean,
                    "best_stable": score_stable,
                    "best_robust": score_robust,
                    "best_minimax": score_minimax,
                }
                
                for strategy_name, strategy in checkpoint_strategies.items():
                    current_score = scores[strategy_name]
                    if strategy.update(current_score, step):
                        logger.info(f"\n{'*'*80}")
                        logger.info(f"🎉 NEW BEST for {strategy_name.upper()}: {current_score:.6f}")
                        logger.info(f"{'*'*80}")
                        logger.info(f"Saving {strategy_name} checkpoint...")
                        
                        checkpoint_dir = cfg.output_dir / "checkpoints" / strategy_name
                        save_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            step=step,
                            cfg=cfg,
                            policy=accelerator.unwrap_model(policy),
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                        )
                
                # Log best scores to WandB
                if wandb_logger:
                    best_scores_log = {
                        "eval/best_score_mean": checkpoint_strategies["best_mean"].best_score,
                        "eval/best_score_stable": checkpoint_strategies["best_stable"].best_score,
                        "eval/best_score_robust": checkpoint_strategies["best_robust"].best_score,
                        "eval/best_score_minimax": checkpoint_strategies["best_minimax"].best_score,
                    }
                    wandb_logger.log_dict(best_scores_log, step)
                
                logger.info(f"\n  📁 BEST CHECKPOINTS SO FAR:")
                for strategy_name, strategy in checkpoint_strategies.items():
                    logger.info(f"     {strategy_name}: {strategy.best_score:.6f} (step {strategy.best_step})")
                
                logger.info(f"{'='*80}\n")
            
            eval_tracker.reset_averages()
            accelerator.wait_for_everyone()

        # NOTE: Traditional periodic checkpointing (last/best) has been replaced
        # by the 3-strategy checkpoint system (best_mean, best_stable, best_robust)
        # which saves checkpoints during evaluation steps above.

        # NOTE: Traditional periodic checkpointing (last/best) has been replaced
        # by the 4-strategy checkpoint system (best_mean, best_stable, best_robust, best_minimax)
        # which saves checkpoints during evaluation steps above.

    if is_main_process:
        logger.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    train()


if __name__ == "__main__":
    main()
