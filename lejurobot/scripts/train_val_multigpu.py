#!/usr/bin/env python
"""
Multi-GPU Training Script with DDP Best Practices.

This script implements distributed training using Hugging Face Accelerate with:
- Gradient Accumulation for larger effective batch sizes
- DistributedSampler with proper epoch handling
- Mixed Precision Training (AMP)
- Optional torch.profiler integration
- SyncBatchNorm support
- Proper metric synchronization across GPUs
- NCCL backend optimization

Usage:
    # Single GPU (falls back gracefully)
    python scripts/train_val_multigpu.py --config configs/train.yaml

    # Multi-GPU with accelerate
    accelerate launch --multi_gpu --num_processes=4 scripts/train_val_multigpu.py ...

    # Or use the provided shell script
    ./run_train_val_multigpu.sh
"""

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

import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any, Optional

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed as accelerate_set_seed
from termcolor import colored
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lerobot.configs import parser
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)

from lejurobot.configs.train import DistributedTrainConfig
from lejurobot.datasets.factory import make_dataset_lejurobot
from lejurobot.datasets.utils import split_train_eval_episodes
from lejurobot.policies.factory import make_lejurobot_policy, make_lejurobot_pre_post_processors


# =============================================================================
# Dataset Loading
# =============================================================================


def load_dataset(
    cfg: DistributedTrainConfig,
    episodes: list[int],
    is_main_process: bool = True,
    accelerator: Optional[Accelerator] = None,
):
    """
    Load the dataset for training and evaluation with proper distributed synchronization.
    
    Only the main process loads/downloads the dataset first to avoid race conditions,
    then all other processes load it from cache.
    
    Args:
        cfg: Training configuration.
        episodes: List of episode indices to include.
        is_main_process: Whether this is the main process.
        accelerator: Accelerator instance for synchronization.
    
    Returns:
        The loaded dataset.
    """
    cfg.dataset.episodes = episodes

    if is_main_process:
        logging.info(f"Creating dataset with {len(episodes)} episodes")
        dataset = make_dataset_lejurobot(cfg)

    if accelerator is not None:
        accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset from cache
    if not is_main_process:
        dataset = make_dataset_lejurobot(cfg)

    return dataset


# =============================================================================
# Training Step with Gradient Accumulation
# =============================================================================


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

    This function executes the forward and backward passes, clips gradients, and steps
    the optimizer and learning rate scheduler. Accelerator handles:
    - Mixed-precision training automatically
    - Gradient synchronization across GPUs
    - Gradient accumulation when configured

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

    # Accelerator handles autocast for mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

    # Use accelerator's backward method (handles gradient scaling for AMP)
    accelerator.backward(loss)

    # Only update weights when gradients are synchronized
    # (after gradient_accumulation_steps batches)
    if accelerator.sync_gradients:
        # Clip gradients if specified
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float("inf"), error_if_nonfinite=False
            )

        # Optimizer step with optional lock for thread safety
        with lock if lock is not None else nullcontext():
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)  # More memory efficient

        # Step through pytorch scheduler at every optimizer step
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update internal buffers if policy has update method (e.g., EMA)
        if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

        train_metrics.grad_norm = grad_norm.item()
    else:
        # Gradient accumulation step - don't record grad_norm yet
        train_metrics.grad_norm = 0.0

    train_metrics.loss = loss.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    
    return train_metrics, output_dict


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_policy(
    eval_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    eval_dataloader: DataLoader,
    preprocessor: Any,
    accelerator: Accelerator,
    num_eval_batches: int = 50,
) -> tuple[MetricsTracker, dict]:
    """
    Evaluates the policy on the evaluation dataset.

    This function runs the policy in eval mode on the evaluation dataset and computes
    metrics. Metrics are properly synchronized across all GPUs using all_reduce.

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
    """
    start_time = time.perf_counter()
    policy.eval()

    total_loss = 0.0
    num_batches = 0
    output_dicts = []

    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if i >= num_eval_batches:
                break

            batch = preprocessor(batch)

            # Let accelerator handle mixed precision
            with accelerator.autocast():
                loss, output_dict = policy.forward(batch)

            total_loss += loss.item()
            num_batches += 1
            output_dicts.append(output_dict)

    # Average the loss over all batches evaluated on this GPU
    local_avg_loss = total_loss / max(num_batches, 1)

    # Synchronize metrics across all GPUs using all_reduce
    if accelerator.num_processes > 1:
        loss_tensor = torch.tensor([local_avg_loss, num_batches], device=accelerator.device)
        # Sum losses and batch counts across all processes
        loss_tensor = accelerator.reduce(loss_tensor, reduction="sum")
        avg_loss = loss_tensor[0].item() / max(loss_tensor[1].item(), 1)
    else:
        avg_loss = local_avg_loss

    # Aggregate output_dicts if needed
    aggregated_output = {}
    if output_dicts:
        for key in output_dicts[0].keys():
            values = [d[key] for d in output_dicts if key in d]
            if values and isinstance(values[0], (int, float)):
                aggregated_output[f"eval_{key}"] = sum(values) / len(values)

    eval_metrics.loss = avg_loss
    eval_metrics.update_s = time.perf_counter() - start_time

    return eval_metrics, aggregated_output


# =============================================================================
# Profiling Context
# =============================================================================


def create_profiler(cfg: DistributedTrainConfig, step: int):
    """
    Create a torch.profiler context manager if profiling is enabled.
    
    Args:
        cfg: Training configuration.
        step: Current training step (for output naming).
    
    Returns:
        A profiler context manager or nullcontext if profiling is disabled.
    """
    if not cfg.enable_profiling:
        return nullcontext()

    profiling_dir = cfg.profiling_output_dir or str(cfg.output_dir / "profiling")
    os.makedirs(profiling_dir, exist_ok=True)

    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=cfg.profiling_warmup_steps,
            active=cfg.profiling_active_steps,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


# =============================================================================
# Distributed Sampler Wrapper
# =============================================================================


class DistributedEpisodeAwareSampler(DistributedSampler):
    """
    A DistributedSampler that wraps EpisodeAwareSampler for multi-GPU training.
    
    This ensures each GPU gets a different subset of the data while respecting
    episode boundaries for proper frame dropping.
    """

    def __init__(
        self,
        dataset,
        episode_from_indices,
        episode_to_indices,
        drop_n_last_frames: int = 0,
        shuffle: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        self.episode_from_indices = episode_from_indices
        self.episode_to_indices = episode_to_indices
        self.drop_n_last_frames = drop_n_last_frames
        self._shuffle = shuffle
        self._seed = seed

    def __iter__(self):
        # Create the base episode-aware indices
        base_sampler = EpisodeAwareSampler(
            self.episode_from_indices,
            self.episode_to_indices,
            drop_n_last_frames=self.drop_n_last_frames,
            shuffle=self._shuffle,
        )
        
        # Set seed based on epoch for reproducibility
        if self._shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed + self.epoch)
            indices = list(base_sampler)
            # Shuffle the indices with the generator
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]
        else:
            indices = list(base_sampler)

        # Subsample for this rank
        indices = indices[self.rank : len(indices) : self.num_replicas]
        
        return iter(indices)

    def __len__(self):
        # Estimate length based on episode ranges and drop_n_last_frames
        total_indices = 0
        for from_idx, to_idx in zip(self.episode_from_indices, self.episode_to_indices):
            episode_len = max(0, to_idx - from_idx - self.drop_n_last_frames)
            total_indices += episode_len
        return total_indices // self.num_replicas


# =============================================================================
# Main Training Function
# =============================================================================


@parser.wrap()
def train(cfg: DistributedTrainConfig, accelerator: Optional[Accelerator] = None):
    """
    Main function to train a policy with multi-GPU DDP support.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop with gradient accumulation.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `DistributedTrainConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # =========================================================================
    # Accelerator Setup with DDP Best Practices
    # =========================================================================
    if accelerator is None:
        # Configure DDP kwargs for optimal performance
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=cfg.find_unused_parameters,
            broadcast_buffers=cfg.broadcast_buffers,
        )
        
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            step_scheduler_with_optimizer=False,  # We handle LR scheduling manually
            kwargs_handlers=[ddp_kwargs],
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    is_main_process = accelerator.is_main_process
    world_size = accelerator.num_processes
    local_rank = accelerator.local_process_index

    # =========================================================================
    # Logging Configuration
    # =========================================================================
    if is_main_process:
        logging.info("=" * 80)
        logging.info("  LejuRobot Multi-GPU Training with DDP Best Practices")
        logging.info("=" * 80)
        logging.info(f"  World Size (num GPUs):       {world_size}")
        logging.info(f"  Local Rank:                  {local_rank}")
        logging.info(f"  Gradient Accumulation Steps: {cfg.gradient_accumulation_steps}")
        logging.info(f"  Batch Size per GPU:          {cfg.batch_size}")
        effective_bs = cfg.batch_size * world_size * cfg.gradient_accumulation_steps
        logging.info(f"  Effective Batch Size:        {effective_bs}")
        logging.info(f"  Mixed Precision:             {accelerator.mixed_precision}")
        if torch.cuda.is_available():
            logging.info(f"  CUDA Device:                 {torch.cuda.get_device_name()}")
        logging.info("=" * 80)
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # =========================================================================
    # Seed and Device Setup
    # =========================================================================
    if cfg.seed is not None:
        # Use accelerate's set_seed which handles distributed seeding properly
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device (automatically assigned per GPU)
    device = accelerator.device
    
    # CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # =========================================================================
    # Dataset Loading with Distributed Synchronization
    # =========================================================================
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset_lejurobot(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset_lejurobot(cfg)

    episodes = list(range(dataset.meta.total_episodes))
    train_episodes, eval_episodes = split_train_eval_episodes(
        episodes, split_ratio=cfg.split_ratio, seed=42
    )

    del dataset

    if is_main_process:
        logging.info(f"Train episodes: {len(train_episodes)}, Eval episodes: {len(eval_episodes)}")

    dataset = load_dataset(cfg, train_episodes, is_main_process, accelerator)
    eval_dataset = load_dataset(cfg, eval_episodes, is_main_process, accelerator)

    # =========================================================================
    # Policy Creation
    # =========================================================================
    if is_main_process:
        logging.info("Creating policy")
    
    policy = make_lejurobot_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    # Convert BatchNorm to SyncBatchNorm if configured (important for multi-GPU consistency)
    if cfg.sync_batch_norms and world_size > 1:
        policy = torch.nn.SyncBatchNorm.convert_sync_batchnorm(policy)
        if is_main_process:
            logging.info("Converted BatchNorm to SyncBatchNorm for multi-GPU training")

    # Wait for all processes to finish policy creation
    accelerator.wait_for_everyone()

    # =========================================================================
    # Preprocessor and Postprocessor Setup
    # =========================================================================
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
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

    # =========================================================================
    # Optimizer and Scheduler
    # =========================================================================
    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # Number of optimizer updates (not forward passes)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # =========================================================================
    # DataLoader Setup with Distributed Sampling
    # =========================================================================
    if hasattr(cfg.policy, "drop_n_last_frames"):
        if is_main_process:
            logging.info(f"Dropping {cfg.policy.drop_n_last_frames} last frames")
        
        # Use distributed-aware sampler for multi-GPU
        if world_size > 1:
            sampler = DistributedEpisodeAwareSampler(
                dataset,
                dataset.meta.episodes["dataset_from_index"],
                dataset.meta.episodes["dataset_to_index"],
                drop_n_last_frames=cfg.policy.drop_n_last_frames,
                shuffle=True,
                num_replicas=world_size,
                rank=local_rank,
                seed=cfg.seed or 0,
            )
            eval_sampler = DistributedEpisodeAwareSampler(
                eval_dataset,
                eval_dataset.meta.episodes["dataset_from_index"],
                eval_dataset.meta.episodes["dataset_to_index"],
                drop_n_last_frames=cfg.policy.drop_n_last_frames,
                shuffle=False,
                num_replicas=world_size,
                rank=local_rank,
                seed=cfg.seed or 0,
            )
        else:
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
                shuffle=False,
            )
        shuffle = False  # Sampler handles shuffling
    else:
        if is_main_process:
            logging.info("Not dropping any frames")
        
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=True,
                seed=cfg.seed or 0,
            )
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=False,
                seed=cfg.seed or 0,
            )
            shuffle = False
        else:
            sampler = None
            eval_sampler = None
            shuffle = True

    # Create DataLoaders with optimized settings
    dataloader = DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=True,  # Important for DDP to ensure equal batch sizes
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,  # Keep workers alive between epochs
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=eval_sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # =========================================================================
    # Prepare with Accelerator
    # =========================================================================
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler, eval_dataloader
    )

    dl_iter = cycle(dataloader)
    policy.train()

    # =========================================================================
    # Metrics Setup
    # =========================================================================
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    eval_metrics = {
        "loss": AverageMeter("eval_loss", ":.3f"),
        "update_s": AverageMeter("eval_s", ":.3f"),
    }

    # Effective batch size accounts for gradient accumulation
    effective_batch_size = cfg.batch_size * world_size * cfg.gradient_accumulation_steps
    
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

    # Track best eval loss for saving best checkpoints
    best_eval_loss = float("inf")

    if is_main_process:
        logging.info("=" * 80)
        logging.info("  Starting Offline Training")
        logging.info("=" * 80)

    # =========================================================================
    # Training Loop with Gradient Accumulation
    # =========================================================================
    
    # Calculate steps considering gradient accumulation
    # Each "step" is an optimizer update, which happens after accumulation_steps forward passes
    forward_step = 0  # Track forward passes for epoch setting
    
    # Set initial epoch for distributed samplers
    current_epoch = 0
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(current_epoch)

    # Optional profiling context
    profiler = create_profiler(cfg, step) if cfg.enable_profiling else nullcontext()

    with profiler:
        for _ in range(step, cfg.steps):
            start_time = time.perf_counter()

            # Gradient accumulation loop
            accumulated_loss = 0.0
            for acc_step in range(cfg.gradient_accumulation_steps):
                batch = next(dl_iter)
                batch = preprocessor(batch)
                forward_step += 1

                # Update epoch counter for sampler (approximation based on dataset size)
                if hasattr(sampler, "set_epoch"):
                    new_epoch = forward_step * cfg.batch_size * world_size // dataset.num_frames
                    if new_epoch > current_epoch:
                        current_epoch = new_epoch
                        sampler.set_epoch(current_epoch)

                # Use accelerator.accumulate for proper gradient accumulation
                with accelerator.accumulate(policy):
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
                    
                    accumulated_loss += train_tracker.loss.val

                start_time = time.perf_counter()

            # Average loss over accumulation steps
            train_tracker.loss = accumulated_loss / cfg.gradient_accumulation_steps

            # Step profiler if enabled
            if cfg.enable_profiling and hasattr(profiler, "step"):
                profiler.step()

            # Note: eval and checkpoint happens *after* the `step`th training update
            step += 1
            train_tracker.step()
            
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            # Log to WandB at every step (if wandb is enabled)
            if wandb_logger and is_main_process:
                wandb_log_dict = {
                    "train/loss": train_tracker.loss.val,
                    "train/grad_norm": train_tracker.grad_norm.val,
                    "train/lr": train_tracker.lr.val,
                    "train/update_s": train_tracker.update_s.val,
                    "train/dataloading_s": train_tracker.dataloading_s.val,
                    "train/epoch": current_epoch,
                }
                if output_dict:
                    for key, value in output_dict.items():
                        if isinstance(value, (int, float)):
                            wandb_log_dict[f"train/{key}"] = value
                wandb_logger.log_dict(wandb_log_dict, step)

            # Log to terminal only at log_freq intervals
            if is_log_step:
                logging.info(train_tracker)
                train_tracker.reset_averages()

            # =================================================================
            # Evaluation Step
            # =================================================================
            if is_eval_step:
                if is_main_process:
                    logging.info(f"\n{'=' * 80}")
                    logging.info(f"  Evaluating policy at step {step}")
                    logging.info(f"{'=' * 80}")

                # Set epoch for eval sampler if distributed
                if hasattr(eval_sampler, "set_epoch"):
                    eval_sampler.set_epoch(current_epoch)

                eval_tracker, eval_output_dict = evaluate_policy(
                    eval_tracker,
                    policy,
                    eval_dataloader,
                    preprocessor,
                    accelerator,
                    num_eval_batches=cfg.num_eval_batches,
                )

                if is_main_process:
                    logging.info(f"\n{'-' * 80}")
                    logging.info(f"  EVALUATION RESULTS AT STEP {step}:")
                    logging.info(f"{'-' * 80}")
                    logging.info(f"    Eval Loss:      {eval_tracker.loss.avg:.4f}")
                    logging.info(f"    Eval Time:      {eval_tracker.update_s.avg:.3f}s")

                    # Log to WandB
                    if wandb_logger:
                        eval_wandb_log_dict = {
                            "eval/loss": eval_tracker.loss.avg,
                            "eval/time_s": eval_tracker.update_s.avg,
                        }

                        if eval_output_dict:
                            for key, value in eval_output_dict.items():
                                clean_key = key.replace("eval_", "")
                                if isinstance(value, (int, float)):
                                    eval_wandb_log_dict[f"eval/{clean_key}"] = value
                                    logging.info(f"    {clean_key.capitalize()}: {value:.4f}")

                        wandb_logger.log_dict(eval_wandb_log_dict, step)

                    # Save best checkpoint if eval loss improved
                    current_eval_loss = eval_tracker.loss.avg
                    if current_eval_loss < best_eval_loss:
                        best_eval_loss = current_eval_loss
                        logging.info(f"\n{'*' * 80}")
                        logging.info(f"  🎉 NEW BEST EVAL LOSS: {best_eval_loss:.4f}")
                        logging.info(f"{'*' * 80}")
                        logging.info("  Saving best checkpoint...")

                        if getattr(cfg, "only_last_best", False):
                            best_checkpoint_dir = cfg.output_dir / "checkpoints" / "best"
                        else:
                            step_identifier = get_step_identifier(step, cfg.steps)
                            best_checkpoint_dir = cfg.output_dir / "checkpoints" / f"best_{step_identifier}"
                        
                        save_checkpoint(
                            checkpoint_dir=best_checkpoint_dir,
                            step=step,
                            cfg=cfg,
                            policy=accelerator.unwrap_model(policy),
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                        )

                    if wandb_logger:
                        wandb_logger.log_dict({"eval/best_loss": best_eval_loss}, step)

                    logging.info(f"{'=' * 80}\n")

                eval_tracker.reset_averages()
                accelerator.wait_for_everyone()

            # =================================================================
            # Checkpoint Saving
            # =================================================================
            if cfg.save_checkpoint and is_saving_step:
                if is_main_process:
                    logging.info(f"Checkpoint policy after step {step}")

                    if getattr(cfg, "only_last_best", False):
                        checkpoint_dir = cfg.output_dir / "checkpoints" / "last"
                    else:
                        checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

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

                    if not getattr(cfg, "only_last_best", False):
                        update_last_checkpoint(checkpoint_dir)

                accelerator.wait_for_everyone()

    # =========================================================================
    # Training Complete
    # =========================================================================
    if is_main_process:
        logging.info("=" * 80)
        logging.info("  Training Complete!")
        logging.info("=" * 80)
        logging.info(f"  Final step: {step}")
        logging.info(f"  Best eval loss: {best_eval_loss:.4f}")

        if cfg.policy.push_to_hub:
            logging.info("  Pushing model to Hugging Face Hub...")
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()

    if is_main_process:
        logging.info("  Cleanup complete. Exiting.")


def main():
    train()


if __name__ == "__main__":
    main()
