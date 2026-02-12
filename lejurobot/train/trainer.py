#!/usr/bin/env python

"""
Trainer class for LejuRobot policies.

This module provides a flexible Trainer class that encapsulates the training loop
and can be extended with different update strategies for different policies.
"""

import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler

from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
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

from lejurobot.configs.train import TrainPipelineConfigLejuRobot
from lejurobot.datasets.factory import make_dataset_lejurobot
from lejurobot.datasets.utils import split_train_eval_episodes
from lejurobot.logger import logger
from lejurobot.policies.factory import make_lejurobot_policy, make_lejurobot_pre_post_processors
from lejurobot.train.strategies_ckpt import (
    CheckpointManager,
    EvalBatchStats,
    create_default_checkpoint_strategies,
)
from lejurobot.train.strategies_ckpt import CheckpointStrategy


# =============================================================================
# Distributed Sampler for Episode-Aware Sampling
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


class Trainer:
    """
    Flexible trainer for LejuRobot policies.
    
    This class encapsulates the training loop and can be extended to support
    different update strategies for different policies. The base implementation
    follows the standard training loop from flowact/train_val.py.
    
    Attributes:
        cfg: Training pipeline configuration.
        accelerator: Accelerator for distributed training and mixed precision.
        is_main_process: Whether this is the main process (for logging).
        device: Device to use for training.
        policy: The policy model to train.
        optimizer: Optimizer for updating policy parameters.
        lr_scheduler: Learning rate scheduler.
        preprocessor: Preprocessor for input data.
        postprocessor: Postprocessor for output data.
        train_dataloader: DataLoader for training data.
        eval_dataloader: DataLoader for evaluation data.
        checkpoint_manager: Manager for checkpoint saving strategies.
        wandb_logger: Weights & Biases logger (optional).
        step: Current training step.
    """
    
    def __init__(
        self,
        cfg: TrainPipelineConfigLejuRobot,
        accelerator: Optional[Accelerator] = None,
        use_distributed: bool = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            cfg: Training pipeline configuration.
            accelerator: Optional Accelerator instance. If None, one will be created.
            use_distributed: Whether to enable distributed training features. If None,
                automatically detected based on number of processes.
        """
        self.cfg = cfg
        self.cfg.validate()
        
        # Create or use provided accelerator
        if accelerator is None:
            # Get distributed config parameters if available
            gradient_accumulation_steps = getattr(cfg, 'gradient_accumulation_steps', 1)
            find_unused_params = getattr(cfg, 'find_unused_parameters', True)
            broadcast_buffers = getattr(cfg, 'broadcast_buffers', True)
            
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=find_unused_params,
                broadcast_buffers=broadcast_buffers,
            )
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                step_scheduler_with_optimizer=False,
                kwargs_handlers=[ddp_kwargs]
            )
        else:
            self.accelerator = accelerator
        
        # Determine if we should use distributed features
        if use_distributed is None:
            # Auto-detect: use distributed if we have multiple processes
            self.use_distributed = self.accelerator.num_processes > 1
        else:
            self.use_distributed = use_distributed
        
        # Initialize logging
        init_logging(accelerator=self.accelerator)
        self.is_main_process = self.accelerator.is_main_process
        
        # Log configuration
        if self.is_main_process:
            logger.info(pformat(self.cfg.to_dict()))
        
        # Initialize wandb
        self._init_wandb()
        
        # Set random seed
        if self.cfg.seed is not None:
            set_seed(self.cfg.seed, accelerator=self.accelerator)
        
        # Setup device
        self.device = self.accelerator.device
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Initialize components (will be set in setup())
        self.dataset = None
        self.eval_dataset = None
        self.policy = None
        self.optimizer = None
        self.lr_scheduler = None
        self.preprocessor = None
        self.postprocessor = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.checkpoint_manager = None
        self.step = 0
        
        # Metrics trackers
        self.train_tracker = None
        self.eval_tracker = None
        
        # Loss key for checkpoint strategies
        self.loss_key = None
        
        # Distributed training state
        self.train_sampler = None
        self.eval_sampler = None
        self.current_epoch = 0
    
    def _init_wandb(self):
        """Initialize Weights & Biases logger."""
        if self.cfg.wandb.enable and self.cfg.wandb.project and self.is_main_process:
            self.wandb_logger = WandBLogger(self.cfg)
        else:
            self.wandb_logger = None
            if self.is_main_process:
                logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    
    def setup(self, loss_key: str = "loss", strategies: list[CheckpointStrategy] = None):
        """
        Setup all components for training.
        
        This includes:
        - Loading datasets
        - Creating policy
        - Creating optimizer and scheduler
        - Creating dataloaders
        - Preparing everything with accelerator
        - Initializing checkpoint manager
        """
        # Load datasets
        self._load_datasets()
        
        # Create policy
        self._create_policy()
        
        # Create optimizer and scheduler
        self._create_optimizer()
        
        # Create dataloaders
        self._create_dataloaders()
        
        # Prepare with accelerator
        self._prepare_with_accelerator()
        
        # Initialize checkpoint manager
        self._init_checkpoint_manager(loss_key, strategies)
        
        # Initialize metrics trackers
        self._init_metrics_trackers()
        
        # Log training info
        self._log_training_info()
    
    def _load_dataset_with_episodes(self, episodes: list[int]):
        """Helper to load dataset with specific episodes."""
        cfg = self.cfg
        cfg.dataset.episodes = episodes
        
        if self.is_main_process:
            dataset = make_dataset_lejurobot(cfg)
        
        self.accelerator.wait_for_everyone()
        
        if not self.is_main_process:
            dataset = make_dataset_lejurobot(cfg)
        
        return dataset

    def _load_datasets(self):
        """Load training and evaluation datasets."""
        if self.is_main_process:
            logger.info("Creating dataset")
            dataset = make_dataset_lejurobot(self.cfg)
        
        self.accelerator.wait_for_everyone()
        
        if not self.is_main_process:
            dataset = make_dataset_lejurobot(self.cfg)
        
        episodes = list(range(dataset.meta.total_episodes))
        train_episodes, eval_episodes = split_train_eval_episodes(
            episodes,
            split_ratio=self.cfg.split_ratio,
            seed=42
        )
        
        del dataset
        
        # Load split datasets
        if self.is_main_process:
            logger.info(f"Loading train dataset with {len(train_episodes)} episodes")
            logger.info(f"Loading eval dataset with {len(eval_episodes)} episodes")
        
        self.dataset = self._load_dataset_with_episodes(train_episodes)
        self.eval_dataset = self._load_dataset_with_episodes(eval_episodes)
    
    def _create_policy(self):
        """Create policy model."""
        if self.is_main_process:
            logger.info("Creating policy")
        
        self.policy = make_lejurobot_policy(
            cfg=self.cfg.policy,
            ds_meta=self.dataset.meta,
        )
        
        # Convert BatchNorm to SyncBatchNorm if using distributed training
        sync_batch_norms = getattr(self.cfg, 'sync_batch_norms', False)
        if self.use_distributed and sync_batch_norms:
            self.policy = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.policy)
            if self.is_main_process:
                logger.info("Converted BatchNorm to SyncBatchNorm for multi-GPU training")
        
        self.accelerator.wait_for_everyone()
        
        # Create processors
        processor_kwargs = {}
        postprocessor_kwargs = {}
        
        if (self.cfg.policy.pretrained_path and not self.cfg.resume) or not self.cfg.policy.pretrained_path:
            processor_kwargs["dataset_stats"] = self.dataset.meta.stats
        
        if self.cfg.policy.pretrained_path is not None:
            processor_kwargs["preprocessor_overrides"] = {
                "device_processor": {"device": self.device.type},
                "normalizer_processor": {
                    "stats": self.dataset.meta.stats,
                    "features": {**self.policy.config.input_features, **self.policy.config.output_features},
                    "norm_map": self.policy.config.normalization_mapping,
                },
            }
            postprocessor_kwargs["postprocessor_overrides"] = {
                "unnormalizer_processor": {
                    "stats": self.dataset.meta.stats,
                    "features": self.policy.config.output_features,
                    "norm_map": self.policy.config.normalization_mapping,
                },
            }
        
        self.preprocessor, self.postprocessor = make_lejurobot_pre_post_processors(
            policy_cfg=self.cfg.policy,
            pretrained_path=self.cfg.policy.pretrained_path,
            **processor_kwargs,
            **postprocessor_kwargs,
        )
    
    def _create_optimizer(self):
        """Create optimizer and learning rate scheduler."""
        if self.is_main_process:
            logger.info("Creating optimizer and scheduler")
        
        self.optimizer, self.lr_scheduler = make_optimizer_and_scheduler(self.cfg, self.policy)
        
        # Load training state if resuming
        if self.cfg.resume:
            self.step, self.optimizer, self.lr_scheduler = load_training_state(
                self.cfg.checkpoint_path,
                self.optimizer,
                self.lr_scheduler
            )
    
    def _create_dataloaders(self):
        """Create training and evaluation dataloaders."""
        world_size = self.accelerator.num_processes
        local_rank = self.accelerator.local_process_index
        
        # Determine sampler configuration
        if hasattr(self.cfg.policy, "drop_n_last_frames"):
            if self.is_main_process:
                logger.info(f"Dropping {self.cfg.policy.drop_n_last_frames} last frames")
            shuffle = False
            
            # Get episode indices
            if hasattr(self.dataset, 'get_episode_data_index_for_sampler'):
                train_from_indices, train_to_indices = self.dataset.get_episode_data_index_for_sampler()
            else:
                train_from_indices = self.dataset.meta.episodes["dataset_from_index"]
                train_to_indices = self.dataset.meta.episodes["dataset_to_index"]
            
            if hasattr(self.eval_dataset, 'get_episode_data_index_for_sampler'):
                eval_from_indices, eval_to_indices = self.eval_dataset.get_episode_data_index_for_sampler()
            else:
                eval_from_indices = self.eval_dataset.meta.episodes["dataset_from_index"]
                eval_to_indices = self.eval_dataset.meta.episodes["dataset_to_index"]
            
            # Use distributed sampler if enabled
            if self.use_distributed:
                if self.is_main_process:
                    logger.info(f"Using DistributedEpisodeAwareSampler for {world_size} processes")
                
                train_sampler = DistributedEpisodeAwareSampler(
                    self.dataset,
                    train_from_indices,
                    train_to_indices,
                    drop_n_last_frames=self.cfg.policy.drop_n_last_frames,
                    shuffle=True,
                    num_replicas=world_size,
                    rank=local_rank,
                    seed=self.cfg.seed or 0,
                )
                
                eval_sampler = DistributedEpisodeAwareSampler(
                    self.eval_dataset,
                    eval_from_indices,
                    eval_to_indices,
                    drop_n_last_frames=self.cfg.policy.drop_n_last_frames,
                    shuffle=False,
                    num_replicas=world_size,
                    rank=local_rank,
                    seed=self.cfg.seed or 0,
                )
            else:
                train_sampler = EpisodeAwareSampler(
                    train_from_indices,
                    train_to_indices,
                    drop_n_last_frames=self.cfg.policy.drop_n_last_frames,
                    shuffle=True,
                )
                
                eval_sampler = EpisodeAwareSampler(
                    eval_from_indices,
                    eval_to_indices,
                    drop_n_last_frames=self.cfg.policy.drop_n_last_frames,
                    shuffle=False,
                )
        else:
            if self.is_main_process:
                logger.info("Not dropping any frames")
            
            # Use standard distributed sampler if enabled
            if self.use_distributed:
                if self.is_main_process:
                    logger.info(f"Using DistributedSampler for {world_size} processes")
                
                train_sampler = DistributedSampler(
                    self.dataset,
                    num_replicas=world_size,
                    rank=local_rank,
                    shuffle=True,
                    seed=self.cfg.seed or 0,
                )
                
                eval_sampler = DistributedSampler(
                    self.eval_dataset,
                    num_replicas=world_size,
                    rank=local_rank,
                    shuffle=False,
                    seed=self.cfg.seed or 0,
                )
                shuffle = False
            else:
                train_sampler = None
                eval_sampler = None
                shuffle = True
        
        # Store samplers for epoch setting in distributed mode
        self.train_sampler = train_sampler
        self.eval_sampler = eval_sampler
        
        # Determine drop_last and persistent_workers based on distributed mode
        drop_last = self.use_distributed  # Important for DDP to ensure equal batch sizes
        persistent_workers = self.use_distributed and self.cfg.num_workers > 0
        
        # Create train dataloader
        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle and not self.cfg.dataset.streaming,
            sampler=train_sampler,
            pin_memory=self.device.type == "cuda",
            drop_last=drop_last,
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
            persistent_workers=persistent_workers,
        )
        
        # Create eval dataloader
        self.eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            sampler=eval_sampler,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
        )
    
    def _prepare_with_accelerator(self):
        """Prepare all components with accelerator."""
        self.accelerator.wait_for_everyone()
        (
            self.policy,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.policy,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.eval_dataloader,
        )
    
    def _init_checkpoint_manager(self, loss_key: str = "loss", strategies: list[CheckpointStrategy] = None):
        """Initialize checkpoint manager with default strategies."""
        # Store loss_key for use in evaluation
        self.loss_key = loss_key
        
        if strategies is None:
            strategies = create_default_checkpoint_strategies(
                loss_key=loss_key,
                lambda_penalty=1.0,
                percentile=90.0,
            )
        self.checkpoint_manager = CheckpointManager(strategies=strategies)
    
    def _init_metrics_trackers(self):
        """Initialize metrics trackers for training and evaluation."""
        # Initialize only basic training metrics (non-policy specific)
        # All policy-specific metrics (including loss) come from output_dict
        train_metrics = {
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
        
        # Initialize only basic eval metrics
        # All policy-specific metrics (including loss) come from output_dict
        eval_metrics = {
            "update_s": AverageMeter("eval_s", ":.3f"),
        }
        
        # Calculate effective batch size accounting for gradient accumulation
        gradient_accumulation_steps = getattr(self.cfg, 'gradient_accumulation_steps', 1)
        effective_batch_size = self.cfg.batch_size * self.accelerator.num_processes * gradient_accumulation_steps
        
        self.train_tracker = MetricsTracker(
            effective_batch_size,
            self.dataset.num_frames,
            self.dataset.num_episodes,
            train_metrics,
            initial_step=self.step,
            accelerator=self.accelerator,
        )
        
        self.eval_tracker = MetricsTracker(
            effective_batch_size,
            self.eval_dataset.num_frames,
            self.eval_dataset.num_episodes,
            eval_metrics,
            initial_step=self.step,
            accelerator=self.accelerator,
        )
        
        # Track dynamically added metrics
        self.dynamic_train_metrics = set()
        self.dynamic_eval_metrics = set()
    
    def _add_dynamic_metric(self, metric_name: str, is_eval: bool = False):
        """
        Dynamically add a new metric to the appropriate tracker.
        
        Args:
            metric_name: Name of the metric to add.
            is_eval: Whether this is an evaluation metric.
        """
        if is_eval:
            if metric_name not in self.dynamic_eval_metrics and metric_name not in self.eval_tracker.metrics:
                # Create a short name for display with eval prefix
                prefix = "eval_"
                max_metric_chars = 8 - len(prefix)
                truncated_name = metric_name[:max_metric_chars] if len(metric_name) > max_metric_chars else metric_name
                short_name = f"{prefix}{truncated_name}"
                self.eval_tracker.metrics[metric_name] = AverageMeter(short_name, ":.3f")
                self.dynamic_eval_metrics.add(metric_name)
        else:
            if metric_name not in self.dynamic_train_metrics and metric_name not in self.train_tracker.metrics:
                # Create a short name for display
                short_name = metric_name[:8] if len(metric_name) > 8 else metric_name
                self.train_tracker.metrics[metric_name] = AverageMeter(short_name, ":.3f")
                self.dynamic_train_metrics.add(metric_name)
    
    def _log_training_info(self):
        """Log information about training setup."""
        if not self.is_main_process:
            return
        
        num_learnable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in self.policy.parameters())
        
        logger.info("=" * 80)
        logger.info(f"  Training Mode: {'DISTRIBUTED' if self.use_distributed else 'SINGLE-GPU'}")
        logger.info("=" * 80)
        logger.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {self.cfg.output_dir}")
        if self.cfg.env is not None:
            logger.info(f"{self.cfg.env.task=}")
        logger.info(f"{self.cfg.steps=} ({format_big_number(self.cfg.steps)})")
        logger.info(f"{self.dataset.num_frames=} ({format_big_number(self.dataset.num_frames)})")
        logger.info(f"{self.dataset.num_episodes=}")
        
        # Calculate effective batch size
        num_processes = self.accelerator.num_processes
        gradient_accumulation_steps = getattr(self.cfg, 'gradient_accumulation_steps', 1)
        effective_bs = self.cfg.batch_size * num_processes * gradient_accumulation_steps
        
        if self.use_distributed:
            logger.info(f"World Size (num GPUs):       {num_processes}")
            logger.info(f"Batch Size per GPU:          {self.cfg.batch_size}")
            logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
            logger.info(f"Effective Batch Size:        {effective_bs}")
            logger.info(f"Mixed Precision:             {self.accelerator.mixed_precision}")
        else:
            logger.info(f"Batch Size: {self.cfg.batch_size}")
            logger.info(f"Effective Batch Size: {effective_bs}")
        
        logger.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logger.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
        logger.info("=" * 80)
    
    def update_policy(
        self,
        batch: Any,
        lock=None,
    ) -> Tuple[dict, dict]:
        """
        Perform a single training step to update the policy.
        
        This method can be overridden by subclasses to implement different
        update strategies for different policies.
        
        Args:
            batch: A batch of training data (already preprocessed).
            lock: Optional lock for thread-safe optimizer updates.
            
        Returns:
            Tuple of (metrics_dict, output_dict) where:
            - metrics_dict contains training metrics (loss, grad_norm, lr, etc.)
            - output_dict contains policy outputs for logging
        """
        start_time = time.perf_counter()
        self.policy.train()
        
        # Forward pass with mixed precision
        with self.accelerator.autocast():
            loss, output_dict = self.policy.forward(batch)
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Gradient clipping
        if self.cfg.optimizer.grad_clip_norm > 0:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.policy.parameters(),
                self.cfg.optimizer.grad_clip_norm
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                float("inf"),
                error_if_nonfinite=False
            )
        
        # Optimizer step
        with lock if lock is not None else nullcontext():
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Learning rate scheduler step
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Update policy internal buffers if needed
        if has_method(self.accelerator.unwrap_model(self.policy, keep_fp32_wrapper=True), "update"):
            self.accelerator.unwrap_model(self.policy, keep_fp32_wrapper=True).update()
        
        # Collect metrics (non-policy specific only)
        # Policy-specific metrics (including loss) should be in output_dict
        metrics_dict = {
            "grad_norm": grad_norm.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "update_s": time.perf_counter() - start_time,
        }
        
        return metrics_dict, output_dict
    
    def evaluate_policy(self, num_eval_batches: Optional[int] = None) -> Tuple[dict, dict, EvalBatchStats]:
        """
        Evaluate the policy on the evaluation dataset.
        
        Args:
            num_eval_batches: Number of batches to evaluate on. If None, uses cfg.num_eval_batches.
            
        Returns:
            Tuple of (metrics_dict, output_dict, batch_stats) where:
            - metrics_dict contains basic evaluation metrics (time, etc.)
            - output_dict contains aggregated policy outputs (including loss and other metrics)
            - batch_stats contains per-batch statistics (used by checkpoint strategies)
        """
        if num_eval_batches is None:
            num_eval_batches = self.cfg.num_eval_batches
        
        start_time = time.perf_counter()
        self.policy.eval()
        
        batch_losses = []
        output_dicts = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.eval_dataloader):
                if i >= num_eval_batches:
                    break
                
                batch = self.preprocessor(batch)
                
                # Forward pass with mixed precision
                with self.accelerator.autocast():
                    loss, output_dict = self.policy.forward(batch)
                
                # Use loss from output_dict with the specified loss_key, otherwise use returned loss
                if output_dict and self.loss_key in output_dict:
                    batch_losses.append(output_dict[self.loss_key])
                else:
                    batch_losses.append(loss.item())
                
                output_dicts.append(output_dict)
        
        # Create batch statistics from loss values (used by checkpoint strategies)
        batch_stats = EvalBatchStats(losses=batch_losses)
        
        # Aggregate output_dicts
        aggregated_output = {}
        if output_dicts:
            for key in output_dicts[0].keys():
                values = [d[key] for d in output_dicts if key in d]
                if values and isinstance(values[0], (int, float)):
                    aggregated_output[key] = sum(values) / len(values)
        
        # Collect metrics (non-policy specific only)
        # Policy-specific metrics (including loss) should be in output_dict
        metrics_dict = {
            "update_s": time.perf_counter() - start_time,
        }
        
        return metrics_dict, aggregated_output, batch_stats
    
    def train(self):
        """
        Main training loop.
        
        This method runs the full training loop, including:
        - Data loading
        - Policy updates
        - Periodic evaluation
        - Checkpoint saving
        - Logging to terminal and WandB
        """
        if self.is_main_process:
            logger.info("Start offline training on a fixed dataset")
        
        # Set initial epoch for distributed samplers
        if self.use_distributed and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(self.current_epoch)
        
        # Create infinite iterator over training data
        dl_iter = cycle(self.train_dataloader)
        
        # Track forward passes for epoch calculation in distributed mode
        forward_step = 0
        
        # Training loop
        for _ in range(self.step, self.cfg.steps):
            # Load batch
            start_time = time.perf_counter()
            batch = next(dl_iter)
            batch = self.preprocessor(batch)
            dataloading_time = time.perf_counter() - start_time
            
            # Track forward steps for distributed epoch setting
            forward_step += 1
            
            # Update epoch for distributed sampler if needed
            if self.use_distributed and hasattr(self.train_sampler, "set_epoch"):
                world_size = self.accelerator.num_processes
                # Calculate new epoch based on dataset coverage
                new_epoch = forward_step * self.cfg.batch_size * world_size // self.dataset.num_frames
                if new_epoch > self.current_epoch:
                    self.current_epoch = new_epoch
                    self.train_sampler.set_epoch(self.current_epoch)
            
            # Update policy
            metrics_dict, output_dict = self.update_policy(batch)
            
            # Update training tracker with basic metrics
            self.train_tracker.grad_norm = metrics_dict["grad_norm"]
            self.train_tracker.lr = metrics_dict["lr"]
            self.train_tracker.update_s = metrics_dict["update_s"]
            self.train_tracker.dataloading_s = dataloading_time
            
            # Update training tracker with all policy metrics from output_dict
            if output_dict:
                for key, value in output_dict.items():
                    if isinstance(value, (int, float)):
                        # Add metric if it doesn't exist
                        self._add_dynamic_metric(key, is_eval=False)
                        # Update the metric value
                        setattr(self.train_tracker, key, value)
            
            # Increment step
            self.step += 1
            self.train_tracker.step()
            
            # Check if logging/evaluation steps
            is_log_step = self.cfg.log_freq > 0 and self.step % self.cfg.log_freq == 0 and self.is_main_process
            is_eval_step = self.cfg.eval_freq > 0 and self.step % self.cfg.eval_freq == 0
            
            # Log to WandB at every step
            if self.wandb_logger and self.is_main_process:
                wandb_log_dict = {
                    "train/grad_norm": self.train_tracker.grad_norm.val,
                    "train/lr": self.train_tracker.lr.val,
                    "train/update_s": self.train_tracker.update_s.val,
                    "train/dataloading_s": self.train_tracker.dataloading_s.val,
                }
                # Add all policy metrics from output_dict with train/ prefix
                if output_dict:
                    for key, value in output_dict.items():
                        if isinstance(value, (int, float)):
                            wandb_log_dict[f"train/{key}"] = value
                self.wandb_logger.log_dict(wandb_log_dict, self.step)
            
            # Log to terminal at log_freq intervals
            if is_log_step:
                logger.info(self.train_tracker)
                self.train_tracker.reset_averages()
            
            # Evaluation step
            if is_eval_step:
                self._run_evaluation()
        
        # End of training
        if self.is_main_process:
            logger.info("End of training")
            
            if self.cfg.policy.push_to_hub:
                unwrapped_policy = self.accelerator.unwrap_model(self.policy)
                unwrapped_policy.push_model_to_hub(self.cfg)
                self.preprocessor.push_to_hub(self.cfg.policy.repo_id)
                self.postprocessor.push_to_hub(self.cfg.policy.repo_id)
        
        # Cleanup
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
    
    def _run_evaluation(self):
        """Run evaluation and handle checkpoint saving."""
        if self.is_main_process:
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating policy at step {self.step}")
            logger.info(f"{'='*80}")
        
        # Set epoch for eval sampler if distributed
        if self.use_distributed and hasattr(self.eval_sampler, "set_epoch"):
            self.eval_sampler.set_epoch(self.current_epoch)
        
        # Run evaluation
        metrics_dict, eval_output_dict, batch_stats = self.evaluate_policy()
        
        # Update eval tracker with basic metrics
        self.eval_tracker.update_s = metrics_dict["update_s"]
        
        # Update eval tracker with all policy metrics from output_dict
        if eval_output_dict:
            for key, value in eval_output_dict.items():
                if isinstance(value, (int, float)):
                    # Add metric if it doesn't exist
                    self._add_dynamic_metric(key, is_eval=True)
                    # Update the metric value
                    setattr(self.eval_tracker, key, value)
        
        if self.is_main_process:
            # Log detailed statistics
            self._log_eval_results(batch_stats, eval_output_dict)
            
            # Update checkpoint strategies and save if needed
            self._handle_checkpoint_saving(batch_stats)
            
            logger.info(f"{'='*80}\n")
        
        self.eval_tracker.reset_averages()
        self.accelerator.wait_for_everyone()
    
    def _log_eval_results(self, batch_stats: EvalBatchStats, eval_output_dict: dict):
        """Log evaluation results to terminal and WandB."""
        # Log to terminal
        logger.info(f"\n{'-'*80}")
        logger.info(f"EVALUATION RESULTS AT STEP {self.step}:")
        logger.info(f"{'-'*80}")
        
        # Log policy-specific metrics (all from output_dict)
        if eval_output_dict:
            logger.info(f"  📈 POLICY METRICS:")
            for key, value in eval_output_dict.items():
                if isinstance(value, (int, float)):
                    logger.info(f"     {key}: {value:.6f}")
        
        logger.info(f"  ⏱️  Eval Time:      {self.eval_tracker.update_s.avg:.3f}s")
        
        # Calculate scores for each strategy
        scores = {}
        logger.info(f"\n  🎯 CHECKPOINT SCORES:")
        for strategy_name, strategy in self.checkpoint_manager.strategies_dict.items():
            score = strategy.compute_score(batch_stats)
            scores[strategy_name] = score
            logger.info(f"     {strategy_name}: {score:.6f}")
        
        # Log to WandB
        if self.wandb_logger:
            eval_wandb_log_dict = {
                "eval/time_s": self.eval_tracker.update_s.avg,
            }
            
            # Add strategy scores
            for strategy_name, score in scores.items():
                eval_wandb_log_dict[f"eval/score_{strategy_name}"] = score
            
            # Add all policy metrics from output_dict with eval/ prefix
            if eval_output_dict:
                for key, value in eval_output_dict.items():
                    if isinstance(value, (int, float)):
                        eval_wandb_log_dict[f"eval/{key}"] = value
            
            self.wandb_logger.log_dict(eval_wandb_log_dict, self.step)
    
    def _handle_checkpoint_saving(self, batch_stats: EvalBatchStats):
        """Update checkpoint strategies and save checkpoints if needed."""
        # Update all strategies
        results = self.checkpoint_manager.update_all(batch_stats, self.step)
        
        # Save checkpoints for strategies that improved
        for strategy_name, (is_new_best, score) in results.items():
            if is_new_best:
                logger.info(f"\n{'*'*80}")
                logger.info(f"🎉 NEW BEST for {strategy_name.upper()}: {score:.6f}")
                logger.info(f"{'*'*80}")
                logger.info(f"Saving {strategy_name} checkpoint...")
                
                checkpoint_dir = Path(self.cfg.output_dir) / "checkpoints" / strategy_name
                self._save_checkpoint(checkpoint_dir)
        
        # Log best scores to WandB
        if self.wandb_logger:
            best_scores_log = {}
            for strategy_name, strategy in self.checkpoint_manager.strategies_dict.items():
                best_scores_log[f"eval/best_{strategy_name}"] = strategy.best_score
            self.wandb_logger.log_dict(best_scores_log, self.step)
        
        # Log best checkpoints summary
        logger.info(f"\n  📁 BEST CHECKPOINTS SO FAR:")
        for strategy_name, strategy in self.checkpoint_manager.strategies_dict.items():
            logger.info(f"     {strategy_name}: {strategy.best_score:.6f} (step {strategy.best_step})")
    
    def _save_checkpoint(self, checkpoint_dir: Path):
        """Save checkpoint to specified directory."""
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=self.step,
            cfg=self.cfg,
            policy=self.accelerator.unwrap_model(self.policy),
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
        )
