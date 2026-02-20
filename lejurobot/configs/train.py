from dataclasses import dataclass
from typing import Optional

from lerobot.configs.train import TrainPipelineConfig


@dataclass
class TrainPipelineConfigLejuRobot(TrainPipelineConfig):
    """
    Base training configuration for LejuRobot (single GPU).
    
    Attributes:
        split_ratio: Fraction of episodes reserved for training (rest is used for eval).
        only_last_best: When True, keeps only 'last' and 'best' checkpoints.
        num_eval_batches: Number of batches to use for evaluation.
        data_augmentation: Enable morphological symmetry data augmentation.
        data_augmentation_temperature: Probability threshold for applying augmentation (0.0-1.0).
    """
    # Train/eval split
    split_ratio: float = 0.8
    
    # Checkpoint management
    only_last_best: bool = True
    
    # Evaluation configuration
    num_eval_batches: int = 50
    
    # Data augmentation configuration
    data_augmentation: bool = False
    data_augmentation_temperature: float = 0.5


@dataclass
class DistributedTrainConfig(TrainPipelineConfigLejuRobot):
    """
    Extended training configuration for distributed (multi-GPU) training with DDP.
    
    Inherits all parameters from TrainPipelineConfigLejuRobot and adds DDP-specific options.
    
    Attributes:
        gradient_accumulation_steps: Number of steps to accumulate gradients before update.
            Effective batch size = batch_size * num_gpus * gradient_accumulation_steps
        sync_batch_norms: Convert BatchNorm to SyncBatchNorm for multi-GPU consistency.
        find_unused_parameters: DDP option to detect unused parameters in backward pass.
            Required for models with conditional computation paths.
        broadcast_buffers: Sync model buffers (like BatchNorm stats) across GPUs.
        enable_profiling: Enable torch.profiler for performance analysis.
        profiling_warmup_steps: Number of warmup steps before profiling starts.
        profiling_active_steps: Number of steps to profile after warmup.
        profiling_output_dir: Directory to save profiling traces (defaults to output_dir/profiling).
    """
    # Gradient Accumulation for larger effective batch sizes
    # Formula: effective_batch = batch_size * world_size * gradient_accumulation_steps
    gradient_accumulation_steps: int = 1
    
    # DDP configuration
    sync_batch_norms: bool = False  # Convert BatchNorm to SyncBatchNorm
    find_unused_parameters: bool = True  # Required for some models with conditional paths
    broadcast_buffers: bool = True  # Sync buffers like BatchNorm running stats
    
    # Profiling configuration (optional, for debugging performance)
    enable_profiling: bool = False
    profiling_warmup_steps: int = 5
    profiling_active_steps: int = 10
    profiling_output_dir: Optional[str] = None
