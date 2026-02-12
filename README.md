# Lejurobot Challenge

## Setup

Create conda environment:
```bash
conda create -n lejurobot python=3.11
conda activate lejurobot
```

Install package:
```bash
pip install .
```

Install FFmpeg (required for torchcodec video backend):
```bash
conda install -n lejurobot -c conda-forge "ffmpeg<8" -y
```

**Note:** FFmpeg version 8+ is not compatible with torchcodec. Use version 7 or earlier (the command above installs FFmpeg 6.1.1 which is compatible).

## Data

Download the dataset from:
https://huggingface.co/datasets/LejuRobotics/kuavo_data_challenge_icra

## Training

The Trainer class supports both single-GPU and distributed multi-GPU training with automatic detection.

### Quick Start Examples

**Example 1: Single-GPU Training**
```yaml
# config_single_gpu.yaml
dataset:
  repo_id: "LejuRobotics/kuavo_data_challenge_icra"
  # ... dataset config

policy:
  name: "act"
  # ... policy config

optimizer:
  name: "adamw"
  lr: 1e-4
  weight_decay: 0.01

steps: 50000
batch_size: 32
log_freq: 100
eval_freq: 500
save_freq: 5000
split_ratio: 0.8
num_eval_batches: 50
```

**Example 2: Multi-GPU Training (4 GPUs)**
```yaml
# config_multi_gpu.yaml
dataset:
  repo_id: "LejuRobotics/kuavo_data_challenge_icra"
  # ... dataset config

policy:
  name: "act"
  # ... policy config

optimizer:
  name: "adamw"
  lr: 1e-4
  weight_decay: 0.01

steps: 50000
batch_size: 8  # Per-GPU batch size
log_freq: 100
eval_freq: 500
save_freq: 5000
split_ratio: 0.8
num_eval_batches: 50

# Multi-GPU specific settings
gradient_accumulation_steps: 4  # Effective batch = 8 × 4 GPUs × 4 = 128
sync_batch_norms: true
find_unused_parameters: true
broadcast_buffers: true
```

Run with:
```bash
# Single GPU
python lejurobot/scripts/train_val.py --config config_single_gpu.yaml

# Multi-GPU (4 GPUs)
accelerate launch --multi_gpu --num_processes=4 lejurobot/scripts/train_val.py --config config_multi_gpu.yaml
```

### Single-GPU Training (Classic Mode)

Use the standard `TrainPipelineConfigLejuRobot` configuration:

```python
from lejurobot.configs.train import TrainPipelineConfigLejuRobot
from lejurobot.train.trainer import Trainer

# Create config
cfg = TrainPipelineConfigLejuRobot(
    dataset=...,
    policy=...,
    optimizer=...,
    steps=10000,
    batch_size=32,
    log_freq=100,
    eval_freq=500,
    # ... other parameters
)

# Create and setup trainer (auto-detects single-GPU mode)
trainer = Trainer(cfg)
trainer.setup(loss_key="loss")

# Start training
trainer.train()
```

Or run from command line:
```bash
# Single GPU
python -m lejurobot.scripts.train_val --config path/to/config.yaml

# Or directly
python lejurobot/scripts/train_val.py --config path/to/config.yaml
```

### Multi-GPU Distributed Training

Use the `DistributedTrainConfig` for multi-GPU with additional DDP options:

```python
from lejurobot.configs.train import DistributedTrainConfig
from lejurobot.train.trainer import Trainer

# Create distributed config
cfg = DistributedTrainConfig(
    dataset=...,
    policy=...,
    optimizer=...,
    steps=10000,
    batch_size=32,  # Per-GPU batch size
    log_freq=100,
    eval_freq=500,
    # Distributed-specific options
    gradient_accumulation_steps=2,  # Effective batch = batch_size * num_gpus * grad_accum
    sync_batch_norms=True,  # Use SyncBatchNorm for multi-GPU consistency
    find_unused_parameters=True,  # DDP option for models with conditional paths
    broadcast_buffers=True,  # Sync BatchNorm running stats across GPUs
    # ... other parameters
)

# Create and setup trainer (auto-detects multi-GPU mode)
trainer = Trainer(cfg)
trainer.setup(loss_key="loss")

# Start training
trainer.train()
```

Launch with accelerate for multi-GPU:
```bash
# Using 4 GPUs (simple method)
accelerate launch --multi_gpu --num_processes=4 lejurobot/scripts/train_val.py --config path/to/config.yaml

# Or create an accelerate config file for more control
accelerate config  # Follow prompts to configure multi-GPU settings
accelerate launch lejurobot/scripts/train_val.py --config path/to/config.yaml

# Example accelerate config for 4 GPUs with mixed precision
cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4
mixed_precision: bf16
gpu_ids: all
EOF

# Launch with custom config
accelerate launch --config_file accelerate_config.yaml lejurobot/scripts/train_val.py --config path/to/config.yaml
```

**Note:** The Trainer automatically detects if it's running in distributed mode by checking `accelerator.num_processes > 1`. You don't need to modify your training script - just use `accelerate launch` to enable multi-GPU.

### How It Works: Auto-Detection

The Trainer automatically detects distributed mode:

1. **Single-GPU Mode** (`num_processes == 1`):
   - Standard samplers and dataloaders
   - Normal BatchNorm
   - No special distributed handling

2. **Multi-GPU Mode** (`num_processes > 1`):
   - Distributed samplers that split data across GPUs
   - Optional SyncBatchNorm for better multi-GPU consistency
   - Automatic epoch tracking for proper shuffling
   - Optimized DataLoader settings

You can also manually control this:
```python
# Force distributed mode (not recommended)
trainer = Trainer(cfg, use_distributed=True)

# Force single-GPU mode (not recommended)
trainer = Trainer(cfg, use_distributed=False)
```

### Key Differences Between Modes

| Feature | Single-GPU | Multi-GPU (Distributed) |
|---------|-----------|------------------------|
| **Sampler** | `EpisodeAwareSampler` or standard | `DistributedEpisodeAwareSampler` or `DistributedSampler` |
| **BatchNorm** | Standard BatchNorm | Optional `SyncBatchNorm` (if `sync_batch_norms=True`) |
| **DataLoader** | `drop_last=False`, no `persistent_workers` | `drop_last=True`, `persistent_workers=True` |
| **Effective Batch Size** | `batch_size` | `batch_size × num_gpus × gradient_accumulation_steps` |
| **Epoch Setting** | Not needed | Automatically updates sampler epoch for proper shuffling |
| **Detection** | Automatic (`num_processes == 1`) | Automatic (`num_processes > 1`) |

### Configuration Parameters

**Common Parameters (TrainPipelineConfigLejuRobot):**
- `split_ratio`: Train/eval split ratio (default: 0.8)
- `only_last_best`: Keep only last and best checkpoints (default: True)
- `num_eval_batches`: Number of batches for evaluation (default: 50)

**Distributed-Only Parameters (DistributedTrainConfig):**
- `gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `sync_batch_norms`: Convert BatchNorm to SyncBatchNorm (default: False)
- `find_unused_parameters`: DDP unused parameter detection (default: True)
- `broadcast_buffers`: Sync model buffers across GPUs (default: True)
- `enable_profiling`: Enable torch.profiler (default: False)
- `profiling_warmup_steps`: Profiler warmup steps (default: 5)
- `profiling_active_steps`: Profiler active steps (default: 10)

### Example: Episode-Aware Sampling with Multi-GPU

When using policies with `drop_n_last_frames` (e.g., ACT, Diffusion), the trainer automatically uses episode-aware sampling that respects episode boundaries:

```python
cfg = DistributedTrainConfig(
    policy=ACTPolicyConfig(
        drop_n_last_frames=8,  # Drop last 8 frames from each episode
        ...
    ),
    ...
)

# Trainer will automatically:
# - Use DistributedEpisodeAwareSampler in multi-GPU mode
# - Use EpisodeAwareSampler in single-GPU mode
# - Properly distribute episodes across GPUs
# - Set epochs for proper shuffling in distributed mode
```

### Best Practices for Multi-GPU Training

1. **Batch Size**: Set `batch_size` to the per-GPU batch size. The effective batch size will be `batch_size × num_gpus × gradient_accumulation_steps`.

2. **Gradient Accumulation**: Use `gradient_accumulation_steps` to increase effective batch size without running out of memory:
   ```python
   cfg = DistributedTrainConfig(
       batch_size=8,  # Per GPU
       gradient_accumulation_steps=4,  # Accumulate 4 steps
       # With 4 GPUs: effective batch = 8 × 4 × 4 = 128
   )
   ```

3. **SyncBatchNorm**: Enable for models with BatchNorm layers to ensure consistency across GPUs:
   ```python
   cfg = DistributedTrainConfig(
       sync_batch_norms=True,  # Recommended for models with BatchNorm
   )
   ```

4. **Find Unused Parameters**: Keep enabled if your model has conditional computation:
   ```python
   cfg = DistributedTrainConfig(
       find_unused_parameters=True,  # Safe default, disable for speed if not needed
   )
   ```

5. **Number of Workers**: Scale with number of GPUs:
   ```python
   cfg = DistributedTrainConfig(
       num_workers=4,  # Per GPU (e.g., 4 GPUs × 4 workers = 16 total workers)
   )
   ```

### Using the Multi-GPU Training Script

For convenience, we provide a pre-configured shell script for multi-GPU training with PI05 policy:

**File:** `train_multigpu.sh`

#### Prerequisites

1. **Configure Accelerate** (one-time setup):
```bash
accelerate config
```

When prompted, a typical multi-GPU setup looks like this:
- **Type of machine**: `multi-GPU` (or equivalent option)
- **Number of machines**: choose based on your setup (often `1`)
- **Check distributed operations for errors**: `NO` (recommended for speed)
- **Use torch dynamo**: `NO`
- **Use DeepSpeed**: `NO`
- **Use FullyShardedDataParallel (FSDP)**: `NO`
- **Use Megatron-LM**: `NO`
- **Number of GPU(s) for distributed training**: set according to your hardware (e.g. `2`, `4`, ...)
- **GPU IDs to use**: `all` (or a comma-separated list of GPU IDs)
- **Enable NUMA efficiency**: `yes` (recommended on NVIDIA hardware)
- **Mixed precision**: `bf16` (recommended; use `fp16` only if needed)

2. **Verify configuration:**
```bash
accelerate env
```

#### Configuration

Edit the script variables according to your setup:

```bash
# Multi-GPU Configuration
NUM_GPUS=4                      # Number of GPUs available
GRADIENT_ACCUMULATION_STEPS=8   # Gradient accumulation for larger effective batch size
BATCH_SIZE_PER_GPU=4            # Batch size per GPU

# Dataset and Experiment
DATASET_REPO_ID="your/dataset"
EXPERIMENT_NAME="pi05"
PRETRAINED_PATH="lerobot/pi05_base"
POLICY_TYPE="pi05"
STEPS=20000
WANDB_ENTITY="your-entity"
WANDB_PROJECT="your-project"
```

#### Understanding Effective Batch Size

The effective batch size is calculated as:
```
Effective Batch Size = BATCH_SIZE_PER_GPU × NUM_GPUS × GRADIENT_ACCUMULATION_STEPS
```

**Example configurations:**

| GPUs | Batch/GPU | Grad Accum | Effective BS | Memory (~GB) | Recommendation |
|------|-----------|------------|--------------|--------------|----------------|
| 2    | 4         | 8          | 64           | ~16GB        | ✅ Minimum acceptable |
| 2    | 4         | 16         | 128          | ~16GB        | ⭐ Recommended |
| 4    | 4         | 4          | 64           | ~16GB        | ✅ Good |
| 4    | 4         | 8          | 128          | ~16GB        | ⭐ Optimal |
| 4    | 8         | 4          | 128          | ~24GB        | ⭐ If you have VRAM |

For **PI05 (Diffusion Policy)**, an effective batch size of **64-128** is recommended for stable training.

#### Running the Script

1. **Make the script executable:**
```bash
chmod +x train_multigpu.sh
```

2. **Run the training:**
```bash
./train_multigpu.sh
```

The script will display the configuration before starting:
```
Configuration:
  GPUs: 4
  Batch size per GPU: 4
  Gradient accumulation steps: 8
  Effective batch size: 128
```

#### Adjusting for Memory Constraints

If you encounter CUDA out-of-memory errors:

1. **Reduce batch size per GPU:**
```bash
BATCH_SIZE_PER_GPU=2
```

2. **Increase gradient accumulation to maintain effective batch size:**
```bash
GRADIENT_ACCUMULATION_STEPS=16  # Now: 2 × 4 × 16 = 128
```

3. **Enable memory-efficient CUDA allocation:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### Learning Rate Considerations

The learning rate may need adjustment based on your effective batch size:

- **Default LR:** `1.5e-5` (conservative, suitable for small batches)
- **For effective BS 128:** Consider `3e-5` to `5e-5`
- **Maximum for diffusion models:** `1e-4`

To adjust, edit the script:
```bash
--policy.optimizer_lr=3e-5 \  # Increased for larger batch size
```

#### Monitoring Training

The script automatically logs to:
- **WandB:** Real-time metrics and visualizations
- **Local logs:** Terminal output with training progress
- **Checkpoints:** Saved to `./outputs/${EXPERIMENT_NAME}/checkpoints/`

**Log frequency:**
- Training metrics: Every 200 steps (configurable with `--log_freq`)
- Evaluation: Every 200 steps (configurable with `--eval_freq`)
- Checkpoints: Every 500 steps (configurable with `--save_freq`)

#### Advanced Options

**Enable profiling for performance analysis:**
```bash
--enable_profiling=true \
--profiling_warmup_steps=5 \
--profiling_active_steps=10 \
--profiling_output_dir=./profiling \
```

**Adjust evaluation:**
```bash
--num_eval_batches=100 \  # Number of batches to evaluate
--eval_freq=100 \          # Evaluate more frequently
```

**Memory optimization:**
```bash
--policy.gradient_checkpointing=true \  # Already enabled
--num_workers=2 \                       # Reduce if CPU memory is limited
```

### Troubleshooting

**Issue: "CUDA out of memory" in multi-GPU mode**
- Reduce `batch_size` (per-GPU batch size)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `num_workers`

**Issue: "RuntimeError: Expected to have finished reduction in the prior iteration"**
- Set `find_unused_parameters=True` in config
- Check if your model has conditional computation paths

**Issue: Different results between single-GPU and multi-GPU**
- Enable `sync_batch_norms=True` if using BatchNorm
- Ensure `seed` is set for reproducibility
- Check that effective batch size is the same

**Issue: Poor GPU utilization**
- Increase `num_workers` (try 4-8 per GPU)
- Enable `persistent_workers=True` (automatic in multi-GPU mode)
- Increase `prefetch_factor` in DataLoader if needed

## Resources

- Official competition repository: https://github.com/LejuRobotics/kuavo_data_challenge/tree/icra
- Simulator: https://github.com/LejuRobotics/kuavo-ros-opensource/tree/opensource/kuavo-data-challenge-icra
