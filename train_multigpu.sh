#!/bin/bash
# Multi-GPU Training Script for PI05 Policy
# 
# This script uses Hugging Face Accelerate to launch distributed training
# across multiple GPUs with gradient accumulation for larger effective batch sizes.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Multi-GPU Configuration
# ============================================================================
NUM_GPUS=2  # Adjust according to your hardware
GRADIENT_ACCUMULATION_STEPS=4  # Increases the effective batch size
BATCH_SIZE_PER_GPU=2  # Batch size per GPU

# ============================================================================
# Dataset and Experiment Configuration
# ============================================================================
DATASET_REPO_ID="leoperezz/LejuRobotTask1"
EXPERIMENT_NAME="pi05"
PRETRAINED_PATH="lerobot/pi05_base"
POLICY_TYPE="pi05"
STEPS=20000
WANDB_ENTITY="icra-lejurobot"
WANDB_PROJECT="task1"

# ============================================================================
# Calculate effective batch size
# Effective BS = BATCH_SIZE_PER_GPU × NUM_GPUS × GRADIENT_ACCUMULATION_STEPS
# Example: 4 × 4 × 8 = 128
# ============================================================================

echo "Configuration:"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "  Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Effective batch size: $((BATCH_SIZE_PER_GPU * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo ""

# ============================================================================
# Launch Multi-GPU Training with Accelerate
# ============================================================================
accelerate launch \
    --mixed_precision=bf16 \
    --multi_gpu \
    --num_processes=${NUM_GPUS} \
    lejurobot/scripts/train_val_multigpu.py \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --dataset.repo_id=${DATASET_REPO_ID} \
    --policy.type=${POLICY_TYPE} \
    --output_dir=./outputs/${EXPERIMENT_NAME} \
    --job_name=${EXPERIMENT_NAME} \
    --policy.repo_id=leoperezz/${EXPERIMENT_NAME} \
    --policy.pretrained_path=${PRETRAINED_PATH} \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.scheduler_decay_steps=${STEPS} \
    --policy.num_inference_steps=5 \
    --policy.optimizer_lr=1.5e-5 \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.entity=${WANDB_ENTITY} \
    --wandb.project=${WANDB_PROJECT} \
    --policy.dtype=bfloat16 \
    --steps=${STEPS} \
    --eval_freq=200 \
    --log_freq=200 \
    --split_ratio=0.9 \
    --batch_size=${BATCH_SIZE_PER_GPU} \
    --save_freq=500 \
    --num_workers=4 \
    --sync_batch_norms=true