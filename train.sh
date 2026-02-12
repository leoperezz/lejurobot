export CUDA_VISIBLE_DEVICES=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


DATASET_REPO_ID="leoperezz/LejuRobotTask1"
EXPERIMENT_NAME="pi05"
PRETRAINED_PATH="lerobot/pi05_base"
POLICY_TYPE="pi05"
STEPS=20000
WANDB_ENTITY="icra-lejurobot"
WANDB_PROJECT="task1"

#python xhuman/policies/pi05/train_val_pi05.py \
#python scripts/train_val.py \
python lejurobot/scripts/train_val.py \
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
    --batch_size=4 \
    --save_freq=500 \