#!/bin/bash
# The projector at initialisation, after one gradient step of one batch.
# Otherwise identical protocol to ft_ph/mh/mg, so the comparison is not about
# hyperparameters — it is about "does the fine-tune add measurable capability
# over an essentially untrained projector on the same backbone."
cd /home/ubuntu/Isaac-GR00T-benchmark || exit 1
export GR00T_VIDEO_BACKEND=pyav TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf /home/ubuntu/ft_step0
.venv/bin/python gr00t/experiment/launch_finetune_lowmem.py \
  --base-model-path /home/ubuntu/models/GR00T-N1.7-3B \
  --dataset-path /home/ubuntu/lift_train_ph \
  --embodiment-tag new_embodiment \
  --modality-config-path /home/ubuntu/lift_modality.py \
  --num-gpus 1 --output-dir /home/ubuntu/ft_step0 \
  --max-steps 1 --save-steps 1 --save-total-limit 1 --seed 42 \
  --learning-rate 1e-4 \
  --global-batch-size 2 --gradient-accumulation-steps 8 \
  --warmup-ratio 0.05 --weight-decay 1e-5 \
  --dataloader-num-workers 2 \
  --shard-size 1024 --episode-sampling-rate 1.0 --num-shards-per-epoch 100000 \
  --no-tune-llm --no-tune-visual --tune-projector --no-tune-diffusion-model \
  --save-only-model
echo "STEP0_TRAIN_EXIT=$?"
