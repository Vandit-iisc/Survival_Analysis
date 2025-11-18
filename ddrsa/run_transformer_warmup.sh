#!/bin/bash

# Train DDRSA Transformer with proper warmup schedule
# This script implements the warmup learning rate schedule that transformers need:
# - Linear warmup for first 4000 steps
# - Cosine decay afterwards

echo "========================================"
echo "Training DDRSA Transformer with Warmup"
echo "========================================"
echo ""
echo "Learning Rate Schedule:"
echo "  - Linear warmup: 0 → 1e-4 over 4000 steps"
echo "  - Cosine decay: 1e-4 → 0 over remaining training"
echo ""

python main.py \
    --model-type transformer \
    --d-model 64 \
    --nhead 4 \
    --num-encoder-layers 2 \
    --num-decoder-layers 2 \
    --dim-feedforward 256 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --lambda-param 0.5 \
    --lookback-window 128 \
    --pred-horizon 100 \
    --num-epochs 200 \
    --warmup-steps 4000 \
    --lr-decay-type cosine \
    --exp-name ddrsa_transformer_warmup \
    --seed 42

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "To visualize training progress:"
echo "  tensorboard --logdir logs/ddrsa_transformer_warmup/tensorboard"
echo ""
echo "To view results:"
echo "  cat logs/ddrsa_transformer_warmup/test_metrics.json"
echo ""
