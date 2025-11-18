#!/bin/bash

# Train DDRSA Transformer with EXACT paper configuration
# This uses all the paper's specifications:
# 1. Warmup learning rate schedule (linear warmup + cosine decay)
# 2. MinMax normalization to [-1, 1] range
# 3. 70/30 train/test split at unit level
# 4. 30% of train units for validation

echo "================================================================"
echo "Training DDRSA Transformer with EXACT Paper Configuration"
echo "================================================================"
echo ""
echo "Data Split (Paper Methodology):"
echo "  - 70% of units (engines) for train/val"
echo "  - 30% of units for test"
echo "  - From train/val, 30% for validation, 70% for training"
echo "  - Final split: ~49% train, ~21% val, ~30% test"
echo ""
echo "Normalization:"
echo "  - MinMaxScaler with range [-1, 1]"
echo ""
echo "Learning Rate Schedule:"
echo "  - Linear warmup: 0 → 1e-4 over 4000 steps"
echo "  - Cosine decay: 1e-4 → 0 over remaining training"
echo ""
echo "Starting training..."
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
    --use-paper-split \
    --use-minmax \
    --exp-name ddrsa_transformer_paper_exact \
    --seed 42

echo ""
echo "================================================================"
echo "Training Complete!"
echo "================================================================"
echo ""
echo "To visualize training progress:"
echo "  tensorboard --logdir logs/ddrsa_transformer_paper_exact/tensorboard"
echo ""
echo "To view results:"
echo "  cat logs/ddrsa_transformer_paper_exact/test_metrics.json"
echo ""
echo "To create hazard rate figures:"
echo "  python create_figures.py --exp-name ddrsa_transformer_paper_exact"
echo ""
