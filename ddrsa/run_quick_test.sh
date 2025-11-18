#!/bin/bash

# Quick test run - 5 epochs to verify everything works
# This should complete in 2-5 minutes

echo "========================================="
echo "DDRSA Quick Test Run"
echo "========================================="
echo ""
echo "This will run a quick 5-epoch test to verify"
echo "that everything is working correctly."
echo ""

DATA_PATH="../Challenge_Data"

python main.py \
    --data-path $DATA_PATH \
    --model-type rnn \
    --rnn-type LSTM \
    --hidden-dim 16 \
    --num-layers 1 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --lambda-param 0.5 \
    --lookback-window 128 \
    --pred-horizon 100 \
    --num-epochs 5 \
    --exp-name quick_test \
    --seed 42

echo ""
echo "========================================="
echo "Quick test complete!"
echo "========================================="
echo ""
echo "Check results:"
echo "  - Logs: logs/quick_test/"
echo "  - Metrics: logs/quick_test/test_metrics.json"
echo "  - TensorBoard: tensorboard --logdir logs/"
echo ""
echo "To run full training (100 epochs):"
echo "  python main.py --num-epochs 100 --exp-name full_training"
