#!/bin/bash

# Quick test run for NASA experiments - 5 epochs to verify everything works
# This should complete in 10-15 minutes (testing all 4 models)

echo "========================================="
echo "NASA Loss Experiments - Quick Test"
echo "========================================="
echo ""
echo "This will run a quick 5-epoch test for all 4 model types"
echo "to verify that everything is working correctly."
echo ""
echo "Models to test:"
echo "  1. LSTM (exact paper implementation)"
echo "  2. GRU/RNN"
echo "  3. Transformer"
echo "  4. ProbSparse Attention (Informer)"
echo ""
echo "========================================="
echo ""

python run_nasa_experiments.py \
    --data-path ../Challenge_Data \
    --num-epochs 5 \
    --experiments all \
    --summary-file nasa_quick_test_summary.json \
    --seed 42

echo ""
echo "========================================="
echo "Quick test complete!"
echo "========================================="
echo ""
echo "Check results:"
echo "  - Summary: nasa_quick_test_summary.json"
echo "  - Logs: logs_nasa/"
echo "  - Figures: logs_nasa/*/figures/"
echo "  - TensorBoard: tensorboard --logdir logs_nasa/"
echo ""
echo "Generate comparison plots:"
echo "  python compare_experiments.py --summary-file nasa_quick_test_summary.json"
echo ""
echo "To run full training (100 epochs):"
echo "  bash run_nasa_all.sh"
echo ""
