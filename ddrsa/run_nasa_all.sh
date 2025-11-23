#!/bin/bash

# Run all NASA experiments - Full training (100 epochs)
# This will train all 4 model types with NASA loss function

echo "========================================="
echo "NASA Loss Experiments - Full Training"
echo "========================================="
echo ""
echo "This will run full 100-epoch training for all 4 model types."
echo "Expected time: 2-4 hours (depending on hardware)"
echo ""
echo "Models to train:"
echo "  1. LSTM (exact paper implementation)"
echo "  2. GRU/RNN"
echo "  3. Transformer"
echo "  4. ProbSparse Attention (Informer)"
echo ""
echo "========================================="
echo ""

python run_nasa_experiments.py \
    --data-path ../Challenge_Data \
    --num-epochs 100 \
    --experiments all \
    --summary-file nasa_full_experiments_summary.json \
    --seed 42

echo ""
echo "========================================="
echo "All experiments complete!"
echo "========================================="
echo ""
echo "Generating comparison plots..."
python compare_experiments.py --summary-file nasa_full_experiments_summary.json --output-dir comparison_plots
echo ""
echo "Check results:"
echo "  - Summary: nasa_full_experiments_summary.json"
echo "  - Logs: logs_nasa/"
echo "  - Figures: logs_nasa/*/figures/"
echo "  - Comparison plots: comparison_plots/"
echo "  - TensorBoard: tensorboard --logdir logs_nasa/"
echo ""
