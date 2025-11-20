#!/bin/bash
# Quick launcher for common parallel experiment scenarios

set -e

echo "=========================================="
echo "DDRSA Parallel Experiment Quick Launcher"
echo "=========================================="
echo ""
echo "Select an experiment to run:"
echo ""
echo "1. Batch Size Study (Small - ~72 experiments, ~6-12 hours on 4 GPUs)"
echo "2. Learning Rate Sweep (Medium - ~216 experiments, ~18-36 hours on 4 GPUs)"
echo "3. NASA Loss Weight Tuning (Small - ~72 experiments, ~6-12 hours on 4 GPUs)"
echo "4. Full Grid Search (Large - ~1,296 experiments, ~80-160 hours on 4 GPUs)"
echo "5. Quick Model Comparison (Tiny - ~12 experiments, ~1-2 hours on 4 GPUs)"
echo "6. Dropout Sensitivity (Small - ~96 experiments, ~8-16 hours on 4 GPUs)"
echo "7. Custom (enter your own parameters)"
echo "8. Exit"
echo ""

read -p "Enter choice [1-8]: " choice

case $choice in
    1)
        echo "Running Batch Size Study..."
        python run_parallel_experiments.py \
            --batch-sizes 32 64 128 256 512 \
            --learning-rates 0.001 \
            --lambda-params 0.75 \
            --nasa-weights 0.1 \
            --dropout-rates 0.1 \
            --output-dir batch_size_study \
            --num-epochs 100

        echo ""
        echo "✓ Complete! Check batch_size_study/analysis_plots/"
        ;;

    2)
        echo "Running Learning Rate Sweep..."
        python run_parallel_experiments.py \
            --batch-sizes 128 \
            --learning-rates 0.00001 0.0001 0.0005 0.001 0.005 0.01 \
            --lambda-params 0.75 \
            --nasa-weights 0.1 \
            --dropout-rates 0.1 \
            --output-dir lr_sweep \
            --num-epochs 100

        echo ""
        echo "✓ Complete! Check lr_sweep/analysis_plots/"
        ;;

    3)
        echo "Running NASA Loss Weight Tuning..."
        python run_parallel_experiments.py \
            --batch-sizes 128 \
            --learning-rates 0.001 \
            --lambda-params 0.75 \
            --nasa-weights 0.0 0.05 0.1 0.2 0.3 0.5 \
            --dropout-rates 0.1 \
            --output-dir nasa_tuning \
            --num-epochs 100

        echo ""
        echo "✓ Complete! Check nasa_tuning/analysis_plots/"
        ;;

    4)
        echo "WARNING: This will run ~1,296 experiments and may take 80-160+ hours!"
        read -p "Are you sure? (yes/no): " confirm

        if [ "$confirm" = "yes" ]; then
            echo "Running Full Grid Search..."
            python run_parallel_experiments.py \
                --batch-sizes 64 128 256 \
                --learning-rates 0.0001 0.0005 0.001 0.005 \
                --lambda-params 0.5 0.75 0.9 \
                --nasa-weights 0.0 0.1 0.2 \
                --dropout-rates 0.0 0.1 0.2 \
                --output-dir full_grid_search \
                --num-epochs 150

            echo ""
            echo "✓ Complete! Check full_grid_search/analysis_plots/"
        else
            echo "Cancelled."
        fi
        ;;

    5)
        echo "Running Quick Model Comparison..."
        python run_parallel_experiments.py \
            --batch-sizes 128 \
            --learning-rates 0.001 \
            --lambda-params 0.75 \
            --nasa-weights 0.1 \
            --dropout-rates 0.1 \
            --output-dir quick_comparison \
            --num-epochs 100

        echo ""
        echo "✓ Complete! Check quick_comparison/analysis_plots/"
        ;;

    6)
        echo "Running Dropout Sensitivity Study..."
        python run_parallel_experiments.py \
            --batch-sizes 128 \
            --learning-rates 0.001 \
            --lambda-params 0.75 \
            --nasa-weights 0.1 \
            --dropout-rates 0.0 0.05 0.1 0.15 0.2 0.3 0.4 0.5 \
            --output-dir dropout_study \
            --num-epochs 100

        echo ""
        echo "✓ Complete! Check dropout_study/analysis_plots/"
        ;;

    7)
        echo "Custom Experiment Configuration"
        echo ""

        read -p "Output directory name: " output_dir
        read -p "Batch sizes (space-separated, e.g., 64 128 256): " batch_sizes
        read -p "Learning rates (space-separated, e.g., 0.001 0.005): " learning_rates
        read -p "Lambda params (space-separated, e.g., 0.5 0.75): " lambda_params
        read -p "NASA weights (space-separated, e.g., 0.0 0.1): " nasa_weights
        read -p "Dropout rates (space-separated, e.g., 0.1 0.2): " dropout_rates
        read -p "Number of epochs (e.g., 100): " num_epochs

        echo "Running custom experiment..."
        python run_parallel_experiments.py \
            --batch-sizes $batch_sizes \
            --learning-rates $learning_rates \
            --lambda-params $lambda_params \
            --nasa-weights $nasa_weights \
            --dropout-rates $dropout_rates \
            --output-dir "$output_dir" \
            --num-epochs $num_epochs

        echo ""
        echo "✓ Complete! Check $output_dir/analysis_plots/"
        ;;

    8)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
