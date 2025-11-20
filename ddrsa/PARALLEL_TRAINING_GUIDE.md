# Parallel Training and Hyperparameter Search Guide

This guide explains how to use the parallel training system for DDRSA models with automated hyperparameter grid search.

## Overview

The parallel training system allows you to:
- **Run multiple experiments simultaneously** using all available GPUs
- **Test different batch sizes** to find optimal memory/performance tradeoff
- **Grid search over hyperparameters** (learning rate, lambda, NASA weight, dropout)
- **Compare model architectures** (LSTM, GRU, Transformer, ProbSparse)
- **Analyze results** with comprehensive visualizations and summary tables

## Quick Start

### 1. Basic Parallel Training (Default Settings)

Run with default hyperparameter grid on all GPUs:

```bash
cd /Users/vandit/Desktop/vandit/Survival_Analysis/Survival_Analysis/ddrsa

python run_parallel_experiments.py
```

This will:
- Use all available GPUs automatically
- Test batch sizes: [64, 128, 256]
- Test learning rates: [0.0005, 0.001, 0.005]
- Test lambda params: [0.5, 0.75]
- Test NASA weights: [0.0, 0.1]
- Test dropout rates: [0.1, 0.2]
- Run on both datasets (turbofan, azure_pm)
- Include all model types (LSTM, GRU, Transformer, ProbSparse)

**Total experiments**: 2 datasets × 6 model variants × 3 batch sizes × 3 LRs × 2 lambdas × 2 NASA × 2 dropouts = **1,296 experiments**

### 2. Custom Hyperparameter Grid

Specify your own hyperparameter ranges:

```bash
python run_parallel_experiments.py \
  --batch-sizes 32 64 128 256 512 \
  --learning-rates 0.0001 0.001 0.01 \
  --lambda-params 0.5 0.75 0.9 \
  --nasa-weights 0.0 0.1 0.2 0.5 \
  --dropout-rates 0.0 0.1 0.2 0.3 \
  --num-epochs 200
```

### 3. Focus on Batch Size Effects

Test only batch size variations (fixed other hyperparameters):

```bash
python run_parallel_experiments.py \
  --batch-sizes 16 32 64 128 256 512 1024 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir batch_size_study
```

### 4. Single Dataset, Specific Models

Run only on turbofan dataset with transformers and ProbSparse:

```bash
python run_parallel_experiments.py \
  --datasets turbofan \
  --include-lstm \
  --include-transformer \
  --no-include-gru \
  --no-include-probsparse
```

### 5. CPU-Only Training

For systems without GPUs:

```bash
python run_parallel_experiments.py \
  --use-cpu \
  --num-workers 4
```

### 6. Control Number of Parallel Jobs

Limit parallel workers (useful if you have 8 GPUs but want to use only 4):

```bash
python run_parallel_experiments.py \
  --num-workers 4
```

## Complete Command Line Options

### Output Settings
```bash
--output-dir DIR          # Output directory (default: parallel_experiments)
--num-epochs N            # Training epochs per experiment (default: 100)
--seed N                  # Random seed (default: 42)
```

### Parallel Execution
```bash
--num-workers N           # Number of parallel workers (default: number of GPUs)
--use-cpu                 # Force CPU usage even if GPUs available
```

### Dataset Selection
```bash
--datasets turbofan azure_pm    # Which datasets to use (default: both)
```

### Model Selection
```bash
--include-lstm            # Include LSTM variants (default: True)
--include-gru             # Include GRU variants (default: True)
--include-transformer     # Include Transformer variants (default: True)
--include-probsparse      # Include ProbSparse variants (default: True)
```

### Hyperparameter Grids
```bash
--batch-sizes 64 128 256              # Batch sizes to test
--learning-rates 0.0005 0.001 0.005   # Learning rates to test
--lambda-params 0.5 0.75              # Lambda parameters to test
--nasa-weights 0.0 0.1                # NASA loss weights (0.0 = disabled)
--dropout-rates 0.1 0.2               # Dropout rates to test
```

## Example Use Cases

### Use Case 1: Find Optimal Batch Size

**Goal**: Determine the best batch size for each model on your hardware.

```bash
python run_parallel_experiments.py \
  --batch-sizes 16 32 64 128 256 512 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir batch_size_optimization
```

Then analyze:
```bash
python analyze_parallel_results.py --output-dir batch_size_optimization
```

Check `batch_size_optimization/analysis_plots/batch_size_analysis.png` for results.

### Use Case 2: Learning Rate Sensitivity

**Goal**: Find optimal learning rate for each model architecture.

```bash
python run_parallel_experiments.py \
  --batch-sizes 128 \
  --learning-rates 0.00001 0.0001 0.0005 0.001 0.005 0.01 0.05 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir lr_sensitivity
```

### Use Case 3: NASA Loss Weight Tuning

**Goal**: Find the best NASA loss weight for predictive maintenance scoring.

```bash
python run_parallel_experiments.py \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.0 0.05 0.1 0.2 0.3 0.5 1.0 \
  --dropout-rates 0.1 \
  --output-dir nasa_weight_tuning
```

### Use Case 4: Full Hyperparameter Search

**Goal**: Comprehensive grid search to find globally optimal configuration.

```bash
python run_parallel_experiments.py \
  --batch-sizes 64 128 256 \
  --learning-rates 0.0001 0.0005 0.001 0.005 \
  --lambda-params 0.5 0.75 0.9 \
  --nasa-weights 0.0 0.1 0.2 \
  --dropout-rates 0.0 0.1 0.2 \
  --output-dir full_grid_search \
  --num-epochs 150
```

**Warning**: This creates 2 × 6 × 3 × 4 × 3 × 3 × 3 = **3,888 experiments**!

### Use Case 5: Quick Model Comparison

**Goal**: Quickly compare all model architectures with reasonable defaults.

```bash
python run_parallel_experiments.py \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir model_comparison \
  --num-epochs 100
```

This creates only 2 × 6 = **12 experiments** (fast baseline).

## Analyzing Results

After experiments complete, analyze with:

```bash
python analyze_parallel_results.py --output-dir parallel_experiments
```

This generates:

### 1. Summary Statistics
- `summary_statistics.csv`: Mean, std, min, max for each metric by model and dataset

### 2. Best Configurations
- `best_configurations.csv`: Top hyperparameters for each model based on MAE and C-Index

### 3. Visualizations (in `analysis_plots/`)

**Hyperparameter Effects**:
- `rul_mae_vs_hyperparameters.png`: Effect of each hyperparameter on MAE
- `rul_rmse_vs_hyperparameters.png`: Effect of each hyperparameter on RMSE
- `concordance_index_vs_hyperparameters.png`: Effect on C-Index

**Model Comparisons**:
- `model_comparison.png`: Boxplots comparing all models across metrics

**Batch Size Analysis**:
- `batch_size_analysis.png`: 4-panel plot showing:
  - Batch size vs MAE
  - Batch size vs RMSE
  - Batch size vs C-Index
  - Batch size vs Training Time

**Learning Rate Analysis**:
- `learning_rate_analysis.png`: LR effects on all metrics (log scale)

**NASA Loss Impact**:
- `nasa_loss_impact.png`: How NASA weight affects metrics and NASA score

**Heatmaps**:
- `hyperparameter_heatmap.png`: Interaction between batch size and learning rate

## Output Directory Structure

```
parallel_experiments/
├── experiment_manifest.json          # Full experiment configuration
├── all_results.json                  # All experiment results
├── summary_statistics.csv            # Summary stats table
├── best_configurations.csv           # Best hyperparameters for each model
├── logs/                             # Individual experiment logs
│   ├── turbofan/
│   │   ├── lstm_basic/
│   │   │   ├── bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id0/
│   │   │   │   ├── checkpoint_best.pt
│   │   │   │   ├── test_metrics.json
│   │   │   │   └── training_log.json
│   │   │   └── ...
│   │   └── ...
│   └── azure_pm/
│       └── ...
└── analysis_plots/                   # Analysis visualizations
    ├── batch_size_analysis.png
    ├── learning_rate_analysis.png
    ├── nasa_loss_impact.png
    ├── model_comparison.png
    ├── hyperparameter_heatmap.png
    └── ...
```

## Performance Expectations

### Training Time Estimates

**Per experiment** (100 epochs):

| Model | Batch Size | GPU | Time |
|-------|-----------|-----|------|
| LSTM Basic | 128 | RTX 4090 | 5-10 min |
| GRU Basic | 128 | RTX 4090 | 5-10 min |
| Transformer Basic | 128 | RTX 4090 | 20-30 min |
| Transformer Deep | 128 | RTX 4090 | 40-60 min |
| ProbSparse Basic | 128 | RTX 4090 | 30-45 min |
| ProbSparse Deep | 128 | RTX 4090 | 60-90 min |

**Total time for default grid** (1,296 experiments):
- **8× RTX 4090s**: ~30-40 hours
- **4× RTX 4090s**: ~60-80 hours
- **2× RTX 4090s**: ~120-160 hours
- **1× RTX 4090**: ~240-320 hours (10-13 days)

### GPU Memory Requirements

| Batch Size | Memory per Experiment |
|------------|-----------------------|
| 32 | ~180 MB |
| 64 | ~240 MB |
| 128 | ~370 MB |
| 256 | ~600 MB |
| 512 | ~1.2 GB |
| 1024 | ~2.4 GB |

**RTX 4090 (24 GB)**: Can run ~12-15 experiments in parallel with batch_size=128
**RTX 4070 Ti (16 GB)**: Can run ~8-10 experiments in parallel with batch_size=128

## Resource Optimization Tips

### 1. Multi-GPU Strategy

If you have multiple GPUs, the script automatically distributes work:

```bash
# 4 GPUs, each runs 1 experiment at a time
python run_parallel_experiments.py --num-workers 4

# 4 GPUs, run 2 experiments per GPU (if memory allows)
python run_parallel_experiments.py --num-workers 8
```

### 2. Batch Size Selection

**Memory-constrained**:
```bash
--batch-sizes 32 64 128
```

**Performance-focused**:
```bash
--batch-sizes 128 256 512
```

**Maximum throughput**:
```bash
--batch-sizes 256 512 1024  # Only if you have sufficient GPU memory
```

### 3. Staged Hyperparameter Search

**Stage 1**: Coarse search
```bash
python run_parallel_experiments.py \
  --batch-sizes 64 256 \
  --learning-rates 0.0001 0.001 0.01 \
  --lambda-params 0.5 0.9 \
  --output-dir stage1_coarse
```

**Stage 2**: Fine-tune around best from Stage 1
```bash
python run_parallel_experiments.py \
  --batch-sizes 128 256 512 \
  --learning-rates 0.0005 0.001 0.002 \
  --lambda-params 0.7 0.75 0.8 \
  --output-dir stage2_fine
```

## Monitoring Progress

### Check Running Experiments

The script prints progress in real-time:
```
[GPU 0] Starting: turbofan/lstm_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id0
[GPU 1] Starting: turbofan/gru_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id1
[GPU 0] ✓ turbofan/lstm_basic/... completed in 8.3 min
[GPU 0] Starting: turbofan/transformer_basic/...
```

### Check GPU Usage

In another terminal:
```bash
watch -n 1 nvidia-smi
```

### Partial Results

Results are saved as experiments complete. You can analyze partial results:

```bash
python analyze_parallel_results.py --output-dir parallel_experiments
```

This works even while experiments are still running!

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1**: Reduce batch size
```bash
--batch-sizes 32 64
```

**Solution 2**: Reduce number of parallel workers
```bash
--num-workers 2  # Instead of 4
```

**Solution 3**: Skip large models
```bash
--no-include-probsparse  # ProbSparse models use most memory
```

### Some Experiments Failing

Check `all_results.json` for error messages:
```bash
cat parallel_experiments/all_results.json | grep -A 5 "FAILED"
```

### Slow Progress

- **Reduce grid size**: Fewer hyperparameter values
- **Increase workers**: More parallel jobs (if GPU memory allows)
- **Skip expensive models**: Remove ProbSparse or deep transformers

## Best Practices

1. **Start small**: Test with 1-2 model variants and small grid first
2. **Monitor resources**: Watch GPU memory and utilization
3. **Staged search**: Coarse → Fine hyperparameter search
4. **Save intermediate results**: Results are saved continuously
5. **Analyze frequently**: Check `analyze_parallel_results.py` output periodically
6. **Use consistent seeds**: For reproducibility across runs

## Example: Complete Workflow

```bash
# 1. Run batch size study
python run_parallel_experiments.py \
  --batch-sizes 32 64 128 256 512 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir batch_study

# 2. Analyze results
python analyze_parallel_results.py --output-dir batch_study

# 3. Based on analysis, run full grid with optimal batch size
python run_parallel_experiments.py \
  --batch-sizes 128 \
  --learning-rates 0.0001 0.0005 0.001 0.005 \
  --lambda-params 0.5 0.75 0.9 \
  --nasa-weights 0.0 0.1 0.2 \
  --dropout-rates 0.0 0.1 0.2 \
  --output-dir full_search \
  --num-epochs 200

# 4. Final analysis
python analyze_parallel_results.py --output-dir full_search

# 5. Check best configurations
cat full_search/best_configurations.csv
```

## Comparison with Sequential Training

**Sequential** (`run_all_experiments.py`):
- 34 experiments × 30 min/experiment = **17 hours** (1 GPU)
- Simple, straightforward
- Good for initial testing

**Parallel** (`run_parallel_experiments.py`):
- 1,296 experiments ÷ 8 GPUs = **162 parallel batches** × 30 min = **81 hours**
- Comprehensive hyperparameter search
- Better final model performance

## Summary

The parallel training system provides:
- ✅ **Automated hyperparameter search** across all configurations
- ✅ **Efficient GPU utilization** with parallel execution
- ✅ **Comprehensive analysis** with visualizations and tables
- ✅ **Flexible configuration** for different experimental needs
- ✅ **Continuous saving** of results for fault tolerance

Use `run_parallel_experiments.py` for hyperparameter tuning and `run_all_experiments.py` for quick baseline testing.
