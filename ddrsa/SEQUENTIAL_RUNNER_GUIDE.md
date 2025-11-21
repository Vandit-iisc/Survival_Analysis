# Sequential Experiment Runner Guide

## Overview

The sequential runner executes experiments **one at a time** on a single GPU, showing **live epoch-by-epoch progress** for each experiment. Perfect for when you want full visibility and control.

## Why Sequential?

✅ **Live epoch progress** - See every epoch's loss, MAE, C-Index in real-time
✅ **Easier to debug** - Problems are immediately visible
✅ **No parallel complexity** - Simpler, more reliable
✅ **Full console output** - Complete training logs
✅ **Automatic checkpointing** - Results saved after each experiment

## Quick Start

### Comprehensive Search (Default)
```bash
python run_sequential_experiments.py
```

**Runs**:
- 2 datasets (turbofan, azure_pm)
- 17 model variants
- 5 batch sizes: [32, 64, 128, 256, 512]
- 5 learning rates: [0.0001, 0.0005, 0.001, 0.005, 0.01]
- 4 lambda values: [0.5, 0.6, 0.75, 0.9]
- 4 NASA weights: [0.0, 0.05, 0.1, 0.2]
- 5 dropout rates: [0.0, 0.1, 0.15, 0.2, 0.3]

**Total**: ~34,000 experiments (yes, comprehensive!)
**Time**: ~11,000 hours (~1.3 years on 1 GPU)

### Turbofan Only (Recommended Start)
```bash
python run_sequential_experiments.py --datasets turbofan
```

**Total**: ~17,000 experiments
**Time**: ~5,500 hours (~230 days on 1 GPU)

### Quick Test (Small Grid)
```bash
python run_sequential_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1
```

**Total**: 17 experiments (one per model variant)
**Time**: ~6 hours

### Batch Size Optimization
```bash
python run_sequential_experiments.py \
  --datasets turbofan \
  --batch-sizes 32 64 128 256 512 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir turbofan_batch_study
```

**Total**: 85 experiments (5 batch sizes × 17 models)
**Time**: ~30 hours

### Learning Rate Optimization
```bash
python run_sequential_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.0001 0.0005 0.001 0.005 0.01 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir turbofan_lr_study
```

**Total**: 85 experiments (5 LRs × 17 models)
**Time**: ~30 hours

### Lambda Parameter Optimization
```bash
python run_sequential_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.3 0.5 0.6 0.75 0.9 1.0 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir turbofan_lambda_study
```

**Total**: 102 experiments (6 lambdas × 17 models)
**Time**: ~35 hours

### NASA Loss Optimization
```bash
python run_sequential_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.0 0.05 0.1 0.15 0.2 0.3 0.5 \
  --dropout-rates 0.1 \
  --output-dir turbofan_nasa_study
```

**Total**: 119 experiments (7 NASA weights × 17 models)
**Time**: ~40 hours

### Dropout Optimization
```bash
python run_sequential_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 \
  --output-dir turbofan_dropout_study
```

**Total**: 153 experiments (9 dropout rates × 17 models)
**Time**: ~50 hours

## What You See (Live Output)

```bash
$ python run_sequential_experiments.py --datasets turbofan --batch-sizes 128

================================================================================
SEQUENTIAL DDRSA EXPERIMENT RUNNER
================================================================================

Configuration:
  Datasets: ['turbofan']
  Batch sizes: [128]
  Learning rates: [0.0001, 0.0005, 0.001, 0.005, 0.01]
  Lambda params: [0.5, 0.6, 0.75, 0.9]
  NASA weights: [0.0, 0.05, 0.1, 0.2]
  Dropout rates: [0.0, 0.1, 0.15, 0.2, 0.3]
  Epochs per experiment: 100
  Output directory: sequential_experiments
================================================================================

Generated 3400 experiments

Estimated time: ~1133.3 hours (47.2 days)
================================================================================

================================================================================
EXPERIMENT 1/3400
================================================================================
Name: turbofan/lstm_paper_exact/bs128_lr0.0001_lam0.5_nasa0.0_drop0.0_id0
Dataset: turbofan
Model: lstm_paper_exact
Batch size: 128
Learning rate: 0.0001
Lambda: 0.5
NASA weight: 0.0
Dropout: 0.0
================================================================================

Using device: cuda
Loading data...
Train batches: 45
Val batches: 12
Test batches: 15
Input dimension: 24

Creating RNN model...
Model created: DDRSA_RNN
Total parameters: 12,345

================================================================================
DDRSA Experiment Configuration
================================================================================
model_type                : rnn
batch_size                : 128
learning_rate             : 0.0001
lambda_param              : 0.5
...
================================================================================

================================================================================
Starting Training
================================================================================

Epoch 1/100
----------
Train Loss: 2.3456 | Val Loss: 2.4567 | RUL MAE: 15.23 | RMSE: 18.45 | C-Index: 0.723
⏱ Epoch time: 3.2s

Epoch 2/100
----------
Train Loss: 2.1234 | Val Loss: 2.2345 | RUL MAE: 14.56 | RMSE: 17.89 | C-Index: 0.745
⏱ Epoch time: 3.1s

Epoch 3/100
----------
Train Loss: 1.9876 | Val Loss: 2.1234 | RUL MAE: 13.89 | RMSE: 17.23 | C-Index: 0.767
⏱ Epoch time: 3.0s

...

Epoch 45/100
----------
Train Loss: 0.8765 | Val Loss: 1.2345 | RUL MAE: 8.45 | RMSE: 11.23 | C-Index: 0.892
✓ New best model saved! (C-Index improved: 0.887 -> 0.892)
⏱ Epoch time: 3.1s

...

Early stopping triggered at epoch 55 (patience: 10)

================================================================================
Final Test Results
================================================================================
rul_mae                       : 8.23
rul_rmse                      : 11.05
concordance_index             : 0.895
expected_tte                  : 67.34
nasa_score                    : 245.67
================================================================================

================================================================================
✓ Experiment completed successfully in 4.5 minutes
================================================================================

================================================================================
EXPERIMENT 2/3400
================================================================================
Name: turbofan/lstm_paper_exact/bs128_lr0.0001_lam0.5_nasa0.05_drop0.0_id1
...
```

## Live Monitoring Features

### ✅ Experiment Header
- Experiment number (X/Total)
- Full configuration details
- Progress tracking
- ETA calculation

### ✅ Live Training Progress
- **Every epoch** shows:
  - Train loss
  - Validation loss
  - RUL MAE
  - RUL RMSE
  - Concordance Index
  - Epoch duration

### ✅ Best Model Tracking
- Notifications when new best model is saved
- Shows improvement metrics

### ✅ Early Stopping
- Shows when early stopping triggers
- Final epoch count

### ✅ Final Results
- Complete test metrics
- Experiment duration

### ✅ Overall Progress
- Completed experiments count
- Average time per experiment
- Total elapsed time
- Estimated time remaining (ETA)

## Stopping and Resuming

### Stop Gracefully
Press `Ctrl+C` during training. The current experiment will **complete** before stopping.

### Resume (Manual)
Currently, you need to:
1. Check `all_results.json` to see which experiments completed
2. Rerun with a different output directory
3. Or manually skip completed experiments

## Comparison: Sequential vs Parallel

| Feature | Sequential | Parallel (4 GPUs) |
|---------|-----------|-------------------|
| **Speed** | 1× | 4× faster |
| **Live epoch progress** | ✅ Yes | ❌ No |
| **Console output** | ✅ Full | ⚠️ Limited |
| **Debugging** | ✅ Easy | ⚠️ Complex |
| **Setup** | ✅ Simple | ⚠️ Requires tmux/screen |
| **GPU utilization** | 1 GPU | 4 GPUs |
| **Monitoring** | ✅ Built-in | Requires separate script |

## Recommended Workflow

### Stage 1: Quick Test (1-2 hours)
```bash
python run_sequential_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1
```
**Purpose**: Verify everything works, get baseline results

### Stage 2: Batch Size Optimization (~30 hours)
```bash
python run_sequential_experiments.py \
  --datasets turbofan \
  --batch-sizes 32 64 128 256 512 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir turbofan_batch_study
```
**Purpose**: Find optimal batch size

### Stage 3: Learning Rate Tuning (~30 hours)
```bash
# Use best batch size from Stage 2
python run_sequential_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.0001 0.0005 0.001 0.005 0.01 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir turbofan_lr_study
```
**Purpose**: Find optimal learning rate

### Stage 4: Fine-Grained Search (~100 hours)
```bash
# Use best batch size and LR from Stages 2 & 3
python run_sequential_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.5 0.6 0.75 0.9 \
  --nasa-weights 0.0 0.05 0.1 0.2 \
  --dropout-rates 0.0 0.1 0.15 0.2 0.3 \
  --output-dir turbofan_fine_tune
```
**Purpose**: Optimize lambda, NASA weight, dropout

### Stage 5: Best Model Final Training (2-3 hours)
```bash
# Use best config from all stages
python main.py \
  --dataset turbofan \
  --model-type transformer \
  --batch-size 128 \
  --learning-rate 0.001 \
  --lambda-param 0.75 \
  --nasa-weight 0.1 \
  --dropout 0.15 \
  --num-epochs 300 \
  --create-visualization \
  --exp-name final_production_model
```
**Purpose**: Production model with full epochs and visualizations

## Tips

### 1. Use tmux/screen for Long Runs
```bash
tmux new -s experiments
python run_sequential_experiments.py --datasets turbofan
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t experiments
```

### 2. Monitor from Another Terminal
```bash
# Watch progress
tail -f sequential_experiments/all_results.json

# Count completed
ls sequential_experiments/logs/*/*/*/*/test_metrics.json | wc -l
```

### 3. Start Small, Scale Up
Don't run the full grid first! Start with quick tests, then expand.

### 4. Save Logs
```bash
python run_sequential_experiments.py 2>&1 | tee experiment_log.txt
```

## Output Structure

```
sequential_experiments/
├── experiment_manifest.json       # All experiment configs
├── all_results.json              # Execution results (updated after each exp)
├── summary_statistics.csv        # Analysis results
├── best_configurations.csv
├── top_20_by_mae.csv
├── top_20_by_cindex.csv
├── top_20_by_nasa_score.csv
├── analysis_plots/               # All visualizations
│   ├── batch_size_analysis.png
│   ├── learning_rate_analysis.png
│   ├── lambda_parameter_analysis.png
│   ├── nasa_loss_impact.png
│   ├── dropout_analysis.png
│   └── ...
└── logs/                         # Individual experiment results
    └── turbofan/
        └── lstm_basic/
            └── bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id0/
                ├── checkpoint_best.pt
                ├── test_metrics.json
                └── training_log.json
```

## Summary

**Use sequential runner when you want**:
- ✅ Live epoch-by-epoch progress visibility
- ✅ Easy debugging and monitoring
- ✅ Simple, reliable execution
- ✅ Full console output
- ✅ Don't have multiple GPUs

**Perfect for**:
- Comprehensive hyperparameter search
- Batch size optimization
- Learning rate tuning
- Lambda/NASA/Dropout optimization
- Single GPU setups

**Start with**:
```bash
python run_sequential_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1
```

Then expand the grid based on initial results!
