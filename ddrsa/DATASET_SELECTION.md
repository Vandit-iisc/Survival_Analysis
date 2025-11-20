# Dataset Selection Guide

## Overview

You can run experiments on specific datasets using the `--datasets` argument.

## Quick Examples

### Turbofan Only
```bash
python run_parallel_experiments.py \
  --datasets turbofan \
  --output-dir turbofan_only
```

### Azure PM Only
```bash
python run_parallel_experiments.py \
  --datasets azure_pm \
  --output-dir azure_pm_only
```

### Both Datasets (Default)
```bash
python run_parallel_experiments.py \
  --datasets turbofan azure_pm
# OR just omit --datasets (both is default)
python run_parallel_experiments.py
```

## Complete Examples

### Example 1: Turbofan Batch Size Study
```bash
python run_parallel_experiments.py \
  --datasets turbofan \
  --batch-sizes 32 64 128 256 512 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir turbofan_batch_study
```

**Creates**: 1 dataset × 6 models × 5 batch sizes = **30 experiments**
**Time**: ~2-4 hours on 4 GPUs (half the time of both datasets!)

---

### Example 2: Azure PM Full Grid Search
```bash
python run_parallel_experiments.py \
  --datasets azure_pm \
  --batch-sizes 64 128 256 \
  --learning-rates 0.0005 0.001 0.005 \
  --lambda-params 0.5 0.6 0.75 \
  --nasa-weights 0.0 0.05 0.1 \
  --dropout-rates 0.1 0.15 0.2 \
  --output-dir azure_pm_comprehensive
```

**Creates**: 1 dataset × 6 models × 3 batch × 3 LR × 3 lambda × 3 NASA × 3 dropout = **729 experiments**
**Time**: ~40-80 hours on 4 GPUs

---

### Example 3: Quick Model Comparison - Turbofan Only
```bash
python run_parallel_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir turbofan_quick_test
```

**Creates**: 1 dataset × 6 models = **6 experiments**
**Time**: ~30-60 minutes on 4 GPUs

---

## Why Run Single Dataset?

### ✅ Faster Experimentation
- **Half the experiments** = half the time
- Quick iteration during development
- Faster hyperparameter tuning

### ✅ Dataset-Specific Optimization
- Find optimal hyperparameters for specific dataset
- Some hyperparameters may work differently on different datasets
- Focus on the dataset you care about most

### ✅ Resource Management
- Save GPU time and electricity
- Run on smaller hardware
- Easier to debug and monitor

### ✅ Sequential Workflow
```bash
# Step 1: Optimize on turbofan first
python run_parallel_experiments.py \
  --datasets turbofan \
  --output-dir turbofan_optimization

# Step 2: Check results
cat turbofan_optimization/best_configurations.csv

# Step 3: Test best configs on azure_pm
python run_parallel_experiments.py \
  --datasets azure_pm \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --output-dir azure_pm_validation
```

---

## Dataset Characteristics

### Turbofan Dataset (NASA C-MAPSS)
- **Source**: NASA Prognostics Data Repository
- **Type**: Aircraft engine degradation
- **Size**: ~260 engines, ~20,000 cycles
- **Features**: 24 sensor readings
- **Difficulty**: Moderate
- **Use Case**: Aerospace predictive maintenance

**Typical Performance**:
- MAE: 8-15 (good models)
- C-Index: 0.85-0.92 (good models)
- Training time: Faster (smaller dataset)

---

### Azure PM Dataset (Microsoft)
- **Source**: Microsoft Azure AI Gallery
- **Type**: Industrial equipment failures
- **Size**: ~100 machines, ~876,000 records
- **Features**: 24 derived features (after preprocessing)
- **Difficulty**: Harder (more complex patterns)
- **Use Case**: General industrial predictive maintenance

**Typical Performance**:
- MAE: 12-20 (good models)
- C-Index: 0.75-0.85 (good models)
- Training time: Slower (larger dataset)

---

## Impact on Experiment Count

### Default Grid (Both Datasets)
```python
datasets = ['turbofan', 'azure_pm']  # 2
models = 6
batch_sizes = [64, 128, 256]  # 3
learning_rates = [0.0005, 0.001, 0.005]  # 3
lambda_params = [0.5, 0.6, 0.75]  # 3
nasa_weights = [0.0, 0.05, 0.1]  # 3
dropout_rates = [0.1, 0.15, 0.2]  # 3

Total = 2 × 6 × 3 × 3 × 3 × 3 × 3 = 1,458 experiments
```

### Turbofan Only
```python
datasets = ['turbofan']  # 1

Total = 1 × 6 × 3 × 3 × 3 × 3 × 3 = 729 experiments (50% reduction!)
```

### Azure PM Only
```python
datasets = ['azure_pm']  # 1

Total = 1 × 6 × 3 × 3 × 3 × 3 × 3 = 729 experiments (50% reduction!)
```

---

## Analysis Differences

### Both Datasets
When you run on both datasets, analysis includes:
- Dataset comparison plot (`dataset_comparison.png`)
- Side-by-side performance metrics
- Generalization insights

### Single Dataset
When you run on one dataset:
- Dataset comparison plot is skipped (message shown)
- All other analyses work the same
- Focused on single-dataset optimization

---

## Command Patterns

### Pattern 1: Focus Then Expand
```bash
# 1. Quick test on turbofan
python run_parallel_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --output-dir turbofan_quick

# 2. Full search on turbofan
python run_parallel_experiments.py \
  --datasets turbofan \
  --output-dir turbofan_full

# 3. Validate on azure_pm with best configs
python run_parallel_experiments.py \
  --datasets azure_pm \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --output-dir azure_pm_validate
```

---

### Pattern 2: Parallel Dataset-Specific Optimization
```bash
# Terminal 1: Optimize for turbofan
python run_parallel_experiments.py \
  --datasets turbofan \
  --num-workers 2 \
  --output-dir turbofan_optimized

# Terminal 2: Optimize for azure_pm
python run_parallel_experiments.py \
  --datasets azure_pm \
  --num-workers 2 \
  --output-dir azure_pm_optimized
```

---

### Pattern 3: Dataset-Specific Model Selection
```bash
# Test if transformers are better for turbofan
python run_parallel_experiments.py \
  --datasets turbofan \
  --include-transformer \
  --no-include-lstm \
  --no-include-gru \
  --output-dir turbofan_transformers_only

# Test if LSTMs are better for azure_pm
python run_parallel_experiments.py \
  --datasets azure_pm \
  --include-lstm \
  --no-include-transformer \
  --no-include-gru \
  --output-dir azure_pm_lstms_only
```

---

## Interactive Menu Option

You can also add dataset selection to the interactive menu:

```bash
$ ./quick_experiments.sh
# Select option 7 (Custom)

Output directory name: turbofan_test
Datasets (turbofan, azure_pm, or both): turbofan
Batch sizes: 128
Learning rates: 0.001
Lambda params: 0.75
NASA weights: 0.1
Dropout rates: 0.1
Number of epochs: 100
```

**Note**: Current version of quick_experiments.sh doesn't have dataset selection yet, but you can add it to option 7 if needed.

---

## Summary

### Turbofan Only
```bash
python run_parallel_experiments.py --datasets turbofan
```
- ✅ 50% fewer experiments
- ✅ Faster results
- ✅ Aerospace-specific optimization

### Azure PM Only
```bash
python run_parallel_experiments.py --datasets azure_pm
```
- ✅ 50% fewer experiments
- ✅ Industrial equipment focus
- ✅ More challenging dataset

### Both Datasets (Default)
```bash
python run_parallel_experiments.py
# OR
python run_parallel_experiments.py --datasets turbofan azure_pm
```
- ✅ Generalization testing
- ✅ Cross-dataset comparison
- ✅ Comprehensive evaluation

---

## Quick Reference

| Command | Experiments | Use Case |
|---------|-------------|----------|
| `--datasets turbofan` | Half | Aerospace focus, faster |
| `--datasets azure_pm` | Half | Industrial focus, harder |
| `--datasets turbofan azure_pm` | Full | Generalization test |
| (no flag) | Full | Default, both datasets |

---

## Complete Example: Turbofan-Only Workflow

```bash
# 1. Quick test (6 experiments, ~30 min)
python run_parallel_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir turbofan_quick

# 2. Check results
cat turbofan_quick/best_configurations.csv
open turbofan_quick/analysis_plots/model_comparison.png

# 3. Full hyperparameter search (729 experiments)
python run_parallel_experiments.py \
  --datasets turbofan \
  --output-dir turbofan_full_search

# 4. Check best configuration
cat turbofan_full_search/top_20_by_mae.csv

# 5. Retrain best model with more epochs
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
  --exp-name turbofan_production_model
```

**Ready to run turbofan-only experiments!**
