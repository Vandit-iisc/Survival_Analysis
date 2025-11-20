# Automatic Analysis Feature

## Overview

The parallel training system now **automatically runs comprehensive analysis** after all experiments complete. No need to run a separate command!

## How It Works

### Before (Two Commands)
```bash
# Step 1: Run experiments
python run_parallel_experiments.py --output-dir my_experiments

# Step 2: Analyze results (manual)
python analyze_parallel_results.py --output-dir my_experiments
```

### After (One Command)
```bash
# One command does everything!
python run_parallel_experiments.py --output-dir my_experiments

# Analysis runs automatically after training completes ✨
```

## What You See

### Example Output

```bash
$ python run_parallel_experiments.py --batch-sizes 128 --output-dir test

================================================================================
PARALLEL EXPERIMENT CONFIGURATION
================================================================================
Total experiments: 12
Datasets: ['turbofan', 'azure_pm']
Batch sizes: [128]
Learning rates: [0.0005, 0.001, 0.005]
...
================================================================================

[GPU 0] Starting: turbofan/lstm_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id0
[GPU 1] Starting: turbofan/gru_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id1
...
[GPU 0] ✓ turbofan/lstm_basic/... completed in 8.3 min
...

================================================================================
EXPERIMENT SUMMARY
================================================================================
Total experiments: 12
Successful: 12
Failed: 0
Exceptions: 0
Average time per experiment: 8.5 minutes
Total time: 1.7 hours
================================================================================
Results saved to: test/all_results.json
================================================================================


================================================================================
RUNNING AUTOMATIC ANALYSIS
================================================================================

Executing: python analyze_parallel_results.py --output-dir test

================================================================================
PARALLEL EXPERIMENT ANALYSIS
================================================================================

Loading results...
Loaded 12 successful experiments

Creating summary statistics...
Summary statistics saved to: test/summary_statistics.csv

Finding best configurations...
Best configurations saved to: test/best_configurations.csv

Creating performance rankings...
Saved: test/top_20_by_mae.csv
Saved: test/top_20_by_cindex.csv
Saved: test/top_20_by_nasa_score.csv

Generating comprehensive visualizations...
  1/13: Hyperparameter effects...
  Saved: test/analysis_plots/rul_mae_vs_hyperparameters.png
  Saved: test/analysis_plots/rul_rmse_vs_hyperparameters.png
  Saved: test/analysis_plots/concordance_index_vs_hyperparameters.png
  2/13: Model comparison...
  Saved: test/analysis_plots/model_comparison.png
  3/13: Batch size analysis...
  Saved: test/analysis_plots/batch_size_analysis.png
  4/13: Learning rate analysis...
  Saved: test/analysis_plots/learning_rate_analysis.png
  5/13: Lambda parameter analysis...
  Saved: test/analysis_plots/lambda_parameter_analysis.png
  6/13: Dropout analysis...
  Saved: test/analysis_plots/dropout_analysis.png
  7/13: NASA loss impact...
  Saved: test/analysis_plots/nasa_loss_impact.png
  8/13: Dataset comparison...
  Saved: test/analysis_plots/dataset_comparison.png
  9/13: Training efficiency...
  Saved: test/analysis_plots/training_efficiency.png
  10/13: Batch size vs LR heatmap...
  Saved: test/analysis_plots/hyperparameter_heatmap.png
  11/13: Lambda vs NASA heatmap...
  12/13: Dropout vs LR heatmap...
  Saved: test/analysis_plots/lambda_nasa_heatmap.png
  Saved: test/analysis_plots/dropout_lr_heatmap.png

================================================================================
ANALYSIS COMPLETE
================================================================================

Generated Outputs:
  - Summary statistics: test/summary_statistics.csv
  - Best configurations: test/best_configurations.csv
  - Top 20 by MAE: test/top_20_by_mae.csv
  - Top 20 by C-Index: test/top_20_by_cindex.csv
  - Top 20 by NASA Score: test/top_20_by_nasa_score.csv
  - Visualizations: test/analysis_plots/

Total visualizations: 13 plots
================================================================================


================================================================================
✓ ANALYSIS COMPLETE
================================================================================

All results available in: /path/to/test
  - Experiment results: test/all_results.json
  - Analysis plots: test/analysis_plots
  - Best configs: test/best_configurations.csv
  - Top 20 by MAE: test/top_20_by_mae.csv
================================================================================
```

## Skip Automatic Analysis (Optional)

If you want to run analysis manually later:

```bash
python run_parallel_experiments.py \
  --output-dir my_experiments \
  --no-auto-analysis

# Then later, run analysis manually
python analyze_parallel_results.py --output-dir my_experiments
```

**Why skip automatic analysis?**
- You want to review raw results first
- Running on a shared cluster and want to defer analysis
- Testing the training system without analysis overhead
- Need to free up resources immediately after training

## Interactive Menu (quick_experiments.sh)

The interactive menu also includes automatic analysis:

```bash
$ ./quick_experiments.sh

==========================================
DDRSA Parallel Experiment Quick Launcher
==========================================

Select an experiment to run:

1. Batch Size Study (Small - ~72 experiments, ~6-12 hours on 4 GPUs)
2. Learning Rate Sweep (Medium - ~216 experiments, ~18-36 hours on 4 GPUs)
3. NASA Loss Weight Tuning (Small - ~72 experiments, ~6-12 hours on 4 GPUs)
4. Full Grid Search (Large - ~1,296 experiments, ~80-160 hours on 4 GPUs)
5. Quick Model Comparison (Tiny - ~12 experiments, ~1-2 hours on 4 GPUs)
6. Dropout Sensitivity (Small - ~96 experiments, ~8-16 hours on 4 GPUs)
7. Custom (enter your own parameters)
8. Exit

Enter choice [1-8]: 5

Running Quick Model Comparison...
[... training happens ...]
[... automatic analysis happens ...]

✓ Complete! Check quick_comparison/analysis_plots/

==========================================
Experiment Complete!
==========================================
```

## Error Handling

If analysis fails for any reason, you'll see:

```
⚠ Analysis failed with error code 1
You can run analysis manually with:
  python analyze_parallel_results.py --output-dir my_experiments
```

The training results are still saved, so you can fix any issues and re-run analysis later.

## Complete Workflow Examples

### Example 1: Quick Model Comparison

```bash
# One command - does everything!
python run_parallel_experiments.py \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir quick_test

# After it finishes, check results:
cat quick_test/top_20_by_mae.csv
open quick_test/analysis_plots/model_comparison.png
```

### Example 2: Batch Size Study

```bash
python run_parallel_experiments.py \
  --batch-sizes 32 64 128 256 512 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir batch_study

# Analysis runs automatically - check results:
open batch_study/analysis_plots/batch_size_analysis.png
```

### Example 3: Full Grid Search

```bash
# Default settings - runs everything
python run_parallel_experiments.py

# Wait for completion (may take 80-160 hours on 4 GPUs)
# Analysis runs automatically at the end

# Check results:
cat parallel_experiments/best_configurations.csv
ls parallel_experiments/analysis_plots/
```

### Example 4: Custom Search with Manual Analysis

```bash
# Run experiments without automatic analysis
python run_parallel_experiments.py \
  --batch-sizes 64 128 \
  --learning-rates 0.0005 0.001 \
  --output-dir custom_search \
  --no-auto-analysis

# Later, run analysis when ready
python analyze_parallel_results.py --output-dir custom_search
```

## Benefits

### ✅ Convenience
**One command** instead of two - set it and forget it!

### ✅ Time Savings
No need to remember to run analysis - it happens automatically

### ✅ Immediate Results
Get comprehensive insights as soon as training finishes

### ✅ Error Reduction
Can't forget to analyze results or use wrong output directory

### ✅ Complete Workflow
Training → Analysis → Results all in one go

## What Gets Analyzed

After training completes, automatic analysis generates:

### 📊 5 CSV Files
1. `summary_statistics.csv` - Stats by model/dataset
2. `best_configurations.csv` - Best hyperparameters per model
3. `top_20_by_mae.csv` - Best overall by MAE
4. `top_20_by_cindex.csv` - Best by C-Index
5. `top_20_by_nasa_score.csv` - Best by NASA score

### 📈 13 Visualizations
1. Hyperparameter effects on MAE (5-panel)
2. Hyperparameter effects on RMSE (5-panel)
3. Hyperparameter effects on C-Index (5-panel)
4. Model comparison boxplots
5. Batch size analysis (4-panel)
6. Learning rate analysis
7. Lambda parameter analysis
8. Dropout analysis
9. NASA loss impact
10. Dataset comparison
11. Training efficiency
12. Batch size × LR heatmap
13. Lambda × NASA and Dropout × LR heatmaps

## Quick Reference

### Run with Automatic Analysis (Default)
```bash
python run_parallel_experiments.py
```

### Skip Automatic Analysis
```bash
python run_parallel_experiments.py --no-auto-analysis
```

### Run Analysis Manually Later
```bash
python analyze_parallel_results.py --output-dir my_experiments
```

### Interactive Menu (Always Includes Analysis)
```bash
./quick_experiments.sh
```

## Summary

🎯 **One command** now does everything:
- Runs all experiments in parallel
- Automatically analyzes results
- Generates all plots and rankings
- Shows you where to find everything

No more manual steps - just run the training and get complete insights automatically!
