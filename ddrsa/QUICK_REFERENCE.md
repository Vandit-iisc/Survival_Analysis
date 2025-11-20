# Quick Reference Card

## 🚀 Common Commands

### Run Default Parallel Experiments (With Auto-Analysis)
```bash
python run_parallel_experiments.py
```
**Creates**: 2 × 6 × 3 × 3 × 3 × 3 × 3 = **1,458 experiments**
**Time**: ~80-160 hours on 4 GPUs
**Analysis**: Runs automatically after training ✨

---

### Run Quick Test (Recommended First Run)
```bash
./quick_experiments.sh
# Select option 5: Quick Model Comparison
```
**Creates**: 12 experiments
**Time**: ~1-2 hours on 4 GPUs
**Analysis**: Included automatically ✨

---

### Manual Analysis (Only If Needed)
```bash
# Usually not needed - analysis runs automatically!
python analyze_parallel_results.py --output-dir parallel_experiments
```
**Generates**: 13 plots + 5 CSV files

---

### Skip Auto-Analysis (Optional)
```bash
python run_parallel_experiments.py --no-auto-analysis
# Then run analysis manually later
```

---

### Turbofan Dataset Only (50% Faster!)
```bash
python run_parallel_experiments.py \
  --datasets turbofan \
  --output-dir turbofan_only
```
**Creates**: 729 experiments (half of default)
**Time**: ~40-80 hours on 4 GPUs

---

### Azure PM Dataset Only
```bash
python run_parallel_experiments.py \
  --datasets azure_pm \
  --output-dir azure_pm_only
```
**Creates**: 729 experiments (half of default)
**Time**: ~40-80 hours on 4 GPUs

---

### Custom Batch Size Study
```bash
python run_parallel_experiments.py \
  --batch-sizes 32 64 128 256 512 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir batch_study
```
**Creates**: 60 experiments
**Time**: ~4-8 hours on 4 GPUs

---

### Turbofan Quick Test
```bash
python run_parallel_experiments.py \
  --datasets turbofan \
  --batch-sizes 128 \
  --learning-rates 0.001 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir turbofan_quick
```
**Creates**: 6 experiments
**Time**: ~30-60 minutes on 4 GPUs

---

### Learning Rate Sweep
```bash
python run_parallel_experiments.py \
  --batch-sizes 128 \
  --learning-rates 0.0001 0.0005 0.001 0.005 0.01 \
  --lambda-params 0.75 \
  --nasa-weights 0.1 \
  --dropout-rates 0.1 \
  --output-dir lr_sweep
```
**Creates**: 60 experiments
**Time**: ~4-8 hours on 4 GPUs

---

### Retrain Best Model
```bash
# First, check best config
cat parallel_experiments/best_configurations.csv

# Then retrain with those hyperparameters
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
  --exp-name final_model
```

---

## 📊 Key Output Files

### Top Priority
```
top_20_by_mae.csv           # Best configs overall
batch_size_analysis.png     # Find optimal batch size
model_comparison.png        # Compare model families
```

### Hyperparameter Tuning
```
learning_rate_analysis.png  # Optimize LR
lambda_parameter_analysis.png  # Optimize lambda
dropout_analysis.png        # Optimize regularization
nasa_loss_impact.png        # Optimize NASA weight
```

### Understanding Tradeoffs
```
training_efficiency.png     # Speed vs accuracy
dataset_comparison.png      # Generalization
hyperparameter_heatmap.png  # Batch × LR interactions
```

### Rankings
```
best_configurations.csv     # Best per model
summary_statistics.csv      # Overall stats
```

---

## 🎯 Decision Tree

### "Which model should I use?"
→ Check `model_comparison.png` (lowest median box)

### "What batch size?"
→ Check `batch_size_analysis.png` (bottom-left panel for speed vs accuracy)

### "What learning rate?"
→ Check `learning_rate_analysis.png` (minimum of curve)

### "Should I use dropout?"
→ Check `dropout_analysis.png` (compare 0.0 vs 0.1-0.2)

### "Should I use NASA loss?"
→ Check `nasa_loss_impact.png` (if entering competition: yes at 0.1)

### "What lambda value?"
→ Check `lambda_parameter_analysis.png` (usually 0.75)

### "Which config is fastest?"
→ Check `training_efficiency.png` (bottom-left corner)

### "What are the top 20 configs overall?"
→ Check `top_20_by_mae.csv` or `top_20_by_cindex.csv`

---

## 🔥 Performance Expectations

### Per Experiment (100 epochs)

| Model | Batch | GPU | Time |
|-------|-------|-----|------|
| LSTM Basic | 128 | RTX 4090 | 5-10 min |
| GRU Basic | 128 | RTX 4090 | 5-10 min |
| Transformer Basic | 128 | RTX 4090 | 20-30 min |
| Transformer Deep | 128 | RTX 4090 | 40-60 min |
| ProbSparse Basic | 128 | RTX 4090 | 30-45 min |
| ProbSparse Deep | 128 | RTX 4090 | 60-90 min |

### Total Time Estimates

| Experiments | 1 GPU | 4 GPUs | 8 GPUs |
|-------------|-------|--------|--------|
| 12 (quick) | 2-4h | 0.5-1h | 0.25-0.5h |
| 60 (focused) | 30h | 8h | 4h |
| 432 (full) | 216h | 55h | 28h |
| 1,458 (default) | 729h | 182h | 91h |

---

## 💡 Pro Tips

### 1. Start Small
```bash
./quick_experiments.sh  # Option 5
```
Test the system before committing to long runs.

### 2. Staged Search
```bash
# Stage 1: Coarse
python run_parallel_experiments.py \
  --batch-sizes 64 256 \
  --learning-rates 0.0001 0.001 0.01

# Stage 2: Fine (around best from Stage 1)
python run_parallel_experiments.py \
  --batch-sizes 128 256 \
  --learning-rates 0.0005 0.001 0.002
```

### 3. Monitor Progress
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### 4. Analyze Partial Results
```bash
# Works even while experiments are running!
python analyze_parallel_results.py --output-dir parallel_experiments
```

### 5. Focus on Your Priority
**If you care about accuracy**: Use `top_20_by_mae.csv`
**If you care about ranking**: Use `top_20_by_cindex.csv`
**If you care about NASA score**: Use `top_20_by_nasa_score.csv`
**If you care about speed**: Use `training_efficiency.png`

---

## 🐛 Troubleshooting

### Out of Memory?
```bash
# Reduce batch sizes
--batch-sizes 32 64 128

# OR reduce parallel workers
--num-workers 2
```

### Too Slow?
```bash
# Reduce hyperparameter grid
--learning-rates 0.001  # Just one value
--lambda-params 0.75    # Just one value

# OR skip expensive models
python run_parallel_experiments.py \
  --include-lstm \
  --include-transformer \
  --no-include-probsparse
```

### Some Experiments Failed?
```bash
# Check errors
cat parallel_experiments/all_results.json | grep "FAILED" -A 5

# Rerun just the failed ones
# (Extract configs from all_results.json and run individually)
```

---

## 📚 Documentation

- **`PARALLEL_TRAINING_GUIDE.md`** - Complete usage guide
- **`ANALYSIS_OUTPUTS.md`** - How to interpret results
- **`PARALLEL_SYSTEM_SUMMARY.md`** - System overview
- **`WHATS_NEW.md`** - Recent enhancements
- **`GPU_RECOMMENDATIONS.md`** - Hardware guide

---

## 🎓 Example Workflow

```bash
# 1. Quick test
./quick_experiments.sh  # Option 5
python analyze_parallel_results.py --output-dir quick_comparison

# 2. Batch size study
python run_parallel_experiments.py \
  --batch-sizes 32 64 128 256 512 \
  --learning-rates 0.001 \
  --output-dir batch_study
python analyze_parallel_results.py --output-dir batch_study

# 3. Full search with optimal batch size (from step 2)
python run_parallel_experiments.py \
  --batch-sizes 128 \
  --learning-rates 0.0001 0.0005 0.001 0.005 \
  --lambda-params 0.5 0.75 0.9 \
  --nasa-weights 0.0 0.1 0.2 \
  --dropout-rates 0.0 0.1 0.2 \
  --output-dir full_search
python analyze_parallel_results.py --output-dir full_search

# 4. Check best config
cat full_search/best_configurations.csv

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
  --exp-name production_model
```

---

## ⚡ Default Hyperparameter Grid

```python
batch_sizes = [64, 128, 256]
learning_rates = [0.0005, 0.001, 0.005]
lambda_params = [0.5, 0.6, 0.75]
nasa_weights = [0.0, 0.05, 0.1]
dropout_rates = [0.1, 0.15, 0.2]
datasets = ['turbofan', 'azure_pm']
models = 6  # 2 LSTM + 2 GRU + 2 Transformer
```

**Total**: 2 × 6 × 3 × 3 × 3 × 3 × 3 = **1,458 experiments**

---

## 🎯 Key Metrics

- **RUL MAE**: Mean Absolute Error (lower is better)
- **RUL RMSE**: Root Mean Squared Error (lower is better)
- **C-Index**: Concordance Index (higher is better, max 1.0)
- **NASA Score**: PHM08 scoring function (lower is better)

Good performance benchmarks:
- MAE < 10 (excellent), < 15 (good), < 20 (acceptable)
- C-Index > 0.8 (excellent), > 0.7 (good), > 0.6 (acceptable)

---

## 🚨 Remember

✅ **Start with quick test** before long runs
✅ **Use staged search** for efficiency
✅ **Monitor GPU usage** with `nvidia-smi`
✅ **Analyze partial results** to check progress
✅ **Check documentation** when stuck

---

**Ready to start?**
```bash
./quick_experiments.sh
```
