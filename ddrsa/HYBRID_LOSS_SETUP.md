# ✅ Hybrid DDRSA+NASA Loss - Setup Complete

## What You Have Now

A complete experimental framework that trains 4 model architectures using **hybrid loss = DDRSA + NASA**.

### 🎯 The Hybrid Loss

```python
Total Loss = DDRSA_Loss + nasa_weight * NASA_Loss
```

**DDRSA Component** (handles censoring & temporal structure):
- `L_DDRSA = λ * loss_z + (1-λ) * (loss_u + loss_c)`
- λ (lambda_param) = 0.5 (default)
- 3 loss components: l_z, l_u, l_c

**NASA Component** (directly optimizes evaluation metric):
- Asymmetric scoring function from PHM08 challenge
- nasa_weight = 0.1 (default)
- Penalizes late predictions more than early ones

### 📦 What's Modified/Created

**Core Files**:
- ✅ `loss.py` - Already has `DDRSALossDetailedWithNASA`
- ✅ `trainer.py` - Already supports hybrid loss via `use_nasa_loss` flag
- ✅ `run_nasa_experiments.py` - **Modified** to use hybrid loss
- ✅ `metrics.py` - Already has NASA evaluation functions

**Shell Scripts**:
- ✅ `run_nasa_quick_test.sh` - Quick 5-epoch test
- ✅ `run_nasa_all.sh` - Full 100-epoch training

**Documentation**:
- ✅ `NASA_EXPERIMENTS_README.md` - **Updated** with hybrid loss info
- ✅ `NASA_EXPERIMENTS_SUMMARY.md` - **Updated** with hybrid loss details
- ✅ `HYBRID_LOSS_SETUP.md` - This file

## 🚀 How to Run

### Quick Test (5 epochs, ~10-15 min)
```bash
cd /Users/vandit/Desktop/vandit/Survival_Analysis/Survival_Analysis/ddrsa
bash run_nasa_quick_test.sh
```

### Full Training (100 epochs, ~2-4 hours)
```bash
bash run_nasa_all.sh
```

### Custom Training
```bash
# Train specific models
python run_nasa_experiments.py --experiments lstm transformer --num-epochs 50

# Change weights
# Edit run_nasa_experiments.py line 52-53:
#   'lambda_param': 0.5,  # DDRSA trade-off
#   'nasa_weight': 0.1,   # NASA weight
```

## 📊 What Gets Trained

4 experiments, each using **hybrid DDRSA+NASA loss**:

1. **hybrid_lstm** - LSTM (16 hidden, paper config)
2. **hybrid_gru** - GRU (16 hidden)
3. **hybrid_transformer** - Transformer (64-dim, 4 heads)
4. **hybrid_probsparse** - ProbSparse/Informer (128-dim, 4 heads)

## 🔧 Configuration

All experiments use:
- **Lookback**: 128 time steps
- **Prediction horizon**: 100 time steps
- **Batch size**: 32
- **Learning rate**: 1e-4
- **lambda_param**: 0.5 (DDRSA loss balance)
- **nasa_weight**: 0.1 (NASA loss weight)

## 📁 Output Structure

```
logs_nasa/
├── hybrid_lstm_20231123_140500/
│   ├── checkpoints/
│   │   └── best_model.pt
│   ├── tensorboard/
│   ├── config.json
│   └── test_metrics.json
├── hybrid_gru_TIMESTAMP/
├── hybrid_transformer_TIMESTAMP/
└── hybrid_probsparse_TIMESTAMP/

nasa_full_experiments_summary.json  # Comparison table
```

## 📈 Metrics Reported

Each experiment reports:
- **NASA Score** - Primary metric (lower is better)
- **DDRSA Loss Components** - loss_z, loss_u, loss_c
- **Standard Metrics** - MSE, MAE, RMSE
- **Error Statistics** - Mean/std error, prediction statistics

## 🌐 Transfer to RunPod

All paths are relative - ready to transfer:

```bash
# Transfer code
scp -r -i ~/.ssh/id_ed25519 ./ddrsa fp6h6x215ubhvr-64411654@ssh.runpod.io:/workspace/survival_analysis/

# Transfer data
scp -r -i ~/.ssh/id_ed25519 ./Challenge_Data fp6h6x215ubhvr-64411654@ssh.runpod.io:/workspace/survival_analysis/

# On RunPod
cd /workspace/survival_analysis/ddrsa
bash run_nasa_quick_test.sh  # Test first
bash run_nasa_all.sh          # Full training
```

## 🎓 Key Advantages

**vs Pure DDRSA**:
- ✅ Directly optimizes NASA evaluation metric
- ✅ Better aligned with final scoring

**vs Pure NASA**:
- ✅ Proper censoring handling (from DDRSA)
- ✅ Better temporal structure modeling
- ✅ More stable training

## 🔍 Adjusting Weights

Want to change the balance?

Edit `run_nasa_experiments.py` line 52-53:

```python
'lambda_param': 0.5,  # 0-1: DDRSA balance (l_z vs l_u+l_c)
'nasa_weight': 0.1,   # 0+: NASA loss weight
```

**Suggestions**:
- **More NASA influence**: `nasa_weight=0.2` or `0.3`
- **Less NASA influence**: `nasa_weight=0.05` or `0.01`
- **DDRSA balance**: Typically keep `lambda_param=0.5`

## ✅ Verification Checklist

Before running on RunPod:

- [ ] Run quick test locally: `bash run_nasa_quick_test.sh`
- [ ] Check output logs appear in `logs_nasa/`
- [ ] Verify TensorBoard works: `tensorboard --logdir logs_nasa/`
- [ ] Check summary JSON is created
- [ ] Transfer to RunPod
- [ ] Run full training: `bash run_nasa_all.sh`

## 📚 Documentation

- **Quick Reference**: `NASA_EXPERIMENTS_SUMMARY.md`
- **Full Guide**: `NASA_EXPERIMENTS_README.md`
- **This File**: `HYBRID_LOSS_SETUP.md`

## 🎉 You're Ready!

Everything is configured for **hybrid DDRSA+NASA loss**. The experiments will:

1. Train 4 model architectures
2. Use combined DDRSA + NASA loss
3. Directly optimize NASA scoring metric
4. Maintain proper censoring handling
5. Save best models and results

Run `bash run_nasa_quick_test.sh` to get started!

---

## Quick Command Reference

```bash
# Quick test (5 epochs)
bash run_nasa_quick_test.sh

# Full training (100 epochs)
bash run_nasa_all.sh

# View results
cat nasa_full_experiments_summary.json
tensorboard --logdir logs_nasa/

# Transfer to RunPod
scp -r -i ~/.ssh/id_ed25519 ./ddrsa user@runpod:/workspace/survival_analysis/
scp -r -i ~/.ssh/id_ed25519 ./Challenge_Data user@runpod:/workspace/survival_analysis/
```
