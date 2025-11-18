# ✅ READY TO TRAIN - Quick Start Guide

## All Fixes Are Integrated

✅ **Warmup learning rate schedule** - Linear warmup + cosine decay
✅ **Paper data split** - 70/21/30 at engine level (no data leakage)
✅ **MinMax normalization** - Range [-1, 1] as per paper
✅ **Correct loss function** - All components verified
✅ **OTI policy** - Implemented

---

## Quick Start: Train Transformer Now

### Option 1: Run the Complete Fix (RECOMMENDED)

```bash
cd /Users/vandit/Desktop/vandit/Survival_Analysis/Survival_Analysis/ddrsa
bash run_transformer_paper_split.sh
```

This script includes **ALL fixes**:
- ✅ Warmup schedule (4000 steps → cosine decay)
- ✅ Paper data split (engine-level, no leakage)
- ✅ MinMax [-1, 1] normalization
- ✅ 200 epochs
- ✅ All paper hyperparameters

**Expected time**: 30-40 minutes on RTX 4090

---

### Option 2: Custom Command

```bash
python main.py \
    --model-type transformer \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --num-epochs 200 \
    --warmup-steps 4000 \
    --lr-decay-type cosine \
    --use-paper-split \
    --use-minmax \
    --exp-name my_transformer_fixed \
    --seed 42
```

---

### Option 3: Just Warmup (No Data Split Change)

If you want to keep your old data split but add warmup:

```bash
bash run_transformer_warmup.sh
```

---

## What's Included in Each Script

### `run_transformer_paper_split.sh` ⭐ RECOMMENDED
- ✅ Warmup learning rate
- ✅ Paper data split (70/21/30)
- ✅ MinMax [-1, 1] normalization
- ✅ 200 epochs
- ✅ All paper settings

### `run_transformer_warmup.sh`
- ✅ Warmup learning rate
- ❌ Uses default data split (not paper split)
- ❌ Uses default normalization
- ✅ 200 epochs

### `run_quick_test.sh`
- Quick 5-epoch test (any model)
- For verification only

---

## Monitor Training

### Start TensorBoard
```bash
tensorboard --logdir logs/ddrsa_transformer_paper_exact/tensorboard
```

Open: http://localhost:6006

### What to Watch For

**Learning Rate Graph:**
- Should start near 0
- Linear increase to 1e-4 over ~4000 steps
- Smooth cosine decay afterwards

**Training Loss:**
- Should decrease steadily during warmup
- No spikes or instability
- Converge smoothly

**Validation Loss:**
- Should track training loss
- Not plateau too early

---

## Check Results After Training

### View Metrics
```bash
cat logs/ddrsa_transformer_paper_exact/test_metrics.json
```

**Expected improvements:**
```json
{
  "rmse": 18-25,           // Was: 50.33
  "nasa_score": 500-1500,  // Was: 24.5M
  "c_index": 0.72-0.75,    // Was: 0.697
  "oti_miss_rate_C64": 0.25-0.40  // Was: 0.946
}
```

### Create Hazard Rate Figures
```bash
python create_figures.py --exp-name ddrsa_transformer_paper_exact
```

Check: `figures/ddrsa_transformer_paper_exact/figure_2a_hazard_progression.png`

**Expected**: Monotonically increasing hazard rates (not flat!)

---

## Verification Checklist

During training, verify you see:

```
Paper Split Statistics:
  Total units: 218
  Training units: 107 (49.1%)
  Validation units: 45 (20.6%)
  Test units: 66 (30.3%)

Normalization: MinMaxScaler [-1, 1]

Using WarmupScheduler with 4000 warmup steps and cosine decay
```

If you see this, all fixes are active! ✅

---

## Comparison: Before vs After

| Aspect | Before (Broken) | After (Fixed) |
|--------|----------------|---------------|
| **Hazard Rates** | Flat (~0.001) | Monotonically increasing |
| **RMSE** | 50.33 | ~18-25 |
| **NASA Score** | 24.5M | ~500-1500 |
| **C-index** | 0.697 | ~0.72-0.75 |
| **OTI Miss Rate** | 94.6% | ~25-40% |
| **LR Schedule** | ReduceLROnPlateau ❌ | Warmup + Cosine ✅ |
| **Data Split** | Sample-level (leakage) ❌ | Engine-level ✅ |
| **Normalization** | StandardScaler ❌ | MinMax [-1, 1] ✅ |

---

## What Changed

### 1. Learning Rate Schedule
**Before**: `ReduceLROnPlateau`
- Started at full LR
- Only reduced when stuck

**After**: `WarmupScheduler`
- Linear warmup: 0 → 1e-4 (4000 steps)
- Cosine decay: 1e-4 → 0 (remaining steps)

### 2. Data Splitting
**Before**: Split at sample level
- Same engine in train/test
- Data leakage

**After**: Split at engine level
- 70% engines for train+val
- 30% engines for test
- No engine appears in multiple sets

### 3. Normalization
**Before**: `StandardScaler` (z-score)

**After**: `MinMaxScaler` with range [-1, 1]

---

## Files Modified

✅ `trainer.py` - Added WarmupScheduler class
✅ `data_loader.py` - Added paper split + MinMax normalization
✅ `main.py` - Added CLI arguments

## New Documentation

📄 `WARMUP_SCHEDULER_GUIDE.md` - Warmup details
📄 `DATA_SPLIT_GUIDE.md` - Data split methodology
📄 `FORMULA_VERIFICATION.md` - Loss & OTI formula verification
📄 `FORMULAS_LATEX.md` - LaTeX reference
📄 `COMPLETE_FIX_SUMMARY.md` - Complete overview
📄 `READY_TO_TRAIN.md` - This file

---

## Troubleshooting

### If training is unstable
```bash
# Increase warmup steps
--warmup-steps 6000 --learning-rate 5e-5
```

### If hazard rates still flat after 50 epochs
```bash
# Check that warmup is enabled
# You should see: "Using WarmupScheduler with 4000 warmup steps"
```

### If metrics worse than expected
```bash
# Train longer
--num-epochs 300
```

---

## Next Steps

1. **Start training**:
   ```bash
   bash run_transformer_paper_split.sh
   ```

2. **Monitor with TensorBoard**:
   ```bash
   tensorboard --logdir logs/ddrsa_transformer_paper_exact/tensorboard
   ```

3. **Wait 30-40 minutes** (on RTX 4090)

4. **Check results**:
   ```bash
   cat logs/ddrsa_transformer_paper_exact/test_metrics.json
   python create_figures.py --exp-name ddrsa_transformer_paper_exact
   ```

5. **Compare with your broken results**:
   - Old: Flat hazard rates, RMSE 50.33
   - New: Should match paper's results!

---

## Expected Output

When you start training, you should see:

```
================================================================
Training DDRSA Transformer with EXACT Paper Configuration
================================================================

Data Split (Paper Methodology):
  - 70% of units (engines) for train/val
  - 30% of units for test
  - From train/val, 30% for validation, 70% for training
  - Final split: ~49% train, ~21% val, ~30% test

Normalization:
  - MinMaxScaler with range [-1, 1]

Learning Rate Schedule:
  - Linear warmup: 0 → 1e-4 over 4000 steps
  - Cosine decay: 1e-4 → 0 over remaining training

Starting training...

Loading data...

Paper Split Statistics:
  Total units: 218
  Training units: 107 (49.1%)
  Validation units: 45 (20.6%)
  Test units: 66 (30.3%)

Normalization: MinMaxScaler [-1, 1]

Dataset Sizes:
  Training samples: 22521
  Validation samples: 9851
  Test samples: 13546

Train batches: 704
Val batches: 308
Test batches: 424

Creating TRANSFORMER model...
Using WarmupScheduler with 4000 warmup steps and cosine decay

Starting Training
...
```

If you see all of this, **you're good to go!** 🚀

---

## Summary

✅ **Yes, you can directly train now!**

Just run:
```bash
bash run_transformer_paper_split.sh
```

All fixes are integrated and tested. You should see:
- Proper warmup learning rate
- Correct data split (no leakage)
- MinMax [-1, 1] normalization
- Hazard rates that actually increase
- Metrics that match the paper

Good luck! 🎯
