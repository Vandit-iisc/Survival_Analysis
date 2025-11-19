# Data Splitting and Normalization - Paper Methodology

## Issues with Previous Implementation

Your previous runs had **critical data leakage issues** that likely contributed to poor results:

### Problem 1: Wrong Normalization
- **Previous**: StandardScaler (z-score normalization)
- **Paper requires**: MinMaxScaler with range **[-1, 1]**

### Problem 2: Data Leakage in Splitting
- **Previous**: Split at **sample level** after creating sequences
  - Same engine's data appears in both train and test
  - Model can "cheat" by learning engine-specific patterns
- **Paper requires**: Split at **unit/engine level** before creating sequences
  - Ensures no engine appears in both train and test
  - Tests true generalization to unseen engines

### Problem 3: Wrong Split Ratios
- **Previous**: 80% train, 20% val (from train.txt), then separate test.txt
- **Paper requires**:
  - 70% of engines for train
  - 30% of engines for test
  - From train engines, 30% for validation

## Paper's Exact Methodology

From the paper (Section 6.2):

> "For both datasets, we train on 70% of randomly selected covariate time-series sequences
> and hold out 30% of the sequences for testing. From the training set a further 30% of the
> sequences are set aside as validation data to tune model parameters and policy thresholds."

### What "sequences" means:
- **Sequence** = entire time-series for one engine/unit
- **NOT** individual time windows within an engine

### Split breakdown:
1. **Total units**: 218 engines in train.txt
2. **Train/Test split**: 70/30 at unit level
   - Training pool: 153 engines (70%)
   - Test set: 65 engines (30%)
3. **Train/Val split**: From 153 training engines, 70/30 split
   - Final training: 107 engines (49% of total)
   - Validation: 46 engines (21% of total)
   - Test: 65 engines (30% of total)

### Normalization:
> "Observations are normalized with min-max transformation that transforms training inputs
> to lie in the range [−1,1]."

- Fit MinMaxScaler on **training engines only**
- Transform validation and test using the same scaler
- Range: **[-1, 1]** (not [0, 1])

## Implementation

### New Function: `get_dataloaders_paper_split()`

Located in `data_loader.py:269-373`, this implements the exact paper methodology:

```python
# 1. Load all data
df = loader.load_data('train.txt')

# 2. Get all unique engine IDs
all_units = df['unit_id'].unique()  # e.g., 218 engines

# 3. First split: 70% train+val, 30% test (at engine level)
train_val_units = all_units[:153]   # 70%
test_units = all_units[153:]        # 30%

# 4. Second split: From train+val, 70% train, 30% val
train_units = train_val_units[:107]  # 70% of 70% = 49%
val_units = train_val_units[107:]    # 30% of 70% = 21%

# 5. Create dataframes (no engine appears in multiple splits)
train_df = df[df['unit_id'].isin(train_units)]
val_df = df[df['unit_id'].isin(val_units)]
test_df = df[df['unit_id'].isin(test_units)]

# 6. Normalize with MinMaxScaler [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
train_df = scaler.fit_transform(train_df)  # Fit on train
val_df = scaler.transform(val_df)          # Transform val
test_df = scaler.transform(test_df)        # Transform test
```

## How to Use

### Option 1: Use Paper Split (Recommended)

```bash
python main.py \
    --model-type transformer \
    --use-paper-split \
    --use-minmax \
    --exp-name transformer_paper_exact
```

This will:
- ✅ Split at engine level (no data leakage)
- ✅ Use 70/21/30 split (train/val/test)
- ✅ Use MinMaxScaler [-1, 1]

### Option 2: Quick Script

```bash
bash run_transformer_paper_split.sh
```

Runs transformer with all paper configurations.

### Option 3: Original Split (for comparison)

```bash
python main.py \
    --model-type transformer \
    --val-split 0.2 \
    --exp-name transformer_original
```

This uses the old method (sample-level split, StandardScaler).

## Expected Differences

### Data Leakage Impact

**With data leakage (old method):**
- Training sees patterns from test engines
- **Artificially better** performance on test set
- **Doesn't generalize** to truly unseen engines
- Can memorize engine-specific degradation patterns

**Without data leakage (paper method):**
- Test engines are completely unseen
- **True generalization** performance
- Slightly worse metrics, but more realistic
- Must learn general degradation patterns

### Normalization Impact

**StandardScaler (old):**
- Centers data at mean=0, std=1
- Can have outliers beyond [-3, 3]
- Different scale per feature

**MinMaxScaler [-1, 1] (paper):**
- All features bounded in [-1, 1]
- Preserves relative distances
- Better for neural networks (stable gradients)
- Matches paper exactly

## Expected Results

### Old Method (with leakage)
```json
{
  "rmse": 50.33,
  "nasa_score": 24.5M,
  "c_index": 0.697
}
```

### Paper Method (without leakage)
Expected to be slightly different but **more realistic**:
```json
{
  "rmse": 18-25,
  "nasa_score": 500-1500,
  "c_index": >0.72
}
```

The metrics should be **similar or better** with proper training (warmup + paper split).

## Why Your Transformer Wasn't Working

Your transformer likely failed due to **combination of issues**:

1. ❌ **No warmup schedule** → unstable training
2. ❌ **Wrong normalization** → different scale than paper
3. ❌ **Data leakage** → confusing learning signal
4. ❌ **Sample-level split** → leaked information

**With all fixes applied:**
1. ✅ **Warmup schedule** → stable transformer training
2. ✅ **MinMax [-1, 1]** → proper input scale
3. ✅ **Engine-level split** → no leakage
4. ✅ **70/21/30 split** → matches paper

## Verification

After training, check the logs for:

```
Paper Split Statistics:
  Total units: 218
  Training units: 107 (49.1%)
  Validation units: 46 (21.1%)
  Test units: 65 (29.8%)

Normalization: MinMaxScaler [-1, 1]

Dataset Sizes:
  Training samples: ~14000
  Validation samples: ~6000
  Test samples: ~9000
```

If you see this, the split is correct!

## 10-Fold Cross-Validation

The paper mentions:

> "This process is repeated to produce 10 random train-validation-test splits."

This means running the entire experiment 10 times with different random seeds, then reporting **mean ± std** of metrics.

### To run 10-fold CV:

```bash
for seed in {0..9}; do
    python main.py \
        --model-type transformer \
        --use-paper-split \
        --use-minmax \
        --seed $seed \
        --exp-name transformer_fold_$seed
done
```

Then aggregate results:
```python
import json
import numpy as np

metrics = []
for seed in range(10):
    with open(f'logs/transformer_fold_{seed}/test_metrics.json') as f:
        metrics.append(json.load(f))

# Compute mean and std for each metric
for key in metrics[0].keys():
    values = [m[key] for m in metrics]
    print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
```

## Summary of Changes

### Files Modified:
1. **data_loader.py**
   - Added `MinMaxScaler` support
   - Added `get_dataloaders_paper_split()` function
   - Updated `normalize_data()` to handle [-1, 1] range
   - Added `use_paper_split` parameter

2. **main.py**
   - Added `--use-paper-split` flag
   - Added `--use-minmax` flag
   - Passes parameters to data loader

### New Scripts:
1. **run_transformer_paper_split.sh**
   - Runs transformer with exact paper config
   - All fixes applied

## Next Steps

1. **Run with paper split:**
   ```bash
   bash run_transformer_paper_split.sh
   ```

2. **Compare with old results:**
   - Your old transformer: RMSE 50.33
   - New transformer: Should be ~18-25

3. **Verify hazard rates:**
   - Should now be monotonically increasing
   - Should match paper's Figure 2a

4. **Run 10-fold CV** (optional):
   - For publication-quality results
   - Reports mean ± std

The combination of **warmup schedule** + **proper data split** + **correct normalization** should give you results that match the paper!
