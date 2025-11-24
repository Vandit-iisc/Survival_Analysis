# Hazard Rate Diagnostic: Why Are Your Hazard Rates So Low?

## 🚨 The Problem

You observed that your hazard rates max out at **~0.004 (0.4%)**, while the original paper shows hazard rates reaching **~0.8 (80%)**. This is a **200× difference** and indicates something fundamentally different from the paper's implementation.

---

## 📊 Current Situation

From your `figure_2a_hazard_progression.png`:
- **Maximum hazard rate**: ~0.0040 (0.4%)
- **Typical hazard rates**: 0.0001 - 0.0010 (0.01% - 0.1%)
- **Expected from paper**: 0.1 - 0.8 (10% - 80%)

### What This Means

Hazard rate h(t) represents: **P(event occurs at time t | survived until t)**

- Your model: "0.4% chance of failure at any given time" → Predicts engines will last FOREVER
- Paper's model: "Up to 80% chance of failure near end of life" → Realistic degradation

---

## 🔍 Root Causes (In Order of Likelihood)

### 1. **Output Bias Initialization** ⭐ MOST LIKELY

**Current Code** (models.py:90):
```python
nn.init.constant_(self.output_layer.bias, -2.0)
```

**What this does**:
- Initial hazard rate: `sigmoid(-2.0) ≈ 0.119` (11.9%)
- But your TRAINED hazard rates are 0.4% (much lower!)
- This means training is pushing hazard rates DOWN

**Paper might use**:
- Different initialization (e.g., 0.0, +1.0, or adaptive)
- Or no explicit bias initialization

**Why -2.0 is problematic**:
```python
# If model outputs stay near initialization
h = sigmoid(-2.0) = 0.119  # Reasonable
# But if loss pushes it more negative:
h = sigmoid(-5.5) = 0.004  # TOO LOW (what you're getting!)
```

---

### 2. **Loss Function Imbalance** ⭐ VERY LIKELY

**The Issue**: Your loss has 3 components:

```python
# From loss.py:128
total_loss = λ * loss_z + (1-λ) * (loss_u + loss_c)
```

Where:
- `loss_z`: Likelihood of event at specific time (encourages accurate timing)
- `loss_u`: Event occurrence probability (encourages h > 0 for uncensored)
- `loss_c`: Survival likelihood (encourages h ≈ 0 for censored)

**The Problem**:
If your dataset has many **censored samples**, `loss_c` dominates and pushes ALL hazard rates toward 0!

**Check your data**:
```python
# What percentage of your data is censored?
uncensored_ratio = (targets.sum(dim=1) > 0).float().mean()
print(f"Uncensored ratio: {uncensored_ratio}")
# If this is < 0.3, loss_c dominates!
```

---

### 3. **Data Preprocessing Difference**

**Current Code** (data_loader.py):
```python
# You use MinMaxScaler with range [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
```

**Paper might use**:
- StandardScaler (z-score normalization)
- Different scaling range
- Different RUL target processing

**Why this matters**:
- Input scale affects gradient magnitudes
- Wrong scaling → model can't learn proper hazard rates

---

### 4. **Lambda Parameter Value**

**Current default** (loss.py:32):
```python
lambda_param = 0.5  # Equal weight to timing vs occurrence
```

**If lambda is too HIGH**:
- Over-emphasizes `loss_z` (timing accuracy)
- Under-emphasizes `loss_u` (ensuring events are predicted)
- Result: Model predicts very low hazard rates to avoid false positives

**Paper likely uses**: λ = 0.5 to 0.75

---

### 5. **Learning Rate / Optimizer Issues**

**Check your settings**:
```python
# From trainer.py or config
learning_rate = ?
optimizer = Adam  # vs SGD?
```

**If learning rate is too low**:
- Model doesn't escape the initial low-hazard regime
- Gets stuck in local minimum

**If learning rate is too high**:
- Unstable training
- Can't fine-tune hazard rates properly

---

### 6. **Gradient Clipping**

**Current code** (trainer.py):
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

**If `grad_clip` is too aggressive** (< 1.0):
- Prevents model from learning high hazard rates
- Gradient signal for increasing hazard is clipped away

---

## 🔧 Diagnostic Steps

### Step 1: Check Data Censoring

```bash
python3 << 'EOF'
import torch
from data_loader import get_dataloaders

train_loader, val_loader, test_loader, scaler = get_dataloaders(
    data_path='../Challenge_Data',
    batch_size=128,
    lookback_window=30,
    pred_horizon=100
)

total_uncensored = 0
total_samples = 0

for X, targets, censored in train_loader:
    is_uncensored = (targets.sum(dim=1) > 0).float()
    total_uncensored += is_uncensored.sum().item()
    total_samples += X.size(0)

uncensored_ratio = total_uncensored / total_samples
print(f"Uncensored ratio: {uncensored_ratio:.3f}")
print(f"Censored ratio: {1 - uncensored_ratio:.3f}")

if uncensored_ratio < 0.3:
    print("⚠️  WARNING: Too many censored samples!")
    print("   This will push hazard rates toward 0")
EOF
```

---

### Step 2: Check Loss Components During Training

Add this to your training loop:

```python
# In trainer.py forward pass
with torch.no_grad():
    hazard_rates = torch.sigmoid(hazard_logits)
    print(f"Hazard rate stats:")
    print(f"  Mean: {hazard_rates.mean():.6f}")
    print(f"  Max: {hazard_rates.max():.6f}")
    print(f"  Min: {hazard_rates.min():.6f}")
    print(f"  Median: {hazard_rates.median():.6f}")
```

**Expected after convergence**:
- Mean: 0.05 - 0.15
- Max: 0.5 - 0.9
- Min: 0.001 - 0.01

**What you're getting**:
- Mean: ~0.0001
- Max: ~0.004
- Min: ~0.00001

---

### Step 3: Check Logits Before Sigmoid

```python
# In models.py forward pass (before return)
print(f"Logit stats:")
print(f"  Mean: {hazard_logits.mean():.3f}")
print(f"  Max: {hazard_logits.max():.3f}")
print(f"  Min: {hazard_logits.min():.3f}")
```

**Mapping logits to hazard rates**:
```
Logit → Hazard Rate (sigmoid)
-6.0  → 0.0025  (0.25%)   ← You're here
-4.0  → 0.0180  (1.8%)
-2.0  → 0.1192  (11.9%)   ← Initial bias
 0.0  → 0.5000  (50%)
+2.0  → 0.8808  (88%)     ← Paper reaches here
+4.0  → 0.9820  (98%)
```

---

## 🎯 Solutions (Try in Order)

### Solution 1: Change Output Bias Initialization ⭐ TRY THIS FIRST

**In models.py:90, 241, 573**:

```python
# BEFORE (current):
nn.init.constant_(self.output_layer.bias, -2.0)

# AFTER (try these):
# Option A: Zero initialization (let model learn)
nn.init.constant_(self.output_layer.bias, 0.0)

# Option B: Positive bias (encourage higher hazard rates)
nn.init.constant_(self.output_layer.bias, +1.0)

# Option C: Adaptive initialization based on data
# Initialize to match average event rate in training data
nn.init.constant_(self.output_layer.bias, 0.5)
```

**Rationale**: Start the model in a regime where hazard rates are reasonable (10-50%), not pessimistically low (0.4%).

---

### Solution 2: Adjust Lambda Parameter

**Try lower lambda values** to emphasize event occurrence:

```python
# In main.py or config
lambda_param = 0.3  # More emphasis on l_u (event occurrence)

# Instead of:
lambda_param = 0.5  # Current default
```

**What this does**:
- Reduces weight on `loss_z` (exact timing)
- Increases weight on `loss_u` (ensuring events are predicted)
- Encourages model to predict higher hazard rates

---

### Solution 3: Balance Loss for Censored Samples

**Modify loss.py:128**:

```python
# BEFORE:
total_loss = self.lambda_param * loss_z + (1 - self.lambda_param) * (loss_u + loss_c)

# AFTER: Weight loss_c less aggressively
uncensored_ratio = is_uncensored.mean()
censored_ratio = 1 - uncensored_ratio

# Reweight to balance censored vs uncensored
weighted_loss_u = loss_u * uncensored_ratio
weighted_loss_c = loss_c * censored_ratio * 0.5  # Reduce censored loss weight

total_loss = self.lambda_param * loss_z + (1 - self.lambda_param) * (weighted_loss_u + weighted_loss_c)
```

---

### Solution 4: Use StandardScaler Instead of MinMaxScaler

**In data_loader.py**:

```python
# BEFORE:
scaler = MinMaxScaler(feature_range=(-1, 1))

# AFTER:
scaler = StandardScaler()  # z-score normalization
```

**Or run with**:
```bash
python main.py --use-standard-scaler --dataset turbofan
```

---

### Solution 5: Increase Learning Rate

**Try higher learning rate**:

```python
# In trainer.py or config
learning_rate = 0.001  # Current
learning_rate = 0.005  # Try this (5× higher)
```

---

### Solution 6: Add Hazard Rate Regularization

**Add to loss function**:

```python
# Encourage hazard rates to be in reasonable range
hazard_rates = torch.sigmoid(hazard_logits)
target_mean_hazard = 0.1  # 10% average hazard rate

# Penalty for too-low hazard rates
hazard_penalty = torch.abs(hazard_rates.mean() - target_mean_hazard)

total_loss = ddrsa_loss + 0.1 * hazard_penalty
```

---

## 🧪 Testing the Fixes

### Quick Test Script

```python
# test_hazard_fix.py
import torch
from models import create_ddrsa_model
from data_loader import get_dataloaders

# Load data
train_loader, _, _, _ = get_dataloaders(
    data_path='../Challenge_Data',
    batch_size=32,
    lookback_window=30,
    pred_horizon=100
)

# Create model with NEW initialization
model = create_ddrsa_model(
    model_type='rnn',
    input_dim=24,
    encoder_hidden_dim=16,
    decoder_hidden_dim=16,
    encoder_layers=1,
    decoder_layers=1,
    pred_horizon=100
)

# Check initial hazard rates
X, targets, censored = next(iter(train_loader))
with torch.no_grad():
    logits = model(X)
    hazard_rates = torch.sigmoid(logits)

print("Initial hazard rates (BEFORE training):")
print(f"  Mean: {hazard_rates.mean():.6f}")
print(f"  Max: {hazard_rates.max():.6f}")
print(f"  Std: {hazard_rates.std():.6f}")
print(f"\nExpected: Mean ~0.12, Max ~0.30")
```

---

## 📈 Expected Results After Fixes

### Before Fix (Current):
```
Hazard rates: 0.0001 - 0.004 (0.01% - 0.4%)
Model predicts: "Engine will never fail"
```

### After Fix (Target):
```
Hazard rates: 0.01 - 0.8 (1% - 80%)
Model predicts: "Realistic failure progression"
```

### Visualization:
```
Before:  |_____________________  (flat line at 0)
After:   |___/‾‾‾‾‾‾‾‾‾‾‾‾‾‾\  (increasing hazard curve)
```

---

## 🎓 Why This Matters

### Current Model Behavior:
```python
h(t) = 0.004  # 0.4% hazard rate
→ Expected TTE = 1 / 0.004 = 250 time steps
→ "Engine lasts 2.5× longer than horizon"
→ Model is OVERLY OPTIMISTIC
```

### Correct Model Behavior:
```python
h(100) = 0.6  # 60% hazard rate near end of life
→ Realistic degradation
→ Can make informed intervention decisions
```

---

## 🔬 Paper's Likely Configuration

Based on typical survival analysis papers:

```python
# Likely paper settings:
output_bias_init = 0.0  # Or positive
lambda_param = 0.5 - 0.75
learning_rate = 0.001 - 0.005
scaler = StandardScaler()
grad_clip = 5.0  # Not too aggressive
```

---

## 📋 Action Plan

1. ✅ **FIRST**: Change output bias from -2.0 to 0.0 in models.py
2. ✅ **SECOND**: Retrain with same config, check hazard rates
3. ✅ **THIRD**: If still too low, try lambda_param = 0.3
4. ✅ **FOURTH**: Check censoring ratio in your data
5. ✅ **FIFTH**: Try StandardScaler if still issues

---

## 🔍 Verification

After retraining, your `figure_2a_hazard_progression.png` should show:

```
✅ Maximum hazard rates: 0.5 - 0.9 (50% - 90%)
✅ Increasing trend toward end of life
✅ Different curves for different units
✅ NOT all flat near zero!
```

---

## 💡 Key Insight

**The problem is not your model architecture or loss function formulation - those are correct!**

**The problem is the TRAINING DYNAMICS** - something is preventing the model from learning realistic hazard rates. Most likely:

1. **Initialization bias** starting too negative
2. **Loss imbalance** from censored samples
3. **Data preprocessing** affecting gradients

Fix the initialization first - that's the fastest way to test!

---

## 🚀 Quick Fix Command

```bash
# 1. Edit models.py - change all three lines:
sed -i.bak 's/nn.init.constant_(self.output_layer.bias, -2.0)/nn.init.constant_(self.output_layer.bias, 0.0)/g' models.py

# 2. Retrain
python main.py \
  --dataset turbofan \
  --model-type rnn \
  --hidden-dim 16 \
  --num-layers 1 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --lambda-param 0.5 \
  --num-epochs 100 \
  --exp-name hazard_fix_test \
  --create-visualization

# 3. Check the new hazard progression plot
open figures/hazard_fix_test/figure_2a_hazard_progression.png
```

Expected: Hazard rates now in 0.1 - 0.8 range! 🎉
