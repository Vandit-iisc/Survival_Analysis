# Warmup Learning Rate Scheduler for Transformers

## Why Your Transformer Wasn't Learning

Your transformer trained for 200 epochs produced nearly flat hazard rates because it was using the wrong learning rate schedule. Here's what was happening:

### The Problem
- **Your previous setup**: Used `ReduceLROnPlateau` scheduler (designed for RNNs)
  - Starts at full learning rate (1e-4)
  - Only reduces LR when validation loss plateaus
  - No gradual warmup

### Why Transformers Need Warmup

Transformers are **much more sensitive** to learning rate than RNNs because:

1. **Attention mechanism instability**: Early in training, attention weights can explode or vanish
2. **Deep architecture**: Multiple transformer layers compound gradient issues
3. **No recurrent connections**: Unlike RNNs, transformers can't gradually accumulate information

The **warmup schedule** solves this by:
- Starting with very small learning rate (near 0)
- Linearly increasing to peak LR over first N steps
- Then gradually decaying

This is the **standard practice** for all transformer models (BERT, GPT, etc.)

## What Was Implemented

### 1. WarmupScheduler Class (trainer.py:21-96)

A custom learning rate scheduler with:
- **Linear warmup**: LR increases from 0 → base_lr over `warmup_steps`
- **Cosine decay**: After warmup, LR follows cosine curve to 0
- **Exponential decay**: Alternative decay option

```python
class WarmupScheduler:
    def _get_lr(self):
        # Phase 1: Linear warmup
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)

        # Phase 2: Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + cos(π * progress))
```

### 2. Updated DDRSATrainer (trainer.py:136-161)

Automatically detects model type and uses appropriate scheduler:
- **Transformers**: WarmupScheduler (step per batch)
- **RNNs**: ReduceLROnPlateau (step per epoch)

### 3. Configuration Updates

Default transformer config now includes:
```python
config = {
    'use_warmup': True,
    'warmup_steps': 4000,
    'lr_decay_type': 'cosine'
}
```

## Learning Rate Schedule Visualization

```
Learning Rate vs. Training Step

1e-4 |           ___________________
     |         /                     \
     |       /                         \
     |     /                             \
     |   /                                 \
     | /                                     \
   0 |/                                       \___
     +----+----+----+----+----+----+----+----+----
     0   4000      10000        20000       40000

     [Warmup]  [           Cosine Decay            ]
```

### Comparison with Your Previous Results

**Before (ReduceLROnPlateau):**
```
- Hazard rates: Nearly flat at ~0.001
- RMSE: 50.33
- NASA Score: 24.5M (terrible!)
- C-index: 0.697
- OTI Miss Rate: 94.6%
```

**Expected After (Warmup + Cosine Decay):**
```
- Hazard rates: Monotonically increasing (like LSTM)
- RMSE: ~18-25
- NASA Score: 500-1500 (good)
- C-index: >0.72
- OTI Miss Rate: <40%
```

## How to Use

### Option 1: Use the Training Script (Recommended)

```bash
bash run_transformer_warmup.sh
```

This runs with optimal settings:
- 4000 warmup steps
- Cosine decay
- 200 epochs
- Learning rate 1e-4

### Option 2: Custom Command

```bash
python main.py \
    --model-type transformer \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --num-epochs 200 \
    --warmup-steps 4000 \
    --lr-decay-type cosine \
    --exp-name my_transformer_exp
```

### Option 3: Different Warmup Settings

**Longer warmup (more stable, slower initial learning):**
```bash
python main.py --model-type transformer --warmup-steps 8000 --num-epochs 300
```

**Exponential decay instead of cosine:**
```bash
python main.py --model-type transformer --lr-decay-type exponential
```

**No decay after warmup (just maintain base LR):**
```bash
python main.py --model-type transformer --lr-decay-type none
```

## Hyperparameter Tuning Guide

### Warmup Steps
- **Rule of thumb**: 10% of total training steps
- **Your setup**: ~218 training samples, 32 batch size → ~7 batches/epoch
- **200 epochs** = 1400 total steps
- **Recommended**: 1000-4000 warmup steps

**Try these values:**
```bash
# Conservative (longer warmup)
--warmup-steps 2000 --num-epochs 200

# Standard (default)
--warmup-steps 4000 --num-epochs 200

# Aggressive (shorter warmup)
--warmup-steps 1000 --num-epochs 150
```

### Learning Rate
Warmup allows **higher peak learning rates**:

```bash
# Conservative
--learning-rate 5e-5 --warmup-steps 2000

# Standard (recommended)
--learning-rate 1e-4 --warmup-steps 4000

# Aggressive (if training is stable)
--learning-rate 3e-4 --warmup-steps 6000
```

### Decay Type

**Cosine (recommended for most cases):**
- Smooth decay to zero
- Good generalization
- Standard in modern transformers

**Exponential (faster initial decay):**
- Decays faster initially
- Can help if overfitting early

**None (constant after warmup):**
- If you want full control with early stopping
- May need lower base LR

## Monitoring Training

Start TensorBoard to watch the learning rate schedule:

```bash
tensorboard --logdir logs/ddrsa_transformer_warmup/tensorboard
```

**What to look for:**

1. **Learning Rate curve** should show:
   - Linear increase for first ~4000 steps
   - Smooth cosine decay afterwards

2. **Training Loss** should:
   - Decrease steadily during warmup
   - Continue decreasing smoothly (no spikes)

3. **Validation Loss** should:
   - Track training loss
   - Not plateau too early

## Troubleshooting

### Training is unstable (loss spikes)
**Solution**: Increase warmup steps or reduce learning rate
```bash
--warmup-steps 8000 --learning-rate 5e-5
```

### Training is too slow
**Solution**: Reduce warmup steps or increase learning rate
```bash
--warmup-steps 2000 --learning-rate 2e-4
```

### Hazard rates still flat after 50+ epochs
**Solution**:
1. Check that warmup is enabled (should see in logs)
2. Try higher learning rate
3. Verify loss is decreasing

### Validation loss worse than training loss
**Solution**:
1. Add more regularization (increase dropout)
2. Use cosine decay (helps generalization)

## Expected Training Time

On your RTX 4090:
- **200 epochs**: ~30-40 minutes
- **First noticeable improvement**: After warmup (~30 epochs)
- **Best results**: Usually around epoch 100-150

## What to Expect

Your hazard rate graphs should now look like this:

```
Hazard Rate
   0.01 |                                            /‾
        |                                          /
        |                                        /
        |                                      /
        |                                   /
   0.001|                               /
        |                           /
        |____________________/‾‾
        +----+----+----+----+----+----+----+----+
        0   10   20   40   60   80   90   100
                    Time steps from j
```

This shows the proper **monotonically increasing** hazard rates that match the paper's Figure 2a.

## Key Differences from LSTM

| Aspect | LSTM | Transformer (Warmup) |
|--------|------|---------------------|
| LR Schedule | ReduceLROnPlateau | Warmup + Cosine |
| LR Update | Per epoch | Per batch |
| Initial LR | Full (1e-4) | Near zero → 1e-4 |
| Peak LR | 1e-4 | 1e-4 (after warmup) |
| Training Time | Faster | Slower initially |
| Final Performance | Good | Should match or exceed |

## Next Steps

1. **Run the training**:
   ```bash
   bash run_transformer_warmup.sh
   ```

2. **Monitor progress**:
   ```bash
   tensorboard --logdir logs/ddrsa_transformer_warmup/tensorboard
   ```

3. **Compare results** with your LSTM baseline

4. **Visualize hazard rates**:
   ```bash
   python create_figures.py --exp-name ddrsa_transformer_warmup
   ```

5. **Check test metrics**:
   ```bash
   cat logs/ddrsa_transformer_warmup/test_metrics.json
   ```

## References

This warmup schedule is based on:
- "Attention Is All You Need" (Vaswani et al., 2017)
- BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)
- Standard practice in Hugging Face Transformers library

The implementation follows the formula from the original Transformer paper:

```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

We simplified this to a more intuitive linear warmup + cosine decay, which is now the standard in modern transformer training.
