# DDRSA Setup Guide

## Step 1: Create Python Environment

```bash
# Create a new conda environment (recommended)
conda create -n ddrsa python=3.9
conda activate ddrsa

# OR use venv
python3 -m venv ddrsa_env
source ddrsa_env/bin/activate  # On macOS/Linux
# ddrsa_env\Scripts\activate  # On Windows
```

## Step 2: Install Dependencies

```bash
cd /Users/vandit/Desktop/vandit/Survival_Analysis/ddrsa

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# OR install PyTorch (GPU version) - if you have CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## Step 3: Verify Installation

```bash
python test_installation.py
```

You should see output like:

```
✓ Imports successful

1. Testing Model Creation...
   Creating DDRSA-RNN...
   ✓ DDRSA-RNN created (XXX parameters)
   Creating DDRSA-Transformer...
   ✓ DDRSA-Transformer created (XXX parameters)

2. Testing Forward Pass...
   ✓ RNN forward pass: torch.Size([8, 128, 24]) → torch.Size([8, 100])
   ✓ Transformer forward pass: torch.Size([8, 128, 24]) → torch.Size([8, 100])

...

✓ All tests passed! DDRSA installation is working correctly.
```

## Step 4: Verify Data

Make sure the NASA Turbofan dataset is in the correct location:

```bash
ls /Users/vandit/Desktop/vandit/Survival_Analysis/Challenge_Data/
```

You should see:
- `train.txt`
- `test.txt`
- `final_test.txt`
- `readme.txt`

## Step 5: Run Your First Experiment

### Quick Test (5 epochs)

```bash
python main.py \
    --model-type rnn \
    --num-epochs 5 \
    --exp-name test_run
```

### Full Training (Paper Configuration)

```bash
python main.py \
    --model-type rnn \
    --rnn-type LSTM \
    --hidden-dim 16 \
    --num-layers 1 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --lambda-param 0.5 \
    --lookback-window 128 \
    --pred-horizon 100 \
    --num-epochs 100 \
    --exp-name ddrsa_lstm_full
```

### Run All Experiments

```bash
# This will run all experiments from the paper
# WARNING: This will take several hours!
bash run_all.sh
```

## Step 6: Monitor Training

In a separate terminal:

```bash
tensorboard --logdir logs/
```

Then open your browser to: `http://localhost:6006`

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution**: Make sure you activated the environment and installed PyTorch:
```bash
conda activate ddrsa
pip install torch
```

### Issue: "FileNotFoundError: train.txt"

**Solution**: Update the data path in your command:
```bash
python main.py --data-path /path/to/your/Challenge_Data
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size:
```bash
python main.py --batch-size 16  # or even 8
```

### Issue: Training is very slow

**Solutions**:
1. Use GPU if available
2. Reduce `num-workers` if using CPU:
   ```bash
   python main.py --num-workers 0 --no-cuda
   ```
3. Reduce model size:
   ```bash
   python main.py --hidden-dim 8 --lookback-window 64
   ```

## Expected Training Time

On a typical setup:

- **CPU (Intel i7)**: ~2-3 hours per 100 epochs
- **GPU (NVIDIA RTX 3080)**: ~30-45 minutes per 100 epochs

## Expected Performance

Based on the paper, you should expect:

- **NASA Score**: ~500-800 (lower is better)
- **RMSE**: ~15-25 cycles
- **C-index**: ~0.70-0.80

Your results may vary depending on hyperparameters and random seed.

## Next Steps

1. **Analyze Results**: Check `logs/your_exp_name/test_metrics.json`
2. **Visualize**: Use TensorBoard to see training curves
3. **Experiment**: Try different hyperparameters
4. **Compare**: Run both RNN and Transformer models

## Common Experiments

### Ablation: Lambda Parameter

```bash
for lambda in 0.1 0.3 0.5 0.7 0.9; do
    python main.py --lambda-param $lambda --exp-name lambda_$lambda
done
```

### Ablation: Hidden Dimension

```bash
for hidden in 8 16 32 64; do
    python main.py --hidden-dim $hidden --exp-name hidden_$hidden
done
```

### Ablation: Lookback Window

```bash
for window in 32 64 128 256; do
    python main.py --lookback-window $window --exp-name window_$window
done
```

## Contact

For issues or questions, please check the main README.md or open an issue.
