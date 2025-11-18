# RunPod Setup Guide for DDRSA Training

## GPU Selection

### Recommended (Best Value)
- **RTX 3070/3080**: $0.20-$0.35/hour - Perfect balance
- **RTX 4070**: $0.35/hour - Newer, efficient
- **RTX 4080**: $0.45/hour - Fast, good value

### Expected Training Time
- **100 epochs**: 15-30 minutes
- **Full experiments (run_all.sh)**: 3-5 hours
- **Total cost**: $0.10-$0.15 per experiment

## Quick Setup on RunPod

### Step 1: Create Pod

1. Go to RunPod.io
2. Select GPU: **RTX 3080** (recommended)
3. Template: **PyTorch 2.0** or **Ubuntu + CUDA**
4. Storage: **20 GB** (enough for all experiments)
5. Click "Deploy"

### Step 2: Connect via SSH or Web Terminal

```bash
# In RunPod terminal
cd /workspace
```

### Step 3: Clone and Setup

```bash
# Upload your code or clone
git clone <your-repo>  # if you pushed to GitHub
# OR
# Upload via RunPod web interface

cd ddrsa

# Install dependencies
pip install torch numpy pandas scikit-learn tqdm tensorboard matplotlib

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 4: Upload Dataset

```bash
# Option 1: Upload via RunPod interface
# Upload to /workspace/Challenge_Data/

# Option 2: Download from cloud
# Example using wget or curl
wget <your-data-url> -O data.zip
unzip data.zip -d Challenge_Data/
```

### Step 5: Test Installation

```bash
python test_installation.py
```

Expected output:
```
✓ GPU detected: NVIDIA GeForce RTX 3080
✓ All tests passed!
```

### Step 6: Run Quick Test

```bash
# 5 epoch test (2-3 minutes)
python main.py --model-type rnn --num-epochs 5 --exp-name gpu_test

# Check GPU utilization
watch -n 1 nvidia-smi
```

### Step 7: Full Training

```bash
# Single experiment (20-30 min)
python main.py --model-type rnn --num-epochs 100 --exp-name ddrsa_rnn

# All experiments (3-5 hours)
bash run_all.sh
```

## Optimization for RunPod

### 1. Use Larger Batch Sizes (More GPU Utilization)

```bash
# Default: batch_size=32
# With RTX 3080 (10GB): Can go up to 128-256

python main.py --batch-size 128 --num-epochs 100
```

### 2. Parallel Experiments

Run multiple experiments in screen sessions:

```bash
# Install screen
apt-get update && apt-get install -y screen

# Start first experiment
screen -S exp1
python main.py --model-type rnn --num-epochs 100 --exp-name rnn
# Detach: Ctrl+A, then D

# Start second experiment
screen -S exp2
python main.py --model-type transformer --num-epochs 100 --exp-name transformer
# Detach: Ctrl+A, then D

# List sessions
screen -ls

# Reattach
screen -r exp1
```

### 3. Mixed Precision Training (Faster)

Enable automatic mixed precision for 2x speedup:

Add to `trainer.py`:
```python
# In __init__
self.scaler = torch.cuda.amp.GradScaler()

# In training loop
with torch.cuda.amp.autocast():
    loss = self.criterion(...)
self.scaler.scale(loss).backward()
self.scaler.step(self.optimizer)
self.scaler.update()
```

Or run with flag (if we implement it):
```bash
python main.py --mixed-precision
```

## Monitoring on RunPod

### Check GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or
nvtop  # if available
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006

# Access via RunPod port forwarding
# Go to: https://<pod-id>.runpod.io:6006
```

### Download Results

```bash
# Create archive
tar -czf results.tar.gz logs/ figures/

# Download via RunPod web interface
# Or use SCP:
scp root@<pod-ip>:/workspace/ddrsa/results.tar.gz ./
```

## Cost Optimization

### Strategy 1: Spot Instances
- **Savings**: 50-70% cheaper
- **Risk**: Can be interrupted
- **Solution**: Save checkpoints frequently (already implemented)

### Strategy 2: Stop When Done
- Use `nohup` to run in background
- Stop pod immediately after training
- Don't leave running overnight!

```bash
# Run in background
nohup bash run_all.sh > training.log 2>&1 &

# Monitor
tail -f training.log

# When done, download results and STOP POD
```

### Strategy 3: Batch Multiple Runs

Instead of starting/stopping pod for each experiment:
```bash
# Run all at once
for lambda in 0.1 0.3 0.5 0.7 0.9; do
    python main.py --lambda-param $lambda --exp-name lambda_$lambda
done
```

## Expected Costs

### Single Experiment (RTX 3080 @ $0.30/hr)
- 100 epochs: ~25 minutes = **$0.12**
- With visualization: +2 minutes = **$0.13**

### All Experiments (15 runs)
- Time: ~6 hours
- Cost: **$1.80**

### With Spot Instance (50% discount)
- Cost: **$0.90**

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python main.py --batch-size 16

# Or reduce model size
python main.py --hidden-dim 8
```

### Slow Training

```bash
# Check GPU usage
nvidia-smi

# Should see:
# - GPU utilization: >80%
# - Memory usage: 3-5 GB
# - Temperature: <80°C

# If low utilization:
# - Increase batch size
# - Reduce num_workers (I/O bottleneck)
python main.py --batch-size 64 --num-workers 2
```

### Can't Access TensorBoard

```bash
# Check RunPod port settings
# Enable port 6006 in pod configuration

# Or use Jupyter notebook for visualization
pip install jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

## Comparison: Local vs RunPod

| Aspect | Local CPU | Local GPU | RunPod RTX 3080 |
|--------|-----------|-----------|-----------------|
| 100 epochs | 2-3 hours | 30-45 min | **20-25 min** |
| Cost | Free | Free | **$0.12** |
| Convenience | Low | Medium | **High** |
| Scalability | None | Limited | **Excellent** |

## Final Recommendation

### Best Setup for Your Case:

1. **GPU**: RTX 3080 (10GB) - **$0.25-$0.35/hour**
2. **Storage**: 20 GB
3. **Template**: PyTorch 2.0 + CUDA 11.8
4. **Instance type**: On-demand (for reliability) or Spot (for cost)

### Expected Workflow:

```bash
# 1. Setup (5 min)
pip install requirements.txt
python test_installation.py

# 2. Quick test (3 min)
python main.py --num-epochs 5

# 3. Full training (25 min)
python main.py --num-epochs 100

# 4. Create figures (2 min)
python create_figures.py --exp-name ddrsa_rnn

# 5. Download results and STOP POD
# Total time: ~35 minutes
# Total cost: ~$0.18
```

### For Multiple Experiments:

```bash
# Run all experiments
bash run_all.sh

# Total time: ~6 hours
# Total cost: ~$1.80 (on-demand) or ~$0.90 (spot)
```

## Summary

✅ **GPU**: RTX 3080 (10GB VRAM)
✅ **Cost**: ~$0.12 per experiment
✅ **Time**: ~20-25 minutes per 100 epochs
✅ **Total for all experiments**: ~$1.80 (6 hours)

This is the sweet spot for your model size and budget!
