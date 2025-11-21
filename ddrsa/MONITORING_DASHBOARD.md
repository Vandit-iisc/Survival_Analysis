# Real-Time Monitoring Dashboard

## Overview

The monitoring dashboard provides **live, real-time updates** on all running experiments with detailed progress information.

## Quick Start

**Terminal 1** (Run experiments):
```bash
python run_parallel_experiments.py --datasets turbofan
```

**Terminal 2** (Monitor progress):
```bash
python monitor_experiments.py --output-dir parallel_experiments
```

## What You See

### Live Dashboard Example

```
================================================================================
              DDRSA PARALLEL EXPERIMENTS - LIVE MONITOR
================================================================================

Overall Progress:
  [████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 42.5%
  Completed: 310 | Running: 4 | Pending: 415 | Total: 729

  Elapsed: 2h 15m 34s | ETA: 3h 8m 12s

────────────────────────────────────────────────────────────────────────────────
GPU Status:
  GPU 0: NVIDIA RTX 4090
    Utilization: 98% | Memory: [████████░░░░░░░░░░░░] (8234/24564 MB) | Temp: 72°C
  GPU 1: NVIDIA RTX 4090
    Utilization: 97% | Memory: [███████░░░░░░░░░░░░░] (7123/24564 MB) | Temp: 70°C
  GPU 2: NVIDIA RTX 4090
    Utilization: 100% | Memory: [████████████░░░░░░░░] (12456/24564 MB) | Temp: 75°C
  GPU 3: NVIDIA RTX 4090
    Utilization: 99% | Memory: [████████░░░░░░░░░░░░] (8567/24564 MB) | Temp: 71°C

────────────────────────────────────────────────────────────────────────────────
Running Experiments (4):
  ▶ turbofan/transformer_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id23
      [████████░░░░░░░] 45/100 | Loss: 1.234/1.456 | MAE: 12.34 | C-Index: 0.876

  ▶ turbofan/lstm_deep/bs256_lr0.005_lam0.6_nasa0.05_drop0.15_id45
      [█████████░░░░░░] 52/100 | Loss: 1.567/1.678 | MAE: 13.67 | C-Index: 0.854

  ▶ turbofan/gru_basic/bs64_lr0.0005_lam0.75_nasa0.0_drop0.1_id78
      [██████░░░░░░░░░] 38/100 | Loss: 1.890/1.987 | MAE: 15.23 | C-Index: 0.823

  ▶ turbofan/probsparse_deep/bs128_lr0.001_lam0.5_nasa0.1_drop0.2_id102
      [███░░░░░░░░░░░░] 18/100 | Loss: 2.345/2.456 | MAE: 18.45 | C-Index: 0.789

────────────────────────────────────────────────────────────────────────────────
Recent Completions (Last 5):
  ✓ turbofan/lstm_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id22
      MAE: 11.23 | RMSE: 14.56 | C-Index: 0.892

  ✓ turbofan/transformer_deep/bs256_lr0.005_lam0.6_nasa0.05_drop0.15_id44
      MAE: 10.87 | RMSE: 13.92 | C-Index: 0.901

  ✓ turbofan/gru_deep/bs128_lr0.001_lam0.75_nasa0.0_drop0.1_id77
      MAE: 12.45 | RMSE: 15.23 | C-Index: 0.878

  ✓ turbofan/lstm_complex/bs64_lr0.0005_lam0.5_nasa0.1_drop0.2_id101
      MAE: 13.67 | RMSE: 16.45 | C-Index: 0.865

  ✓ turbofan/transformer_basic/bs512_lr0.001_lam0.75_nasa0.1_drop0.1_id123
      MAE: 11.56 | RMSE: 14.89 | C-Index: 0.887

────────────────────────────────────────────────────────────────────────────────
Best Results So Far:
  Best MAE: 10.87 - turbofan/transformer_deep/bs256_lr0.005_lam0.6_nasa0.05...
  Best C-Index: 0.901 - turbofan/transformer_deep/bs256_lr0.005_lam0.6_nasa0.05...

────────────────────────────────────────────────────────────────────────────────
Last updated: 2025-11-20 19:45:23 | Refresh rate: 2s | Press Ctrl+C to exit
================================================================================
```

## Features

### ✅ Overall Progress
- **Progress bar** showing completion percentage
- **Experiment counts**: Completed, Running, Pending, Total
- **Time tracking**: Elapsed time and estimated time remaining (ETA)

### ✅ GPU Status (Real-Time)
- **Per-GPU monitoring**: Utilization, memory usage, temperature
- **Color-coded**: Green (good), Yellow (moderate), Red (high)
- **Memory bars**: Visual representation of GPU memory usage

### ✅ Running Experiments
- **Live epoch progress**: See current epoch out of total
- **Training metrics**: Loss, validation loss, MAE, C-Index
- **Progress bars**: Visual epoch completion
- **Shows up to 8 concurrent experiments**

### ✅ Recent Completions
- **Last 5 completed experiments**
- **Final metrics**: MAE, RMSE, C-Index
- **Quick performance overview**

### ✅ Best Results Tracker
- **Best MAE so far**: Lowest error achieved
- **Best C-Index so far**: Highest survival ranking
- **Updates in real-time** as experiments complete

### ✅ Auto-Refresh
- **2-second updates** (configurable)
- **Clean display** with color coding
- **Progress bars** for visual feedback

## Usage Options

### Basic Usage
```bash
python monitor_experiments.py --output-dir parallel_experiments
```

### Custom Refresh Rate
```bash
# Faster updates (1 second)
python monitor_experiments.py --output-dir parallel_experiments --refresh-rate 1

# Slower updates (5 seconds - less CPU usage)
python monitor_experiments.py --output-dir parallel_experiments --refresh-rate 5
```

### Different Output Directory
```bash
python monitor_experiments.py --output-dir turbofan_only --refresh-rate 2
```

## Complete Workflow

### Two-Terminal Setup

**Terminal 1: Run Experiments**
```bash
cd /Users/vandit/Desktop/vandit/Survival_Analysis/Survival_Analysis/ddrsa

python run_parallel_experiments.py \
  --datasets turbofan \
  --batch-sizes 64 128 256 \
  --output-dir turbofan_search
```

**Terminal 2: Monitor Progress**
```bash
cd /Users/vandit/Desktop/vandit/Survival_Analysis/Survival_Analysis/ddrsa

python monitor_experiments.py \
  --output-dir turbofan_search \
  --refresh-rate 2
```

### Using tmux (Recommended for Long Runs)

```bash
# Create a tmux session with 2 panes
tmux new -s experiments

# In first pane: Run experiments
python run_parallel_experiments.py

# Split screen (Ctrl+B, then ")
# In second pane: Monitor
python monitor_experiments.py

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t experiments
```

### Using screen

```bash
# Start experiments in background
screen -S experiments -d -m python run_parallel_experiments.py

# Start monitor in foreground
python monitor_experiments.py

# Check on experiments: screen -r experiments
```

## What Each Section Shows

### 1. Overall Progress
```
Overall Progress:
  [████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 42.5%
  Completed: 310 | Running: 4 | Pending: 415 | Total: 729
  Elapsed: 2h 15m 34s | ETA: 3h 8m 12s
```

**Tells you**:
- How far through all experiments you are
- Exact counts of different states
- How long it's been running
- When it will likely finish

### 2. GPU Status
```
GPU 0: NVIDIA RTX 4090
  Utilization: 98% | Memory: [████████░░░░░░░░░░░░] (8234/24564 MB) | Temp: 72°C
```

**Tells you**:
- If GPUs are being utilized (should be >90%)
- Memory usage (helps detect memory leaks)
- Temperature (warns if overheating)

**Color coding**:
- **Green**: Optimal (util >80%, temp <70°C)
- **Yellow**: Moderate (util 50-80%, temp 70-80°C)
- **Red**: Concerning (util <50%, temp >80°C)

### 3. Running Experiments
```
▶ turbofan/transformer_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id23
    [████████░░░░░░░] 45/100 | Loss: 1.234/1.456 | MAE: 12.34 | C-Index: 0.876
```

**Tells you**:
- **Which experiment** is running
- **Current epoch** out of total (45/100)
- **Training/Validation loss** (should decrease over time)
- **Current metrics**: MAE (lower is better), C-Index (higher is better)

### 4. Recent Completions
```
✓ turbofan/lstm_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id22
    MAE: 11.23 | RMSE: 14.56 | C-Index: 0.892
```

**Tells you**:
- **What just finished**
- **Final performance**: MAE, RMSE, C-Index
- **Quick comparison** of recent results

### 5. Best Results
```
Best MAE: 10.87 - turbofan/transformer_deep/bs256_lr0.005_lam0.6_nasa0.05...
Best C-Index: 0.901 - turbofan/transformer_deep/bs256_lr0.005_lam0.6_nasa0.05...
```

**Tells you**:
- **Best model so far** (by MAE)
- **Best ranking model** (by C-Index)
- **What configuration** achieved it

## Interpreting the Display

### Good Signs ✅
- GPU utilization >90% (green)
- Experiments progressing through epochs
- Loss values decreasing
- MAE decreasing, C-Index increasing
- ETA countdown decreasing

### Warning Signs ⚠️
- GPU utilization <50% (red) - check for issues
- Experiments stuck at same epoch - possible hang
- Loss not decreasing - possible bad hyperparameters
- High temperature >85°C - check cooling

### When to Intervene
- **All GPUs idle**: Experiments may have crashed
- **Loss = NaN**: Training instability, may need to stop
- **Memory usage growing**: Possible memory leak
- **Temperature >90°C**: Stop and check cooling

## Exit and Resume

### Exit Monitor
Press `Ctrl+C` to exit the monitor

**Note**: This only stops the monitor, not the experiments. Experiments continue running in the background.

### Resume Monitoring
Just run the monitor command again:
```bash
python monitor_experiments.py --output-dir parallel_experiments
```

It will pick up where it left off and show current status.

## Troubleshooting

### Monitor shows "Waiting for experiments to start..."
**Cause**: Manifest file doesn't exist yet
**Solution**: Wait a few seconds for experiments to initialize

### No GPU info shown
**Cause**: `nvidia-smi` not available or not in PATH
**Solution**: GPU monitoring will be skipped, other info still works

### Monitor shows old data
**Cause**: Experiments finished but monitor still running
**Solution**: Exit with Ctrl+C. Check final results in analysis plots.

### Terminal too narrow
**Solution**: Resize terminal window or maximize it for best display

## Advanced Usage

### Monitor Multiple Experiment Sets
```bash
# Terminal 1: Monitor first set
python monitor_experiments.py --output-dir turbofan_experiments

# Terminal 2: Monitor second set
python monitor_experiments.py --output-dir azure_pm_experiments
```

### Log Monitor Output
```bash
python monitor_experiments.py --output-dir parallel_experiments 2>&1 | tee monitor.log
```

### Run Monitor in Background (Not Recommended)
```bash
# Better to use tmux/screen instead
nohup python monitor_experiments.py --output-dir parallel_experiments &
```

## Summary

The monitoring dashboard provides:

✅ **Real-time progress** - See experiments complete in real-time
✅ **Detailed metrics** - Epoch-by-epoch training progress
✅ **GPU monitoring** - Utilization, memory, temperature
✅ **ETA calculation** - Know when experiments will finish
✅ **Best results tracking** - See top performers as they complete
✅ **Visual feedback** - Progress bars and color coding

**Run it in a separate terminal while experiments execute for complete visibility into your parallel training!**
