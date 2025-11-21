# Live Monitoring Guide

## Current Live Output

The parallel experiment runner already shows **live status updates**:

✅ **Experiment start/completion** - See when each experiment starts and finishes
✅ **GPU assignment** - Know which GPU is running which experiment
✅ **Duration** - See how long each experiment took
✅ **Success/Failure** - Immediate notification of failures

## What You See Now

```bash
[GPU 0] Starting: turbofan/lstm_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id0
[GPU 1] Starting: turbofan/gru_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id1
[GPU 2] Starting: turbofan/transformer_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id2
[GPU 3] Starting: turbofan/lstm_deep/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id3

[GPU 0] ✓ turbofan/lstm_basic/... completed in 8.3 min
[GPU 0] Starting: turbofan/probsparse_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id4

[GPU 1] ✓ turbofan/gru_basic/... completed in 7.9 min
[GPU 1] Starting: turbofan/lstm_basic/bs64_lr0.001_lam0.75_nasa0.1_drop0.1_id5
```

## Monitor GPU Usage (Separate Terminal)

While experiments run, open another terminal and monitor GPU utilization:

```bash
watch -n 1 nvidia-smi
```

**Output**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 4090     On   | 00000000:01:00.0 Off |                  Off |
| 30%   45C    P2    89W / 450W |   1234MiB / 24564MiB |     98%      Default |
|   1  NVIDIA RTX 4090     On   | 00000000:02:00.0 Off |                  Off |
| 30%   43C    P2    85W / 450W |   1156MiB / 24564MiB |     97%      Default |
|   2  NVIDIA RTX 4090     On   | 00000000:03:00.0 Off |                  Off |
| 35%   48C    P2   145W / 450W |   2345MiB / 24564MiB |    100%      Default |
|   3  NVIDIA RTX 4090     On   | 00000000:04:00.0 Off |                  Off |
| 32%   46C    P2    92W / 450W |   1289MiB / 24564MiB |     99%      Default |
+-----------------------------------------------------------------------------+
```

## Monitor Individual Experiment Logs

Each experiment saves a detailed training log. You can follow it in real-time:

```bash
# Find the experiment directory
ls parallel_experiments/logs/turbofan/lstm_basic/

# Follow the training log (while it's running)
tail -f parallel_experiments/logs/turbofan/lstm_basic/bs128_lr0.001_lam0.75_nasa0.1_drop0.1_id0/training_log.json
```

**Output** (real-time):
```json
{"epoch": 1, "train_loss": 2.345, "val_loss": 2.567, "rul_mae": 15.23, "concordance_index": 0.78}
{"epoch": 2, "train_loss": 2.123, "val_loss": 2.345, "rul_mae": 14.56, "concordance_index": 0.80}
{"epoch": 3, "train_loss": 1.987, "val_loss": 2.234, "rul_mae": 13.89, "concordance_index": 0.82}
...
```

## Monitor Progress with Shell Script

Create a monitoring script:

```bash
#!/bin/bash
# monitor_experiments.sh

OUTPUT_DIR=${1:-parallel_experiments}

echo "Monitoring experiments in: $OUTPUT_DIR"
echo ""

while true; do
    clear
    echo "================================"
    echo "PARALLEL EXPERIMENTS MONITOR"
    echo "================================"
    echo ""

    # Count experiments
    total=$(cat $OUTPUT_DIR/experiment_manifest.json 2>/dev/null | grep -c "\"name\":" || echo "0")
    completed=$(ls -1 $OUTPUT_DIR/logs/*/*/*/*/test_metrics.json 2>/dev/null | wc -l)

    echo "Progress: $completed / $total experiments"
    echo ""

    # Show recent completions
    echo "Recent completions:"
    ls -lt $OUTPUT_DIR/logs/*/*/*/*/test_metrics.json 2>/dev/null | head -5 | awk '{print $9}' | sed 's|/test_metrics.json||'
    echo ""

    # Show GPU usage
    echo "GPU Usage:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -4

    sleep 5
done
```

**Usage**:
```bash
chmod +x monitor_experiments.sh
./monitor_experiments.sh parallel_experiments
```

## Check Partial Results (While Running)

You can analyze partial results while experiments are still running:

```bash
# Run this in another terminal while experiments are running
python analyze_parallel_results.py --output-dir parallel_experiments
```

This will analyze all **completed** experiments so far and show you interim results!

## Monitor Specific Metrics

### Check Best Results So Far
```bash
# While experiments run, check what's best so far
python -c "
import json
import glob

results = []
for f in glob.glob('parallel_experiments/logs/*/*/*/*/test_metrics.json'):
    try:
        with open(f) as file:
            data = json.load(file)
            data['path'] = f
            results.append(data)
    except:
        pass

if results:
    best = min(results, key=lambda x: x['rul_mae'])
    print(f'Best MAE so far: {best[\"rul_mae\"]:.2f}')
    print(f'From: {best[\"path\"]}')
"
```

### Count Completed Experiments
```bash
ls parallel_experiments/logs/*/*/*/*/test_metrics.json 2>/dev/null | wc -l
```

### Check for Failures
```bash
grep "FAILED\|EXCEPTION" parallel_experiments/all_results.json 2>/dev/null
```

## tmux/screen for Long-Running Experiments

For very long experiments, use tmux or screen to keep them running even if you disconnect:

### Using tmux
```bash
# Start a new tmux session
tmux new -s experiments

# Run experiments
python run_parallel_experiments.py

# Detach: Press Ctrl+B, then D
# Reattach later:
tmux attach -t experiments
```

### Using screen
```bash
# Start a new screen session
screen -S experiments

# Run experiments
python run_parallel_experiments.py

# Detach: Press Ctrl+A, then D
# Reattach later:
screen -r experiments
```

## Email Notifications (Optional)

Get notified when experiments complete:

```bash
# At the end of your command
python run_parallel_experiments.py && \
  echo "Experiments complete!" | mail -s "Training Done" your@email.com
```

Or use a service like ntfy.sh:
```bash
python run_parallel_experiments.py && \
  curl -d "Experiments complete!" ntfy.sh/your-topic
```

## Logging to File

Save all output to a log file while still seeing it on screen:

```bash
python run_parallel_experiments.py 2>&1 | tee experiment_run.log
```

Now you can:
- See output live in terminal
- Check log file later: `cat experiment_run.log`
- Search log: `grep "FAILED" experiment_run.log`

## Summary: What You Can Monitor

### ✅ Already Live
- Experiment start/completion messages
- GPU assignments
- Duration per experiment
- Success/failure status

### ✅ Available in Separate Terminal
- GPU utilization (`nvidia-smi`)
- Individual training logs (`tail -f`)
- Partial analysis results
- Completion progress

### ✅ After Completion
- Full analysis with all plots
- Best configurations
- Complete rankings

## Quick Monitor Setup

**Terminal 1** (Run experiments):
```bash
python run_parallel_experiments.py --datasets turbofan
```

**Terminal 2** (Monitor GPUs):
```bash
watch -n 1 nvidia-smi
```

**Terminal 3** (Monitor progress):
```bash
watch -n 10 "ls parallel_experiments/logs/*/*/*/*/test_metrics.json 2>/dev/null | wc -l"
```

**Terminal 4** (Follow a specific experiment):
```bash
# Wait for first experiment to start, then:
tail -f parallel_experiments/logs/turbofan/lstm_basic/bs*/training_log.json
```

This gives you **comprehensive real-time monitoring** of all aspects of your parallel training!
