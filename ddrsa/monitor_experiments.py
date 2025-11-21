#!/usr/bin/env python
"""
Real-time Progress Monitor for Parallel DDRSA Experiments

Shows live status of all running experiments including:
- Overall progress (X/Y experiments complete)
- Per-GPU status and current experiment
- Latest epoch progress for each running experiment
- Recent completions
- GPU utilization
- Estimated time remaining

Usage:
    python monitor_experiments.py --output-dir parallel_experiments
"""

import argparse
import json
import os
import time
import glob
from datetime import datetime, timedelta
import subprocess
import sys

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def get_terminal_width():
    """Get terminal width"""
    try:
        return os.get_terminal_size().columns
    except:
        return 80


def parse_training_log(log_path):
    """Parse latest entries from training log"""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return None

            # Get last line with valid JSON
            for line in reversed(lines[-5:]):
                try:
                    return json.loads(line.strip())
                except:
                    continue
            return None
    except:
        return None


def get_experiment_status(exp_dir):
    """Get status of a single experiment"""
    status = {
        'name': os.path.basename(exp_dir),
        'state': 'unknown',
        'progress': None,
        'metrics': None
    }

    # Check if completed
    test_metrics_path = os.path.join(exp_dir, 'test_metrics.json')
    if os.path.exists(test_metrics_path):
        status['state'] = 'completed'
        try:
            with open(test_metrics_path, 'r') as f:
                status['metrics'] = json.load(f)
        except:
            pass
        return status

    # Check if running
    training_log_path = os.path.join(exp_dir, 'training_log.json')
    if os.path.exists(training_log_path):
        status['state'] = 'running'
        latest = parse_training_log(training_log_path)
        if latest:
            status['progress'] = latest
        return status

    # Check if started
    checkpoint_dir = os.path.join(exp_dir)
    if os.path.exists(checkpoint_dir):
        status['state'] = 'started'
        return status

    return status


def get_gpu_info():
    """Get GPU utilization info"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'mem_used': int(parts[2]),
                        'mem_total': int(parts[3]),
                        'util': int(parts[4]),
                        'temp': int(parts[5])
                    })
            return gpus
        return None
    except:
        return None


def format_time(seconds):
    """Format seconds to human readable time"""
    if seconds is None:
        return "Unknown"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def draw_progress_bar(current, total, width=40):
    """Draw a progress bar"""
    if total == 0:
        return "[" + " " * width + "] 0%"

    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    percentage = 100 * current / total
    return f"[{bar}] {percentage:.1f}%"


def monitor_experiments(output_dir, refresh_rate=2):
    """Main monitoring loop"""

    start_time = time.time()

    while True:
        clear_screen()
        term_width = get_terminal_width()

        # Header
        print("=" * term_width)
        print(f"{BOLD}{CYAN}DDRSA PARALLEL EXPERIMENTS - LIVE MONITOR{RESET}".center(term_width))
        print("=" * term_width)
        print()

        # Load manifest
        manifest_path = os.path.join(output_dir, 'experiment_manifest.json')
        if not os.path.exists(manifest_path):
            print(f"{YELLOW}Waiting for experiments to start...{RESET}")
            print(f"Looking for: {manifest_path}")
            time.sleep(refresh_rate)
            continue

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        total_experiments = manifest['total_experiments']

        # Scan all experiment directories
        log_dir = os.path.join(output_dir, 'logs')
        all_exp_dirs = glob.glob(os.path.join(log_dir, '*/*/*/*/*'))

        # Get status for each
        completed = []
        running = []
        pending = []

        for exp_dir in all_exp_dirs:
            status = get_experiment_status(exp_dir)

            if status['state'] == 'completed':
                completed.append(status)
            elif status['state'] in ['running', 'started']:
                running.append(status)
            else:
                pending.append(status)

        num_completed = len(completed)
        num_running = len(running)
        num_pending = total_experiments - num_completed - num_running

        # Overall Progress
        print(f"{BOLD}Overall Progress:{RESET}")
        print(f"  {draw_progress_bar(num_completed, total_experiments, 50)}")
        print(f"  Completed: {GREEN}{num_completed}{RESET} | Running: {YELLOW}{num_running}{RESET} | Pending: {BLUE}{num_pending}{RESET} | Total: {total_experiments}")
        print()

        # Elapsed time and ETA
        elapsed = time.time() - start_time
        if num_completed > 0:
            avg_time_per_exp = elapsed / num_completed
            remaining = num_pending + num_running
            eta_seconds = avg_time_per_exp * remaining
            print(f"  Elapsed: {format_time(elapsed)} | ETA: {format_time(eta_seconds)}")
        else:
            print(f"  Elapsed: {format_time(elapsed)} | ETA: Calculating...")
        print()

        # GPU Status
        print("─" * term_width)
        print(f"{BOLD}GPU Status:{RESET}")
        gpus = get_gpu_info()
        if gpus:
            for gpu in gpus:
                util_color = GREEN if gpu['util'] > 80 else YELLOW if gpu['util'] > 50 else RED
                temp_color = RED if gpu['temp'] > 80 else YELLOW if gpu['temp'] > 70 else GREEN

                mem_pct = 100 * gpu['mem_used'] / gpu['mem_total']
                mem_bar = draw_progress_bar(gpu['mem_used'], gpu['mem_total'], 20)

                print(f"  GPU {gpu['index']}: {gpu['name']}")
                print(f"    Utilization: {util_color}{gpu['util']:3d}%{RESET} | "
                      f"Memory: {mem_bar} ({gpu['mem_used']}/{gpu['mem_total']} MB) | "
                      f"Temp: {temp_color}{gpu['temp']}°C{RESET}")
        else:
            print(f"  {YELLOW}GPU info unavailable{RESET}")
        print()

        # Running Experiments
        print("─" * term_width)
        print(f"{BOLD}Running Experiments ({num_running}):{RESET}")
        if running:
            for i, exp in enumerate(running[:8], 1):  # Show max 8
                exp_name = exp['name'][:60]  # Truncate long names

                if exp['progress']:
                    epoch = exp['progress'].get('epoch', '?')
                    total_epochs = exp['progress'].get('total_epochs', '?')
                    train_loss = exp['progress'].get('train_loss', 0)
                    val_loss = exp['progress'].get('val_loss', 0)
                    mae = exp['progress'].get('rul_mae', 0)
                    cindex = exp['progress'].get('concordance_index', 0)

                    # Determine if total_epochs is available
                    if total_epochs != '?':
                        epoch_progress = draw_progress_bar(epoch, total_epochs, 15)
                    else:
                        epoch_progress = f"Epoch {epoch}"

                    print(f"  {YELLOW}▶{RESET} {exp_name}")
                    print(f"      {epoch_progress} | "
                          f"Loss: {train_loss:.3f}/{val_loss:.3f} | "
                          f"MAE: {mae:.2f} | C-Index: {cindex:.3f}")
                else:
                    print(f"  {YELLOW}▶{RESET} {exp_name}")
                    print(f"      Starting...")

            if num_running > 8:
                print(f"  ... and {num_running - 8} more")
        else:
            print(f"  {YELLOW}No experiments currently running{RESET}")
        print()

        # Recent Completions
        print("─" * term_width)
        print(f"{BOLD}Recent Completions (Last 5):{RESET}")
        if completed:
            # Sort by most recent
            recent = sorted(completed, key=lambda x: x['name'])[-5:]
            for exp in reversed(recent):
                exp_name = exp['name'][:60]

                if exp['metrics']:
                    mae = exp['metrics'].get('rul_mae', 0)
                    rmse = exp['metrics'].get('rul_rmse', 0)
                    cindex = exp['metrics'].get('concordance_index', 0)

                    print(f"  {GREEN}✓{RESET} {exp_name}")
                    print(f"      MAE: {mae:.2f} | RMSE: {rmse:.2f} | C-Index: {cindex:.3f}")
                else:
                    print(f"  {GREEN}✓{RESET} {exp_name}")
        else:
            print(f"  {YELLOW}No experiments completed yet{RESET}")
        print()

        # Best Results So Far
        if completed and any(exp['metrics'] for exp in completed):
            print("─" * term_width)
            print(f"{BOLD}Best Results So Far:{RESET}")

            # Find best by MAE
            completed_with_metrics = [exp for exp in completed if exp['metrics']]
            if completed_with_metrics:
                best_mae = min(completed_with_metrics, key=lambda x: x['metrics'].get('rul_mae', float('inf')))
                best_cindex = max(completed_with_metrics, key=lambda x: x['metrics'].get('concordance_index', 0))

                print(f"  {GREEN}Best MAE:{RESET} {best_mae['metrics']['rul_mae']:.2f} - {best_mae['name'][:50]}")
                print(f"  {GREEN}Best C-Index:{RESET} {best_cindex['metrics']['concordance_index']:.3f} - {best_cindex['name'][:50]}")
            print()

        # Footer
        print("─" * term_width)
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
              f"Refresh rate: {refresh_rate}s | Press Ctrl+C to exit")
        print("=" * term_width)

        # Check if all done
        if num_completed == total_experiments:
            print()
            print(f"{GREEN}{BOLD}🎉 ALL EXPERIMENTS COMPLETE!{RESET}")
            print(f"\nResults saved to: {output_dir}")
            print(f"Check analysis plots in: {os.path.join(output_dir, 'analysis_plots')}")
            break

        # Wait before next update
        time.sleep(refresh_rate)


def main():
    parser = argparse.ArgumentParser(description='Real-time monitor for parallel DDRSA experiments')
    parser.add_argument('--output-dir', type=str, default='parallel_experiments',
                       help='Output directory containing experiments')
    parser.add_argument('--refresh-rate', type=int, default=2,
                       help='Screen refresh rate in seconds (default: 2)')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"{RED}Error: Output directory not found: {args.output_dir}{RESET}")
        print("\nMake sure experiments are running and the directory exists.")
        sys.exit(1)

    try:
        monitor_experiments(args.output_dir, args.refresh_rate)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(0)


if __name__ == '__main__':
    main()
