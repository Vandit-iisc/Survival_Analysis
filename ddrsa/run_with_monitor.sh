#!/bin/bash
# Automated launcher that runs experiments and monitoring dashboard together

set -e

# Parse arguments
OUTPUT_DIR="parallel_experiments"
EXTRA_ARGS=""

# Check if user provided arguments
if [ $# -gt 0 ]; then
    # If first arg looks like an output dir flag
    for arg in "$@"; do
        if [[ $arg == --output-dir=* ]]; then
            OUTPUT_DIR="${arg#*=}"
        elif [[ $arg == "--output-dir" ]]; then
            shift
            OUTPUT_DIR="$1"
        fi
    done
    EXTRA_ARGS="$@"
fi

echo "=========================================="
echo "DDRSA Parallel Experiments with Live Monitor"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Extra arguments: $EXTRA_ARGS"
echo ""

# Check if tmux is available
if command -v tmux &> /dev/null; then
    echo "Using tmux for split-screen view..."
    echo ""

    # Create a new tmux session with two panes
    tmux new-session -d -s ddrsa_experiments

    # Run experiments in first pane
    tmux send-keys -t ddrsa_experiments "python run_parallel_experiments.py $EXTRA_ARGS" C-m

    # Split window horizontally
    tmux split-window -h -t ddrsa_experiments

    # Wait a moment for experiments to initialize
    tmux send-keys -t ddrsa_experiments "echo 'Waiting for experiments to start...'; sleep 3" C-m

    # Run monitor in second pane
    tmux send-keys -t ddrsa_experiments "python monitor_experiments.py --output-dir $OUTPUT_DIR" C-m

    # Attach to the session
    echo "Starting tmux session..."
    echo ""
    echo "Commands:"
    echo "  - Switch panes: Ctrl+B, then arrow keys"
    echo "  - Detach: Ctrl+B, then D"
    echo "  - Reattach: tmux attach -t ddrsa_experiments"
    echo "  - Kill session: tmux kill-session -t ddrsa_experiments"
    echo ""
    sleep 2
    tmux attach-session -t ddrsa_experiments

elif command -v screen &> /dev/null; then
    echo "Using screen for split-screen view..."
    echo ""

    # Create a screen session
    screen -dmS ddrsa_experiments

    # Run experiments
    screen -S ddrsa_experiments -X stuff "python run_parallel_experiments.py $EXTRA_ARGS\n"

    # Split screen
    screen -S ddrsa_experiments -X split

    # Switch to new region
    screen -S ddrsa_experiments -X focus

    # Run monitor
    screen -S ddrsa_experiments -X stuff "sleep 3; python monitor_experiments.py --output-dir $OUTPUT_DIR\n"

    echo "Starting screen session..."
    echo ""
    echo "Commands:"
    echo "  - Switch panes: Ctrl+A, then Tab"
    echo "  - Detach: Ctrl+A, then D"
    echo "  - Reattach: screen -r ddrsa_experiments"
    echo ""
    sleep 2
    screen -r ddrsa_experiments

else
    echo "Neither tmux nor screen found!"
    echo ""
    echo "Please install tmux or screen for split-screen monitoring:"
    echo "  macOS: brew install tmux"
    echo "  Linux: sudo apt install tmux"
    echo ""
    echo "Or run manually in two terminals:"
    echo ""
    echo "  Terminal 1: python run_parallel_experiments.py $EXTRA_ARGS"
    echo "  Terminal 2: python monitor_experiments.py --output-dir $OUTPUT_DIR"
    echo ""
    exit 1
fi
