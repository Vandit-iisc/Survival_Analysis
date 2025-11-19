#!/usr/bin/env python
"""
Run all experiment variants for both datasets (turbofan and azure_pm).
Results are stored in 20_nov_experiments with logical subfolder structure.

Folder structure:
20_nov_experiments/
├── turbofan/
│   ├── transformer_basic/
│   ├── transformer_deep/
│   ├── probsparse_basic/
│   ├── lstm_basic/
│   ├── gru_basic/
│   └── ...
└── azure_pm/
    └── (same variants)

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --device cpu
    python run_all_experiments.py --datasets turbofan --variants lstm_basic transformer_basic
    python run_all_experiments.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# Base directory for experiments
BASE_DIR = "../20_nov_experiments"

# Dataset configurations
DATASETS = {
    "turbofan": {
        "data_path": "../Challenge_Data",
        "use_paper_split": True,
    },
    "azure_pm": {
        "data_path": "../AMLWorkshop/Data",
        "use_paper_split": False,
    }
}

# Model variant configurations
VARIANTS = {
    # ==================== Transformer variants ====================
    "transformer_basic": {
        "model_type": "transformer",
        "num_epochs": 200,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "d_model": 64,
        "nhead": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "activation": "relu",
        "batch_size": 32,
        "learning_rate": 0.0001,
    },
    "transformer_deep": {
        "model_type": "transformer",
        "num_epochs": 250,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "d_model": 64,
        "nhead": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "activation": "relu",
        "batch_size": 32,
        "learning_rate": 0.0001,
    },
    "transformer_wide": {
        "model_type": "transformer",
        "num_epochs": 200,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "d_model": 128,
        "nhead": 8,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "activation": "relu",
        "batch_size": 24,
        "learning_rate": 0.00008,
    },
    "transformer_gelu": {
        "model_type": "transformer",
        "num_epochs": 200,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "d_model": 64,
        "nhead": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "activation": "gelu",
        "batch_size": 32,
        "learning_rate": 0.0001,
    },
    "transformer_complex": {
        "model_type": "transformer",
        "num_epochs": 300,
        "num_encoder_layers": 6,
        "num_decoder_layers": 4,
        "d_model": 128,
        "nhead": 8,
        "dim_feedforward": 512,
        "dropout": 0.15,
        "activation": "relu",
        "batch_size": 16,
        "learning_rate": 0.00005,
    },

    # ==================== ProbSparse/Informer variants ====================
    "probsparse_basic": {
        "model_type": "probsparse",
        "num_epochs": 200,
        "num_encoder_layers": 2,
        "num_decoder_layers": 1,
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 2048,
        "dropout": 0.05,
        "activation": "gelu",
        "batch_size": 32,
        "learning_rate": 0.0001,
        "factor": 5,  # ProbSparse attention factor
    },
    "probsparse_deep": {
        "model_type": "probsparse",
        "num_epochs": 250,
        "num_encoder_layers": 4,
        "num_decoder_layers": 2,
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 2048,
        "dropout": 0.05,
        "activation": "gelu",
        "batch_size": 24,
        "learning_rate": 0.00008,
        "factor": 5,
    },

    # ==================== LSTM variants ====================
    "lstm_paper_exact": {
        "model_type": "rnn",
        "rnn_type": "LSTM",
        "num_epochs": 200,
        "hidden_dim": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "batch_size": 512,
        "learning_rate": 0.01,
    },
    "lstm_basic": {
        "model_type": "rnn",
        "rnn_type": "LSTM",
        "num_epochs": 200,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_size": 512,
        "learning_rate": 0.01,
    },
    "lstm_deep": {
        "model_type": "rnn",
        "rnn_type": "LSTM",
        "num_epochs": 250,
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.2,
        "batch_size": 512,
        "learning_rate": 0.01,
    },
    "lstm_wide": {
        "model_type": "rnn",
        "rnn_type": "LSTM",
        "num_epochs": 200,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_size": 512,
        "learning_rate": 0.01,
    },
    "lstm_complex": {
        "model_type": "rnn",
        "rnn_type": "LSTM",
        "num_epochs": 300,
        "hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.3,
        "batch_size": 256,
        "learning_rate": 0.008,
    },

    # ==================== GRU variants ====================
    "gru_basic": {
        "model_type": "rnn",
        "rnn_type": "GRU",
        "num_epochs": 200,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_size": 512,
        "learning_rate": 0.01,
    },
    "gru_deep": {
        "model_type": "rnn",
        "rnn_type": "GRU",
        "num_epochs": 250,
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.2,
        "batch_size": 512,
        "learning_rate": 0.01,
    },
    "gru_wide": {
        "model_type": "rnn",
        "rnn_type": "GRU",
        "num_epochs": 200,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_size": 512,
        "learning_rate": 0.01,
    },
    "gru_complex": {
        "model_type": "rnn",
        "rnn_type": "GRU",
        "num_epochs": 300,
        "hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.3,
        "batch_size": 256,
        "learning_rate": 0.008,
    },
}

def build_command(dataset, variant, config, dataset_config, args):
    """Build the command for running a single experiment."""
    exp_name = f"{dataset}_{variant}"
    output_dir = os.path.join(BASE_DIR, dataset, variant)

    cmd = [
        sys.executable, "main.py",
        "--dataset", dataset,
        "--data-path", dataset_config["data_path"],
        "--model-type", config["model_type"],
        "--num-epochs", str(config["num_epochs"]),
        "--patience", str(args.patience),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--pred-horizon", str(args.pred_horizon),
        "--lambda-param", str(args.lambda_param),
        "--lookback-window", str(args.lookback_window),
        "--output-dir", output_dir,
        "--exp-name", exp_name,
        "--device", args.device,
        "--use-minmax",
        "--create-visualization",
    ]

    # Add paper split for turbofan
    if dataset_config.get("use_paper_split"):
        cmd.append("--use-paper-split")

    # Model-specific arguments
    if config["model_type"] in ["transformer", "probsparse"]:
        cmd.extend([
            "--num-encoder-layers", str(config["num_encoder_layers"]),
            "--num-decoder-layers", str(config["num_decoder_layers"]),
            "--d-model", str(config["d_model"]),
            "--nhead", str(config["nhead"]),
            "--dim-feedforward", str(config["dim_feedforward"]),
            "--dropout", str(config["dropout"]),
            "--activation", config["activation"],
        ])
        # ProbSparse specific
        if config["model_type"] == "probsparse" and "factor" in config:
            cmd.extend(["--factor", str(config["factor"])])
    else:  # RNN (LSTM/GRU)
        cmd.extend([
            "--rnn-type", config["rnn_type"],
            "--hidden-dim", str(config["hidden_dim"]),
            "--num-layers", str(config["num_layers"]),
            "--dropout", str(config["dropout"]),
        ])

    return cmd, exp_name, output_dir

def run_experiment(cmd, exp_name, output_dir, show_output=False):
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    try:
        if show_output:
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        else:
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True
            )

        elapsed_time = time.time() - start_time

        if result.returncode != 0:
            print(f"FAILED: {exp_name} (took {elapsed_time:.2f}s)")
            if not show_output and result.stderr:
                # Save error log
                error_file = os.path.join(output_dir, "error.log")
                with open(error_file, "w") as f:
                    f.write(result.stderr)
                print(f"Error log saved to: {error_file}")
            return {
                "status": "failed",
                "time": elapsed_time,
                "error": result.stderr if not show_output else None
            }
        else:
            print(f"SUCCESS: {exp_name} (took {elapsed_time:.2f}s)")
            return {
                "status": "success",
                "time": elapsed_time
            }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"ERROR: {exp_name}: {str(e)}")
        return {
            "status": "error",
            "time": elapsed_time,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Run all DDRSA experiments.')

    # Experiment selection
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to run (default: all)")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Specific variants to run (default: all)")

    # Training parameters (paper-exact defaults)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--pred-horizon", type=int, default=64,
                        help="Prediction horizon")
    parser.add_argument("--lambda-param", type=float, default=0.75,
                        help="Lambda parameter")
    parser.add_argument("--lookback-window", type=int, default=128,
                        help="Lookback window size")
    parser.add_argument("--device", default="cuda",
                        help="Device to use (cuda/cpu)")

    # Output control
    parser.add_argument("--show-output", action="store_true",
                        help="Show real-time output from experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    args = parser.parse_args()

    # Select datasets and variants
    datasets_to_run = args.datasets if args.datasets else list(DATASETS.keys())
    variants_to_run = args.variants if args.variants else list(VARIANTS.keys())

    # Validate selections
    for d in datasets_to_run:
        if d not in DATASETS:
            print(f"Error: Unknown dataset '{d}'. Available: {list(DATASETS.keys())}")
            sys.exit(1)
    for v in variants_to_run:
        if v not in VARIANTS:
            print(f"Error: Unknown variant '{v}'. Available: {list(VARIANTS.keys())}")
            sys.exit(1)

    # Print header
    print(f"\n{'#'*60}")
    print(f"# DDRSA Experiment Runner")
    print(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    print(f"\nDatasets: {datasets_to_run}")
    print(f"Variants: {variants_to_run}")
    print(f"Total experiments: {len(datasets_to_run) * len(variants_to_run)}")
    print(f"Device: {args.device}")
    print(f"Output directory: {os.path.abspath(BASE_DIR)}")

    if args.dry_run:
        print("\n*** DRY RUN - Commands will be printed but not executed ***\n")

    # Create base output directory
    os.makedirs(BASE_DIR, exist_ok=True)

    # Run experiments
    results = {}
    total = len(datasets_to_run) * len(variants_to_run)
    completed = 0
    overall_start = time.time()

    for dataset in datasets_to_run:
        results[dataset] = {}
        dataset_config = DATASETS[dataset]

        for variant in variants_to_run:
            completed += 1
            print(f"\nProgress: {completed}/{total}")

            config = VARIANTS[variant]
            cmd, exp_name, output_dir = build_command(
                dataset, variant, config, dataset_config, args
            )

            if args.dry_run:
                print(f"\nCommand for {exp_name}:")
                print(" \\\n    ".join(cmd))
                results[dataset][variant] = {"status": "dry_run", "time": 0}
            else:
                result = run_experiment(cmd, exp_name, output_dir, args.show_output)
                results[dataset][variant] = result

    overall_time = time.time() - overall_start

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    success_count = 0
    fail_count = 0

    for dataset in datasets_to_run:
        print(f"\n{dataset}:")
        for variant in variants_to_run:
            result = results[dataset][variant]
            status = result["status"]
            time_taken = result.get("time", 0)

            if status == "success":
                success_count += 1
                print(f"  {variant}: SUCCESS ({time_taken:.2f}s)")
            elif status == "dry_run":
                print(f"  {variant}: DRY RUN")
            else:
                fail_count += 1
                print(f"  {variant}: FAILED ({time_taken:.2f}s)")

    print(f"\n{'='*60}")
    if not args.dry_run:
        print(f"Total: {success_count} succeeded, {fail_count} failed")
        print(f"Total time: {overall_time:.2f}s ({overall_time/60:.2f} min)")
    print(f"Results saved to: {os.path.abspath(BASE_DIR)}")
    print(f"{'='*60}")

    # Save summary
    summary_file = os.path.join(BASE_DIR, "experiment_summary.json")
    summary = {
        "date": datetime.now().isoformat(),
        "datasets": datasets_to_run,
        "variants": variants_to_run,
        "total_time": overall_time,
        "success_count": success_count,
        "fail_count": fail_count,
        "results": results,
        "args": vars(args)
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4, default=str)
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
