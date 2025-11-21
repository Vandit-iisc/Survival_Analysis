"""
Sequential Experiment Runner with Live Progress Display
Runs DDRSA experiments one at a time with full hyperparameter grid search

Shows live epoch-by-epoch progress for each experiment
Includes: batch size optimization, learning rate tuning, lambda optimization,
          NASA loss tuning, dropout optimization
"""

import argparse
import os
import json
import itertools
import subprocess
import time
from datetime import datetime
import sys


def generate_experiments(args):
    """Generate all experiment configurations"""

    # Define comprehensive hyperparameter grids
    batch_sizes = args.batch_sizes
    learning_rates = args.learning_rates
    lambda_params = args.lambda_params
    nasa_weights = args.nasa_weights
    dropout_rates = args.dropout_rates
    datasets = args.datasets

    # Define all model variants
    model_configs = []

    if args.include_lstm:
        model_configs.extend([
            {
                'model_type': 'rnn',
                'rnn_type': 'LSTM',
                'hidden_dim': 16,
                'num_layers': 1,
                'variant': 'lstm_paper_exact'
            },
            {
                'model_type': 'rnn',
                'rnn_type': 'LSTM',
                'hidden_dim': 128,
                'num_layers': 2,
                'variant': 'lstm_basic'
            },
            {
                'model_type': 'rnn',
                'rnn_type': 'LSTM',
                'hidden_dim': 256,
                'num_layers': 4,
                'variant': 'lstm_deep'
            },
            {
                'model_type': 'rnn',
                'rnn_type': 'LSTM',
                'hidden_dim': 256,
                'num_layers': 2,
                'variant': 'lstm_wide'
            },
            {
                'model_type': 'rnn',
                'rnn_type': 'LSTM',
                'hidden_dim': 512,
                'num_layers': 4,
                'variant': 'lstm_complex'
            },
        ])

    if args.include_gru:
        model_configs.extend([
            {
                'model_type': 'rnn',
                'rnn_type': 'GRU',
                'hidden_dim': 128,
                'num_layers': 2,
                'variant': 'gru_basic'
            },
            {
                'model_type': 'rnn',
                'rnn_type': 'GRU',
                'hidden_dim': 256,
                'num_layers': 4,
                'variant': 'gru_deep'
            },
            {
                'model_type': 'rnn',
                'rnn_type': 'GRU',
                'hidden_dim': 256,
                'num_layers': 2,
                'variant': 'gru_wide'
            },
            {
                'model_type': 'rnn',
                'rnn_type': 'GRU',
                'hidden_dim': 512,
                'num_layers': 4,
                'variant': 'gru_complex'
            },
        ])

    if args.include_transformer:
        model_configs.extend([
            {
                'model_type': 'transformer',
                'd_model': 64,
                'nhead': 4,
                'num_encoder_layers': 2,
                'num_decoder_layers': 2,
                'activation': 'gelu',
                'variant': 'transformer_basic'
            },
            {
                'model_type': 'transformer',
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 6,
                'num_decoder_layers': 4,
                'activation': 'gelu',
                'variant': 'transformer_deep'
            },
            {
                'model_type': 'transformer',
                'd_model': 256,
                'nhead': 8,
                'num_encoder_layers': 4,
                'num_decoder_layers': 4,
                'activation': 'gelu',
                'variant': 'transformer_wide'
            },
            {
                'model_type': 'transformer',
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 4,
                'num_decoder_layers': 4,
                'activation': 'gelu',
                'variant': 'transformer_gelu'
            },
            {
                'model_type': 'transformer',
                'd_model': 256,
                'nhead': 16,
                'num_encoder_layers': 8,
                'num_decoder_layers': 6,
                'activation': 'gelu',
                'variant': 'transformer_complex'
            },
        ])

    if args.include_probsparse:
        model_configs.extend([
            {
                'model_type': 'probsparse',
                'd_model': 512,
                'nhead': 8,
                'num_encoder_layers': 2,
                'variant': 'probsparse_basic'
            },
            {
                'model_type': 'probsparse',
                'd_model': 512,
                'nhead': 8,
                'num_encoder_layers': 4,
                'variant': 'probsparse_deep'
            },
        ])

    # Generate all combinations
    experiments = []
    exp_id = 0

    for dataset in datasets:
        for model_config in model_configs:
            for batch_size, lr, lambda_p, nasa_w, dropout in itertools.product(
                batch_sizes, learning_rates, lambda_params, nasa_weights, dropout_rates
            ):
                config = {
                    'dataset': dataset,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'lambda_param': lambda_p,
                    'nasa_weight': nasa_w,
                    'dropout': dropout,
                    'seed': args.seed,
                    **model_config
                }

                exp_name = (
                    f"{dataset}/"
                    f"{model_config['variant']}/"
                    f"bs{batch_size}_lr{lr}_lam{lambda_p}_nasa{nasa_w}_drop{dropout}_id{exp_id}"
                )

                experiments.append((exp_name, config))
                exp_id += 1

    return experiments


def run_experiment(exp_name, config, args):
    """Run a single experiment with live output"""

    # Build command
    cmd = [
        sys.executable, "main.py",
        "--dataset", config['dataset'],
        "--model-type", config['model_type'],
        "--batch-size", str(config['batch_size']),
        "--learning-rate", str(config['learning_rate']),
        "--lambda-param", str(config['lambda_param']),
        "--num-epochs", str(args.num_epochs),
        "--exp-name", exp_name,
        "--output-dir", args.output_dir,
        "--seed", str(config['seed']),
    ]

    # Add optional hyperparameters
    if 'dropout' in config:
        cmd.extend(["--dropout", str(config['dropout'])])

    if 'nasa_weight' in config and config['nasa_weight'] > 0:
        cmd.append("--use-nasa-loss")
        cmd.extend(["--nasa-weight", str(config['nasa_weight'])])

    # Add model-specific parameters
    if config['model_type'] == 'rnn':
        cmd.extend(["--hidden-dim", str(config.get('hidden_dim', 128))])
        cmd.extend(["--num-layers", str(config.get('num_layers', 2))])
        cmd.extend(["--rnn-type", config.get('rnn_type', 'LSTM')])
    elif config['model_type'] == 'transformer':
        cmd.extend(["--d-model", str(config.get('d_model', 128))])
        cmd.extend(["--nhead", str(config.get('nhead', 8))])
        cmd.extend(["--num-encoder-layers", str(config.get('num_encoder_layers', 4))])
        cmd.extend(["--num-decoder-layers", str(config.get('num_decoder_layers', 4))])
        cmd.extend(["--activation", config.get('activation', 'gelu')])
    elif config['model_type'] == 'probsparse':
        cmd.extend(["--d-model", str(config.get('d_model', 512))])
        cmd.extend(["--nhead", str(config.get('nhead', 8))])
        cmd.extend(["--num-encoder-layers", str(config.get('num_encoder_layers', 2))])

    # Run experiment with live output
    start_time = time.time()

    try:
        # Run with stdout/stderr forwarded to console (live output!)
        result = subprocess.run(cmd, check=False)

        duration = time.time() - start_time

        if result.returncode == 0:
            status = "SUCCESS"
            print(f"\n{'='*80}")
            print(f"✓ Experiment completed successfully in {duration/60:.1f} minutes")
            print(f"{'='*80}\n")
        else:
            status = "FAILED"
            print(f"\n{'='*80}")
            print(f"✗ Experiment FAILED with return code {result.returncode}")
            print(f"{'='*80}\n")

        return {
            'exp_name': exp_name,
            'status': status,
            'duration': duration,
            'config': config,
            'returncode': result.returncode
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✗ Experiment EXCEPTION: {str(e)}")
        print(f"{'='*80}\n")

        return {
            'exp_name': exp_name,
            'status': 'EXCEPTION',
            'duration': duration,
            'config': config,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Sequential DDRSA Hyperparameter Grid Search with Live Progress')

    # Output settings
    parser.add_argument('--output-dir', type=str, default='sequential_experiments',
                       help='Output directory for all experiments')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Dataset selection
    parser.add_argument('--datasets', nargs='+', default=['turbofan', 'azure_pm'],
                       choices=['turbofan', 'azure_pm'],
                       help='Datasets to run experiments on')

    # Model selection
    parser.add_argument('--include-lstm', action='store_true', default=True,
                       help='Include LSTM variants')
    parser.add_argument('--include-gru', action='store_true', default=True,
                       help='Include GRU variants')
    parser.add_argument('--include-transformer', action='store_true', default=True,
                       help='Include Transformer variants')
    parser.add_argument('--include-probsparse', action='store_true', default=True,
                       help='Include ProbSparse variants')

    # Hyperparameter grids - COMPREHENSIVE
    parser.add_argument('--batch-sizes', nargs='+', type=int,
                       default=[32, 64, 128, 256, 512],
                       help='Batch sizes to test (default: comprehensive)')
    parser.add_argument('--learning-rates', nargs='+', type=float,
                       default=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                       help='Learning rates to test (default: comprehensive)')
    parser.add_argument('--lambda-params', nargs='+', type=float,
                       default=[0.5, 0.6, 0.75, 0.9],
                       help='Lambda parameters to test (default: comprehensive)')
    parser.add_argument('--nasa-weights', nargs='+', type=float,
                       default=[0.0, 0.05, 0.1, 0.2],
                       help='NASA loss weights to test (default: comprehensive)')
    parser.add_argument('--dropout-rates', nargs='+', type=float,
                       default=[0.0, 0.1, 0.15, 0.2, 0.3],
                       help='Dropout rates to test (default: comprehensive)')

    # Analysis
    parser.add_argument('--no-auto-analysis', action='store_true',
                       help='Skip automatic analysis after training')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("SEQUENTIAL DDRSA EXPERIMENT RUNNER")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Datasets: {args.datasets}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Learning rates: {args.learning_rates}")
    print(f"  Lambda params: {args.lambda_params}")
    print(f"  NASA weights: {args.nasa_weights}")
    print(f"  Dropout rates: {args.dropout_rates}")
    print(f"  Epochs per experiment: {args.num_epochs}")
    print(f"  Output directory: {args.output_dir}")
    print("="*80 + "\n")

    # Generate all experiments
    experiments = generate_experiments(args)
    total_experiments = len(experiments)

    print(f"Generated {total_experiments} experiments\n")

    # Calculate approximate time
    avg_time_per_exp = 20  # minutes (rough estimate)
    total_hours = (total_experiments * avg_time_per_exp) / 60
    print(f"Estimated time: ~{total_hours:.1f} hours ({total_hours/24:.1f} days)")
    print("="*80 + "\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save experiment manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': total_experiments,
        'args': vars(args),
        'experiments': [{'name': name, 'config': config} for name, config in experiments]
    }

    with open(os.path.join(args.output_dir, 'experiment_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # Run experiments sequentially
    results = []
    start_time = time.time()

    for idx, (exp_name, config) in enumerate(experiments, 1):
        print("\n" + "="*80)
        print(f"EXPERIMENT {idx}/{total_experiments}")
        print("="*80)
        print(f"Name: {exp_name}")
        print(f"Dataset: {config['dataset']}")
        print(f"Model: {config['variant']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Lambda: {config['lambda_param']}")
        print(f"NASA weight: {config['nasa_weight']}")
        print(f"Dropout: {config['dropout']}")

        # Calculate progress and ETA
        elapsed = time.time() - start_time
        if idx > 1:
            avg_time_per_exp = elapsed / (idx - 1)
            remaining = total_experiments - idx + 1
            eta_seconds = avg_time_per_exp * remaining
            eta_hours = eta_seconds / 3600
            print(f"\nProgress: {idx-1}/{total_experiments} complete")
            print(f"Elapsed: {elapsed/3600:.1f} hours")
            print(f"ETA: {eta_hours:.1f} hours")

        print("="*80 + "\n")

        # Run experiment
        result = run_experiment(exp_name, config, args)
        results.append(result)

        # Save results after each experiment (fault tolerance)
        results_file = os.path.join(args.output_dir, 'all_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Final summary
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)

    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']
    exceptions = [r for r in results if r['status'] == 'EXCEPTION']

    print(f"\nTotal experiments: {total_experiments}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Exceptions: {len(exceptions)}")
    print(f"\nTotal time: {total_time/3600:.1f} hours ({total_time/3600/24:.1f} days)")

    if successful:
        avg_time = sum(r['duration'] for r in successful) / len(successful)
        print(f"Average time per experiment: {avg_time/60:.1f} minutes")

    print("="*80 + "\n")

    # Automatically run analysis
    if successful and not args.no_auto_analysis:
        print("\n" + "="*80)
        print("RUNNING AUTOMATIC ANALYSIS")
        print("="*80 + "\n")

        try:
            analysis_cmd = [
                sys.executable,
                "analyze_parallel_results.py",
                "--output-dir", args.output_dir
            ]

            result = subprocess.run(analysis_cmd, check=True)

            if result.returncode == 0:
                print("\n" + "="*80)
                print("✓ ANALYSIS COMPLETE")
                print("="*80)
                print(f"\nAll results available in: {os.path.abspath(args.output_dir)}")
                print(f"  - Analysis plots: {os.path.join(args.output_dir, 'analysis_plots')}")
                print(f"  - Best configs: {os.path.join(args.output_dir, 'best_configurations.csv')}")
                print(f"  - Top 20 by MAE: {os.path.join(args.output_dir, 'top_20_by_mae.csv')}")
                print("="*80 + "\n")

        except Exception as e:
            print(f"\n⚠ Analysis failed: {str(e)}")
            print("You can run analysis manually with:")
            print(f"  python analyze_parallel_results.py --output-dir {args.output_dir}")


if __name__ == '__main__':
    main()
