"""
Parallel Experiment Runner with Hyperparameter Grid Search
Runs multiple DDRSA experiments simultaneously with varied batch sizes and hyperparameters
"""

import argparse
import os
import json
import itertools
import subprocess
import time
from multiprocessing import Process, Queue, cpu_count
from datetime import datetime
import torch

def get_available_gpus():
    """Get list of available GPU indices"""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def worker(gpu_id, task_queue, results_queue, args):
    """Worker process that runs experiments on a specific GPU"""
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill to stop worker
            break

        exp_name, config = task
        print(f"[GPU {gpu_id}] Starting: {exp_name}")

        # Build command
        cmd = [
            "python", "main.py",
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
            cmd.extend(["--activation", "gelu"])
        elif config['model_type'] == 'probsparse':
            cmd.extend(["--d-model", str(config.get('d_model', 512))])
            cmd.extend(["--nhead", str(config.get('nhead', 8))])
            cmd.extend(["--num-encoder-layers", str(config.get('num_encoder_layers', 2))])

        # Add GPU setting
        if gpu_id == "cpu":
            cmd.append("--no-cuda")

        # Set environment variable for GPU
        env = os.environ.copy()
        if gpu_id != "cpu":
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Run experiment
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )
            duration = time.time() - start_time

            if result.returncode == 0:
                status = "SUCCESS"
                print(f"[GPU {gpu_id}] ✓ {exp_name} completed in {duration/60:.1f} min")
            else:
                status = "FAILED"
                print(f"[GPU {gpu_id}] ✗ {exp_name} FAILED")
                print(f"Error: {result.stderr[:500]}")

            results_queue.put({
                'exp_name': exp_name,
                'status': status,
                'duration': duration,
                'config': config,
                'returncode': result.returncode,
                'stdout': result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                'stderr': result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
            })

        except Exception as e:
            duration = time.time() - start_time
            print(f"[GPU {gpu_id}] ✗ {exp_name} EXCEPTION: {str(e)}")
            results_queue.put({
                'exp_name': exp_name,
                'status': 'EXCEPTION',
                'duration': duration,
                'config': config,
                'error': str(e)
            })


def generate_experiments(args):
    """Generate all experiment configurations based on hyperparameter grid"""

    # Define hyperparameter grids
    batch_sizes = args.batch_sizes
    learning_rates = args.learning_rates
    lambda_params = args.lambda_params
    nasa_weights = args.nasa_weights
    dropout_rates = args.dropout_rates
    datasets = args.datasets

    # Define model variants to test
    model_configs = []

    if args.include_lstm:
        model_configs.extend([
            {'model_type': 'rnn', 'rnn_type': 'LSTM', 'hidden_dim': 128, 'num_layers': 2, 'variant': 'lstm_basic'},
            {'model_type': 'rnn', 'rnn_type': 'LSTM', 'hidden_dim': 256, 'num_layers': 4, 'variant': 'lstm_deep'},
        ])

    if args.include_gru:
        model_configs.extend([
            {'model_type': 'rnn', 'rnn_type': 'GRU', 'hidden_dim': 128, 'num_layers': 2, 'variant': 'gru_basic'},
            {'model_type': 'rnn', 'rnn_type': 'GRU', 'hidden_dim': 256, 'num_layers': 4, 'variant': 'gru_deep'},
        ])

    if args.include_transformer:
        model_configs.extend([
            {'model_type': 'transformer', 'd_model': 128, 'nhead': 8, 'num_encoder_layers': 4, 'num_decoder_layers': 4, 'variant': 'transformer_basic'},
            {'model_type': 'transformer', 'd_model': 256, 'nhead': 8, 'num_encoder_layers': 6, 'num_decoder_layers': 6, 'variant': 'transformer_deep'},
        ])

    if args.include_probsparse:
        model_configs.extend([
            {'model_type': 'probsparse', 'd_model': 512, 'nhead': 8, 'num_encoder_layers': 2, 'variant': 'probsparse_basic'},
            {'model_type': 'probsparse', 'd_model': 512, 'nhead': 8, 'num_encoder_layers': 4, 'variant': 'probsparse_deep'},
        ])

    # Generate all combinations
    experiments = []
    exp_id = 0

    for dataset in datasets:
        for model_config in model_configs:
            for batch_size, lr, lambda_p, nasa_w, dropout in itertools.product(
                batch_sizes, learning_rates, lambda_params, nasa_weights, dropout_rates
            ):
                # Create experiment config
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

                # Create experiment name
                exp_name = (
                    f"{dataset}/"
                    f"{model_config['variant']}/"
                    f"bs{batch_size}_lr{lr}_lam{lambda_p}_nasa{nasa_w}_drop{dropout}_id{exp_id}"
                )

                experiments.append((exp_name, config))
                exp_id += 1

    return experiments


def main():
    parser = argparse.ArgumentParser(description='Parallel DDRSA Hyperparameter Grid Search')

    # Output settings
    parser.add_argument('--output-dir', type=str, default='parallel_experiments',
                       help='Output directory for all experiments')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Parallel execution settings
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of parallel workers (default: number of GPUs, or 1 if CPU)')
    parser.add_argument('--use-cpu', action='store_true',
                       help='Force CPU usage even if GPUs are available')
    parser.add_argument('--no-auto-analysis', action='store_true',
                       help='Skip automatic analysis after training (run manually later)')

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

    # Hyperparameter grids
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[64, 128, 256],
                       help='Batch sizes to test')
    parser.add_argument('--learning-rates', nargs='+', type=float, default=[0.0005, 0.001, 0.005],
                       help='Learning rates to test')
    parser.add_argument('--lambda-params', nargs='+', type=float, default=[0.5,0.6,0.75],
                       help='Lambda parameters to test')
    parser.add_argument('--nasa-weights', nargs='+', type=float, default=[0.0,0.05,0.1],
                       help='NASA loss weights to test (0.0 = disabled)')
    parser.add_argument('--dropout-rates', nargs='+', type=float, default=[0.1, 0.15, 0.2],
                       help='Dropout rates to test')

    args = parser.parse_args()

    # Determine available compute resources
    if args.use_cpu:
        available_devices = ["cpu"]
        if args.num_workers is None:
            args.num_workers = 1
    else:
        available_devices = get_available_gpus()
        if not available_devices:
            print("No GPUs available, falling back to CPU")
            available_devices = ["cpu"]
            if args.num_workers is None:
                args.num_workers = 1
        else:
            if args.num_workers is None:
                args.num_workers = len(available_devices)
            print(f"Found {len(available_devices)} GPUs: {available_devices}")

    print(f"Using {args.num_workers} parallel workers")

    # Generate all experiments
    experiments = generate_experiments(args)
    total_experiments = len(experiments)

    print("\n" + "="*80)
    print("PARALLEL EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Total experiments: {total_experiments}")
    print(f"Datasets: {args.datasets}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Learning rates: {args.learning_rates}")
    print(f"Lambda params: {args.lambda_params}")
    print(f"NASA weights: {args.nasa_weights}")
    print(f"Dropout rates: {args.dropout_rates}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Output directory: {args.output_dir}")
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

    # Create task and results queues
    task_queue = Queue()
    results_queue = Queue()

    # Add all experiments to task queue
    for exp in experiments:
        task_queue.put(exp)

    # Add poison pills to stop workers
    for _ in range(args.num_workers):
        task_queue.put(None)

    # Start worker processes
    workers = []
    for i in range(args.num_workers):
        # Assign GPU to worker (round-robin if more workers than GPUs)
        if available_devices[0] == "cpu":
            device_id = "cpu"
        else:
            device_id = available_devices[i % len(available_devices)]

        p = Process(target=worker, args=(device_id, task_queue, results_queue, args))
        p.start()
        workers.append(p)
        print(f"Started worker {i+1}/{args.num_workers} on device {device_id}")

    # Wait for all workers to finish
    for p in workers:
        p.join()

    # Collect all results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Save results
    results_file = os.path.join(args.output_dir, 'all_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']
    exceptions = [r for r in results if r['status'] == 'EXCEPTION']

    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Exceptions: {len(exceptions)}")

    if successful:
        total_time = sum(r['duration'] for r in successful)
        avg_time = total_time / len(successful)
        print(f"Average time per experiment: {avg_time/60:.1f} minutes")
        print(f"Total time: {total_time/3600:.1f} hours")

    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  - {r['exp_name']}")

    if exceptions:
        print("\nExceptions:")
        for r in exceptions:
            print(f"  - {r['exp_name']}: {r.get('error', 'Unknown error')}")

    print("="*80)
    print(f"Results saved to: {results_file}")
    print("="*80 + "\n")

    # Automatically run analysis if we have successful experiments
    if successful and not args.no_auto_analysis:
        print("\n" + "="*80)
        print("RUNNING AUTOMATIC ANALYSIS")
        print("="*80 + "\n")

        try:
            import subprocess
            import sys

            analysis_cmd = [
                sys.executable,
                "analyze_parallel_results.py",
                "--output-dir", args.output_dir
            ]

            print("Executing: " + " ".join(analysis_cmd))
            print("")

            result = subprocess.run(analysis_cmd, check=True)

            if result.returncode == 0:
                print("\n" + "="*80)
                print("✓ ANALYSIS COMPLETE")
                print("="*80)
                print(f"\nAll results available in: {os.path.abspath(args.output_dir)}")
                print(f"  - Experiment results: {results_file}")
                print(f"  - Analysis plots: {os.path.join(args.output_dir, 'analysis_plots')}")
                print(f"  - Best configs: {os.path.join(args.output_dir, 'best_configurations.csv')}")
                print(f"  - Top 20 by MAE: {os.path.join(args.output_dir, 'top_20_by_mae.csv')}")
                print("="*80 + "\n")

        except subprocess.CalledProcessError as e:
            print(f"\n⚠ Analysis failed with error code {e.returncode}")
            print("You can run analysis manually with:")
            print(f"  python analyze_parallel_results.py --output-dir {args.output_dir}")
        except Exception as e:
            print(f"\n⚠ Could not run analysis: {str(e)}")
            print("You can run analysis manually with:")
            print(f"  python analyze_parallel_results.py --output-dir {args.output_dir}")
    elif successful and args.no_auto_analysis:
        print("\n" + "="*80)
        print("Automatic analysis skipped (--no-auto-analysis flag used)")
        print("="*80)
        print("\nTo analyze results, run:")
        print(f"  python analyze_parallel_results.py --output-dir {args.output_dir}")
        print("="*80 + "\n")
    else:
        print("\n⚠ No successful experiments to analyze")
        print("Check the errors above and try again.")


if __name__ == '__main__':
    main()
