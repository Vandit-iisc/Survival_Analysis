"""
Run all experiments with NASA Loss Function
Tests 4 model architectures on Turbofan dataset with NASA scoring loss
"""

import argparse
import torch
import os
import json
import numpy as np
import random
from datetime import datetime

from data_loader import get_dataloaders
from models import create_ddrsa_model
from loss import DDRSALossDetailedWithNASA
from trainer import DDRSATrainer
from visualization import (
    plot_hazard_progression,
    plot_oti_policy,
    plot_training_curves,
    create_all_visualizations
)


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_experiment_configs():
    """
    Define all experiment configurations

    Returns 4 experiments (one for each model type):
    1. LSTM (exact paper implementation)
    2. RNN/GRU
    3. Transformer
    4. ProbSparse Attention (Informer)
    """

    base_config = {
        'lookback_window': 128,
        'pred_horizon': 100,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'grad_clip': 1.0,
        'patience': 20,
        'save_interval': 10,
        'lambda_param': 0.5,  # For DDRSA loss (l_z vs l_u+l_c)
        'nasa_weight': 0.1,   # Weight for NASA loss component
    }

    experiments = []

    # 1. LSTM (Paper Implementation)
    experiments.append({
        **base_config,
        'model_type': 'rnn',
        'exp_name': 'hybrid_lstm',
        'description': 'LSTM with DDRSA+NASA hybrid loss (exact paper config)',
        'model_params': {
            'encoder_hidden_dim': 16,
            'decoder_hidden_dim': 16,
            'encoder_layers': 1,
            'decoder_layers': 1,
            'dropout': 0.1,
            'rnn_type': 'LSTM'
        }
    })

    # 2. GRU/RNN
    experiments.append({
        **base_config,
        'model_type': 'rnn',
        'exp_name': 'hybrid_gru',
        'description': 'GRU with DDRSA+NASA hybrid loss',
        'model_params': {
            'encoder_hidden_dim': 16,
            'decoder_hidden_dim': 16,
            'encoder_layers': 1,
            'decoder_layers': 1,
            'dropout': 0.1,
            'rnn_type': 'GRU'
        }
    })

    # 3. Transformer
    experiments.append({
        **base_config,
        'model_type': 'transformer',
        'exp_name': 'hybrid_transformer',
        'description': 'Transformer with DDRSA+NASA hybrid loss',
        'model_params': {
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'activation': 'relu'
        }
    })

    # 4. ProbSparse Attention (Informer)
    experiments.append({
        **base_config,
        'model_type': 'probsparse',
        'exp_name': 'hybrid_probsparse',
        'description': 'ProbSparse/Informer with DDRSA+NASA hybrid loss',
        'model_params': {
            'd_model': 128,
            'nhead': 4,
            'num_encoder_layers': 2,
            'decoder_hidden_dim': 128,
            'decoder_layers': 1,
            'dim_feedforward': 512,
            'dropout': 0.05,
            'activation': 'gelu',
            'factor': 5
        }
    })

    return experiments


def run_experiment(config, data_path, num_epochs, device, val_split=0.2, num_workers=4, generate_plots=True):
    """
    Run a single experiment

    Args:
        config: Experiment configuration dictionary
        data_path: Path to dataset
        num_epochs: Number of training epochs
        device: Device to train on
        val_split: Validation split ratio
        num_workers: Number of data loading workers
        generate_plots: Whether to generate visualization plots

    Returns:
        test_metrics: Dictionary of test metrics
        log_dir: Path to log directory
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT: {config['exp_name']}")
    print("="*80)
    print(f"Description: {config['description']}")
    print(f"Model Type: {config['model_type']}")
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))
    print("="*80 + "\n")

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        data_path=data_path,
        batch_size=config['batch_size'],
        lookback_window=config['lookback_window'],
        pred_horizon=config['pred_horizon'],
        train_file='train.txt',
        test_file='test.txt',
        val_split=val_split,
        num_workers=num_workers
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get input dimension
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"Input dimension: {input_dim}")

    # Create model
    print(f"\nCreating {config['model_type'].upper()} model...")
    model = create_ddrsa_model(
        model_type=config['model_type'],
        input_dim=input_dim,
        pred_horizon=config['pred_horizon'],
        **config['model_params']
    )

    print(f"Model created: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join('logs_nasa', f"{config['exp_name']}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Add hybrid loss flag to config
    config['use_nasa_loss'] = True  # Enable hybrid DDRSA+NASA loss

    # Create trainer (using standard DDRSA trainer with hybrid loss)
    trainer = DDRSATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        log_dir=log_dir
    )

    # Train model
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")

    test_metrics = trainer.train(num_epochs=num_epochs)

    # Print final results
    print("\n" + "="*80)
    print(f"FINAL TEST RESULTS - {config['exp_name']}")
    print("="*80)
    for key, value in test_metrics.items():
        print(f"{key:30s}: {value:.4f}")
    print("="*80 + "\n")

    print(f"Results saved to: {log_dir}")

    # Generate visualizations if requested
    if generate_plots:
        print("\n" + "="*80)
        print("Generating Visualizations")
        print("="*80 + "\n")

        figures_dir = os.path.join(log_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        try:
            # Create all visualizations
            print("Creating comprehensive visualizations...")
            create_all_visualizations(
                model=trainer.model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                log_dir=log_dir,
                output_dir=figures_dir,
                use_train_data=False  # Use test data for visualizations
            )

            print(f"\n✓ Visualizations saved to: {figures_dir}/")
            print("Generated figures:")
            for fig_file in sorted(os.listdir(figures_dir)):
                if fig_file.endswith('.png'):
                    print(f"  - {fig_file}")

        except Exception as e:
            print(f"Warning: Error generating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()

        print("="*80 + "\n")

    return test_metrics, log_dir


def main():
    parser = argparse.ArgumentParser(
        description='Run NASA Loss experiments on Turbofan dataset'
    )

    # Data arguments
    parser.add_argument('--data-path', type=str,
                       default='../Challenge_Data',
                       help='Path to data directory')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')

    # Experiment selection
    parser.add_argument('--experiments', type=str, nargs='+',
                       choices=['lstm', 'gru', 'transformer', 'probsparse', 'all'],
                       default=['all'],
                       help='Which experiments to run')

    # Visualization
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable automatic plot generation')

    # Output
    parser.add_argument('--summary-file', type=str,
                       default='nasa_experiments_summary.json',
                       help='File to save summary of all experiments')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Get experiment configurations
    all_configs = get_experiment_configs()

    # Filter experiments based on selection
    if 'all' in args.experiments:
        configs_to_run = all_configs
    else:
        exp_map = {
            'lstm': 'hybrid_lstm',
            'gru': 'hybrid_gru',
            'transformer': 'hybrid_transformer',
            'probsparse': 'hybrid_probsparse'
        }
        selected_names = [exp_map[e] for e in args.experiments]
        configs_to_run = [c for c in all_configs if c['exp_name'] in selected_names]

    print("\n" + "="*80)
    print("NASA LOSS EXPERIMENTS - Turbofan Dataset")
    print("="*80)
    print(f"Total experiments to run: {len(configs_to_run)}")
    print(f"Experiments: {[c['exp_name'] for c in configs_to_run]}")
    print(f"Number of epochs per experiment: {args.num_epochs}")
    print(f"Device: {device}")
    print("="*80 + "\n")

    # Run all experiments
    results_summary = {}

    for i, config in enumerate(configs_to_run):
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT {i+1}/{len(configs_to_run)}")
        print(f"{'='*80}\n")

        try:
            test_metrics, log_dir = run_experiment(
                config=config,
                data_path=args.data_path,
                num_epochs=args.num_epochs,
                device=device,
                val_split=args.val_split,
                num_workers=args.num_workers,
                generate_plots=not args.no_plots
            )

            results_summary[config['exp_name']] = {
                'status': 'completed',
                'metrics': test_metrics,
                'log_dir': log_dir,
                'config': config
            }

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR in experiment {config['exp_name']}")
            print(f"{'='*80}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

            results_summary[config['exp_name']] = {
                'status': 'failed',
                'error': str(e),
                'config': config
            }

    # Save summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)

    with open(args.summary_file, 'w') as f:
        json.dump(results_summary, f, indent=4)

    print(f"\nSummary saved to: {args.summary_file}")

    # Print comparison table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"{'Experiment':<25} {'Status':<12} {'NASA Score':<15} {'MSE':<15}")
    print("-"*80)

    for exp_name, result in results_summary.items():
        if result['status'] == 'completed':
            metrics = result['metrics']
            nasa_score = metrics.get('nasa_score', 'N/A')
            mse = metrics.get('mse', 'N/A')
            print(f"{exp_name:<25} {result['status']:<12} {nasa_score:<15.4f} {mse:<15.4f}")
        else:
            print(f"{exp_name:<25} {result['status']:<12} {'N/A':<15} {'N/A':<15}")

    print("="*80)


if __name__ == '__main__':
    main()
