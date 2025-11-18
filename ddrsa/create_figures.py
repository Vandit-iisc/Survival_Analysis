"""
Script to create all figures from the paper after training
Run this after your model has finished training
"""

import argparse
import torch
import os

from models import create_ddrsa_model
from data_loader import get_dataloaders
from visualization import create_all_visualizations


def main(args):
    """Create all visualizations from trained model"""

    print("="*80)
    print("Creating Visualizations from Trained Model")
    print("="*80)
    print(f"Experiment: {args.exp_name}")
    print(f"Log directory: {args.log_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load configuration
    import json
    config_path = os.path.join(args.log_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Load test data
    print("Loading test data...")
    _, _, test_loader, _ = get_dataloaders(
        data_path=args.data_path,
        batch_size=config['batch_size'],
        lookback_window=config['lookback_window'],
        pred_horizon=config['pred_horizon'],
        num_workers=args.num_workers
    )

    # Get input dimension
    sample_batch = next(iter(test_loader))
    input_dim = sample_batch[0].shape[-1]

    # Create model
    print(f"Creating {config['model_type'].upper()} model...")
    model_kwargs = {k: v for k, v in config.items()
                   if k not in ['model_type', 'batch_size', 'learning_rate',
                               'weight_decay', 'lambda_param', 'grad_clip',
                               'patience', 'save_interval', 'lookback_window']}

    if config['model_type'] == 'rnn':
        from models import DDRSA_RNN
        model = DDRSA_RNN(
            input_dim=input_dim,
            encoder_hidden_dim=config['encoder_hidden_dim'],
            decoder_hidden_dim=config['decoder_hidden_dim'],
            encoder_layers=config['encoder_layers'],
            decoder_layers=config['decoder_layers'],
            pred_horizon=config['pred_horizon'],
            dropout=config['dropout'],
            rnn_type=config['rnn_type']
        )
    elif config['model_type'] == 'transformer':
        from models import DDRSA_Transformer
        model = DDRSA_Transformer(
            input_dim=input_dim,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            dim_feedforward=config.get('dim_feedforward', 256),
            pred_horizon=config['pred_horizon'],
            dropout=config['dropout']
        )

    # Load checkpoint
    checkpoint_path = os.path.join(args.log_dir, 'checkpoints', 'best_model.pt')
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded (epoch {checkpoint['epoch']+1}, val_loss: {checkpoint['best_val_loss']:.4f})")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create all visualizations
    create_all_visualizations(
        model=model,
        test_loader=test_loader,
        log_dir=args.log_dir,
        output_dir=args.output_dir
    )

    print()
    print("="*80)
    print("✓ All visualizations created successfully!")
    print("="*80)
    print(f"\nFigures saved to: {args.output_dir}/")
    print("\nGenerated files:")
    for file in os.listdir(args.output_dir):
        if file.endswith('.png'):
            print(f"  - {file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create visualizations from trained DDRSA model'
    )

    parser.add_argument('--exp-name', type=str, required=True,
                       help='Experiment name (log subdirectory)')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Log directory (default: logs/<exp-name>)')
    parser.add_argument('--data-path', type=str,
                       default='../Challenge_Data',
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for figures (default: figures/<exp-name>)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    args = parser.parse_args()

    # Set default log_dir if not provided
    if args.log_dir is None:
        args.log_dir = f'logs/{args.exp_name}'

    # Set default output_dir if not provided
    if args.output_dir is None:
        args.output_dir = f'figures/{args.exp_name}'

    main(args)
