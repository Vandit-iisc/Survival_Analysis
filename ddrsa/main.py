"""
Main experiment runner for DDRSA
Reproduces experiments from the paper on NASA Turbofan dataset
"""

import argparse
import torch
import os
import json
import numpy as np
import random

from data_loader import get_dataloaders
from models import create_ddrsa_model
from trainer import DDRSATrainer, get_default_config


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


def main(args):
    """Main training function"""

    # Set seed for reproducibility
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Get configuration
    config = get_default_config(model_type=args.model_type)

    # Override config with command line arguments
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.lambda_param is not None:
        config['lambda_param'] = args.lambda_param
    if args.lookback_window is not None:
        config['lookback_window'] = args.lookback_window
    if args.pred_horizon is not None:
        config['pred_horizon'] = args.pred_horizon
    if args.warmup_steps is not None:
        config['warmup_steps'] = args.warmup_steps
    if args.lr_decay_type is not None:
        config['lr_decay_type'] = args.lr_decay_type

    # Store num_epochs in config for scheduler calculations
    config['num_epochs'] = args.num_epochs

    # Additional model-specific overrides
    if args.model_type == 'rnn':
        if args.hidden_dim is not None:
            config['encoder_hidden_dim'] = args.hidden_dim
            config['decoder_hidden_dim'] = args.hidden_dim
        if args.num_layers is not None:
            config['encoder_layers'] = args.num_layers
            config['decoder_layers'] = args.num_layers
        if args.rnn_type is not None:
            config['rnn_type'] = args.rnn_type
    elif args.model_type == 'transformer':
        if args.d_model is not None:
            config['d_model'] = args.d_model
        if args.nhead is not None:
            config['nhead'] = args.nhead
        if args.num_layers is not None:
            config['num_encoder_layers'] = args.num_layers
            config['num_decoder_layers'] = args.num_layers

    print("\n" + "="*80)
    print("DDRSA Experiment Configuration")
    print("="*80)
    for key, value in config.items():
        print(f"{key:25s}: {value}")
    print("="*80 + "\n")

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        data_path=args.data_path,
        batch_size=config['batch_size'],
        lookback_window=config['lookback_window'],
        pred_horizon=config['pred_horizon'],
        train_file='train.txt',
        test_file='test.txt',
        val_split=args.val_split,
        num_workers=args.num_workers,
        use_paper_split=args.use_paper_split,
        random_seed=args.seed,
        use_minmax=args.use_minmax
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get input dimension from data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"Input dimension: {input_dim}")

    # Create model
    print(f"\nCreating {args.model_type.upper()} model...")
    model_kwargs = {k: v for k, v in config.items()
                   if k not in ['model_type', 'batch_size', 'learning_rate',
                               'weight_decay', 'lambda_param', 'grad_clip',
                               'patience', 'save_interval', 'lookback_window']}
    model = create_ddrsa_model(
        model_type=args.model_type,
        input_dim=input_dim,
        **model_kwargs
    )

    print(f"Model created: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    log_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

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

    test_metrics = trainer.train(num_epochs=args.num_epochs)

    # Print final results
    print("\n" + "="*80)
    print("Final Test Results")
    print("="*80)
    for key, value in test_metrics.items():
        print(f"{key:30s}: {value:.4f}")
    print("="*80 + "\n")

    print(f"Results saved to: {log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DDRSA model on NASA Turbofan dataset')

    # Data arguments
    parser.add_argument('--data-path', type=str,
                       default='../Challenge_Data',
                       help='Path to data directory')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio (only used if not using paper split)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use-paper-split', action='store_true',
                       help='Use paper splitting methodology: 70/30 train/test at unit level, then 30%% validation from train')
    parser.add_argument('--use-minmax', action='store_true', default=True,
                       help='Use MinMaxScaler [-1, 1] instead of StandardScaler (paper default: True)')
    parser.add_argument('--use-standard-scaler', dest='use_minmax', action='store_false',
                       help='Use StandardScaler (z-score) instead of MinMaxScaler')

    # Model arguments
    parser.add_argument('--model-type', type=str, default='rnn',
                       choices=['rnn', 'transformer'],
                       help='Model architecture type')
    parser.add_argument('--hidden-dim', type=int, default=None,
                       help='Hidden dimension (for RNN)')
    parser.add_argument('--num-layers', type=int, default=None,
                       help='Number of layers')
    parser.add_argument('--rnn-type', type=str, default=None,
                       choices=['LSTM', 'GRU'],
                       help='Type of RNN')
    parser.add_argument('--d-model', type=int, default=None,
                       help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=None,
                       help='Number of attention heads')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lambda-param', type=float, default=None,
                       help='Lambda parameter for DDRSA loss')
    parser.add_argument('--lookback-window', type=int, default=None,
                       help='Lookback window size')
    parser.add_argument('--pred-horizon', type=int, default=None,
                       help='Prediction horizon')
    parser.add_argument('--warmup-steps', type=int, default=None,
                       help='Number of warmup steps for transformer (default: 4000)')
    parser.add_argument('--lr-decay-type', type=str, default=None,
                       choices=['cosine', 'exponential', 'none'],
                       help='Learning rate decay type after warmup')

    # Experiment arguments
    parser.add_argument('--exp-name', type=str, default='ddrsa_experiment',
                       help='Experiment name')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for logs and checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')

    args = parser.parse_args()

    main(args)
