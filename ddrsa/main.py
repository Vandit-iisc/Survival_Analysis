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
from azure_pm_loader import get_azure_pm_dataloaders
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
    if args.patience is not None:
        config['patience'] = args.patience
    if args.no_early_stopping:
        config['use_early_stopping'] = False
    else:
        config['use_early_stopping'] = True

    # NASA loss configuration
    if args.use_nasa_loss:
        config['use_nasa_loss'] = True
        config['nasa_weight'] = args.nasa_weight

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
        if args.num_encoder_layers is not None:
            config['num_encoder_layers'] = args.num_encoder_layers
        if args.num_decoder_layers is not None:
            config['num_decoder_layers'] = args.num_decoder_layers
        if args.dim_feedforward is not None:
            config['dim_feedforward'] = args.dim_feedforward
        if args.dropout is not None:
            config['dropout'] = args.dropout
        if args.activation is not None:
            config['activation'] = args.activation
        # Legacy: --num-layers sets both encoder and decoder layers
        if args.num_layers is not None:
            config['num_encoder_layers'] = args.num_layers
            config['num_decoder_layers'] = args.num_layers
    elif args.model_type == 'probsparse':
        if args.d_model is not None:
            config['d_model'] = args.d_model
        if args.nhead is not None:
            config['nhead'] = args.nhead
        if args.num_encoder_layers is not None:
            config['num_encoder_layers'] = args.num_encoder_layers
        if args.hidden_dim is not None:
            config['decoder_hidden_dim'] = args.hidden_dim
        if args.num_layers is not None:
            config['decoder_layers'] = args.num_layers
        if args.dim_feedforward is not None:
            config['dim_feedforward'] = args.dim_feedforward
        if args.dropout is not None:
            config['dropout'] = args.dropout
        if args.activation is not None:
            config['activation'] = args.activation
        if args.factor is not None:
            config['factor'] = args.factor

    print("\n" + "="*80)
    print("DDRSA Experiment Configuration")
    print("="*80)
    for key, value in config.items():
        print(f"{key:25s}: {value}")
    print("="*80 + "\n")

    # Create data loaders
    print("Loading data...")

    if args.dataset == 'turbofan':
        # NASA Turbofan dataset
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
    elif args.dataset == 'azure_pm':
        # Azure Predictive Maintenance dataset
        train_loader, val_loader, test_loader, scaler = get_azure_pm_dataloaders(
            data_path=args.data_path,
            batch_size=config['batch_size'],
            lookback_window=config['lookback_window'],
            pred_horizon=config['pred_horizon'],
            num_workers=args.num_workers,
            random_seed=args.seed,
            use_minmax=args.use_minmax,
            downsample_factor=args.downsample_factor
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get input dimension from data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"Input dimension: {input_dim}")

    # Create model
    print(f"\nCreating {args.model_type.upper()} model...")
    # Filter out training-specific parameters, only pass model architecture parameters
    exclude_keys = ['model_type', 'batch_size', 'learning_rate', 'weight_decay',
                   'lambda_param', 'grad_clip', 'patience', 'save_interval',
                   'lookback_window', 'use_warmup', 'warmup_steps', 'lr_decay_type',
                   'num_epochs', 'use_early_stopping']
    model_kwargs = {k: v for k, v in config.items() if k not in exclude_keys}
    model = create_ddrsa_model(
        model_type=args.model_type,
        input_dim=input_dim,
        **model_kwargs
    )

    print(f"Model created: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create output directory structure
    # If output_dir is specified, use it as parent directory in Survival_Analysis folder
    # Otherwise, use default logs directory in ddrsa folder
    if args.output_dir:
        # Use parent directory (../output_dir relative to ddrsa)
        base_dir = os.path.join('..', args.output_dir)
        log_dir = os.path.join(base_dir, args.log_dir, args.exp_name)
    else:
        log_dir = os.path.join(args.log_dir, args.exp_name)

    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(log_dir)}")

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

    # Create visualizations if requested
    if args.create_visualization:
        print("\n" + "="*80)
        print("Creating Visualizations")
        print("="*80 + "\n")

        from visualization import create_all_visualizations
        import matplotlib.pyplot as plt

        # Create figures directory
        if args.output_dir:
            figures_dir = os.path.join(args.output_dir, 'figures', args.exp_name)
        else:
            figures_dir = f'figures/{args.exp_name}'
        os.makedirs(figures_dir, exist_ok=True)

        # Plot and save loss curves
        loss_fig = trainer.plot_loss_curves(
            save_path=os.path.join(figures_dir, 'loss_curves.png')
        )
        if loss_fig:
            plt.close(loss_fig)

        # === BEST MODEL VISUALIZATIONS ===
        print("\n--- Best Model Visualizations ---")
        # Model is already loaded with best weights from training

        # Create subdirectory for best model
        best_figures_dir = os.path.join(figures_dir, 'best_model')
        os.makedirs(best_figures_dir, exist_ok=True)

        # Create visualizations (test data)
        create_all_visualizations(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            log_dir=log_dir,
            output_dir=best_figures_dir,
            use_train_data=False
        )

        # Create visualizations (train data)
        create_all_visualizations(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            log_dir=log_dir,
            output_dir=best_figures_dir,
            use_train_data=True
        )

        # === LAST EPOCH MODEL VISUALIZATIONS ===
        print("\n--- Last Epoch Model Visualizations (Overfitting Analysis) ---")

        # Load last epoch model
        trainer.load_checkpoint('last_model.pt')

        # Create subdirectory for last model
        last_figures_dir = os.path.join(figures_dir, 'last_model')
        os.makedirs(last_figures_dir, exist_ok=True)

        # Create visualizations (test data)
        create_all_visualizations(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            log_dir=log_dir,
            output_dir=last_figures_dir,
            use_train_data=False
        )

        # Create visualizations (train data)
        create_all_visualizations(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            log_dir=log_dir,
            output_dir=last_figures_dir,
            use_train_data=True
        )

        print(f"\nFigures saved to: {os.path.abspath(figures_dir)}")
        print(f"  - Best model figures: {os.path.join(figures_dir, 'best_model')}")
        print(f"  - Last model figures: {os.path.join(figures_dir, 'last_model')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DDRSA model on NASA Turbofan dataset')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='turbofan',
                       choices=['turbofan', 'azure_pm'],
                       help='Dataset to use: turbofan (NASA) or azure_pm (Microsoft)')
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
    parser.add_argument('--downsample-factor', type=int, default=5,
                       help='Down-sample factor for Azure PM dataset (5 = every 15 hours)')

    # Model arguments
    parser.add_argument('--model-type', type=str, default='rnn',
                       choices=['rnn', 'transformer', 'probsparse'],
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
    parser.add_argument('--num-encoder-layers', type=int, default=None,
                       help='Number of transformer encoder layers')
    parser.add_argument('--num-decoder-layers', type=int, default=None,
                       help='Number of transformer decoder layers')
    parser.add_argument('--dim-feedforward', type=int, default=None,
                       help='Dimension of feedforward network in transformer')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--activation', type=str, default=None,
                       choices=['relu', 'gelu'],
                       help='Activation function for transformer (default: relu)')
    parser.add_argument('--factor', type=int, default=None,
                       help='ProbSparse attention factor (controls sparsity, default: 5)')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lambda-param', type=float, default=None,
                       help='Lambda parameter for DDRSA loss')
    parser.add_argument('--use-nasa-loss', action='store_true',
                       help='Include NASA scoring function in loss')
    parser.add_argument('--nasa-weight', type=float, default=0.1,
                       help='Weight for NASA loss component (default: 0.1)')
    parser.add_argument('--lookback-window', type=int, default=None,
                       help='Lookback window size')
    parser.add_argument('--pred-horizon', type=int, default=None,
                       help='Prediction horizon')
    parser.add_argument('--warmup-steps', type=int, default=None,
                       help='Number of warmup steps for transformer (default: 4000)')
    parser.add_argument('--lr-decay-type', type=str, default=None,
                       choices=['cosine', 'exponential', 'none'],
                       help='Learning rate decay type after warmup')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Disable early stopping')

    # Experiment arguments
    parser.add_argument('--exp-name', type=str, default='ddrsa_experiment',
                       help='Experiment name')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Parent output directory (default: ../results). All logs and figures will be stored here.')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Subdirectory name for logs and checkpoints (default: logs)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')
    parser.add_argument('--create-visualization', action='store_true',
                       help='Create visualizations after training')

    args = parser.parse_args()

    main(args)
