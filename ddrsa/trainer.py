"""
Training script for DDRSA models
Implements the training procedure as described in the paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import json
import numpy as np
from tqdm import tqdm
import time
import math

from loss import DDRSALossDetailed, compute_expected_tte
from metrics import evaluate_model, compute_oti_metrics


class WarmupScheduler:
    """
    Learning rate scheduler with linear warmup and cosine/exponential decay

    This is the standard transformer learning rate schedule:
    - Linear warmup for warmup_steps
    - Cosine or exponential decay afterwards
    """

    def __init__(self, optimizer, d_model=64, warmup_steps=4000, decay_type='cosine', total_steps=None):
        """
        Args:
            optimizer: PyTorch optimizer
            d_model: Model dimension (for scaling, optional)
            warmup_steps: Number of warmup steps
            decay_type: 'cosine' or 'exponential'
            total_steps: Total training steps (needed for cosine decay)
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.decay_type = decay_type
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self):
        """Update learning rate"""
        self.current_step += 1
        lr = self._get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        """Calculate learning rate for current step"""
        step = self.current_step

        # Linear warmup
        if step < self.warmup_steps:
            # Linear increase from 0 to base_lr
            return self.base_lr * (step / self.warmup_steps)

        # After warmup, apply decay
        if self.decay_type == 'cosine':
            # Cosine annealing
            if self.total_steps is None:
                # If total_steps not provided, use a gentle decay
                progress = (step - self.warmup_steps) / (10000 - self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        elif self.decay_type == 'exponential':
            # Exponential decay
            decay_factor = (self.warmup_steps / step) ** 0.5
            return self.base_lr * decay_factor

        else:
            # No decay, just maintain base_lr after warmup
            return self.base_lr

    def state_dict(self):
        """Return state dict for checkpointing"""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'base_lr': self.base_lr
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.base_lr = state_dict['base_lr']


class DDRSATrainer:
    """Trainer class for DDRSA models"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 config, device='cuda', log_dir='logs'):
        """
        Args:
            model: DDRSA model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
            device: Device to train on
            log_dir: Directory for logs and checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.log_dir = log_dir

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)

        # Loss function
        self.criterion = DDRSALossDetailed(lambda_param=config.get('lambda_param', 0.5))

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-6)
        )

        # Learning rate scheduler - use warmup for transformers, ReduceLROnPlateau for RNNs
        self.use_warmup = config.get('use_warmup', False)

        if self.use_warmup:
            # Transformer-style warmup scheduler
            warmup_steps = config.get('warmup_steps', 4000)
            total_steps = config.get('num_epochs', 100) * len(train_loader)
            decay_type = config.get('lr_decay_type', 'cosine')

            self.scheduler = WarmupScheduler(
                self.optimizer,
                d_model=config.get('d_model', 64),
                warmup_steps=warmup_steps,
                decay_type=decay_type,
                total_steps=total_steps
            )
            print(f"Using WarmupScheduler with {warmup_steps} warmup steps and {decay_type} decay")
        else:
            # RNN-style ReduceLROnPlateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
            print("Using ReduceLROnPlateau scheduler")

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Save config
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {
            'loss_z': 0,
            'loss_u': 0,
            'loss_c': 0
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1} [Train]')

        for batch_idx, (sequences, targets, censored) in enumerate(pbar):
            # Move data to device
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            censored = censored.to(self.device)

            # Forward pass
            hazard_logits = self.model(sequences)

            # Compute loss
            loss, loss_dict = self.criterion(hazard_logits, targets, censored)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip', 1.0)
            )

            self.optimizer.step()

            # Update learning rate (for warmup scheduler, step per batch)
            if self.use_warmup:
                self.scheduler.step()

            # Accumulate losses
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Average losses
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches

        return avg_loss, loss_components

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        loss_components = {
            'loss_z': 0,
            'loss_u': 0,
            'loss_c': 0
        }

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch+1} [Val]')

            for batch_idx, (sequences, targets, censored) in enumerate(pbar):
                # Move data to device
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                censored = censored.to(self.device)

                # Forward pass
                hazard_logits = self.model(sequences)

                # Compute loss
                loss, loss_dict = self.criterion(hazard_logits, targets, censored)

                # Accumulate losses
                total_loss += loss.item()
                for key in loss_components:
                    loss_components[key] += loss_dict[key]

                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})

        # Average losses
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches

        return avg_loss, loss_components

    def train(self, num_epochs):
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()

            # Train
            train_loss, train_components = self.train_epoch()

            # Validate
            val_loss, val_components = self.validate_epoch()

            # Learning rate scheduler step (for ReduceLROnPlateau, step per epoch)
            if not self.use_warmup:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

            epoch_time = time.time() - start_time

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            for key in train_components:
                self.writer.add_scalar(f'Train/{key}', train_components[key], epoch)
                self.writer.add_scalar(f'Val/{key}', val_components[key], epoch)

            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1

            # Save latest checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

            # Early stopping
            if self.patience_counter >= self.config.get('patience', 20):
                print(f"\nEarly stopping after {epoch+1} epochs")
                break

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Load best model for final evaluation
        self.load_checkpoint('best_model.pt')

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate_test()

        return test_metrics

    def evaluate_test(self):
        """Evaluate model on test set"""
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_censored = []

        with torch.no_grad():
            for sequences, targets, censored in tqdm(self.test_loader, desc='Testing'):
                sequences = sequences.to(self.device)

                # Forward pass
                hazard_logits = self.model(sequences)

                all_predictions.append(hazard_logits.cpu())
                all_targets.append(targets)
                all_censored.append(censored)

        # Concatenate all batches
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        censored = torch.cat(all_censored, dim=0)

        # Compute metrics
        metrics = evaluate_model(predictions, targets, censored)

        # Compute OTI policy metrics
        oti_metrics = compute_oti_metrics(predictions, targets, censored)

        # Combine metrics
        all_metrics = {**metrics, **oti_metrics}

        # Print metrics
        print("\nTest Metrics:")
        for key, value in all_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save metrics
        serializable_metrics = {k: float(v) for k, v in all_metrics.items()}
        with open(os.path.join(self.log_dir, 'test_metrics.json'), 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

        return all_metrics

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', filename)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, checkpoint_path)

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', filename)
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {self.epoch}")


def get_default_config(model_type='rnn'):
    """
    Get default configuration matching the paper

    From paper Section 6.2:
    - Encoder: LSTM with 128 step look-back, 16 hidden units
    - Decoder: LSTM with 16 hidden units
    - Learning rate: 1e-4
    - Batch size: 32
    - Lambda (α): 0.5
    """
    config = {
        'model_type': model_type,
        'lookback_window': 128,
        'pred_horizon': 100,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'lambda_param': 0.5,
        'grad_clip': 1.0,
        'patience': 20,
        'save_interval': 10,
    }

    if model_type == 'rnn':
        config.update({
            'encoder_hidden_dim': 16,
            'decoder_hidden_dim': 16,
            'encoder_layers': 1,
            'decoder_layers': 1,
            'dropout': 0.1,
            'rnn_type': 'LSTM'
        })
    elif model_type == 'transformer':
        config.update({
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            # Transformer-specific training settings
            'use_warmup': True,
            'warmup_steps': 4000,
            'lr_decay_type': 'cosine'
        })

    return config
