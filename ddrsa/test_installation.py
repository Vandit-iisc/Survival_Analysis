"""
Test script to verify DDRSA installation and components
"""

import torch
import numpy as np
print("✓ Imports successful")

# Test model creation
from models import create_ddrsa_model
print("\n1. Testing Model Creation...")

# Test RNN model
print("   Creating DDRSA-RNN...")
model_rnn = create_ddrsa_model(
    'rnn',
    input_dim=24,
    encoder_hidden_dim=16,
    decoder_hidden_dim=16,
    pred_horizon=100
)
print(f"   ✓ DDRSA-RNN created ({sum(p.numel() for p in model_rnn.parameters()):,} parameters)")

# Test Transformer model
print("   Creating DDRSA-Transformer...")
model_transformer = create_ddrsa_model(
    'transformer',
    input_dim=24,
    d_model=64,
    nhead=4,
    pred_horizon=100
)
print(f"   ✓ DDRSA-Transformer created ({sum(p.numel() for p in model_transformer.parameters()):,} parameters)")

# Test forward pass
print("\n2. Testing Forward Pass...")
batch_size = 8
lookback = 128
input_dim = 24
pred_horizon = 100

x = torch.randn(batch_size, lookback, input_dim)

output_rnn = model_rnn(x)
assert output_rnn.shape == (batch_size, pred_horizon), f"RNN output shape mismatch: {output_rnn.shape}"
print(f"   ✓ RNN forward pass: {x.shape} → {output_rnn.shape}")

output_transformer = model_transformer(x)
assert output_transformer.shape == (batch_size, pred_horizon), f"Transformer output shape mismatch: {output_transformer.shape}"
print(f"   ✓ Transformer forward pass: {x.shape} → {output_transformer.shape}")

# Test loss function
print("\n3. Testing Loss Function...")
from loss import DDRSALossDetailed

criterion = DDRSALossDetailed(lambda_param=0.5)

# Create dummy targets
targets = torch.zeros(batch_size, pred_horizon)
censored = torch.ones(batch_size, pred_horizon)

# Make some samples uncensored
for i in range(batch_size // 2):
    event_time = torch.randint(0, pred_horizon, (1,)).item()
    targets[i, event_time] = 1
    censored[i, :event_time+1] = 0

loss, loss_dict = criterion(output_rnn, targets, censored)
print(f"   ✓ Loss computed: {loss.item():.4f}")
print(f"   Loss components:")
for key, value in loss_dict.items():
    print(f"     - {key}: {value:.4f}")

# Test expected TTE
print("\n4. Testing Expected TTE Computation...")
from loss import compute_expected_tte

expected_tte = compute_expected_tte(output_rnn)
assert expected_tte.shape == (batch_size,), f"TTE shape mismatch: {expected_tte.shape}"
print(f"   ✓ Expected TTE computed: shape {expected_tte.shape}")
print(f"   Sample TTEs: {expected_tte[:3].detach().numpy()}")

# Test data loader
print("\n5. Testing Data Loader...")
from data_loader import TurbofanDataLoader
import os

data_path = "/Users/vandit/Desktop/vandit/Survival_Analysis/Challenge_Data"
if os.path.exists(data_path):
    loader = TurbofanDataLoader(data_path, lookback_window=128, pred_horizon=100)
    print("   ✓ TurbofanDataLoader created")

    # Check if train.txt exists
    train_file = os.path.join(data_path, 'train.txt')
    if os.path.exists(train_file):
        print(f"   ✓ Found train.txt")

        # Load a small sample
        train_df = loader.load_data('train.txt')
        print(f"   ✓ Loaded training data: {len(train_df)} samples")
        print(f"   Number of engines: {train_df['unit_id'].nunique()}")
    else:
        print(f"   ⚠ train.txt not found at {train_file}")
else:
    print(f"   ⚠ Data path not found: {data_path}")
    print("   Skipping data loader test")

# Test metrics
print("\n6. Testing Metrics...")
from metrics import evaluate_model, compute_oti_metrics

metrics = evaluate_model(output_rnn, targets, censored)
print("   ✓ Model evaluation metrics:")
for key, value in metrics.items():
    print(f"     - {key}: {value:.4f}")

oti_metrics = compute_oti_metrics(output_rnn, targets, censored, cost_values=[64, 128])
print("   ✓ OTI metrics:")
for key, value in oti_metrics.items():
    print(f"     - {key}: {value:.4f}")

print("\n" + "="*70)
print("✓ All tests passed! DDRSA installation is working correctly.")
print("="*70)
print("\nYou can now run experiments with:")
print("  python main.py --model-type rnn --num-epochs 100")
print("\nOr run all experiments with:")
print("  bash run_all.sh")
