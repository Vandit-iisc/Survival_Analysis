"""
Quick Diagnostic: Why Are Hazard Rates So Low?
Run this to identify the root cause
"""

import torch
import sys
from data_loader import get_dataloaders
from models import create_ddrsa_model
from loss import DDRSALoss
import numpy as np


def diagnose_hazard_rates():
    """Comprehensive diagnostic of hazard rate issues"""

    print("="*80)
    print("HAZARD RATE DIAGNOSTIC")
    print("="*80)

    # ========== Step 1: Check Data Censoring ==========
    print("\n" + "="*80)
    print("STEP 1: Checking Data Censoring Ratio")
    print("="*80)

    try:
        train_loader, val_loader, test_loader, scaler = get_dataloaders(
            data_path='../Challenge_Data',
            batch_size=128,
            lookback_window=30,
            pred_horizon=100
        )

        total_uncensored = 0
        total_samples = 0

        for X, targets, censored in train_loader:
            is_uncensored = (targets.sum(dim=1) > 0).float()
            total_uncensored += is_uncensored.sum().item()
            total_samples += X.size(0)

        uncensored_ratio = total_uncensored / total_samples
        censored_ratio = 1 - uncensored_ratio

        print(f"\n✓ Data loaded successfully")
        print(f"  Total samples: {total_samples}")
        print(f"  Uncensored: {int(total_uncensored)} ({uncensored_ratio:.1%})")
        print(f"  Censored: {int(total_samples - total_uncensored)} ({censored_ratio:.1%})")

        if censored_ratio > 0.7:
            print(f"\n⚠️  WARNING: High censoring ratio ({censored_ratio:.1%})!")
            print("   This pushes hazard rates toward 0 via loss_c term")
            print("   RECOMMENDATION: Reduce loss_c weight or increase lambda_param")
        else:
            print(f"\n✓ Censoring ratio is reasonable ({censored_ratio:.1%})")

    except Exception as e:
        print(f"\n✗ Could not load data: {e}")
        print("  Skipping data diagnostics...")
        train_loader = None

    # ========== Step 2: Check Model Initialization ==========
    print("\n" + "="*80)
    print("STEP 2: Checking Model Output Initialization")
    print("="*80)

    try:
        # Create model
        model = create_ddrsa_model(
            model_type='rnn',
            input_dim=24,
            encoder_hidden_dim=16,
            decoder_hidden_dim=16,
            encoder_layers=1,
            decoder_layers=1,
            pred_horizon=100,
            rnn_type='LSTM'
        )

        # Check output layer bias
        output_bias = model.output_layer.bias.data[0].item()
        initial_hazard = torch.sigmoid(torch.tensor(output_bias)).item()

        print(f"\n✓ Model created successfully")
        print(f"  Output bias: {output_bias:.4f}")
        print(f"  Initial hazard rate: sigmoid({output_bias:.4f}) = {initial_hazard:.6f} ({initial_hazard*100:.2f}%)")

        if output_bias < -1.0:
            print(f"\n⚠️  WARNING: Output bias is very negative ({output_bias:.4f})!")
            print(f"   This initializes hazard rates at {initial_hazard*100:.2f}%")
            print("   RECOMMENDATION: Change to 0.0 or positive value")
            print("\n   Fix in models.py (lines 90, 241, 573):")
            print("   nn.init.constant_(self.output_layer.bias, 0.0)  # Change from -2.0")
        else:
            print(f"\n✓ Output bias is reasonable ({output_bias:.4f})")

    except Exception as e:
        print(f"\n✗ Could not create model: {e}")
        model = None

    # ========== Step 3: Check Initial Predictions ==========
    if train_loader is not None and model is not None:
        print("\n" + "="*80)
        print("STEP 3: Checking Initial Model Predictions (Before Training)")
        print("="*80)

        try:
            # Get a batch
            X, targets, censored = next(iter(train_loader))

            # Forward pass (no training)
            with torch.no_grad():
                logits = model(X)
                hazard_rates = torch.sigmoid(logits)

            # Statistics
            mean_h = hazard_rates.mean().item()
            max_h = hazard_rates.max().item()
            min_h = hazard_rates.min().item()
            std_h = hazard_rates.std().item()
            median_h = hazard_rates.median().item()

            print(f"\n✓ Initial hazard rate statistics:")
            print(f"  Mean: {mean_h:.6f} ({mean_h*100:.3f}%)")
            print(f"  Median: {median_h:.6f} ({median_h*100:.3f}%)")
            print(f"  Max: {max_h:.6f} ({max_h*100:.3f}%)")
            print(f"  Min: {min_h:.6f} ({min_h*100:.3f}%)")
            print(f"  Std: {std_h:.6f}")

            print(f"\n✓ Logit statistics:")
            print(f"  Mean: {logits.mean().item():.3f}")
            print(f"  Max: {logits.max().item():.3f}")
            print(f"  Min: {logits.min().item():.3f}")

            # Check if reasonable
            if mean_h < 0.05:
                print(f"\n⚠️  WARNING: Mean hazard rate is too low ({mean_h*100:.3f}%)!")
                print("   Expected: 5-20% initially")
                print("   This suggests initialization issue")
            elif mean_h > 0.5:
                print(f"\n⚠️  WARNING: Mean hazard rate is too high ({mean_h*100:.3f}%)!")
                print("   Expected: 5-20% initially")
            else:
                print(f"\n✓ Initial hazard rates are in reasonable range")

        except Exception as e:
            print(f"\n✗ Could not get predictions: {e}")

    # ========== Step 4: Check Loss Function Behavior ==========
    if train_loader is not None and model is not None:
        print("\n" + "="*80)
        print("STEP 4: Checking Loss Function Components")
        print("="*80)

        try:
            # Get a batch
            X, targets, censored = next(iter(train_loader))

            # Forward pass
            with torch.no_grad():
                logits = model(X)

            # Compute loss
            criterion = DDRSALoss(lambda_param=0.5)
            loss = criterion(logits, targets, censored)

            print(f"\n✓ Loss computed successfully")
            print(f"  Total loss: {loss.item():.6f}")

            # Compute components manually
            hazard_rates = torch.sigmoid(logits)
            log_survival = torch.cumsum(torch.log(1 - hazard_rates + 1e-7), dim=1)
            survival_probs = torch.exp(log_survival)
            is_uncensored = (targets.sum(dim=1) > 0).float()

            final_survival = survival_probs[:, -1]
            l_u = 1 - final_survival
            l_c = final_survival

            print(f"\n✓ Loss components:")
            print(f"  l_u (event occurs): mean = {l_u.mean().item():.6f}")
            print(f"  l_c (survival): mean = {l_c.mean().item():.6f}")
            print(f"  Uncensored ratio: {is_uncensored.mean().item():.3f}")

            # Check if loss_c dominates
            if is_uncensored.mean() < 0.3:
                print(f"\n⚠️  WARNING: Loss_c likely dominates due to high censoring!")
                print("   This encourages low hazard rates")

        except Exception as e:
            print(f"\n✗ Could not compute loss: {e}")

    # ========== Step 5: Summary and Recommendations ==========
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)

    print("\nBased on diagnostics, the most likely causes are:\n")

    print("1. ⭐ OUTPUT BIAS INITIALIZATION")
    print("   Current: -2.0 → Initial hazard ~12%")
    print("   If trained hazard is 0.4%, model is learning LOWER rates")
    print("   FIX: Change to 0.0 or positive in models.py\n")

    print("2. ⭐ LOSS IMBALANCE FROM CENSORING")
    print("   If censoring > 70%, loss_c dominates")
    print("   FIX: Reduce lambda_param or reweight loss_c\n")

    print("3. DATA PREPROCESSING")
    print("   Check if MinMaxScaler vs StandardScaler matters")
    print("   FIX: Try --use-standard-scaler flag\n")

    print("4. LAMBDA PARAMETER")
    print("   Current default: 0.5")
    print("   FIX: Try 0.3 to emphasize event occurrence\n")

    print("\n" + "="*80)
    print("RECOMMENDED ACTION PLAN")
    print("="*80)

    print("""
1. IMMEDIATE FIX (5 minutes):
   Edit models.py lines 90, 241, 573:
   Change: nn.init.constant_(self.output_layer.bias, -2.0)
   To:     nn.init.constant_(self.output_layer.bias, 0.0)

2. RETRAIN:
   python main.py \\
     --dataset turbofan \\
     --model-type rnn \\
     --hidden-dim 16 \\
     --num-layers 1 \\
     --lambda-param 0.5 \\
     --num-epochs 100 \\
     --exp-name hazard_fix_test \\
     --create-visualization

3. VERIFY:
   Check figures/hazard_fix_test/figure_2a_hazard_progression.png
   Expected: Maximum hazard rates 0.5-0.9 (not 0.004!)

4. IF STILL TOO LOW:
   Try lambda_param=0.3 or add hazard regularization
""")

    print("="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == '__main__':
    diagnose_hazard_rates()
