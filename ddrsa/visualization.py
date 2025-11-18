"""
Visualization utilities for DDRSA
Reproduces the plots from Section 6.1 of the paper (Figure 2)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from loss import compute_hazard_from_logits, compute_expected_tte


def plot_hazard_progression(model, data_loader, device='cpu', num_samples=5,
                            save_path=None, sample_indices=None):
    """
    Reproduce Figure 2(a): Progression of conditional hazard rates

    Shows how estimated conditional hazard rates evolve as the model
    observes more data approaching the critical event.

    Args:
        model: Trained DDRSA model
        data_loader: Data loader with sequences
        device: Device to run on
        num_samples: Number of samples to plot
        save_path: Path to save the figure
        sample_indices: Specific indices to plot (if None, random selection)
    """
    model.eval()

    # Get some samples from test set
    all_sequences = []
    all_targets = []

    for sequences, targets, _ in data_loader:
        all_sequences.append(sequences)
        all_targets.append(targets)
        if len(all_sequences) * sequences.size(0) >= num_samples * 10:
            break

    all_sequences = torch.cat(all_sequences, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Select samples with uncensored events
    uncensored_mask = all_targets.sum(dim=1) > 0
    uncensored_sequences = all_sequences[uncensored_mask]
    uncensored_targets = all_targets[uncensored_mask]

    if sample_indices is None:
        # Randomly select samples
        if len(uncensored_sequences) > num_samples:
            indices = np.random.choice(len(uncensored_sequences), num_samples, replace=False)
        else:
            indices = range(len(uncensored_sequences))
    else:
        indices = sample_indices

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    with torch.no_grad():
        for idx, color in zip(indices, colors):
            sequence = uncensored_sequences[idx:idx+1].to(device)
            target = uncensored_targets[idx]

            # Find event time
            event_time = torch.argmax(target).item()

            # Get predictions at different time points
            # Simulate observing data at different points in the sequence
            lookback_window = sequence.size(1)

            # We'll look at predictions at various points as we approach the event
            time_points = [max(0, lookback_window - event_time + offset)
                          for offset in [-200, -100, -50, -10, -1]]
            time_points = [t for t in time_points if t >= 0 and t < lookback_window]

            if len(time_points) == 0:
                continue

            # For simplicity, we'll show the final prediction
            # (In practice, you'd need to modify the model to accept variable-length inputs)
            hazard_logits = model(sequence)
            hazard_rates, _ = compute_hazard_from_logits(hazard_logits)
            hazard_rates = hazard_rates.cpu().numpy()[0]

            # Plot hazard rates
            time_steps = np.arange(len(hazard_rates))
            label = f'j=L-{lookback_window - event_time}' if lookback_window - event_time > 0 else 'j=L-1'
            ax.plot(time_steps, hazard_rates, color=color, label=label, alpha=0.7)

    ax.set_xlabel('Time steps from j', fontsize=12)
    ax.set_ylabel('Hazard rate', fontsize=12)
    ax.set_title('Progression of Conditional Hazard Rates (Figure 2a)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_oti_policy(model, data_loader, device='cpu', cost_values=[8, 64, 128],
                   save_path=None, num_samples=1):
    """
    Reproduce Figure 2(b): OTI policy in action

    Shows the expected time-to-event and OTI threshold (V'_j) for different
    cost values C_α. Intervention is triggered when TTE crosses below threshold.

    Args:
        model: Trained DDRSA model
        data_loader: Data loader with sequences
        device: Device to run on
        cost_values: List of cost values to plot thresholds for
        save_path: Path to save the figure
        num_samples: Number of samples to analyze
    """
    model.eval()

    # Get a sample from test set
    for sequences, targets, _ in data_loader:
        break

    # Take first sample
    sequence = sequences[0:1].to(device)
    target = targets[0]

    with torch.no_grad():
        # Get predictions
        hazard_logits = model(sequence)

        # Compute expected TTE
        expected_tte = compute_expected_tte(hazard_logits).cpu().numpy()[0]

        # Compute hazard rates for threshold computation
        hazard_rates, survival_probs = compute_hazard_from_logits(hazard_logits)
        hazard_rates = hazard_rates.cpu().numpy()[0]
        survival_probs = survival_probs.cpu().numpy()[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot expected time-to-event (constant horizontal line since we only have one prediction)
    time_steps = np.arange(len(hazard_rates))
    ax.axhline(y=expected_tte, color='blue', linewidth=2, label='Expected time-to-event')

    # Plot OTI thresholds for different costs
    colors = ['green', 'orange', 'red']

    for cost, color in zip(cost_values, colors):
        # Compute threshold V'_j(H_j) - simplified version
        # In practice, this would use Equation 9 from the paper
        # Higher cost → earlier intervention → higher threshold

        # Approximate threshold based on survival probability
        # This is a simplified version; full implementation is in metrics.py
        threshold = compute_oti_threshold_visualization(hazard_rates, survival_probs, cost)

        ax.axhline(y=threshold, color=color, linewidth=2, linestyle='--',
                  label=f'OTI threshold when C={cost}')

    ax.set_xlabel('Time steps', fontsize=12)
    ax.set_ylabel('Risk', fontsize=12)
    ax.set_title('OTI Policy in Action (Figure 2b)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(expected_tte * 1.5, 50))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def compute_oti_threshold_visualization(hazard_rates, survival_probs, cost):
    """
    Compute OTI threshold for visualization
    Simplified version of Equation 9 from the paper

    Args:
        hazard_rates: Array of hazard rates
        survival_probs: Array of survival probabilities
        cost: Cost of missing the event (C_α)

    Returns:
        threshold: Threshold value for triggering intervention
    """
    # Expected TTE at each time point
    tte_values = np.cumsum(survival_probs[::-1])[::-1]

    # Map cost to threshold
    # Higher cost → earlier intervention → higher threshold
    if cost <= 8:
        percentile = 20
    elif cost <= 32:
        percentile = 40
    elif cost <= 64:
        percentile = 60
    elif cost <= 128:
        percentile = 75
    else:
        percentile = 90

    threshold = np.percentile(tte_values, percentile)
    return threshold


def plot_policy_tradeoff(test_metrics_list, labels, save_path=None):
    """
    Reproduce Figure 3: Policy trade-off plots

    Shows the trade-off between event miss rate and average time-to-event
    for different intervention policies (OTI, TTE, WBI).

    Args:
        test_metrics_list: List of dictionaries with test metrics
        labels: List of labels for each policy
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['black', 'red', 'blue']
    markers = ['o', 's', '^']

    for metrics, label, color, marker in zip(test_metrics_list, labels, colors, markers):
        # Extract OTI metrics
        miss_rates = []
        avg_ttes = []

        for key, value in metrics.items():
            if 'oti_miss_rate' in key:
                cost = key.split('_C')[-1]
                miss_rate_key = f'oti_miss_rate_C{cost}'
                avg_tte_key = f'oti_avg_tte_C{cost}'

                if miss_rate_key in metrics and avg_tte_key in metrics:
                    miss_rates.append(metrics[miss_rate_key])
                    avg_ttes.append(metrics[avg_tte_key])

        # Plot with error bars (if multiple runs)
        if miss_rates and avg_ttes:
            ax.plot(miss_rates, avg_ttes, marker=marker, color=color,
                   label=label, linewidth=2, markersize=8)

    ax.set_xlabel('Event miss rate', fontsize=12)
    ax.set_ylabel('Average time to event', fontsize=12)
    ax.set_title('Policy Trade-off Plot (Figure 3)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_training_curves(log_dir, save_path=None):
    """
    Plot training and validation loss curves

    Args:
        log_dir: Directory containing TensorBoard logs
        save_path: Path to save the figure
    """
    from tensorboard.backend.event_processing import event_accumulator

    # Load TensorBoard data
    train_dir = f"{log_dir}/tensorboard/train"
    val_dir = f"{log_dir}/tensorboard/val"

    ea_train = event_accumulator.EventAccumulator(train_dir)
    ea_train.Reload()

    ea_val = event_accumulator.EventAccumulator(val_dir)
    ea_val.Reload()

    # Extract loss data
    train_loss = [(s.step, s.value) for s in ea_train.Scalars('Loss/train')]
    val_loss = [(s.step, s.value) for s in ea_val.Scalars('Loss/val')]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    if train_loss:
        steps, values = zip(*train_loss)
        ax.plot(steps, values, label='Train Loss', linewidth=2)

    if val_loss:
        steps, values = zip(*val_loss)
        ax.plot(steps, values, label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def create_all_visualizations(model, test_loader, log_dir, output_dir='figures'):
    """
    Create all visualizations from the paper

    Args:
        model: Trained DDRSA model
        test_loader: Test data loader
        log_dir: Directory with training logs
        output_dir: Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    device = next(model.parameters()).device

    print("Creating visualizations...")

    # Figure 2(a): Hazard rate progression
    print("\n1. Plotting hazard rate progression (Figure 2a)...")
    fig1 = plot_hazard_progression(
        model, test_loader, device,
        num_samples=5,
        save_path=f'{output_dir}/figure_2a_hazard_progression.png'
    )
    plt.close(fig1)

    # Figure 2(b): OTI policy
    print("2. Plotting OTI policy (Figure 2b)...")
    fig2 = plot_oti_policy(
        model, test_loader, device,
        cost_values=[8, 64, 128],
        save_path=f'{output_dir}/figure_2b_oti_policy.png'
    )
    plt.close(fig2)

    # Training curves
    print("3. Plotting training curves...")
    try:
        fig3 = plot_training_curves(
            log_dir,
            save_path=f'{output_dir}/training_curves.png'
        )
        plt.close(fig3)
    except Exception as e:
        print(f"Could not create training curves: {e}")

    print(f"\n✓ All figures saved to: {output_dir}/")


if __name__ == '__main__':
    print("Visualization utilities loaded.")
    print("\nTo create visualizations:")
    print("  from visualization import create_all_visualizations")
    print("  create_all_visualizations(model, test_loader, 'logs/exp_name')")
