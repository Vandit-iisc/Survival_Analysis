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
    Plots hazard rates at exactly j=L-200, j=L-100, j=L-50, j=L-10, j=L-1

    Args:
        model: Trained DDRSA model
        data_loader: Data loader with sequences
        device: Device to run on
        num_samples: Number of samples to plot (not used, kept for compatibility)
        save_path: Path to save the figure
        sample_indices: Specific indices to plot (not used, kept for compatibility)
    """
    model.eval()

    # Get samples from dataset - collect enough batches to find uncensored samples
    all_sequences = []
    all_targets = []

    # Collect more batches to ensure we find uncensored samples
    max_batches = 50
    for i, (sequences, targets, _) in enumerate(data_loader):
        all_sequences.append(sequences)
        all_targets.append(targets)
        if i >= max_batches:
            break

    if len(all_sequences) == 0:
        print("Warning: No data found in data loader")
        return plt.figure()

    all_sequences = torch.cat(all_sequences, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Select samples with uncensored events
    uncensored_mask = all_targets.sum(dim=1) > 0
    uncensored_sequences = all_sequences[uncensored_mask]
    uncensored_targets = all_targets[uncensored_mask]

    if len(uncensored_sequences) == 0:
        print(f"Warning: No uncensored samples found in {len(all_sequences)} samples.")
        return plt.figure()

    # Find a sample with event time > 200 to show all j values
    # Also prioritize samples that have actual data (non-zero values) throughout the sequence
    best_idx = 0
    best_event_time = 0
    best_data_length = 0

    for i in range(len(uncensored_sequences)):
        event_time = torch.argmax(uncensored_targets[i]).item()
        # Count non-zero timesteps to find sample with most actual data
        seq = uncensored_sequences[i]
        # Check how many timesteps have non-zero data (sum across features)
        data_length = (seq.abs().sum(dim=-1) > 0).sum().item()

        # Prefer samples with more data, then longer event times
        if data_length > best_data_length or (data_length == best_data_length and event_time > best_event_time):
            best_event_time = event_time
            best_data_length = data_length
            best_idx = i

    sequence = uncensored_sequences[best_idx:best_idx+1].to(device)
    target = uncensored_targets[best_idx]
    event_time = torch.argmax(target).item()
    lookback_window = sequence.size(1)

    print(f"Selected sample with event_time={event_time}, data_length={best_data_length}, lookback_window={lookback_window}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define the exact j values we want to plot (as offsets from L)
    # Adjust based on lookback window size
    if lookback_window >= 200:
        j_offsets = [200, 100, 50, 10, 1]
    elif lookback_window >= 100:
        j_offsets = [lookback_window - 1, 100, 50, 10, 1]
    else:
        # For smaller lookback windows (e.g., 128)
        j_offsets = [lookback_window - 1, lookback_window // 2, lookback_window // 4, 10, 1]

    # Filter out any offsets larger than lookback_window
    j_offsets = [o for o in j_offsets if o <= lookback_window]

    colors = plt.cm.viridis(np.linspace(0, 1, len(j_offsets)))

    with torch.no_grad():
        for offset, color in zip(j_offsets, colors):
            # Calculate how much of the sequence to use
            # j = L - offset means we're at position (total_length - offset)
            # We need to truncate the sequence to simulate being at that position

            if offset > lookback_window:
                continue  # Skip if we don't have enough data

            # Truncate sequence to simulate being at j = L - offset
            # We take the first (lookback_window - offset) time steps
            truncated_length = lookback_window - offset + 1
            if truncated_length <= 0:
                continue

            # Pad the truncated sequence to match expected input size
            truncated_seq = sequence[:, :truncated_length, :]

            # Pad to original size (model expects fixed input size)
            if truncated_length < lookback_window:
                padding = torch.zeros(1, lookback_window - truncated_length, sequence.size(2), device=device)
                padded_seq = torch.cat([padding, truncated_seq], dim=1)
            else:
                padded_seq = truncated_seq

            # Get hazard predictions
            hazard_logits = model(padded_seq)
            hazard_rates, _ = compute_hazard_from_logits(hazard_logits)
            hazard_rates = hazard_rates.cpu().numpy()[0]

            # Plot hazard rates
            time_steps = np.arange(len(hazard_rates))
            label = f'j=L-{offset}'
            ax.plot(time_steps, hazard_rates, color=color, label=label, alpha=0.7, linewidth=1.5)

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

    Shows how expected TTE decreases over time as we approach the event,
    along with OTI thresholds for different costs.

    Args:
        model: Trained DDRSA model
        data_loader: Data loader with sequences
        device: Device to run on
        cost_values: List of cost values to plot thresholds for
        save_path: Path to save the figure
        num_samples: Number of samples to analyze
    """
    model.eval()

    # Get samples from data loader - search through multiple batches for uncensored samples
    sequence = None
    target = None
    event_time = None
    lookback = None

    for sequences, targets, _ in data_loader:
        # Find an uncensored sample (has actual event)
        uncensored_mask = targets.sum(dim=1) > 0
        if uncensored_mask.sum() > 0:
            sample_idx = uncensored_mask.nonzero(as_tuple=True)[0][0].item()
            sequence = sequences[sample_idx:sample_idx+1]
            target = targets[sample_idx]
            event_time = torch.argmax(target).item()
            lookback = sequence.size(1)
            break

    # If no uncensored sample found in any batch, use first sample from first batch
    if sequence is None:
        print("Warning: No uncensored samples found in entire dataset, using first sample")
        for sequences, targets, _ in data_loader:
            sequence = sequences[0:1]
            target = targets[0]
            event_time = torch.argmax(target).item() if target.sum() > 0 else 50
            lookback = sequence.size(1)
            break

    # Simulate sequential observations leading up to the event
    # In the paper, they show expected TTE at each observation cycle j
    # The TTE decreases as j increases (we're getting closer to the event)
    # The zigzag pattern comes from actual model predictions at each time step
    time_points = []
    tte_values = []

    with torch.no_grad():
        # Compute expected TTE at different observation points
        # by truncating the sequence and padding

        # Sample points throughout the sequence
        num_points = min(100, lookback)
        step_size = max(1, lookback // num_points)

        # Move sequence to device once
        sequence = sequence.to(device)

        for j in range(1, lookback + 1, step_size):
            # At observation cycle j, we have seen j time steps of data
            # Truncate sequence to first j elements and pad
            truncated_seq = sequence[:, :j, :]

            # Pad to original size (model expects fixed input size)
            if j < lookback:
                padding = torch.zeros(1, lookback - j, sequence.size(2), device=device)
                padded_seq = torch.cat([padding, truncated_seq], dim=1)
            else:
                padded_seq = truncated_seq

            # Get model prediction
            hazard_logits = model(padded_seq)
            expected_tte = compute_expected_tte(hazard_logits).cpu().numpy()[0]

            time_points.append(j)
            tte_values.append(expected_tte)

    # Get hazard rates for threshold computation
    with torch.no_grad():
        hazard_logits = model(sequence.to(device))
        hazard_rates, survival_probs = compute_hazard_from_logits(hazard_logits)
        hazard_rates = hazard_rates.cpu().numpy()[0]
        survival_probs = survival_probs.cpu().numpy()[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot expected TTE over time (decreasing as we approach event)
    ax.plot(time_points, tte_values, color='blue', linewidth=2,
            label='Expected time-to-event', marker='o', markersize=4)

    # Plot OTI thresholds for different costs (horizontal lines)
    colors = ['green', 'orange', 'red']

    for cost, color in zip(cost_values, colors):
        # Compute threshold
        threshold = compute_oti_threshold_visualization(hazard_rates, survival_probs, cost)

        ax.axhline(y=threshold, color=color, linewidth=2, linestyle='--',
                  label=f'OTI threshold when C={cost}')

    ax.set_xlabel('Observation cycle j', fontsize=12)
    ax.set_ylabel('Expected TTE', fontsize=12)
    ax.set_title('OTI Policy in Action (Figure 2b)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set axis limits to match paper (x: 0-250, y: 0-120)
    if len(tte_values) > 0:
        ax.set_ylim(0, max(max(tte_values) * 1.1, 120))
        ax.set_xlim(0, max(time_points) * 1.05)

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


def create_all_visualizations(model, train_loader, val_loader, test_loader,
                            log_dir, output_dir='figures', use_train_data=False):
    """
    Create all visualizations from the paper

    Args:
        model: Trained DDRSA model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        log_dir: Directory with training logs
        output_dir: Directory to save figures
        use_train_data: If True, use training data for visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    device = next(model.parameters()).device

    # Choose which data loader to use for visualizations
    data_loader = train_loader if use_train_data else test_loader
    data_type = "train" if use_train_data else "test"

    print(f"Creating visualizations using {data_type} data...")

    # Figure 2(a): Hazard rate progression
    print(f"\n1. Plotting hazard rate progression (Figure 2a) on {data_type} data...")
    fig1 = plot_hazard_progression(
        model, data_loader, device,
        num_samples=5,
        save_path=f'{output_dir}/figure_2a_hazard_progression_{data_type}.png'
    )
    plt.close(fig1)

    # Figure 2(b): OTI policy
    print(f"2. Plotting OTI policy (Figure 2b) on {data_type} data...")
    fig2 = plot_oti_policy(
        model, data_loader, device,
        cost_values=[8, 64, 128],
        save_path=f'{output_dir}/figure_2b_oti_policy_{data_type}.png'
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
