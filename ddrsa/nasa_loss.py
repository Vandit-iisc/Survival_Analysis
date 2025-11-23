"""
NASA/PHM08 Loss Function for Survival Analysis
Pure NASA scoring function for training (without DDRSA loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NASALoss(nn.Module):
    """
    NASA/PHM08 Challenge Scoring Function as a loss

    Asymmetric scoring that heavily penalizes late predictions:
    - Early prediction (predicted > true): s = exp(-d/13) - 1
    - Late prediction (predicted < true): s = exp(d/10) - 1

    where d = predicted - true

    This version includes auxiliary losses to help training stability
    """

    def __init__(self, mse_weight=0.1):
        """
        Args:
            mse_weight: Weight for MSE auxiliary loss (for training stability)
        """
        super(NASALoss, self).__init__()
        self.mse_weight = mse_weight

    def forward(self, hazard_logits, targets, censored):
        """
        Compute NASA loss from hazard predictions

        Args:
            hazard_logits: Predicted hazard logits (batch_size, pred_horizon)
            targets: Ground truth event times (batch_size, pred_horizon)
            censored: Censoring indicators (batch_size, pred_horizon)

        Returns:
            total_loss: Scalar loss value
        """
        # Convert hazard logits to expected TTE
        predicted_tte = self._compute_expected_tte(hazard_logits)

        # Get true event times (only for uncensored samples)
        is_uncensored = (targets.sum(dim=1) > 0).float()

        if is_uncensored.sum() == 0:
            # No uncensored samples in batch - return zero loss
            return torch.tensor(0.0, device=hazard_logits.device, requires_grad=True)

        # Extract true event times for uncensored samples
        true_event_times = torch.argmax(targets, dim=1).float()

        # Filter to uncensored samples only
        uncensored_mask = is_uncensored == 1
        predicted_tte_uncensored = predicted_tte[uncensored_mask]
        true_tte_uncensored = true_event_times[uncensored_mask]

        # Compute NASA score
        d = predicted_tte_uncensored - true_tte_uncensored

        # Early predictions (d > 0): lighter penalty
        early_mask = d > 0
        early_score = torch.exp(-d[early_mask] / 13.0) - 1

        # Late predictions (d <= 0): heavier penalty
        late_mask = d <= 0
        late_score = torch.exp(-d[late_mask] / 10.0) - 1

        # Combine scores
        total_score = torch.zeros_like(d)
        total_score[early_mask] = early_score
        total_score[late_mask] = late_score

        nasa_loss = total_score.mean()

        # Add MSE auxiliary loss for training stability
        mse_loss = F.mse_loss(predicted_tte_uncensored, true_tte_uncensored)

        total_loss = nasa_loss + self.mse_weight * mse_loss

        return total_loss

    def _compute_expected_tte(self, hazard_logits):
        """
        Compute expected time-to-event from hazard logits

        T_j = Σ_{k=0}^{L_max-1} ∏_{m=0}^{k} (1 - h_j(m))

        Args:
            hazard_logits: Tensor of shape (batch_size, pred_horizon)

        Returns:
            expected_tte: Expected time-to-event for each sample
        """
        # Convert logits to hazard rates
        hazard_rates = torch.sigmoid(hazard_logits)

        # Compute survival probabilities
        # S(k) = ∏_{m=0}^{k} (1 - h(m))
        log_survival = torch.cumsum(torch.log(1 - hazard_rates + 1e-7), dim=1)
        survival_probs = torch.exp(log_survival)

        # Add S(0) = 1 at the beginning
        batch_size = hazard_logits.size(0)
        survival_with_zero = torch.cat([
            torch.ones(batch_size, 1, device=hazard_logits.device),
            survival_probs
        ], dim=1)

        # Sum over all time steps
        expected_tte = survival_with_zero.sum(dim=1)

        return expected_tte


class NASALossDetailed(NASALoss):
    """
    Extended NASA loss that returns individual loss components for monitoring
    """

    def forward(self, hazard_logits, targets, censored):
        """
        Compute NASA loss with detailed components

        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary with individual loss components
        """
        # Convert hazard logits to expected TTE
        predicted_tte = self._compute_expected_tte(hazard_logits)

        # Get true event times
        is_uncensored = (targets.sum(dim=1) > 0).float()

        if is_uncensored.sum() == 0:
            # No uncensored samples
            loss_dict = {
                'total_loss': 0.0,
                'nasa_loss': 0.0,
                'mse_loss': 0.0,
                'uncensored_ratio': 0.0,
                'mean_predicted_tte': predicted_tte.mean().item(),
                'mean_true_tte': 0.0
            }
            return torch.tensor(0.0, device=hazard_logits.device, requires_grad=True), loss_dict

        # Extract true event times for uncensored samples
        true_event_times = torch.argmax(targets, dim=1).float()

        # Filter to uncensored samples
        uncensored_mask = is_uncensored == 1
        predicted_tte_uncensored = predicted_tte[uncensored_mask]
        true_tte_uncensored = true_event_times[uncensored_mask]

        # Compute NASA score
        d = predicted_tte_uncensored - true_tte_uncensored

        early_mask = d > 0
        early_score = torch.exp(-d[early_mask] / 13.0) - 1

        late_mask = d <= 0
        late_score = torch.exp(-d[late_mask] / 10.0) - 1

        total_score = torch.zeros_like(d)
        total_score[early_mask] = early_score
        total_score[late_mask] = late_score

        nasa_loss = total_score.mean()

        # MSE loss
        mse_loss = F.mse_loss(predicted_tte_uncensored, true_tte_uncensored)

        # Total loss
        total_loss = nasa_loss + self.mse_weight * mse_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'nasa_loss': nasa_loss.item(),
            'mse_loss': mse_loss.item(),
            'uncensored_ratio': is_uncensored.mean().item(),
            'mean_predicted_tte': predicted_tte_uncensored.mean().item(),
            'mean_true_tte': true_tte_uncensored.mean().item(),
            'early_predictions_ratio': early_mask.float().mean().item()
        }

        return total_loss, loss_dict


if __name__ == '__main__':
    # Test NASA loss
    batch_size = 8
    pred_horizon = 100

    # Create dummy data
    hazard_logits = torch.randn(batch_size, pred_horizon)

    # Create targets (some uncensored, some censored)
    targets = torch.zeros(batch_size, pred_horizon)
    censored = torch.ones(batch_size, pred_horizon)

    # Make half the samples uncensored with events at random times
    for i in range(batch_size // 2):
        event_time = torch.randint(20, 80, (1,)).item()
        targets[i, event_time] = 1
        censored[i, :event_time+1] = 0

    # Test loss
    print("Testing NASA Loss...")
    criterion = NASALossDetailed(mse_weight=0.1)
    loss, loss_dict = criterion(hazard_logits, targets, censored)

    print(f"\nTotal Loss: {loss.item():.4f}")
    print(f"\nLoss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
