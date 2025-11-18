"""
DDRSA Loss Function
Implements the loss function from Equation 12 in the paper:
"When to Intervene: Learning Optimal Intervention Policies for Critical Events"

Loss components:
- l_z: likelihood of event occurrence at specific time
- l_u: event rate (for uncensored samples)
- l_c: censored likelihood (for censored samples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDRSALoss(nn.Module):
    """
    DDRSA Loss Function (Equation 12 from paper)

    L_f = -λ log l_z - (1-λ)[(1-c) log l_u + c log l_c]

    where:
    - l_z: likelihood of event at time l (uncensored)
    - l_u: event rate (probability of event occurring at any time)
    - l_c: survival likelihood (censored samples)
    - λ: trade-off parameter (alpha in paper)
    - c: censoring indicator (1 if censored, 0 if uncensored)
    """

    def __init__(self, lambda_param=0.5):
        """
        Args:
            lambda_param: Trade-off parameter λ (default 0.5 as in paper)
        """
        super(DDRSALoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, hazard_logits, targets, censored):
        """
        Compute DDRSA loss

        Args:
            hazard_logits: Predicted hazard logits of shape (batch_size, pred_horizon)
            targets: Ground truth event times of shape (batch_size, pred_horizon)
                    targets[i, k] = 1 if event occurs at time k, 0 otherwise
            censored: Censoring indicators of shape (batch_size, pred_horizon)
                     censored[i, k] = 1 if censored at time k, 0 if observed

        Returns:
            total_loss: Scalar loss value
        """
        batch_size = hazard_logits.size(0)
        pred_horizon = hazard_logits.size(1)

        # Convert logits to hazard rates using sigmoid
        # h_j(k) = σ(logit)
        hazard_rates = torch.sigmoid(hazard_logits)

        # Compute survival probabilities
        # S(k) = ∏_{m=0}^{k} (1 - h(m))
        log_survival = torch.cumsum(torch.log(1 - hazard_rates + 1e-7), dim=1)
        survival_probs = torch.exp(log_survival)

        # Determine uncensored and censored samples
        # A sample is uncensored if any target is 1
        is_uncensored = (targets.sum(dim=1) > 0).float()  # Shape: (batch_size,)
        is_censored = 1 - is_uncensored

        # ====== Compute l_z (likelihood of event at specific time) ======
        # l_z = ∏_{k=0}^{l-1} (1 - h(k)) * h(l)
        # where l is the event time

        # For uncensored samples, find event time
        event_times = torch.argmax(targets, dim=1)  # Shape: (batch_size,)

        # Gather hazard rates at event times
        hazard_at_event = torch.gather(hazard_rates, 1, event_times.unsqueeze(1)).squeeze(1)

        # Gather survival probability up to (but not including) event time
        # We need S(l-1) for uncensored samples
        event_times_minus_one = torch.clamp(event_times - 1, min=0)
        survival_before_event = torch.gather(
            torch.cat([torch.ones(batch_size, 1, device=hazard_logits.device),
                      survival_probs[:, :-1]], dim=1),
            1,
            event_times_minus_one.unsqueeze(1)
        ).squeeze(1)

        # l_z for each sample
        l_z = survival_before_event * hazard_at_event
        l_z = torch.clamp(l_z, min=1e-7)  # Numerical stability

        # Only compute for uncensored samples
        loss_z = -torch.log(l_z) * is_uncensored
        loss_z = loss_z.mean()

        # ====== Compute l_u (event rate for uncensored) ======
        # l_u = 1 - ∏_{k=0}^{L_max-1} (1 - h(k))
        # This is the probability that event occurs at ANY time in the horizon

        final_survival = survival_probs[:, -1]  # S(L_max - 1)
        l_u = 1 - final_survival
        l_u = torch.clamp(l_u, min=1e-7)

        # Only compute for uncensored samples
        loss_u = -torch.log(l_u) * is_uncensored
        loss_u = loss_u.mean()

        # ====== Compute l_c (censored likelihood) ======
        # l_c = ∏_{k=0}^{L_max-1} (1 - h(k))
        # This is the survival probability (no event occurs)

        l_c = final_survival
        l_c = torch.clamp(l_c, min=1e-7)

        # Only compute for censored samples
        loss_c = -torch.log(l_c) * is_censored
        loss_c = loss_c.mean()

        # ====== Combine losses according to Equation 12 ======
        # L_f = λ * loss_z + (1-λ) * [loss_u + loss_c]
        # Note: We include both loss_u and loss_c in the second term
        # because the original equation has (1-c)*log(l_u) + c*log(l_c)
        # which sums to loss_u (for uncensored) + loss_c (for censored)

        total_loss = self.lambda_param * loss_z + (1 - self.lambda_param) * (loss_u + loss_c)

        return total_loss


class DDRSALossDetailed(DDRSALoss):
    """
    Extended DDRSA loss that returns individual loss components for monitoring
    """

    def forward(self, hazard_logits, targets, censored):
        """
        Compute DDRSA loss with detailed components

        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary with individual loss components
        """
        batch_size = hazard_logits.size(0)
        pred_horizon = hazard_logits.size(1)

        hazard_rates = torch.sigmoid(hazard_logits)
        log_survival = torch.cumsum(torch.log(1 - hazard_rates + 1e-7), dim=1)
        survival_probs = torch.exp(log_survival)

        is_uncensored = (targets.sum(dim=1) > 0).float()
        is_censored = 1 - is_uncensored

        # l_z
        event_times = torch.argmax(targets, dim=1)
        hazard_at_event = torch.gather(hazard_rates, 1, event_times.unsqueeze(1)).squeeze(1)
        event_times_minus_one = torch.clamp(event_times - 1, min=0)
        survival_before_event = torch.gather(
            torch.cat([torch.ones(batch_size, 1, device=hazard_logits.device),
                      survival_probs[:, :-1]], dim=1),
            1,
            event_times_minus_one.unsqueeze(1)
        ).squeeze(1)

        l_z = survival_before_event * hazard_at_event
        l_z = torch.clamp(l_z, min=1e-7)
        loss_z = -torch.log(l_z) * is_uncensored
        loss_z_mean = loss_z.mean()

        # l_u
        final_survival = survival_probs[:, -1]
        l_u = 1 - final_survival
        l_u = torch.clamp(l_u, min=1e-7)
        loss_u = -torch.log(l_u) * is_uncensored
        loss_u_mean = loss_u.mean()

        # l_c
        l_c = final_survival
        l_c = torch.clamp(l_c, min=1e-7)
        loss_c = -torch.log(l_c) * is_censored
        loss_c_mean = loss_c.mean()

        # Total loss
        total_loss = self.lambda_param * loss_z_mean + (1 - self.lambda_param) * (loss_u_mean + loss_c_mean)

        loss_dict = {
            'total_loss': total_loss.item(),
            'loss_z': loss_z_mean.item(),
            'loss_u': loss_u_mean.item(),
            'loss_c': loss_c_mean.item(),
            'uncensored_ratio': is_uncensored.mean().item()
        }

        return total_loss, loss_dict


def compute_hazard_from_logits(hazard_logits):
    """
    Convert hazard logits to hazard rates and survival probabilities

    Args:
        hazard_logits: Tensor of shape (batch_size, pred_horizon)

    Returns:
        hazard_rates: h_j(k) for each k
        survival_probs: S_j(k) = P(T > j+k | X_j, T > j)
    """
    hazard_rates = torch.sigmoid(hazard_logits)

    # Compute survival probabilities
    # S(k) = ∏_{m=0}^{k} (1 - h(m))
    log_survival = torch.cumsum(torch.log(1 - hazard_rates + 1e-7), dim=1)
    survival_probs = torch.exp(log_survival)

    return hazard_rates, survival_probs


def compute_expected_tte(hazard_logits):
    """
    Compute expected time-to-event (TTE) from hazard logits

    This is used in the OTI policy (Equation 10 in paper):
    T_j = Σ_{k=0}^{L_max-1} ∏_{m=0}^{k} (1 - h_j(m))

    Args:
        hazard_logits: Tensor of shape (batch_size, pred_horizon)

    Returns:
        expected_tte: Expected time-to-event for each sample
    """
    _, survival_probs = compute_hazard_from_logits(hazard_logits)

    # Add S(0) = 1 at the beginning
    batch_size = hazard_logits.size(0)
    survival_with_zero = torch.cat([
        torch.ones(batch_size, 1, device=hazard_logits.device),
        survival_probs
    ], dim=1)

    # Sum over all time steps
    expected_tte = survival_with_zero.sum(dim=1)

    return expected_tte


if __name__ == '__main__':
    # Test loss function
    batch_size = 8
    pred_horizon = 100

    # Create dummy data
    hazard_logits = torch.randn(batch_size, pred_horizon)

    # Create targets (some uncensored, some censored)
    targets = torch.zeros(batch_size, pred_horizon)
    censored = torch.ones(batch_size, pred_horizon)

    # Make half the samples uncensored with events at random times
    for i in range(batch_size // 2):
        event_time = torch.randint(0, pred_horizon, (1,)).item()
        targets[i, event_time] = 1
        censored[i, :event_time+1] = 0

    # Test loss
    print("Testing DDRSA Loss...")
    criterion = DDRSALossDetailed(lambda_param=0.5)
    loss, loss_dict = criterion(hazard_logits, targets, censored)

    print(f"Total Loss: {loss.item():.4f}")
    print(f"Loss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # Test TTE computation
    print("\nTesting Expected TTE computation...")
    expected_tte = compute_expected_tte(hazard_logits)
    print(f"Expected TTE shape: {expected_tte.shape}")
    print(f"Expected TTE (first 5): {expected_tte[:5]}")
