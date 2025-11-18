"""
Evaluation metrics and OTI (Optimal Timed Intervention) policy
Implements evaluation metrics and intervention policies from the paper
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from loss import compute_hazard_from_logits, compute_expected_tte


def evaluate_model(predictions, targets, censored):
    """
    Evaluate model predictions

    Args:
        predictions: Hazard logits of shape (N, pred_horizon)
        targets: Ground truth targets of shape (N, pred_horizon)
        censored: Censoring indicators of shape (N, pred_horizon)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Convert to numpy for easier processing
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(censored, torch.Tensor):
        censored = censored.cpu().numpy()

    # Convert back to torch for hazard computation
    predictions_tensor = torch.FloatTensor(predictions)
    hazard_rates, survival_probs = compute_hazard_from_logits(predictions_tensor)
    hazard_rates = hazard_rates.numpy()
    survival_probs = survival_probs.numpy()

    # Compute expected TTE
    expected_tte_tensor = compute_expected_tte(predictions_tensor)
    expected_tte = expected_tte_tensor.numpy()

    # Get true RUL (time to event)
    true_rul = []
    for i in range(len(targets)):
        if targets[i].sum() > 0:
            # Uncensored: find event time
            true_rul.append(np.argmax(targets[i]))
        else:
            # Censored: use prediction horizon as upper bound
            true_rul.append(len(targets[i]))
    true_rul = np.array(true_rul)

    # Compute metrics
    metrics = {}

    # RUL prediction metrics
    # Only evaluate on uncensored samples for fairness
    uncensored_mask = targets.sum(axis=1) > 0

    if uncensored_mask.sum() > 0:
        metrics['rmse'] = np.sqrt(mean_squared_error(
            true_rul[uncensored_mask],
            expected_tte[uncensored_mask]
        ))
        metrics['mae'] = mean_absolute_error(
            true_rul[uncensored_mask],
            expected_tte[uncensored_mask]
        )

        # NASA scoring function (asymmetric)
        errors = expected_tte[uncensored_mask] - true_rul[uncensored_mask]
        nasa_score = compute_nasa_score(errors)
        metrics['nasa_score'] = nasa_score

    # Survival prediction metrics
    # C-index (concordance index)
    c_index = compute_concordance_index(expected_tte, true_rul, uncensored_mask)
    metrics['c_index'] = c_index

    # Integrated Brier Score
    ibs = compute_integrated_brier_score(survival_probs, targets, censored)
    metrics['integrated_brier_score'] = ibs

    return metrics


def compute_nasa_score(errors, a1=13, a2=10):
    """
    Compute NASA asymmetric scoring function

    From the PHM08 challenge:
    s = Σ exp(d/a1) - 1  for d < 0 (early predictions)
    s = Σ exp(d/a2) - 1  for d ≥ 0 (late predictions)

    where d = estimated RUL - true RUL

    Args:
        errors: Array of prediction errors (estimated - true)
        a1: Parameter for early predictions (default 13)
        a2: Parameter for late predictions (default 10)

    Returns:
        score: NASA score (lower is better)
    """
    score = 0
    for error in errors:
        if error < 0:
            # Early prediction
            score += np.exp(-error / a1) - 1
        else:
            # Late prediction
            score += np.exp(error / a2) - 1

    return score


def compute_concordance_index(predicted_time, true_time, event_observed):
    """
    Compute concordance index (C-index) for survival analysis

    Args:
        predicted_time: Predicted time to event
        true_time: True time to event
        event_observed: Whether event was observed (not censored)

    Returns:
        c_index: Concordance index (between 0 and 1, higher is better)
    """
    n = len(predicted_time)
    concordant = 0
    permissible = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Only consider pairs where at least one event is observed
            if event_observed[i] or event_observed[j]:
                # Check if ordering is correct
                if true_time[i] < true_time[j]:
                    permissible += 1
                    if predicted_time[i] < predicted_time[j]:
                        concordant += 1
                    elif predicted_time[i] == predicted_time[j]:
                        concordant += 0.5
                elif true_time[i] > true_time[j]:
                    permissible += 1
                    if predicted_time[i] > predicted_time[j]:
                        concordant += 1
                    elif predicted_time[i] == predicted_time[j]:
                        concordant += 0.5

    if permissible == 0:
        return 0.5  # Random performance if no permissible pairs

    c_index = concordant / permissible
    return c_index


def compute_integrated_brier_score(survival_probs, targets, censored, time_points=None):
    """
    Compute Integrated Brier Score for survival predictions

    Args:
        survival_probs: Predicted survival probabilities (N, pred_horizon)
        targets: Ground truth targets (N, pred_horizon)
        censored: Censoring indicators (N, pred_horizon)
        time_points: Specific time points to evaluate (if None, use all)

    Returns:
        ibs: Integrated Brier Score
    """
    n_samples, pred_horizon = survival_probs.shape

    if time_points is None:
        time_points = range(pred_horizon)

    brier_scores = []

    for t in time_points:
        # True event status at time t
        # 1 if event occurred by time t, 0 otherwise
        true_status = (targets[:, :t+1].sum(axis=1) > 0).astype(float)

        # Predicted survival at time t
        pred_survival = survival_probs[:, t]

        # Brier score at time t
        # Only consider samples that are not censored before time t
        not_censored_before_t = (censored[:, :t+1].sum(axis=1) < (t + 1))

        if not_censored_before_t.sum() > 0:
            brier_t = ((true_status - pred_survival) ** 2)[not_censored_before_t].mean()
            brier_scores.append(brier_t)

    # Integrate over time
    ibs = np.mean(brier_scores) if len(brier_scores) > 0 else 0.0

    return ibs


def compute_oti_metrics(predictions, targets, censored, cost_values=[8, 16, 32, 64, 128, 256]):
    """
    Compute OTI (Optimal Timed Intervention) policy metrics

    Implements the OTI policy from Corollary 4.1.1 in the paper:
    φ*_j(X_j) = 1(T_j ≤ V'_j(H_j))

    Args:
        predictions: Hazard logits of shape (N, pred_horizon)
        targets: Ground truth targets of shape (N, pred_horizon)
        censored: Censoring indicators of shape (N, pred_horizon)
        cost_values: List of cost values C_α to evaluate

    Returns:
        oti_metrics: Dictionary of OTI policy metrics for different costs
    """
    if isinstance(predictions, torch.Tensor):
        predictions_tensor = predictions
    else:
        predictions_tensor = torch.FloatTensor(predictions)

    # Compute expected TTE
    expected_tte = compute_expected_tte(predictions_tensor).detach().numpy()

    # Get true event times
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(censored, torch.Tensor):
        censored = censored.cpu().numpy()

    true_event_times = []
    for i in range(len(targets)):
        if targets[i].sum() > 0:
            true_event_times.append(np.argmax(targets[i]))
        else:
            true_event_times.append(len(targets[i]))  # Censored
    true_event_times = np.array(true_event_times)

    oti_metrics = {}

    # For each cost value, compute intervention metrics
    for cost in cost_values:
        # Compute threshold V'_j(H_j) based on cost
        # This is a simplified version; full implementation would use Equation 9
        threshold = compute_oti_threshold(predictions_tensor, cost)

        # Trigger intervention when expected TTE <= threshold
        interventions = expected_tte <= threshold

        # Compute metrics
        # Miss rate: fraction of samples where intervention is too late
        missed = (expected_tte > true_event_times).astype(float)
        miss_rate = missed.mean()

        # Average time to event for interventions
        intervened_tte = expected_tte[interventions]
        avg_tte = intervened_tte.mean() if len(intervened_tte) > 0 else 0

        oti_metrics[f'oti_miss_rate_C{cost}'] = miss_rate
        oti_metrics[f'oti_avg_tte_C{cost}'] = avg_tte

    return oti_metrics


def compute_oti_threshold(hazard_logits, cost, pred_horizon=None):
    """
    Compute OTI threshold V'_j(H_j) from Equation 9 in the paper

    This is a simplified version that uses expected TTE percentiles
    A full implementation would compute the exact continuation value function

    Args:
        hazard_logits: Hazard logits tensor
        cost: Cost of missing the event (C_α)
        pred_horizon: Prediction horizon

    Returns:
        threshold: Threshold value for triggering intervention
    """
    expected_tte = compute_expected_tte(hazard_logits).numpy()

    # Map cost to threshold using percentiles
    # Higher cost → earlier intervention → higher threshold
    # This is an approximation; exact computation requires Equation 9
    cost_to_percentile = {
        8: 0.1,
        16: 0.2,
        32: 0.3,
        64: 0.5,
        128: 0.7,
        256: 0.9
    }

    percentile = cost_to_percentile.get(cost, 0.5)
    threshold = np.percentile(expected_tte, percentile * 100)

    return threshold


def compute_intervention_policy_metrics(predictions, targets, thresholds):
    """
    Compute metrics for different intervention policies

    Args:
        predictions: Model predictions
        targets: Ground truth
        thresholds: List of thresholds to evaluate

    Returns:
        metrics: Dictionary of metrics for each threshold
    """
    expected_tte = compute_expected_tte(predictions).numpy()

    # Get true event times
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    true_event_times = []
    for i in range(len(targets)):
        if targets[i].sum() > 0:
            true_event_times.append(np.argmax(targets[i]))
        else:
            true_event_times.append(len(targets[i]))
    true_event_times = np.array(true_event_times)

    metrics = {}

    for threshold in thresholds:
        # Intervention triggered when expected TTE <= threshold
        interventions = expected_tte <= threshold

        # Miss rate
        missed = (expected_tte > true_event_times).astype(float)
        miss_rate = missed.mean()

        # Average residual time to event
        residual_tte = np.maximum(true_event_times - expected_tte, 0)
        avg_residual_tte = residual_tte.mean()

        # Precision and recall
        true_positives = ((expected_tte <= threshold) & (expected_tte <= true_event_times)).sum()
        false_positives = ((expected_tte <= threshold) & (expected_tte > true_event_times)).sum()
        false_negatives = ((expected_tte > threshold) & (expected_tte <= true_event_times)).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        metrics[f'threshold_{threshold}'] = {
            'miss_rate': miss_rate,
            'avg_residual_tte': avg_residual_tte,
            'precision': precision,
            'recall': recall
        }

    return metrics
