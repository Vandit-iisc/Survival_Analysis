# DDRSA Formula Verification: Code vs Paper

## Summary

I've reviewed both the **DDRSA Loss Function** and **OTI Policy** implementations. Here's what I found:

### ✅ Correctly Implemented
1. **Hazard rate computation**
2. **Survival probability computation**
3. **Expected TTE computation**
4. **Loss component l_z** (mostly correct)
5. **Loss component l_u** (correct)
6. **Loss component l_c** (correct)

### ⚠️ Issues Found

1. **Loss function formula**: Minor interpretation issue in combining components
2. **OTI threshold computation**: Uses approximation instead of exact formula

---

## 1. Hazard Rates and Survival Functions

### Paper Definition

**Hazard Rate** (Equation 3):
```latex
h_j(k) = \sigma(\text{logit}_j(k))
```
where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

**Survival Function** (Equation 4):
```latex
S_j(k) = P(T > j+k \mid X_j, T > j) = \prod_{m=0}^{k} (1 - h_j(m))
```

### Code Implementation (loss.py:56-63)

```python
# Convert logits to hazard rates using sigmoid
hazard_rates = torch.sigmoid(hazard_logits)

# Compute survival probabilities
# S(k) = ∏_{m=0}^{k} (1 - h(m))
log_survival = torch.cumsum(torch.log(1 - hazard_rates + 1e-7), dim=1)
survival_probs = torch.exp(log_survival)
```

### LaTeX Conversion

```latex
h_j(k) &= \sigma(\text{logit}_j(k)) \\
S_j(k) &= \exp\left(\sum_{m=0}^{k} \log(1 - h_j(m))\right) = \prod_{m=0}^{k} (1 - h_j(m))
```

**✅ CORRECT**: Numerically stable log-sum-exp trick

---

## 2. Expected Time-to-Event (TTE)

### Paper Definition (Equation 10)

```latex
T_j = \mathbb{E}[T - j \mid X_j, T > j] = \sum_{k=0}^{L_{\max}-1} \prod_{m=0}^{k} (1 - h_j(m)) = \sum_{k=0}^{L_{\max}-1} S_j(k)
```

Note: Some versions include $S_j(0) = 1$ explicitly.

### Code Implementation (loss.py:219-244)

```python
def compute_expected_tte(hazard_logits):
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
```

### LaTeX Conversion

```latex
T_j = \sum_{k=0}^{L_{\max}} S_j(k) \quad \text{where } S_j(0) = 1
```

**✅ CORRECT**: Includes initial survival probability $S_j(0) = 1$

---

## 3. DDRSA Loss Function

### Paper Definition (Equation 12)

The paper states:
```latex
\mathcal{L}_f = -\alpha \log \ell_z - (1-\alpha) \left[(1-c) \log \ell_u + c \log \ell_c\right]
```

where:
- $\ell_z$: Likelihood of event at specific time $\ell$ (for uncensored samples)
- $\ell_u$: Event rate (probability event occurs in prediction horizon)
- $\ell_c$: Survival likelihood (no event occurs)
- $c$: Censoring indicator (1 if censored, 0 if observed)
- $\alpha$: Trade-off parameter (called $\lambda$ in code)

### Component Definitions

**Component $\ell_z$ (Equation 11):**
```latex
\ell_z = P(T = j + \ell \mid X_j, T > j) = h_j(\ell) \prod_{m=0}^{\ell-1} (1 - h_j(m)) = h_j(\ell) \cdot S_j(\ell-1)
```

**Component $\ell_u$:**
```latex
\ell_u = P(T \leq j + L_{\max} \mid X_j, T > j) = 1 - \prod_{m=0}^{L_{\max}-1} (1 - h_j(m)) = 1 - S_j(L_{\max}-1)
```

**Component $\ell_c$:**
```latex
\ell_c = P(T > j + L_{\max} \mid X_j, T > j) = \prod_{m=0}^{L_{\max}-1} (1 - h_j(m)) = S_j(L_{\max}-1)
```

### Code Implementation (loss.py:70-127)

**Component l_z:**
```python
# l_z = h(l) * S(l-1)
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
loss_z = -torch.log(l_z) * is_uncensored
```

**Component l_u:**
```python
# l_u = 1 - S(L_max - 1)
final_survival = survival_probs[:, -1]
l_u = 1 - final_survival
loss_u = -torch.log(l_u) * is_uncensored
```

**Component l_c:**
```python
# l_c = S(L_max - 1)
l_c = final_survival
loss_c = -torch.log(l_c) * is_censored
```

**Total Loss:**
```python
total_loss = self.lambda_param * loss_z + (1 - self.lambda_param) * (loss_u + loss_c)
```

### LaTeX Conversion of Code

```latex
\ell_z &= h_j(\ell) \cdot S_j(\ell - 1) \\
\ell_u &= 1 - S_j(L_{\max} - 1) \\
\ell_c &= S_j(L_{\max} - 1) \\
\\
\text{loss}_z &= -\log(\ell_z) \cdot \mathbb{1}_{\{c=0\}} \\
\text{loss}_u &= -\log(\ell_u) \cdot \mathbb{1}_{\{c=0\}} \\
\text{loss}_c &= -\log(\ell_c) \cdot \mathbb{1}_{\{c=1\}} \\
\\
\mathcal{L}_{\text{total}} &= \lambda \cdot \text{loss}_z + (1 - \lambda) \cdot (\text{loss}_u + \text{loss}_c)
```

### ⚠️ Issue with Loss Function

**Paper formula:**
```latex
\mathcal{L}_f = -\alpha \log \ell_z - (1-\alpha) \left[(1-c) \log \ell_u + c \log \ell_c\right]
```

**Code implementation (expanded):**
```latex
\mathcal{L}_{\text{code}} = \lambda \cdot (-\log \ell_z) \cdot \mathbb{1}_{c=0} + (1-\lambda) \cdot \left[(-\log \ell_u) \cdot \mathbb{1}_{c=0} + (-\log \ell_c) \cdot \mathbb{1}_{c=1}\right]
```

Simplifying the code version:
```latex
\mathcal{L}_{\text{code}} = -\lambda \log \ell_z \cdot \mathbb{1}_{c=0} - (1-\lambda) \left[\log \ell_u \cdot \mathbb{1}_{c=0} + \log \ell_c \cdot \mathbb{1}_{c=1}\right]
```

**Paper version (per sample):**
```latex
\mathcal{L}_{\text{paper}} = -\alpha \log \ell_z - (1-\alpha) \left[(1-c) \log \ell_u + c \log \ell_c\right]
```

When $c = 0$ (uncensored):
```latex
\mathcal{L}_{\text{paper}} &= -\alpha \log \ell_z - (1-\alpha) \log \ell_u \\
\mathcal{L}_{\text{code}} &= -\lambda \log \ell_z - (1-\lambda) \log \ell_u
```
**✅ CORRECT** for uncensored samples

When $c = 1$ (censored):
```latex
\mathcal{L}_{\text{paper}} &= -\alpha \log \ell_z - (1-\alpha) \log \ell_c \\
\mathcal{L}_{\text{code}} &= 0 - (1-\lambda) \log \ell_c = -(1-\lambda) \log \ell_c
```

**⚠️ ISSUE**: For censored samples, the paper includes $-\alpha \log \ell_z$ but:
- Censored samples don't have a defined event time $\ell$
- The code correctly sets `loss_z = 0` for censored samples
- But this means the code doesn't match the literal formula

**However**, this is likely a **correct interpretation** because:
1. $\ell_z$ is only defined for uncensored samples (where event time is known)
2. For censored samples, $\ell_z$ is meaningless
3. The paper likely means the formula applies only to uncensored samples for the $\ell_z$ term

**Conclusion**: Code implementation is **practically correct** but differs slightly from literal paper formula for censored samples.

---

## 4. OTI (Optimal Timed Intervention) Policy

### Paper Definition (Corollary 4.1.1)

**Optimal Policy:**
```latex
\phi^*_j(X_j) = \mathbb{1}\{T_j \leq V'_j(H_j)\}
```

Intervene when expected TTE $T_j$ is less than or equal to threshold $V'_j(H_j)$.

**Threshold Function** (Equation 9):
```latex
V'_j(H_j) = \inf\left\{v : v \geq \mathbb{E}_{T|X_j, T>j}\left[\min(T-j, L_{\max}) - C_\alpha \mathbb{1}\{T-j < v\} \mid H_j\right]\right\}
```

where:
- $C_\alpha$: Cost of missing the event
- $H_j$: Hazard rate history
- $v$: Threshold value

### Code Implementation (metrics.py:200-296)

**Main OTI Function:**
```python
def compute_oti_metrics(predictions, targets, censored, cost_values=[8, 16, 32, 64, 128, 256]):
    # Compute expected TTE
    expected_tte = compute_expected_tte(predictions_tensor).detach().numpy()

    # For each cost value, compute intervention metrics
    for cost in cost_values:
        # Compute threshold V'_j(H_j) based on cost
        threshold = compute_oti_threshold(predictions_tensor, cost)

        # Trigger intervention when expected TTE <= threshold
        interventions = expected_tte <= threshold
```

**Threshold Computation:**
```python
def compute_oti_threshold(hazard_logits, cost, pred_horizon=None):
    expected_tte = compute_expected_tte(hazard_logits).numpy()

    # Map cost to threshold using percentiles
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
```

### LaTeX Conversion of Code

**Policy:**
```latex
\phi_j(X_j) = \mathbb{1}\{T_j \leq \tau_C\}
```
where $\tau_C$ is an approximation of $V'_j(H_j)$ using percentiles.

**Threshold Approximation:**
```latex
\tau_C = \text{percentile}_{p(C)}(T_j) \quad \text{where } p(C) \in [0.1, 0.9]
```

### ⚠️ Issue with OTI Implementation

**Problem**: The code uses a **heuristic percentile-based threshold** instead of computing the exact continuation value function from Equation 9.

**Paper's exact formula:**
```latex
V'_j(H_j) = \inf\left\{v : v \geq \mathbb{E}_{T|X_j, T>j}\left[\min(T-j, L_{\max}) - C_\alpha \mathbb{1}\{T-j < v\} \mid H_j\right]\right\}
```

**Code's approximation:**
```latex
V'_j(H_j) \approx \text{percentile}_{p(C_\alpha)}(\{T_j^{(i)}\}_{i=1}^N)
```

**Why this is an issue:**
1. The exact formula requires computing expectation over the learned hazard distribution
2. The threshold should depend on $H_j$ (hazard history), not just population percentiles
3. Different samples should potentially have different thresholds

**Impact:**
- OTI metrics (miss rate, avg TTE) will be approximate
- Results won't exactly match paper's theoretical optimal policy
- But still provides a reasonable intervention strategy

---

## Correct Implementation of OTI Threshold

To properly implement Equation 9, you would need:

```python
def compute_oti_threshold_exact(hazard_logits, cost):
    """
    Exact implementation of Equation 9 from the paper

    V'_j(H_j) = inf{v : v >= E[min(T-j, L_max) - C_α * 1{T-j < v} | H_j]}
    """
    hazard_rates, survival_probs = compute_hazard_from_logits(hazard_logits)
    batch_size, pred_horizon = hazard_logits.shape

    # For each possible threshold value v
    thresholds = np.arange(0, pred_horizon + 1)

    optimal_threshold = np.zeros(batch_size)

    for batch_idx in range(batch_size):
        best_v = pred_horizon

        for v in thresholds:
            # Compute expectation: E[min(T-j, L_max) - C_α * 1{T-j < v}]
            expected_value = 0

            for k in range(pred_horizon):
                # Probability of event at time k
                if k == 0:
                    prob_event_at_k = hazard_rates[batch_idx, k]
                else:
                    prob_event_at_k = (hazard_rates[batch_idx, k] *
                                      survival_probs[batch_idx, k-1])

                # Contribution to expectation
                event_time = min(k, pred_horizon)
                penalty = cost if k < v else 0
                expected_value += prob_event_at_k * (event_time - penalty)

            # Add contribution from no event (survival past L_max)
            prob_no_event = survival_probs[batch_idx, -1]
            expected_value += prob_no_event * pred_horizon

            # Check if this v satisfies the condition
            if v >= expected_value:
                best_v = v
                break

        optimal_threshold[batch_idx] = best_v

    return optimal_threshold
```

### LaTeX for Exact Implementation

```latex
V'_j(H_j) &= \inf\left\{v : v \geq \mathbb{E}[f(T, v)]\right\} \\
\text{where } f(T, v) &= \min(T-j, L_{\max}) - C_\alpha \mathbb{1}\{T-j < v\} \\
\\
\mathbb{E}[f(T, v)] &= \sum_{k=0}^{L_{\max}-1} P(T=j+k) \cdot [k - C_\alpha \mathbb{1}\{k < v\}] \\
&\quad + P(T > j + L_{\max}) \cdot L_{\max} \\
\\
P(T=j+k) &= h_j(k) \prod_{m=0}^{k-1}(1-h_j(m)) = h_j(k) \cdot S_j(k-1)
```

---

## Summary Table

| Component | Paper Formula | Code Implementation | Status |
|-----------|---------------|---------------------|--------|
| Hazard rates | $h_j(k) = \sigma(\text{logit})$ | `torch.sigmoid(logits)` | ✅ CORRECT |
| Survival function | $S_j(k) = \prod_{m=0}^k (1-h_j(m))$ | `exp(cumsum(log(1-h)))` | ✅ CORRECT |
| Expected TTE | $T_j = \sum_{k=0}^{L_{\max}} S_j(k)$ | `survival_with_zero.sum()` | ✅ CORRECT |
| Loss $\ell_z$ | $h_j(\ell) \cdot S_j(\ell-1)$ | `hazard_at_event * survival_before_event` | ✅ CORRECT |
| Loss $\ell_u$ | $1 - S_j(L_{\max}-1)$ | `1 - final_survival` | ✅ CORRECT |
| Loss $\ell_c$ | $S_j(L_{\max}-1)$ | `final_survival` | ✅ CORRECT |
| Total loss | $-\alpha \log \ell_z - (1-\alpha)[(1-c)\log \ell_u + c \log \ell_c]$ | Code implements correctly for each case | ✅ MOSTLY CORRECT* |
| OTI policy | $\phi^*_j = \mathbb{1}\{T_j \leq V'_j(H_j)\}$ | `expected_tte <= threshold` | ✅ CORRECT |
| OTI threshold | Equation 9 (exact formula) | Percentile-based approximation | ⚠️ APPROXIMATION |

\* Minor difference for censored samples, but likely correct interpretation

---

## Recommendations

### 1. Loss Function
**Current implementation is acceptable** but could be clarified:
- For uncensored samples: matches paper exactly
- For censored samples: only uses $\ell_c$, which is correct since $\ell_z$ is undefined

### 2. OTI Threshold
**Should implement exact formula** for publication-quality results:
- Current percentile approach is a heuristic
- Exact implementation would compute continuation value per sample
- See "Correct Implementation of OTI Threshold" section above

### 3. Priority
- **High**: Keep current loss function (it's correct)
- **Medium**: Implement exact OTI threshold if you need exact paper reproduction
- **Low**: Everything else is correctly implemented

The main formulas (hazard, survival, TTE, loss components) are all **correctly implemented**!
