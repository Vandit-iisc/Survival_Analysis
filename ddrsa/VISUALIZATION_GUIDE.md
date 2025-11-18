# Visualization Guide - Reproducing Paper Figures

This guide shows how to reproduce the exact figures from Section 6 of the NeurIPS 2022 paper.

## Installation

First, install matplotlib if you haven't already:

```bash
pip install matplotlib
```

## Quick Start - Create All Figures

After training your model, run:

```bash
python create_figures.py --exp-name quick_test
```

This will generate all figures in `figures/quick_test/`:
- `figure_2a_hazard_progression.png` - Section 6.1, Figure 2(a)
- `figure_2b_oti_policy.png` - Section 6.1, Figure 2(b)
- `training_curves.png` - Training/validation loss

## Figures from the Paper

### Figure 2(a): Progression of Conditional Hazard Rates

**What it shows**: How the predicted hazard rates evolve as we get closer to the critical event.

**From paper**: "Figure 2(a) shows the evolution of estimated conditional hazard rate functions. We see that the onset of the critical event is clearly reflected in the progression of the estimated vectors H_j."

**To create**:
```python
from visualization import plot_hazard_progression
from data_loader import get_dataloaders
import torch

# Load model and data
model = ...  # Load your trained model
_, _, test_loader, _ = get_dataloaders(data_path='Challenge_Data')

# Create figure
plot_hazard_progression(
    model,
    test_loader,
    num_samples=5,
    save_path='figures/hazard_progression.png'
)
```

**What you'll see**: Multiple curves showing hazard rates at different time points (j=L-200, j=L-100, etc.)

### Figure 2(b): OTI Policy in Action

**What it shows**: The OTI policy threshold compared to expected time-to-event for different cost values.

**From paper**: "Figure 2(b) plots the estimated time-to-event at each time step... As one attempts to intervene closer to the event, the probability of missing the event goes up."

**To create**:
```python
from visualization import plot_oti_policy

plot_oti_policy(
    model,
    test_loader,
    cost_values=[8, 64, 128],
    save_path='figures/oti_policy.png'
)
```

**What you'll see**:
- Blue line: Expected time-to-event (T_j)
- Dashed lines: OTI thresholds for different costs
- Intervention triggered when blue line crosses below threshold

### Figure 3: Policy Trade-off Plots

**What it shows**: Comparison of OTI policy vs. baselines (miss rate vs. average time-to-event)

**From paper**: "Figure 3 summarizes the evaluation results in the form of policy trade-off curves."

**To create**: This is automatically generated from test metrics after running experiments:

```python
from visualization import plot_policy_tradeoff
import json

# Load test metrics from different experiments
with open('logs/oti_experiment/test_metrics.json') as f:
    oti_metrics = json.load(f)

plot_policy_tradeoff(
    test_metrics_list=[oti_metrics],
    labels=['OTI-DDRSA-RNN'],
    save_path='figures/tradeoff.png'
)
```

## Complete Workflow

### 1. Train Model

```bash
python main.py \
    --model-type rnn \
    --num-epochs 100 \
    --exp-name paper_reproduction
```

### 2. Create All Figures

```bash
python create_figures.py --exp-name paper_reproduction
```

Output:
```
figures/paper_reproduction/
├── figure_2a_hazard_progression.png
├── figure_2b_oti_policy.png
├── training_curves.png
└── (more figures)
```

### 3. View Results

```bash
open figures/paper_reproduction/
```

## Advanced Usage

### Custom Visualization

```python
import torch
from models import create_ddrsa_model
from data_loader import get_dataloaders
from visualization import (
    plot_hazard_progression,
    plot_oti_policy,
    create_all_visualizations
)

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = create_ddrsa_model('rnn', input_dim=24, encoder_hidden_dim=16)
checkpoint = torch.load('logs/my_exp/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load data
_, _, test_loader, _ = get_dataloaders(data_path='Challenge_Data')

# Create specific figure
plot_hazard_progression(
    model=model,
    data_loader=test_loader,
    device=device,
    num_samples=10,  # More samples
    save_path='custom_hazard.png'
)
```

### Compare Multiple Models

```python
# Create comparison plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot for RNN model
plot_hazard_progression(model_rnn, test_loader, save_path='rnn_hazards.png')

# Plot for Transformer model
plot_hazard_progression(model_transformer, test_loader, save_path='transformer_hazards.png')
```

### Export for Publication

High-quality figures for papers:

```python
plot_hazard_progression(
    model,
    test_loader,
    save_path='paper_figure.png'
)

# Or save as PDF for LaTeX
import matplotlib.pyplot as plt
plt.savefig('paper_figure.pdf', dpi=300, bbox_inches='tight', format='pdf')
```

## Interpreting the Figures

### Figure 2(a): Hazard Rate Progression

**What to look for:**
1. **Early time points** (j=L-200): Flat, low hazard rates → system is healthy
2. **Approaching event** (j=L-50, j=L-10): Rising hazard rates → degradation detected
3. **Near event** (j=L-1): Sharp peak → critical event imminent

**Good model**: Clear progression from flat to peaked distributions

### Figure 2(b): OTI Policy

**What to look for:**
1. **Blue line declining**: Expected TTE decreases as we approach event
2. **Threshold crossings**: Intervention triggered at different times for different costs
3. **Higher cost → Earlier intervention**: Red threshold (C=128) > Orange (C=64) > Green (C=8)

**Trade-off**:
- Intervene too early: Waste resources
- Intervene too late: Miss the event

### Figure 3: Trade-off Curves

**What to look for:**
1. **Lower is better**: Lower miss rate AND lower avg TTE
2. **OTI should dominate**: Curve should be below baselines
3. **Pareto frontier**: Different cost values trace the optimal trade-off

## Troubleshooting

### "No module named 'matplotlib'"
```bash
pip install matplotlib
```

### "FileNotFoundError: best_model.pt"
Make sure your model finished training:
```bash
ls logs/your_exp_name/checkpoints/
```

### Figures look different from paper
This is normal! Reasons:
- Different dataset (NASA Turbofan vs. MIMIC-III)
- Different random initialization
- Different hyperparameters

The **structure** should be similar (progression, thresholds, etc.)

### Memory error when creating figures
Reduce batch size or number of samples:
```python
plot_hazard_progression(model, test_loader, num_samples=3)  # Fewer samples
```

## Figure Quality Settings

### For presentations (screen):
```python
plt.savefig('figure.png', dpi=150)
```

### For papers (print):
```python
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

### For posters (large format):
```python
plt.savefig('figure.png', dpi=600, bbox_inches='tight')
```

## Summary

✅ **Figure 2(a)**: `plot_hazard_progression()` - Shows hazard rate evolution
✅ **Figure 2(b)**: `plot_oti_policy()` - Shows OTI policy thresholds
✅ **Figure 3**: `plot_policy_tradeoff()` - Shows policy comparison
✅ **Training curves**: Automatically from TensorBoard logs

All figures can be created with one command:
```bash
python create_figures.py --exp-name your_experiment
```

Happy visualizing! 📊
