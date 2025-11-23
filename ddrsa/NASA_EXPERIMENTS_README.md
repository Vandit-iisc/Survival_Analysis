# Hybrid DDRSA+NASA Loss Function Experiments

## Overview

This directory contains a clean experimental setup for training survival analysis models using **hybrid loss: DDRSA + NASA** scoring function.

### What's Different

- **Hybrid Loss**: Combines DDRSA loss (l_z, l_u, l_c) + NASA asymmetric scoring
- **4 Model Architectures**: Tests LSTM, GRU, Transformer, and ProbSparse Attention
- **Configurable Weights**: lambda_param for DDRSA, nasa_weight for NASA component
- **Clean Setup**: No hyperparameter tuning, just direct comparison of architectures
- **Turbofan Dataset**: All experiments run on NASA's C-MAPSS Turbofan dataset

## Model Architectures Tested

### 1. **LSTM** (Exact Paper Implementation)
- Encoder: 1-layer LSTM with 16 hidden units, 128 lookback window
- Decoder: 1-layer LSTM with 16 hidden units
- From: "When to Intervene" paper (NeurIPS 2022)

### 2. **GRU/RNN**
- Same architecture as LSTM but using GRU cells
- Encoder: 1-layer GRU with 16 hidden units
- Decoder: 1-layer GRU with 16 hidden units

### 3. **Transformer**
- Encoder-Decoder Transformer architecture
- 64-dimensional embeddings, 4 attention heads
- 2 encoder layers, 2 decoder layers
- 256-dimensional feedforward network

### 4. **ProbSparse Attention** (Informer)
- ProbSparse self-attention mechanism (O(L log L) complexity)
- 128-dimensional embeddings, 4 attention heads
- 2 encoder layers with distilling
- LSTM decoder with 128 hidden units
- From: "Informer" paper (AAAI 2021)

## Hybrid Loss Function

The hybrid loss combines two components:

### 1. DDRSA Loss (from paper)
```
L_DDRSA = λ * loss_z + (1-λ) * (loss_u + loss_c)
```
Where:
- `loss_z`: Likelihood of event at specific time (uncensored)
- `loss_u`: Event rate (probability of event occurring)
- `loss_c`: Survival likelihood (censored samples)
- `λ` (lambda_param): Trade-off parameter (default 0.5)

### 2. NASA Scoring Function
```
For each prediction:
- Early (predicted > true): score = exp(-error/13) - 1
- Late (predicted ≤ true):  score = exp(error/10) - 1
```

### Combined Loss
```
Total Loss = L_DDRSA + nasa_weight * L_NASA
```

**Default**: `lambda_param=0.5`, `nasa_weight=0.1`

The NASA component adds direct optimization for the final evaluation metric while DDRSA handles censoring and temporal structure.

## Files Created

### Core Implementation
- `loss.py` - Contains `DDRSALossDetailedWithNASA` (hybrid loss)
- `trainer.py` - Updated trainer with hybrid loss support
- `run_nasa_experiments.py` - Main experiment runner with automatic visualization
- `metrics.py` - Updated with NASA evaluation functions
- `visualization.py` - Plotting functions for analysis
- `compare_experiments.py` - Generate comparison plots across experiments

### Shell Scripts
- `run_nasa_quick_test.sh` - Quick 5-epoch test of all 4 models (~10-15 min)
- `run_nasa_all.sh` - Full 100-epoch training of all 4 models (~2-4 hours)

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm tensorboard
```

### 2. Quick Test (5 epochs)

```bash
# Test all 4 models with 5 epochs each
bash run_nasa_quick_test.sh
```

This will:
- Train all 4 model types for 5 epochs each
- Save results to `nasa_quick_test_summary.json`
- Create logs in `logs_nasa/`
- **Automatically generate visualizations** for each model
- Take about 10-15 minutes

### 3. Full Training (100 epochs)

```bash
# Full training for all 4 models
bash run_nasa_all.sh
```

This will:
- Train all 4 model types for 100 epochs each
- Save results to `nasa_full_experiments_summary.json`
- Create logs in `logs_nasa/`
- **Automatically generate visualizations** for each model
- **Generate comparison plots** across all models
- Take about 2-4 hours (depending on hardware)

### 4. Custom Experiments

```bash
# Train only LSTM and Transformer
python run_nasa_experiments.py \
    --experiments lstm transformer \
    --num-epochs 50 \
    --data-path ../Challenge_Data

# Train without generating plots (faster)
python run_nasa_experiments.py \
    --experiments all \
    --num-epochs 100 \
    --no-plots

# Generate comparison plots from existing results
python compare_experiments.py \
    --summary-file nasa_full_experiments_summary.json \
    --output-dir comparison_plots
```

## Output Structure

```
logs_nasa/
├── hybrid_lstm_20231123_140500/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── checkpoint_epoch_*.pt
│   ├── tensorboard/
│   ├── figures/                    # 🆕 Auto-generated visualizations
│   │   ├── figure_2a_hazard_progression_test.png
│   │   ├── figure_2b_oti_policy_test.png
│   │   └── training_curves.png
│   ├── config.json
│   └── test_metrics.json
├── hybrid_gru_20231123_142000/
│   └── figures/                    # Each model gets its own figures
├── hybrid_transformer_20231123_144000/
│   └── figures/
└── hybrid_probsparse_20231123_150000/
    └── figures/

comparison_plots/                   # 🆕 Cross-model comparisons
├── summary_table.png
├── nasa_score_comparison.png
├── mse_comparison.png
├── mae_comparison.png
└── rmse_comparison.png

nasa_full_experiments_summary.json  # Comparison of all experiments
```

## Monitoring Training & Results

### TensorBoard

```bash
# View training curves for all experiments
tensorboard --logdir logs_nasa/

# Open browser to http://localhost:6006
```

### Check Results

```bash
# View summary of all experiments
cat nasa_full_experiments_summary.json

# View specific experiment results
cat logs_nasa/hybrid_lstm_*/test_metrics.json
```

### Visualizations

**Automatic Generation**: Each experiment automatically generates:
- **Hazard Progression Plot** - Shows how hazard rates evolve over time
- **OTI Policy Plot** - Optimal intervention timing for different costs
- **Training Curves** - Loss and metrics during training

**Comparison Plots**: After all experiments complete:
```bash
# Generate cross-model comparison plots
python compare_experiments.py \
    --summary-file nasa_full_experiments_summary.json \
    --output-dir comparison_plots
```

This creates:
- Summary table with all metrics
- Bar charts comparing NASA score, MSE, MAE, RMSE
- Easy identification of best model

**View Figures**:
```bash
# Individual model visualizations
open logs_nasa/hybrid_lstm_*/figures/*.png

# Comparison plots
open comparison_plots/*.png
```

## Evaluation Metrics

Each experiment reports:

- **NASA Score**: Primary metric (lower is better)
- **MSE**: Mean squared error of TTE predictions
- **MAE**: Mean absolute error
- **RMSE**: Root mean squared error
- **Mean Error**: Average prediction error
- **Std Error**: Standard deviation of errors
- **Mean Predicted TTE**: Average predicted time-to-event
- **Mean True TTE**: Average actual time-to-event

## Expected Results

Based on the paper and literature:

| Model | Expected NASA Score | Expected RMSE |
|-------|-------------------|---------------|
| LSTM | ~1000-1500 | ~15-20 |
| GRU | ~1000-1500 | ~15-20 |
| Transformer | ~1200-1700 | ~16-22 |
| ProbSparse | ~1100-1600 | ~15-21 |

*Note: These are rough estimates. Actual results will vary.*

## Configuration Details

### Common Hyperparameters (All Models)
- Lookback window: 128 time steps
- Prediction horizon: 100 time steps
- Batch size: 32
- Learning rate: 1e-4 with ReduceLROnPlateau
- Weight decay: 1e-6
- Gradient clipping: 1.0
- Early stopping patience: 20 epochs
- Validation split: 20%

### Hybrid Loss Parameters
- **lambda_param**: 0.5 (DDRSA trade-off: l_z vs l_u+l_c)
- **nasa_weight**: 0.1 (weight for NASA component)
- Early prediction penalty: a1 = 13
- Late prediction penalty: a2 = 10

## Comparison with Original DDRSA

This setup uses **hybrid DDRSA+NASA loss**, which extends the original DDRSA:

| Feature | Original DDRSA | Hybrid (This) |
|---------|----------------|---------------|
| Loss Components | 3 (l_z, l_u, l_c) | 4 (l_z, l_u, l_c, NASA) |
| Censoring Handling | Explicit | Explicit (via DDRSA) |
| NASA Score | Not optimized | Directly optimized |
| Optimization | Multi-objective | Multi-objective + metric |
| Training Stability | High | High |

**Advantage**: The hybrid loss directly optimizes the NASA evaluation metric while maintaining proper censoring handling from DDRSA.

## Transfer to RunPod

All paths are relative (`../Challenge_Data`), so the code is ready to transfer:

```bash
# From local machine
scp -r -i ~/.ssh/id_ed25519 ./ddrsa user@runpod:/workspace/survival_analysis/
scp -r -i ~/.ssh/id_ed25519 ./Challenge_Data user@runpod:/workspace/survival_analysis/
```

On RunPod:
```bash
cd /workspace/survival_analysis/ddrsa
bash run_nasa_quick_test.sh  # Test first
bash run_nasa_all.sh          # Then full training
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `run_nasa_experiments.py` (line 30)
- Reduce model dimensions for Transformer/ProbSparse

### Training Instability
- Increase MSE weight in `nasa_loss.py` (default 0.1)
- Reduce learning rate
- Add more warmup steps

### Poor Performance
- Check data loading: `python -c "from data_loader import get_dataloaders; print('OK')"`
- Verify NASA score calculation: `python nasa_loss.py`
- Ensure sufficient training epochs (100+)

## Citation

If using this code, please cite:

```bibtex
@inproceedings{kiyasseh2022intervene,
  title={When to Intervene: Learning Optimal Intervention Policies for Critical Events},
  author={Kiyasseh, Dani and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Questions?

Check the main README.md or open an issue in the repository.
