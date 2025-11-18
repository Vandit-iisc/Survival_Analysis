# DDRSA Implementation Notes

## Overview

This is a complete, faithful implementation of the DDRSA (Dynamic Deep Recurrent Survival Analysis) architecture from the NeurIPS 2022 paper "When to Intervene: Learning Optimal Intervention Policies for Critical Events".

## Implementation Fidelity to Paper

### 1. Architecture (Figure 1, Section 5.1)

✅ **Implemented Exactly as Described**

#### DDRSA-RNN (`models.py:17-95`)
- **Encoder RNN**: Maps covariate history X_{j-K}...X_j to hidden state Z_j
- **Decoder RNN**: Takes Z_j and generates hazard rates h_j(k) for k=0...L_max-1
- **Architecture Details**:
  - Encoder: LSTM/GRU with configurable hidden dimension
  - Decoder: LSTM/GRU taking replicated encoder output
  - Output Layer: Linear layer mapping to hazard logits

#### DDRSA-Transformer (`models.py:98-190`)
- **Encoder**: Transformer encoder with positional encoding
- **Decoder**: Transformer decoder with learnable query embeddings
- **Same seq2seq philosophy** as RNN variant

### 2. Loss Function (Equation 12, Section 5.1)

✅ **Implemented Exactly** (`loss.py:14-127`)

```python
L_f = -λ log l_z - (1-λ)[(1-c) log l_u + c log l_c]
```

**Three Components**:
1. **l_z**: Likelihood of event at specific time (uncensored samples)
   - `l_z = ∏_{k=0}^{l-1} (1 - h(k)) * h(l)`

2. **l_u**: Event rate (uncensored samples)
   - `l_u = 1 - ∏_{k=0}^{L_max-1} (1 - h(k))`

3. **l_c**: Survival likelihood (censored samples)
   - `l_c = ∏_{k=0}^{L_max-1} (1 - h(k))`

### 3. OTI Policy (Corollary 4.1.1, Section 4)

✅ **Implemented** (`metrics.py:138-228`)

**Optimal Policy**: Intervene when `T_j ≤ V'_j(H_j)`

Where:
- `T_j`: Expected time-to-event (Equation 10)
- `V'_j(H_j)`: Continuation value function (Equation 9)

**Expected TTE Computation**:
```python
T_j = Σ_{k=0}^{L_max-1} ∏_{m=0}^{k} (1 - h_j(m))
```

### 4. Training Procedure (Section 6.2)

✅ **Implemented with Paper's Hyperparameters** (`trainer.py`)

| Hyperparameter | Paper Value | Implementation |
|----------------|-------------|----------------|
| Lookback Window | 128 steps | ✅ Default: 128 |
| Hidden Dimension | 16 units | ✅ Default: 16 |
| Batch Size | 32 | ✅ Default: 32 |
| Learning Rate | 1e-4 | ✅ Default: 1e-4 |
| Lambda (α) | 0.5 | ✅ Default: 0.5 |
| Optimizer | Adam | ✅ Adam with weight decay |

### 5. Evaluation Metrics (Section 6.2)

✅ **Implemented** (`metrics.py`)

1. **NASA Scoring Function** (asymmetric RUL error)
   - Early predictions penalized less than late predictions
   - Uses a1=13, a2=10 as in PHM08 challenge

2. **RMSE and MAE** on RUL predictions

3. **C-index** (Concordance Index) for survival analysis

4. **Integrated Brier Score** for survival probability calibration

5. **OTI Metrics**: Miss rate and average TTE for different costs

## Dataset Preprocessing

### NASA Turbofan Dataset (`data_loader.py`)

✅ **Proper Preprocessing Pipeline**

1. **Load data**: 26 columns (unit_id, time, 3 settings, 21 sensors)
2. **Calculate RUL**: Max cycles - current cycle for each engine
3. **Normalize**: StandardScaler on settings and sensors
4. **Create sequences**:
   - Lookback window: Past K time steps
   - Targets: Hazard labels for next L_max time steps
5. **Handle censoring**: Mark censored vs uncensored samples

## Key Design Decisions

### 1. Survival Analysis Formulation

Unlike traditional RUL regression, we use **survival analysis**:
- Model hazard rates at each future time step
- Handle censored data properly
- Enable OTI policy computation

### 2. Encoder-Decoder Architecture

**Why seq2seq?**
- Encoder: Summarize variable-length history into fixed representation
- Decoder: Generate hazard rates for all future time steps in parallel
- Allows dynamic horizon prediction

### 3. Loss Function Design

**Three-component loss balances**:
- Event timing (l_z): When exactly did it happen?
- Event occurrence (l_u): Did it happen at all?
- Censoring (l_c): What if we don't observe the event?

## Differences from Paper (Improvements)

### 1. Data Handling
- **Paper**: Used MIMIC-III and HiRiD medical datasets
- **Our Implementation**: Adapted for NASA Turbofan (run-to-failure)
- **Improvement**: More thorough data normalization and handling

### 2. Model Flexibility
- **Paper**: Primarily LSTM
- **Our Implementation**: Both LSTM, GRU, and Transformer
- **Improvement**: Easy to compare architectures

### 3. Evaluation
- **Paper**: Focus on medical intervention metrics
- **Our Implementation**: NASA scoring + survival metrics
- **Improvement**: Domain-appropriate evaluation

### 4. Training Infrastructure
- **Our Implementation**: TensorBoard logging, checkpointing, early stopping
- **Improvement**: Better experiment tracking and reproducibility

## Code Organization

```
ddrsa/
├── data_loader.py       # Dataset loading and preprocessing
│   ├── TurbofanDataLoader   # Main data loading class
│   ├── TurbofanDataset       # PyTorch Dataset
│   └── get_dataloaders       # Convenience function
│
├── models.py            # Model architectures
│   ├── DDRSA_RNN            # RNN-based encoder-decoder
│   ├── DDRSA_Transformer    # Transformer-based encoder-decoder
│   └── create_ddrsa_model   # Factory function
│
├── loss.py              # DDRSA loss function
│   ├── DDRSALoss            # Basic loss (Equation 12)
│   ├── DDRSALossDetailed    # Loss with component logging
│   └── compute_expected_tte  # TTE computation (Equation 10)
│
├── metrics.py           # Evaluation metrics
│   ├── evaluate_model            # Comprehensive evaluation
│   ├── compute_oti_metrics       # OTI policy metrics
│   ├── compute_nasa_score        # PHM08 scoring
│   └── compute_concordance_index # C-index
│
├── trainer.py           # Training loop
│   ├── DDRSATrainer         # Main trainer class
│   └── get_default_config   # Paper's hyperparameters
│
└── main.py              # Experiment runner
```

## Testing and Validation

### Unit Tests (`test_installation.py`)

1. ✅ Model creation (RNN and Transformer)
2. ✅ Forward pass correctness
3. ✅ Loss computation
4. ✅ TTE computation
5. ✅ Data loading
6. ✅ Metrics computation

### Integration Tests

Run a quick experiment to verify end-to-end:

```bash
python main.py --num-epochs 5 --batch-size 8
```

## Performance Considerations

### Memory Optimization
- Gradient clipping prevents explosion
- Batch padding removed dynamically
- Checkpoint saving prevents memory leaks

### Speed Optimization
- Multi-worker data loading (configurable)
- GPU support with automatic detection
- Efficient tensor operations

### Numerical Stability
- Log-space computations for survival probabilities
- Clamping for hazard rates (avoid log(0))
- Gradient clipping

## Reproducibility

✅ **Full reproducibility ensured**:

1. **Random seed setting** in all components
2. **Deterministic algorithms** when possible
3. **Config saving** for every experiment
4. **Checkpoint versioning**

## Future Extensions

### Possible Improvements

1. **Multi-task Learning**: Predict multiple failure modes
2. **Attention Visualization**: Interpret what the model focuses on
3. **Uncertainty Quantification**: Bayesian neural networks
4. **Online Learning**: Update model with streaming data
5. **Transfer Learning**: Pre-train on one dataset, fine-tune on another

## Citation

If you use this implementation, please cite both the original paper and this implementation:

```bibtex
@inproceedings{venkata2022ddrsa,
  title={When to Intervene: Learning Optimal Intervention Policies for Critical Events},
  author={Damera Venkata, Niranjan and Bhattacharyya, Chiranjib},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Acknowledgments

This implementation is based on the theoretical framework developed by Niranjan Damera Venkata and Chiranjib Bhattacharyya. The code is written for educational and research purposes.
