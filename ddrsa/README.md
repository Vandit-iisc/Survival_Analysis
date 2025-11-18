# DDRSA: Dynamic Deep Recurrent Survival Analysis

Implementation of the DDRSA architecture from the NeurIPS 2022 paper:
**"When to Intervene: Learning Optimal Intervention Policies for Critical Events"**
by Niranjan Damera Venkata and Chiranjib Bhattacharyya

## Overview

This implementation reproduces the experiments from the paper on the NASA Turbofan Engine Degradation dataset (PHM08 Challenge). It includes:

- **DDRSA-RNN**: Encoder-decoder architecture with LSTM/GRU
- **DDRSA-Transformer**: Encoder-decoder architecture with Transformers
- **DDRSA Loss**: Exact implementation of Equation 12 from the paper
- **OTI Policy**: Optimal Timed Intervention policy (Corollary 4.1.1)
- **Evaluation Metrics**: NASA scoring, C-index, Integrated Brier Score

## Architecture

### DDRSA-RNN (Figure 1 in paper)

```
Input Sequence (X_j-K ... X_j)
    ↓
[Encoder RNN] → Hidden State (Z_j)
    ↓
[Decoder RNN] → Hazard Rates (h_j(0), h_j(1), ..., h_j(L_max-1))
    ↓
Output Layer
```

### Key Components

1. **Encoder**: Maps covariate history to hidden representation
2. **Decoder**: Generates conditional hazard rates for future time steps
3. **Loss Function**: Three-component loss (l_z, l_u, l_c) from Equation 12

## Dataset

**NASA Turbofan Engine Degradation Dataset (PHM08)**

- **Training engines**: 218
- **Test engines**: 218
- **Features**: 3 operational settings + 21 sensor measurements
- **Task**: Predict Remaining Useful Life (RUL) before engine failure

## Installation

```bash
# Create conda environment
conda create -n ddrsa python=3.9
conda activate ddrsa

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn tqdm tensorboard
```

## Quick Start

### 1. Train DDRSA-RNN (LSTM)

```bash
python main.py \
    --model-type rnn \
    --rnn-type LSTM \
    --hidden-dim 16 \
    --num-layers 1 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --lambda-param 0.5 \
    --lookback-window 128 \
    --pred-horizon 100 \
    --num-epochs 100 \
    --exp-name ddrsa_lstm
```

### 2. Train DDRSA-RNN (GRU)

```bash
python main.py \
    --model-type rnn \
    --rnn-type GRU \
    --hidden-dim 16 \
    --num-layers 1 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --lambda-param 0.5 \
    --lookback-window 128 \
    --pred-horizon 100 \
    --num-epochs 100 \
    --exp-name ddrsa_gru
```

### 3. Train DDRSA-Transformer

```bash
python main.py \
    --model-type transformer \
    --d-model 64 \
    --nhead 4 \
    --num-layers 2 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --lambda-param 0.5 \
    --lookback-window 128 \
    --pred-horizon 100 \
    --num-epochs 100 \
    --exp-name ddrsa_transformer
```

## File Structure

```
ddrsa/
├── data_loader.py      # NASA Turbofan dataset loader
├── models.py           # DDRSA-RNN and DDRSA-Transformer architectures
├── loss.py             # DDRSA loss function (Equation 12)
├── metrics.py          # Evaluation metrics and OTI policy
├── trainer.py          # Training loop and model checkpointing
├── main.py             # Main experiment runner
├── run_all.sh          # Script to run all experiments
└── README.md           # This file
```

## Hyperparameters (from Paper)

### Default Configuration (Section 6.2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Lookback Window (K) | 128 | Number of past time steps |
| Prediction Horizon (L_max) | 100 | Maximum prediction horizon |
| Encoder Hidden Dim | 16 | LSTM hidden units |
| Decoder Hidden Dim | 16 | LSTM hidden units |
| Batch Size | 32 | Training batch size |
| Learning Rate | 1e-4 | Adam optimizer learning rate |
| Lambda (λ) | 0.5 | Trade-off in loss function |
| Dropout | 0.1 | Dropout rate |

## Loss Function (Equation 12)

```
L_f = -λ log l_z - (1-λ)[(1-c) log l_u + c log l_c]

where:
- l_z: likelihood of event at specific time (uncensored)
- l_u: event rate (probability of event occurring)
- l_c: survival likelihood (censored samples)
- λ: trade-off parameter
- c: censoring indicator
```

## OTI Policy (Corollary 4.1.1)

The optimal intervention policy triggers intervention when:

```
T_j ≤ V'_j(H_j)

where:
- T_j: expected time-to-event at time j
- V'_j(H_j): continuation value function (threshold)
```

## Evaluation Metrics

1. **NASA Score**: Asymmetric scoring function from PHM08 challenge
2. **RMSE**: Root Mean Squared Error on RUL predictions
3. **MAE**: Mean Absolute Error on RUL predictions
4. **C-index**: Concordance index for survival analysis
5. **Integrated Brier Score**: Survival probability calibration
6. **OTI Metrics**: Miss rate and average TTE for different costs

## Results

Results will be saved in the `logs/` directory:

```
logs/
├── ddrsa_experiment/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── checkpoint_epoch_*.pt
│   ├── tensorboard/
│   ├── config.json
│   └── test_metrics.json
```

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir logs/
```

## Creating Visualizations (Section 6.1 Figures)

After training, reproduce the exact figures from the paper:

```bash
# Install matplotlib if needed
pip install matplotlib

# Create all figures (Figure 2a, 2b, 3, etc.)
python create_figures.py --exp-name your_experiment_name
```

This creates:
- **Figure 2(a)**: Progression of conditional hazard rates
- **Figure 2(b)**: OTI policy in action with thresholds
- **Figure 3**: Policy trade-off curves
- **Training curves**: Loss over epochs

See `VISUALIZATION_GUIDE.md` for detailed instructions.

## Testing Pre-trained Model

```python
import torch
from models import create_ddrsa_model
from data_loader import get_dataloaders

# Load checkpoint
checkpoint = torch.load('logs/ddrsa_experiment/checkpoints/best_model.pt')

# Create model
model = create_ddrsa_model('rnn', input_dim=24, encoder_hidden_dim=16)
model.load_state_dict(checkpoint['model_state_dict'])

# Load test data
_, _, test_loader, _ = get_dataloaders(data_path='Challenge_Data')

# Evaluate
model.eval()
# ... evaluation code
```

## Paper Citation

```bibtex
@inproceedings{venkata2022ddrsa,
  title={When to Intervene: Learning Optimal Intervention Policies for Critical Events},
  author={Damera Venkata, Niranjan and Bhattacharyya, Chiranjib},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Dataset Citation

```bibtex
@inproceedings{saxena2008damage,
  title={Damage propagation modeling for aircraft engine run-to-failure simulation},
  author={Saxena, Abhinav and Goebel, Kai and Simon, Don and Eklund, Neil},
  booktitle={International Conference on Prognostics and Health Management},
  year={2008}
}
```

## License

This implementation is for research purposes. Please cite the original paper if you use this code.

## Contact

For questions or issues, please open an issue on GitHub.
