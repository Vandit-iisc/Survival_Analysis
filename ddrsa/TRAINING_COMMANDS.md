# DDRSA Training Commands Reference

Complete reference for all training commands with different model configurations.

---

## Table of Contents
1. [Dataset Selection](#dataset-selection)
2. [Paper-Exact Configurations](#paper-exact-configurations)
3. [Transformer Models](#transformer-models)
4. [LSTM Models](#lstm-models)
5. [GRU Models](#gru-models)
6. [Command Line Arguments](#command-line-arguments)
7. [Output Directory Structure](#output-directory-structure)

---

## Dataset Selection

All commands support both datasets. Simply add the `--dataset` and `--data-path` arguments:

### NASA Turbofan (Default)
```bash
--dataset turbofan --data-path ../Challenge_Data
```

### Azure Predictive Maintenance
```bash
--dataset azure_pm --data-path ../AMLWorkshop/Data
```

**Note:** All model configurations below work with both datasets. Just add the appropriate dataset arguments.

---

## Paper-Exact Configurations

These configurations match the exact hyperparameters from Appendix B of the NIPS 2022 paper.

### Paper-Exact RNN (NASA Turbofan)

Based on Appendix B: hidden_dim=16, batch_size=512, lr=0.01, pred_horizon=64, lambda=0.75, patience=10

```bash
python main.py \
    --dataset turbofan \
    --data-path ../Challenge_Data \
    --model-type rnn \
    --rnn-type LSTM \
    --num-epochs 200 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --hidden-dim 16 \
    --num-layers 1 \
    --batch-size 512 \
    --learning-rate 0.01 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --lookback-window 128 \
    --output-dir paper_exact \
    --exp-name turbofan_paper_exact \
    --create-visualization
```

### Paper-Exact RNN (Azure PM)

Same hyperparameters applied to Azure PM dataset:

```bash
python main.py \
    --dataset azure_pm \
    --data-path ../AMLWorkshop/Data \
    --model-type rnn \
    --rnn-type LSTM \
    --num-epochs 200 \
    --patience 10 \
    --use-minmax \
    --hidden-dim 16 \
    --num-layers 1 \
    --batch-size 512 \
    --learning-rate 0.01 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --lookback-window 128 \
    --output-dir paper_exact \
    --exp-name azure_pm_paper_exact \
    --create-visualization
```

### Important: Hazard Rate Magnitude

If your hazard rates are in the range of 0-0.1 instead of the paper's 0-0.6, check these key parameters:

| Parameter | Default (Now Paper-Exact) | Impact on Hazard Rates |
|-----------|---------------------------|------------------------|
| `pred_horizon` | **64** | Shorter horizon → higher per-step hazard rates |
| `lambda_param` | **0.75** | Weights l_z more (precise event time prediction) |
| `batch_size` | **512** | Better gradient estimates |
| `learning_rate` (RNN) | **0.01** | Faster convergence to proper hazard magnitudes |
| `patience` | **10** | More aggressive early stopping |

**Why pred_horizon matters:**
- With `pred_horizon=100`: model spreads failure probability over 100 steps → lower per-step hazard
- With `pred_horizon=64`: model must predict higher hazard at failure time to capture event

**Quick fix command:**
```bash
python main.py \
    --model-type rnn \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --batch-size 512 \
    --learning-rate 0.01 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --num-epochs 200 \
    --exp-name paper_defaults \
    --create-visualization
```

Or simply use defaults (now paper-exact):
```bash
python main.py --model-type rnn --use-paper-split --use-minmax --num-epochs 200 --exp-name test --create-visualization
```

---

## Transformer Models

**Note:** Add `--dataset azure_pm --data-path ../AMLWorkshop/Data` for Azure PM dataset.

All transformer commands now include paper-exact parameters: `--pred-horizon 64 --lambda-param 0.75 --patience 10`

### Basic Transformer (Default Configuration)

```bash
python main.py \
    --model-type transformer \
    --num-epochs 200 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --num-encoder-layers 2 \
    --num-decoder-layers 2 \
    --d-model 64 \
    --nhead 4 \
    --dim-feedforward 256 \
    --dropout 0.1 \
    --activation relu \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name transformer_basic \
    --create-visualization
```

### Deep Transformer (More Layers)

```bash
python main.py \
    --model-type transformer \
    --num-epochs 250 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --num-encoder-layers 4 \
    --num-decoder-layers 4 \
    --d-model 64 \
    --nhead 4 \
    --dim-feedforward 256 \
    --dropout 0.1 \
    --activation relu \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name transformer_deep \
    --create-visualization
```

### Wide Transformer (Larger Dimensions)

```bash
python main.py \
    --model-type transformer \
    --num-epochs 200 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --num-encoder-layers 2 \
    --num-decoder-layers 2 \
    --d-model 128 \
    --nhead 8 \
    --dim-feedforward 512 \
    --dropout 0.1 \
    --activation relu \
    --batch-size 24 \
    --learning-rate 0.00008 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name transformer_wide \
    --create-visualization
```

### Complex Transformer (Deep + Wide)

```bash
python main.py \
    --model-type transformer \
    --num-epochs 300 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --num-encoder-layers 6 \
    --num-decoder-layers 4 \
    --d-model 128 \
    --nhead 8 \
    --dim-feedforward 512 \
    --dropout 0.15 \
    --activation relu \
    --batch-size 16 \
    --learning-rate 0.00005 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name transformer_complex \
    --create-visualization
```

### Transformer with GELU Activation

```bash
python main.py \
    --model-type transformer \
    --num-epochs 200 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --num-encoder-layers 2 \
    --num-decoder-layers 2 \
    --d-model 64 \
    --nhead 4 \
    --dim-feedforward 256 \
    --dropout 0.1 \
    --activation gelu \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name transformer_gelu \
    --create-visualization
```

### Deep Transformer with GELU

```bash
python main.py \
    --model-type transformer \
    --num-epochs 300 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --num-encoder-layers 6 \
    --num-decoder-layers 4 \
    --d-model 128 \
    --nhead 8 \
    --dim-feedforward 512 \
    --dropout 0.2 \
    --activation gelu \
    --batch-size 16 \
    --learning-rate 0.00005 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name transformer_deep_gelu \
    --create-visualization
```

### Very Deep Encoder Transformer

```bash
python main.py \
    --model-type transformer \
    --num-epochs 300 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --num-encoder-layers 8 \
    --num-decoder-layers 2 \
    --d-model 64 \
    --nhead 4 \
    --dim-feedforward 256 \
    --dropout 0.1 \
    --activation relu \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name transformer_8enc \
    --create-visualization
```

### High Dropout Transformer (Regularization)

```bash
python main.py \
    --model-type transformer \
    --num-epochs 250 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --num-encoder-layers 4 \
    --num-decoder-layers 4 \
    --d-model 128 \
    --nhead 8 \
    --dim-feedforward 512 \
    --dropout 0.3 \
    --activation gelu \
    --batch-size 16 \
    --learning-rate 0.00005 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name transformer_high_dropout \
    --create-visualization
```

---

## LSTM Models

**Note:** Add `--dataset azure_pm --data-path ../AMLWorkshop/Data` for Azure PM dataset.

All LSTM commands now include paper-exact parameters: `--pred-horizon 64 --lambda-param 0.75 --patience 10 --batch-size 512 --learning-rate 0.01`

### Basic LSTM (Default Configuration)

```bash
python main.py \
    --model-type rnn \
    --rnn-type LSTM \
    --num-epochs 200 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --hidden-dim 128 \
    --num-layers 2 \
    --dropout 0.1 \
    --batch-size 512 \
    --learning-rate 0.01 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name lstm_basic \
    --create-visualization
```

### Deep LSTM (More Layers)

```bash
python main.py \
    --model-type rnn \
    --rnn-type LSTM \
    --num-epochs 250 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --hidden-dim 128 \
    --num-layers 4 \
    --dropout 0.2 \
    --batch-size 512 \
    --learning-rate 0.01 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name lstm_deep \
    --create-visualization
```

### Wide LSTM (Larger Hidden Dimension)

```bash
python main.py \
    --model-type rnn \
    --rnn-type LSTM \
    --num-epochs 200 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --hidden-dim 256 \
    --num-layers 2 \
    --dropout 0.1 \
    --batch-size 512 \
    --learning-rate 0.01 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name lstm_wide \
    --create-visualization
```

### Complex LSTM (Deep + Wide)

```bash
python main.py \
    --model-type rnn \
    --rnn-type LSTM \
    --num-epochs 300 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --hidden-dim 256 \
    --num-layers 4 \
    --dropout 0.3 \
    --batch-size 256 \
    --learning-rate 0.008 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name lstm_complex \
    --create-visualization
```

### Small LSTM (Fast Training)

```bash
python main.py \
    --model-type rnn \
    --rnn-type LSTM \
    --num-epochs 100 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --hidden-dim 64 \
    --num-layers 1 \
    --dropout 0.1 \
    --batch-size 512 \
    --learning-rate 0.01 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name lstm_small \
    --create-visualization
```

---

## GRU Models

**Note:** Add `--dataset azure_pm --data-path ../AMLWorkshop/Data` for Azure PM dataset.

All GRU commands now include paper-exact parameters: `--pred-horizon 64 --lambda-param 0.75 --patience 10 --batch-size 512 --learning-rate 0.01`

### Basic GRU

```bash
python main.py \
    --model-type rnn \
    --rnn-type GRU \
    --num-epochs 200 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --hidden-dim 128 \
    --num-layers 2 \
    --dropout 0.1 \
    --batch-size 512 \
    --learning-rate 0.01 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name gru_basic \
    --create-visualization
```

### Deep GRU

```bash
python main.py \
    --model-type rnn \
    --rnn-type GRU \
    --num-epochs 250 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --hidden-dim 128 \
    --num-layers 4 \
    --dropout 0.2 \
    --batch-size 512 \
    --learning-rate 0.01 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name gru_deep \
    --create-visualization
```

### Wide GRU

```bash
python main.py \
    --model-type rnn \
    --rnn-type GRU \
    --num-epochs 200 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --hidden-dim 256 \
    --num-layers 2 \
    --dropout 0.1 \
    --batch-size 512 \
    --learning-rate 0.01 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name gru_wide \
    --create-visualization
```

### Complex GRU

```bash
python main.py \
    --model-type rnn \
    --rnn-type GRU \
    --num-epochs 300 \
    --patience 10 \
    --use-paper-split \
    --use-minmax \
    --hidden-dim 256 \
    --num-layers 4 \
    --dropout 0.3 \
    --batch-size 256 \
    --learning-rate 0.008 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir experiment_run1 \
    --exp-name gru_complex \
    --create-visualization
```

---

## Training Without Early Stopping

To train for the full number of epochs without early stopping:

```bash
python main.py \
    --model-type transformer \
    --num-epochs 500 \
    --no-early-stopping \
    --use-paper-split \
    --use-minmax \
    --num-encoder-layers 4 \
    --num-decoder-layers 4 \
    --d-model 128 \
    --nhead 8 \
    --dim-feedforward 512 \
    --dropout 0.1 \
    --activation gelu \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --pred-horizon 64 \
    --lambda-param 0.75 \
    --output-dir full_training \
    --exp-name transformer_500epochs \
    --create-visualization
```

---

## Command Line Arguments

### Data Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset to use (turbofan, azure_pm) | `turbofan` |
| `--data-path` | Path to data directory | `../Challenge_Data` |
| `--use-paper-split` | Use paper's data splitting methodology | False |
| `--use-minmax` | Use MinMaxScaler [-1, 1] | True |
| `--use-standard-scaler` | Use StandardScaler instead | False |
| `--num-workers` | Data loading workers | 4 |
| `--downsample-factor` | Down-sample factor for Azure PM (5=15 hours) | 5 |

### Transformer Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--num-encoder-layers` | Number of encoder layers | 2 |
| `--num-decoder-layers` | Number of decoder layers | 2 |
| `--d-model` | Model dimension | 64 |
| `--nhead` | Number of attention heads | 4 |
| `--dim-feedforward` | FFN dimension | 256 |
| `--dropout` | Dropout rate | 0.1 |
| `--activation` | Activation function (relu/gelu) | relu |

### RNN Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--rnn-type` | RNN type (LSTM/GRU) | LSTM |
| `--hidden-dim` | Hidden dimension | 128 |
| `--num-layers` | Number of RNN layers | 2 |
| `--dropout` | Dropout rate | 0.1 |

### Training Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--batch-size` | Batch size | 32 |
| `--learning-rate` | Learning rate | 0.0001 (transformer), 0.001 (RNN) |
| `--num-epochs` | Number of epochs | 100 |
| `--patience` | Early stopping patience | 20 |
| `--no-early-stopping` | Disable early stopping | False |
| `--warmup-steps` | Warmup steps for transformer | 4000 |
| `--lr-decay-type` | LR decay type (cosine/exponential/none) | cosine |

### Output Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--output-dir` | Parent output directory | None (uses ddrsa/logs) |
| `--exp-name` | Experiment name | ddrsa_experiment |
| `--create-visualization` | Create visualizations after training | False |

---

## Output Directory Structure

When using `--output-dir experiment_run1`:

```
Survival_Analysis/
├── ddrsa/                          # Code directory
├── experiment_run1/                # Your output folder
│   ├── logs/
│   │   └── <exp-name>/
│   │       ├── checkpoints/
│   │       │   └── best_model.pt
│   │       ├── config.json
│   │       ├── test_metrics.json
│   │       └── loss_history.json
│   └── figures/
│       └── <exp-name>/
│           ├── loss_curves.png
│           ├── figure_2a_hazard_progression_test.png
│           ├── figure_2a_hazard_progression_train.png
│           ├── figure_2b_oti_policy_test.png
│           └── figure_2b_oti_policy_train.png
```

---

## Generating Figures Separately

If you trained without `--create-visualization`, generate figures later:

```bash
python create_figures.py \
    --exp-name <your-exp-name> \
    --parent-dir <your-output-dir>

# For training data visualizations:
python create_figures.py \
    --exp-name <your-exp-name> \
    --parent-dir <your-output-dir> \
    --use-train-data
```

---

## Model Comparison Table

| Model | Layers | Dimensions | Parameters (approx) | Training Time |
|-------|--------|------------|---------------------|---------------|
| Transformer Basic | 2+2 | 64 | ~200K | Fast |
| Transformer Deep | 4+4 | 64 | ~400K | Medium |
| Transformer Wide | 2+2 | 128 | ~800K | Medium |
| Transformer Complex | 6+4 | 128 | ~1.5M | Slow |
| LSTM Basic | 2 | 128 | ~300K | Fast |
| LSTM Deep | 4 | 128 | ~600K | Medium |
| LSTM Wide | 2 | 256 | ~1M | Medium |
| GRU Basic | 2 | 128 | ~250K | Fast |
| GRU Deep | 4 | 128 | ~500K | Medium |

---

## Tips

1. **Start simple**: Begin with basic configurations and increase complexity if needed
2. **Monitor overfitting**: If val loss increases while train loss decreases, increase dropout
3. **GELU for transformers**: Often performs better than ReLU for attention-based models
4. **Reduce batch size**: For larger models, reduce batch size to fit in memory
5. **Lower learning rate**: For more complex models, use lower learning rate
6. **Increase patience**: Deeper models need more epochs to converge

---

## Quick Start Commands

### Fastest Training (for testing):
```bash
python main.py --model-type rnn --rnn-type LSTM --num-epochs 50 --hidden-dim 64 --num-layers 1 --output-dir quick_test --exp-name lstm_fast --create-visualization
```

### Best Transformer (recommended):
```bash
python main.py --model-type transformer --num-epochs 200 --patience 30 --use-paper-split --num-encoder-layers 4 --num-decoder-layers 4 --d-model 128 --nhead 8 --dim-feedforward 512 --dropout 0.1 --activation gelu --batch-size 16 --learning-rate 0.00005 --output-dir experiments --exp-name transformer_best --create-visualization
```

### Best LSTM (recommended):
```bash
python main.py --model-type rnn --rnn-type LSTM --num-epochs 200 --patience 30 --use-paper-split --hidden-dim 256 --num-layers 3 --dropout 0.2 --batch-size 32 --learning-rate 0.001 --output-dir experiments --exp-name lstm_best --create-visualization
```
