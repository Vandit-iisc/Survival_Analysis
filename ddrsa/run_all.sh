#!/bin/bash

# Run all DDRSA experiments
# This script reproduces the experiments from the paper

# Set data path
DATA_PATH="/Users/vandit/Desktop/vandit/Survival_Analysis/Challenge_Data"

echo "========================================="
echo "DDRSA Experiments - NASA Turbofan Dataset"
echo "========================================="
echo ""

# Experiment 1: DDRSA-LSTM (default from paper)
echo "Experiment 1: DDRSA-LSTM (Paper Configuration)"
echo "-------------------------------------------"
python main.py \
    --data-path $DATA_PATH \
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
    --exp-name ddrsa_lstm_paper \
    --seed 42

echo ""
echo "========================================="
echo ""

# Experiment 2: DDRSA-GRU
echo "Experiment 2: DDRSA-GRU"
echo "-------------------------------------------"
python main.py \
    --data-path $DATA_PATH \
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
    --exp-name ddrsa_gru \
    --seed 42

echo ""
echo "========================================="
echo ""

# Experiment 3: DDRSA-Transformer
echo "Experiment 3: DDRSA-Transformer"
echo "-------------------------------------------"
python main.py \
    --data-path $DATA_PATH \
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
    --exp-name ddrsa_transformer \
    --seed 42

echo ""
echo "========================================="
echo ""

# Experiment 4: Ablation - Different Lambda values
echo "Experiment 4: Ablation Study - Lambda Parameter"
echo "-------------------------------------------"

for lambda in 0.1 0.3 0.5 0.7 0.9; do
    echo "Training with lambda = $lambda"
    python main.py \
        --data-path $DATA_PATH \
        --model-type rnn \
        --rnn-type LSTM \
        --hidden-dim 16 \
        --num-layers 1 \
        --batch-size 32 \
        --learning-rate 1e-4 \
        --lambda-param $lambda \
        --lookback-window 128 \
        --pred-horizon 100 \
        --num-epochs 100 \
        --exp-name ddrsa_lstm_lambda_$lambda \
        --seed 42
    echo ""
done

echo "========================================="
echo ""

# Experiment 5: Ablation - Different Hidden Dimensions
echo "Experiment 5: Ablation Study - Hidden Dimension"
echo "-------------------------------------------"

for hidden in 8 16 32 64; do
    echo "Training with hidden_dim = $hidden"
    python main.py \
        --data-path $DATA_PATH \
        --model-type rnn \
        --rnn-type LSTM \
        --hidden-dim $hidden \
        --num-layers 1 \
        --batch-size 32 \
        --learning-rate 1e-4 \
        --lambda-param 0.5 \
        --lookback-window 128 \
        --pred-horizon 100 \
        --num-epochs 100 \
        --exp-name ddrsa_lstm_hidden_$hidden \
        --seed 42
    echo ""
done

echo "========================================="
echo "All experiments completed!"
echo "========================================="
