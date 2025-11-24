# DDRSA Architecture Diagrams

This document provides detailed architectural diagrams for all three model variants implemented in the DDRSA system.

## Overview of All Architectures

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DDRSA Model Variants                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. DDRSA-RNN        : LSTM/GRU Encoder-Decoder                        │
│  2. DDRSA-Transformer: Self-Attention Encoder-Decoder                   │
│  3. DDRSA-ProbSparse : Informer with Sparse Attention                  │
│                                                                         │
│  All variants output: Conditional hazard rates h_j(k) for k=1..L_max   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. DDRSA-RNN Architecture (LSTM/GRU)

### High-Level Overview

```
Input Covariates          Encoder              Decoder              Hazard Rates
─────────────────         ────────             ────────             ────────────

X_j = [x₁, x₂, ..., xₜ]
  │
  │ (Batch, T, 24)
  │
  ▼
┌─────────────────┐
│  Encoder RNN    │
│  (LSTM/GRU)     │
│                 │
│  Hidden: 16-512 │
│  Layers: 1-4    │
└────────┬────────┘
         │
         │ Z_j (Encoded State)
         │ (Batch, Hidden)
         │
         ├──────────────────────────────────────┐
         │                                      │
         │ Replicate L_max times                │
         │                                      │
         ▼                                      │
┌─────────────────┐                             │
│  Decoder RNN    │◄────────────────────────────┘
│  (LSTM/GRU)     │
│                 │
│  Hidden: 16-512 │
│  Layers: 1-4    │
└────────┬────────┘
         │
         │ (Batch, L_max, Hidden)
         │
         ▼
┌─────────────────┐
│  Linear Layer   │
│  + Sigmoid      │
└────────┬────────┘
         │
         ▼
    h_j(1), h_j(2), ..., h_j(L_max)
    (Batch, L_max, 1)
```

### Detailed Layer-by-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DDRSA-RNN Detailed Architecture                    │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT LAYER
───────────
Shape: (Batch=32, Lookback=128, Features=24)
Example: [32, 128, 24] - 32 samples, 128 time steps, 24 sensor readings

          │
          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ ENCODER RNN (LSTM or GRU)                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Configuration Options:                                                     │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │ Variant          │ Hidden Dim │ Layers │ Parameters           │         │
│  ├───────────────────────────────────────────────────────────────┤         │
│  │ paper_exact      │     16     │   1    │    ~10K              │         │
│  │ basic            │    128     │   2    │   ~200K              │         │
│  │ deep             │    256     │   4    │   ~1M                │         │
│  │ wide             │    256     │   2    │   ~500K              │         │
│  │ complex          │    512     │   4    │   ~4M                │         │
│  └───────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  LSTM Cell (if rnn_type='LSTM'):                                           │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  i_t = σ(W_ii·x_t + b_ii + W_hi·h_(t-1) + b_hi)    [Input]  │          │
│  │  f_t = σ(W_if·x_t + b_if + W_hf·h_(t-1) + b_hf)    [Forget] │          │
│  │  g_t = tanh(W_ig·x_t + b_ig + W_hg·h_(t-1) + b_hg) [Cell]   │          │
│  │  o_t = σ(W_io·x_t + b_io + W_ho·h_(t-1) + b_ho)    [Output] │          │
│  │  c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t                             │          │
│  │  h_t = o_t ⊙ tanh(c_t)                                       │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  Input:  (Batch, Lookback, Features=24)                                    │
│  Output: (Batch, Hidden_Dim)  ← Takes final hidden state                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          │ Z_j = Encoded representation
          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ REPLICATION LAYER                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Z_j.unsqueeze(1).repeat(1, L_max, 1)                                      │
│                                                                             │
│  Input:  (Batch, Hidden_Dim)                                               │
│  Output: (Batch, L_max, Hidden_Dim)                                        │
│                                                                             │
│  Example: L_max=350 (prediction horizon)                                   │
│  Creates 350 copies of the encoded state as decoder input                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECODER RNN (LSTM or GRU)                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Same architecture as encoder (can have different size)                    │
│                                                                             │
│  Processes each future time step sequentially:                             │
│  h_j(1) from Z_j                                                           │
│  h_j(2) from h_j(1)                                                        │
│  ...                                                                        │
│  h_j(L_max) from h_j(L_max-1)                                              │
│                                                                             │
│  Input:  (Batch, L_max, Hidden_Dim)                                        │
│  Output: (Batch, L_max, Hidden_Dim)  ← All hidden states                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ DROPOUT LAYER                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Rate: 0.0 - 0.3 (configurable)                                            │
│  Applied element-wise for regularization                                   │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT LAYER                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Linear(Hidden_Dim → 1) + Sigmoid                                          │
│                                                                             │
│  logits = W·h + b  (where b initialized to -2.0)                           │
│  h(k) = σ(logits) = 1 / (1 + exp(-logits))                                 │
│                                                                             │
│  Output: (Batch, L_max, 1) → Conditional hazard rates                      │
│                                                                             │
│  Note: Bias initialized to -2.0 so σ(-2) ≈ 0.12 (reasonable starting rate) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          │
          ▼

OUTPUT: Hazard Rates h_j(k) for k = 1, 2, ..., L_max
Shape: (Batch, L_max, 1)
```

---

## 2. DDRSA-Transformer Architecture

### High-Level Overview

```
Input Covariates          Encoder                  Decoder              Hazard Rates
─────────────────         ────────                 ────────             ────────────

X_j = [x₁, x₂, ..., xₜ]
  │
  │ (Batch, T, 24)
  │
  ▼
┌──────────────────┐
│ Input Embedding  │
│ 24 → d_model     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Positional       │
│ Encoding         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Transformer      │
│ Encoder          │
│ (N layers)       │
│                  │
│ Multi-Head       │
│ Self-Attention   │
└────────┬─────────┘
         │
         │ Memory (encoded)
         │
         ├────────────────────────┐
         │                        │
         ▼                        ▼
┌──────────────────┐     ┌──────────────────┐
│ Decoder Input    │     │ Cross-Attention  │
│ (Learned Query)  │     │ to Memory        │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └────────────┬───────────┘
                      │
                      ▼
         ┌──────────────────┐
         │ Transformer      │
         │ Decoder          │
         │ (M layers)       │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Linear + Sigmoid │
         └────────┬─────────┘
                  │
                  ▼
    h_j(1), h_j(2), ..., h_j(L_max)
```

### Detailed Layer-by-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DDRSA-Transformer Detailed Architecture                  │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT LAYER
───────────
Shape: (Batch=32, Lookback=128, Features=24)

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUT EMBEDDING LAYER                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Linear(24 → d_model)                                                       │
│                                                                             │
│  Configuration Options:                                                     │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │ Variant     │ d_model │ Heads │ Enc Layers │ Dec Layers │ Params│       │
│  ├───────────────────────────────────────────────────────────────┤         │
│  │ basic       │   64    │   4   │     2      │     2      │ ~200K │       │
│  │ deep        │  128    │   8   │     6      │     4      │ ~1.5M │       │
│  │ wide        │  256    │   8   │     4      │     4      │ ~3M   │       │
│  │ gelu        │  128    │   8   │     4      │     4      │ ~1.8M │       │
│  │ complex     │  256    │  16   │     8      │     6      │ ~8M   │       │
│  └───────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  Output: (Batch, Lookback, d_model)                                        │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ POSITIONAL ENCODING                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))                             │
│  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))                             │
│                                                                             │
│  Adds position information to embeddings                                   │
│  Shape: (Lookback, d_model) - added to input embeddings                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ TRANSFORMER ENCODER (N layers)                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Each Encoder Layer Contains:                                              │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │ 1. MULTI-HEAD SELF-ATTENTION                                  │         │
│  │    ┌──────────────────────────────────────────────────────┐   │         │
│  │    │ For each head h (h = 1..num_heads):                  │   │         │
│  │    │                                                       │   │         │
│  │    │   Q_h = X·W^Q_h  (Query)                             │   │         │
│  │    │   K_h = X·W^K_h  (Key)                               │   │         │
│  │    │   V_h = X·W^V_h  (Value)                             │   │         │
│  │    │                                                       │   │         │
│  │    │   Attention_h = softmax(Q_h·K_h^T / √d_k) · V_h      │   │         │
│  │    │                                                       │   │         │
│  │    │ Concat all heads and project:                        │   │         │
│  │    │   MultiHead = Concat(head_1, ..., head_H) · W^O      │   │         │
│  │    └──────────────────────────────────────────────────────┘   │         │
│  │                                                                │         │
│  │ 2. ADD & NORM (Residual Connection + Layer Normalization)     │         │
│  │    output = LayerNorm(X + MultiHead)                          │         │
│  │                                                                │         │
│  │ 3. FEED-FORWARD NETWORK                                        │         │
│  │    FFN(x) = GELU(x·W_1 + b_1)·W_2 + b_2                       │         │
│  │    (d_model → d_ff → d_model)                                 │         │
│  │    where d_ff = 4 * d_model (default)                         │         │
│  │                                                                │         │
│  │ 4. ADD & NORM                                                  │         │
│  │    output = LayerNorm(X + FFN(X))                             │         │
│  │                                                                │         │
│  └────────────────────────────────────────────────────────────────         │
│                                                                             │
│  Repeat for N encoder layers (2-8 layers depending on variant)             │
│                                                                             │
│  Output: Encoded memory (Batch, Lookback, d_model)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │ Encoder Memory
          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECODER INPUT PREPARATION                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Learned query embedding: nn.Parameter (1, L_max, d_model)                 │
│  Expanded to batch size: (Batch, L_max, d_model)                           │
│                                                                             │
│  + Positional encoding for target sequence                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ TRANSFORMER DECODER (M layers)                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Each Decoder Layer Contains:                                              │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │ 1. MASKED MULTI-HEAD SELF-ATTENTION                           │         │
│  │    (Same as encoder, but with causal mask)                    │         │
│  │    Prevents attending to future positions                     │         │
│  │                                                                │         │
│  │ 2. ADD & NORM                                                  │         │
│  │                                                                │         │
│  │ 3. MULTI-HEAD CROSS-ATTENTION                                 │         │
│  │    ┌──────────────────────────────────────────────────────┐   │         │
│  │    │ Q = Decoder_output · W^Q  (from decoder)             │   │         │
│  │    │ K = Encoder_memory · W^K  (from encoder)             │   │         │
│  │    │ V = Encoder_memory · W^V  (from encoder)             │   │         │
│  │    │                                                       │   │         │
│  │    │ Attention = softmax(Q·K^T / √d_k) · V                │   │         │
│  │    └──────────────────────────────────────────────────────┘   │         │
│  │                                                                │         │
│  │ 4. ADD & NORM                                                  │         │
│  │                                                                │         │
│  │ 5. FEED-FORWARD NETWORK (with GELU activation)                │         │
│  │                                                                │         │
│  │ 6. ADD & NORM                                                  │         │
│  │                                                                │         │
│  └────────────────────────────────────────────────────────────────         │
│                                                                             │
│  Repeat for M decoder layers (2-6 layers depending on variant)             │
│                                                                             │
│  Output: (Batch, L_max, d_model)                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT LAYER                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Linear(d_model → 1) + Sigmoid                                             │
│  Bias initialized to -2.0                                                  │
│                                                                             │
│  Output: (Batch, L_max, 1) → Conditional hazard rates                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

OUTPUT: Hazard Rates h_j(k) for k = 1, 2, ..., L_max
```

---

## 3. DDRSA-ProbSparse (Informer) Architecture

### High-Level Overview

```
Input Covariates          ProbSparse Encoder         Decoder              Hazard Rates
─────────────────         ──────────────────         ────────             ────────────

X_j = [x₁, x₂, ..., xₜ]
  │
  │ (Batch, T, 24)
  │
  ▼
┌──────────────────┐
│ Input Embedding  │
│ 24 → d_model     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Positional       │
│ Encoding         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ProbSparse       │  ← O(L log L) complexity
│ Encoder Layer 1  │
│ + Distilling     │
└────────┬─────────┘
         │ L/2 (downsampled)
         ▼
┌──────────────────┐
│ ProbSparse       │
│ Encoder Layer 2  │
│ + Distilling     │
└────────┬─────────┘
         │ L/4 (downsampled)
         ▼
         ...
         │
         │ Encoded Memory
         │
         ▼
┌──────────────────┐
│ LSTM Decoder     │  ← Efficient for long horizons
│ (reuses RNN)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Linear + Sigmoid │
└────────┬─────────┘
         │
         ▼
    h_j(1), h_j(2), ..., h_j(L_max)
```

### Detailed Layer-by-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  DDRSA-ProbSparse (Informer) Detailed Architecture          │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT LAYER
───────────
Shape: (Batch=32, Lookback=128, Features=24)

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUT EMBEDDING LAYER                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Linear(24 → d_model)                                                       │
│                                                                             │
│  Configuration Options:                                                     │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │ Variant     │ d_model │ Heads │ Enc Layers │ Parameters        │         │
│  ├───────────────────────────────────────────────────────────────┤         │
│  │ basic       │   512   │   8   │     2      │    ~8M            │         │
│  │ deep        │   512   │   8   │     4      │   ~15M            │         │
│  └───────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  Output: (Batch, Lookback, d_model=512)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ POSITIONAL ENCODING                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Same as Transformer                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ PROBSPARSE ENCODER LAYER 1                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │ 1. PROBSPARSE SELF-ATTENTION (Key Innovation!)                │         │
│  │    ┌──────────────────────────────────────────────────────┐   │         │
│  │    │ Instead of computing all L×L attention scores:       │   │         │
│  │    │                                                       │   │         │
│  │    │ Step 1: Sample Top-u queries by sparsity measure     │   │         │
│  │    │         M(q_i) = max_j(Q·K^T/√d) - mean_j(Q·K^T/√d)  │   │         │
│  │    │         u = c·ln(L) where c is 'factor' param        │   │         │
│  │    │                                                       │   │         │
│  │    │ Step 2: These u queries attend to ALL keys           │   │         │
│  │    │         A_sparse = softmax(Q_top·K^T/√d)·V           │   │         │
│  │    │                                                       │   │         │
│  │    │ Step 3: Other queries use mean pooling               │   │         │
│  │    │                                                       │   │         │
│  │    │ Complexity: O(L·ln(L)) instead of O(L²)              │   │         │
│  │    └──────────────────────────────────────────────────────┘   │         │
│  │                                                                │         │
│  │ 2. ADD & NORM                                                  │         │
│  │                                                                │         │
│  │ 3. FEED-FORWARD NETWORK                                        │         │
│  │    FFN(x) = GELU(x·W_1 + b_1)·W_2 + b_2                       │         │
│  │                                                                │         │
│  │ 4. ADD & NORM                                                  │         │
│  │                                                                │         │
│  └────────────────────────────────────────────────────────────────         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ DISTILLING (Convolution Downsampling)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1D Convolution with MaxPooling                                            │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │ Conv1d(d_model, d_model, kernel=3, padding=1)            │              │
│  │ ELU activation                                            │              │
│  │ MaxPool1d(kernel_size=3, stride=2, padding=1)            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                             │
│  Reduces sequence length: L → L/2                                          │
│  Focuses on dominant features                                              │
│                                                                             │
│  Input:  (Batch, L, d_model)                                               │
│  Output: (Batch, L/2, d_model)                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ PROBSPARSE ENCODER LAYER 2                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Same structure as Layer 1                                                  │
│  Input: (Batch, L/2, d_model)                                              │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ DISTILLING                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Reduces: L/2 → L/4                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          │ (Repeat for N encoder layers)
          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ ATTENTION POOLING                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Learned pooling to combine downsampled representations                    │
│  Output: (Batch, d_model) - Single encoded vector                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ LSTM DECODER (Reuses DDRSA-RNN Decoder)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Why LSTM decoder:                                                          │
│  - ProbSparse encoder: Efficient for long input sequences                  │
│  - LSTM decoder: Efficient for long output sequences                       │
│  - Best of both worlds!                                                     │
│                                                                             │
│  1. Replicate encoded vector L_max times                                   │
│  2. Pass through LSTM decoder                                              │
│  3. Generate hazard rates sequentially                                     │
│                                                                             │
│  Input:  (Batch, d_model) → replicated to (Batch, L_max, d_model)          │
│  Output: (Batch, L_max, decoder_hidden_dim)                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT LAYER                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Linear(decoder_hidden_dim → 1) + Sigmoid                                  │
│  Output: (Batch, L_max, 1) → Conditional hazard rates                      │
└─────────────────────────────────────────────────────────────────────────────┘

          │
          ▼

OUTPUT: Hazard Rates h_j(k) for k = 1, 2, ..., L_max
```

### ProbSparse Attention Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              ProbSparse Attention Mechanism (Key Innovation)                │
└─────────────────────────────────────────────────────────────────────────────┘

STANDARD ATTENTION (O(L²) complexity):
───────────────────────────────────────

  All L queries attend to all L keys:

  Q = [q₁, q₂, q₃, ..., q_L]
  K = [k₁, k₂, k₃, ..., k_L]
  V = [v₁, v₂, v₃, ..., v_L]

  Attention_Matrix[L×L] = softmax(Q·K^T / √d_k)
  Output = Attention_Matrix · V

  Computational Cost: O(L²·d)


PROBSPARSE ATTENTION (O(L·log L) complexity):
──────────────────────────────────────────────

  Step 1: Measure "sparsity" of each query
  ┌─────────────────────────────────────────────────────────────┐
  │ For each query q_i:                                         │
  │                                                              │
  │   Compute: Q_i·K^T = [s_i1, s_i2, ..., s_iL]               │
  │                                                              │
  │   Sparsity Measure M(q_i):                                  │
  │     M(q_i) = max(q_i·K^T) - (1/L)·sum(q_i·K^T)             │
  │                                                              │
  │   High M(q_i) → query focuses on few keys (sparse)          │
  │   Low M(q_i)  → query distributes attention (uniform)       │
  └─────────────────────────────────────────────────────────────┘

  Step 2: Select Top-u sparse queries
  ┌─────────────────────────────────────────────────────────────┐
  │ u = c·ln(L)  where c is 'factor' parameter (default: 5)    │
  │                                                              │
  │ Top-u = indices of u queries with highest M(q_i)            │
  │                                                              │
  │ Example: L=128 → u = 5·ln(128) ≈ 24 queries                │
  └─────────────────────────────────────────────────────────────┘

  Step 3: Compute attention
  ┌─────────────────────────────────────────────────────────────┐
  │ For top-u queries:                                           │
  │   Compute full attention: softmax(Q_top·K^T/√d_k)·V         │
  │                                                              │
  │ For remaining (L-u) queries:                                 │
  │   Use mean pooling: V_bar = (1/L)·sum(V)                    │
  └─────────────────────────────────────────────────────────────┘

  Computational Cost: O(L·ln(L)·d)


COMPARISON:
───────────

  Input Length (L)  │  Standard O(L²)  │  ProbSparse O(L·ln(L))  │  Speedup
  ──────────────────┼──────────────────┼─────────────────────────┼──────────
        128         │     16,384       │          ~896           │   18×
        256         │     65,536       │         ~1,988          │   33×
        512         │    262,144       │         ~4,546          │   58×
       1024         │  1,048,576       │        ~10,240          │  102×
```

---

## Loss Functions

All three architectures use the same DDRSA loss function:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DDRSA LOSS FUNCTION                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Given hazard rates: h_j(k) for k = 1, 2, ..., L_max

Step 1: Compute Survival Probabilities
───────────────────────────────────────
  S_j(k) = ∏_{m=0}^{k-1} (1 - h_j(m))    (probability of surviving until k)

  S_j(0) = 1  (alive at current time)
  S_j(1) = 1 - h_j(0)
  S_j(2) = S_j(1) · (1 - h_j(1))
  ...

Step 2: Compute Expected Time-to-Event (TTE)
─────────────────────────────────────────────
  E[T_j] = ∑_{k=0}^{L_max} S_j(k)

  Interpretation: Expected remaining lifetime

Step 3: DDRSA Loss (Combines Hazard and Survival)
──────────────────────────────────────────────────
  For each sample j:

  If NOT censored (event observed at time t*):
    L_hazard = -log(h_j(t*))              ← Maximize hazard at failure time
    L_surv   = -∑_{k=0}^{t*-1} log(1 - h_j(k))  ← Maximize survival before

  If censored (survived past observation):
    L_hazard = 0                          ← No failure observed
    L_surv   = -∑_{k=0}^{T_obs} log(1 - h_j(k))  ← Maximize survival

  Combined Loss:
    L_DDRSA = λ·L_surv + (1-λ)·L_hazard

  where λ ∈ [0, 1] balances survival vs hazard
  (default: λ = 0.75, i.e., 75% survival, 25% hazard)

Step 4: Batch Loss
──────────────────
  L_total = (1/N) · ∑_{j=1}^{N} L_DDRSA(j)

OPTIONAL: Add NASA Scoring Function Loss
─────────────────────────────────────────
  If --use-nasa-loss flag is set:

  NASA_score = ∑_j scoring_function(E[T_j], t_true_j)

  where scoring_function(predicted, true):
    if predicted > true (early):
      s = exp(-(predicted - true) / 13) - 1
    else (late):
      s = exp((predicted - true) / 10) - 1

  Final Loss:
    L_final = L_DDRSA + nasa_weight · NASA_score

  (default: nasa_weight = 0.1)
```

---

## Comparison Table

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Architecture Comparison Summary                           │
├──────────────┬────────────────┬─────────────────┬────────────────────────────┤
│ Feature      │  DDRSA-RNN     │ DDRSA-Transform │ DDRSA-ProbSparse           │
├──────────────┼────────────────┼─────────────────┼────────────────────────────┤
│ Encoder      │ LSTM/GRU       │ Transformer     │ ProbSparse Attention       │
│              │ Layers: 1-4    │ Layers: 2-8     │ Layers: 2-4                │
│              │                │                 │ + Distilling               │
├──────────────┼────────────────┼─────────────────┼────────────────────────────┤
│ Decoder      │ LSTM/GRU       │ Transformer     │ LSTM (hybrid)              │
│              │ Layers: 1-4    │ Layers: 2-6     │ Layers: 2-4                │
├──────────────┼────────────────┼─────────────────┼────────────────────────────┤
│ Complexity   │ O(T·d²)        │ O(T²·d)         │ O(T·log(T)·d)              │
│ (Encoder)    │                │                 │                            │
├──────────────┼────────────────┼─────────────────┼────────────────────────────┤
│ Parameters   │ 10K - 4M       │ 200K - 8M       │ 8M - 15M                   │
├──────────────┼────────────────┼─────────────────┼────────────────────────────┤
│ Best For     │ Short sequences│ Medium sequences│ Long sequences             │
│              │ Simple patterns│ Complex patterns│ Very long sequences        │
├──────────────┼────────────────┼─────────────────┼────────────────────────────┤
│ Training     │ Fastest        │ Medium          │ Slower                     │
│ Speed        │ 5-10 min       │ 20-40 min       │ 40-90 min                  │
├──────────────┼────────────────┼─────────────────┼────────────────────────────┤
│ Memory Usage │ Lowest         │ High (O(T²))    │ Medium (O(T·log T))        │
│              │ ~180-600 MB    │ ~600-1200 MB    │ ~800-2400 MB               │
├──────────────┼────────────────┼─────────────────┼────────────────────────────┤
│ Activation   │ Tanh (LSTM)    │ GELU            │ GELU + ELU                 │
├──────────────┼────────────────┼─────────────────┼────────────────────────────┤
│ Positional   │ Implicit       │ Sinusoidal      │ Sinusoidal                 │
│ Encoding     │ (in RNN)       │                 │                            │
├──────────────┼────────────────┼─────────────────┼────────────────────────────┤
│ Key Feature  │ Simple,        │ Self-attention  │ Sparse attention           │
│              │ Sequential     │ Global context  │ O(L log L) complexity      │
└──────────────┴────────────────┴─────────────────┴────────────────────────────┘
```

---

## Common Components Across All Architectures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SHARED COMPONENTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 1. OUTPUT LAYER (All models)                                               │
│    ───────────────────────────────────────────────────────────────          │
│    Linear(hidden_dim → 1) + Sigmoid                                        │
│    Bias initialized to -2.0 → σ(-2) ≈ 0.12 (reasonable starting hazard)   │
│                                                                             │
│ 2. LOSS FUNCTION (All models)                                              │
│    ───────────────────────────────────────────────────────────────          │
│    DDRSA Loss: L = λ·L_survival + (1-λ)·L_hazard                          │
│    Optional: + nasa_weight · NASA_score                                    │
│                                                                             │
│ 3. OPTIMIZATION (All models)                                               │
│    ───────────────────────────────────────────────────────────────          │
│    Adam optimizer                                                           │
│    Learning rate: 0.0001 - 0.01 (model dependent)                         │
│    Warmup + Cosine/Exponential decay (for Transformers)                   │
│    Gradient clipping: 1.0                                                  │
│                                                                             │
│ 4. REGULARIZATION (All models)                                             │
│    ───────────────────────────────────────────────────────────────          │
│    Dropout: 0.0 - 0.3                                                      │
│    Early stopping: patience = 10-20 epochs                                 │
│                                                                             │
│ 5. INPUT/OUTPUT SHAPES (All models)                                        │
│    ───────────────────────────────────────────────────────────────          │
│    Input:  (Batch, Lookback, Features) = (32, 128, 24)                    │
│    Output: (Batch, L_max, 1) = (32, 350, 1)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Selection Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WHICH MODEL SHOULD I USE?                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Use DDRSA-RNN (LSTM/GRU) when:                                             │
│ ✓ You have short to medium sequences (T < 256)                            │
│ ✓ You want fast training (minutes per experiment)                         │
│ ✓ You have limited GPU memory                                             │
│ ✓ Your patterns are relatively simple/sequential                          │
│ ✓ You want a robust, proven architecture                                  │
│                                                                             │
│ Use DDRSA-Transformer when:                                                │
│ ✓ You need to capture long-range dependencies                             │
│ ✓ Your data has complex, non-sequential patterns                          │
│ ✓ You have sufficient GPU memory (>8GB)                                   │
│ ✓ Training time is not critical                                           │
│ ✓ You want state-of-the-art performance                                   │
│                                                                             │
│ Use DDRSA-ProbSparse when:                                                 │
│ ✓ You have very long sequences (T > 512)                                  │
│ ✓ You need efficient attention mechanism                                  │
│ ✓ You want the best of RNN (decoder) + Transformer (encoder)              │
│ ✓ Your sequences have hierarchical/multi-scale patterns                   │
│ ✓ Memory efficiency matters but you still want attention                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Locations

```
Implementation Files:
─────────────────────
models.py:632-639     - create_ddrsa_model() factory function
models.py:16-144      - DDRSA_RNN class
models.py:147-279     - DDRSA_Transformer class
models.py:282-617     - DDRSA_ProbSparse class

loss.py:1-245         - DDRSALoss and DDRSALossDetailed
loss.py:248-428       - NASA scoring function variants

trainer.py:1-320      - DDRSATrainer class (training loop)
```

This completes the architectural diagrams for all three DDRSA model variants!
