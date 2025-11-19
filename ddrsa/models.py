"""
DDRSA (Dynamic Deep Recurrent Survival Analysis) Model Architecture
Implements both RNN-based and Transformer-based variants as described in the paper:
"When to Intervene: Learning Optimal Intervention Policies for Critical Events" (NeurIPS 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ==================== DDRSA-RNN ====================

class DDRSA_RNN(nn.Module):
    """
    Dynamic Deep Recurrent Survival Analysis with RNN Encoder-Decoder

    Architecture (from Figure 1 in paper):
    - Encoder RNN: Maps covariate history X_j to hidden state Z_j
    - Decoder RNN: Produces conditional hazard rates h_j(k) for future time steps
    """

    def __init__(self, input_dim, encoder_hidden_dim=16, decoder_hidden_dim=16,
                 encoder_layers=1, decoder_layers=1, pred_horizon=100,
                 dropout=0.1, rnn_type='LSTM'):
        """
        Args:
            input_dim: Dimension of input features
            encoder_hidden_dim: Hidden dimension for encoder RNN
            decoder_hidden_dim: Hidden dimension for decoder RNN
            encoder_layers: Number of encoder RNN layers
            decoder_layers: Number of decoder RNN layers
            pred_horizon: Maximum prediction horizon (L_max)
            dropout: Dropout rate
            rnn_type: Type of RNN ('LSTM' or 'GRU')
        """
        super(DDRSA_RNN, self).__init__()

        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.pred_horizon = pred_horizon
        self.rnn_type = rnn_type

        # Encoder RNN
        if rnn_type == 'LSTM':
            self.encoder = nn.LSTM(
                input_dim,
                encoder_hidden_dim,
                encoder_layers,
                batch_first=True,
                dropout=dropout if encoder_layers > 1 else 0
            )
        elif rnn_type == 'GRU':
            self.encoder = nn.GRU(
                input_dim,
                encoder_hidden_dim,
                encoder_layers,
                batch_first=True,
                dropout=dropout if encoder_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        # Decoder RNN (DRSA-RNN from paper)
        if rnn_type == 'LSTM':
            self.decoder = nn.LSTM(
                encoder_hidden_dim,  # Input is replicated encoder output
                decoder_hidden_dim,
                decoder_layers,
                batch_first=True,
                dropout=dropout if decoder_layers > 1 else 0
            )
        elif rnn_type == 'GRU':
            self.decoder = nn.GRU(
                encoder_hidden_dim,
                decoder_hidden_dim,
                decoder_layers,
                batch_first=True,
                dropout=dropout if decoder_layers > 1 else 0
            )

        # Output layer: Maps decoder hidden state to hazard rate
        self.output_layer = nn.Linear(decoder_hidden_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, lookback_window, input_dim)

        Returns:
            hazard_logits: Tensor of shape (batch_size, pred_horizon)
        """
        batch_size = x.size(0)

        # Encode covariate history
        encoder_output, encoder_hidden = self.encoder(x)

        # Get final encoder hidden state Z_j
        if self.rnn_type == 'LSTM':
            z_j = encoder_hidden[0][-1]  # Shape: (batch_size, encoder_hidden_dim)
        else:  # GRU
            z_j = encoder_hidden[-1]

        # Replicate Z_j for each decoder time step
        # Shape: (batch_size, pred_horizon, encoder_hidden_dim)
        decoder_input = z_j.unsqueeze(1).repeat(1, self.pred_horizon, 1)

        # Decode to get hazard rates
        decoder_output, _ = self.decoder(decoder_input)

        # Apply dropout
        decoder_output = self.dropout(decoder_output)

        # Map to hazard rates
        # Shape: (batch_size, pred_horizon, 1)
        hazard_logits = self.output_layer(decoder_output)

        # Squeeze last dimension
        # Shape: (batch_size, pred_horizon)
        hazard_logits = hazard_logits.squeeze(-1)

        return hazard_logits


# ==================== DDRSA-Transformer ====================

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DDRSA_Transformer(nn.Module):
    """
    Dynamic Deep Recurrent Survival Analysis with Transformer Encoder-Decoder

    Architecture:
    - Transformer Encoder: Maps covariate history X_j to hidden representations
    - Transformer Decoder: Produces conditional hazard rates h_j(k) for future time steps
    """

    def __init__(self, input_dim, d_model=64, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=256, pred_horizon=100,
                 dropout=0.1, activation='relu'):
        """
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of transformer embeddings
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            pred_horizon: Maximum prediction horizon (L_max)
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super(DDRSA_Transformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_horizon = pred_horizon

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding for encoder
        self.encoder_pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Decoder input embedding (learnable query embeddings for future time steps)
        self.decoder_queries = nn.Parameter(torch.randn(pred_horizon, d_model))

        # Positional encoding for decoder
        self.decoder_pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Output layer: Maps decoder output to hazard rate
        self.output_layer = nn.Linear(d_model, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, lookback_window, input_dim)

        Returns:
            hazard_logits: Tensor of shape (batch_size, pred_horizon)
        """
        batch_size = x.size(0)

        # Embed input
        x = self.input_embedding(x)

        # Add positional encoding to encoder input
        x = self.encoder_pos_encoding(x)

        # Encode covariate history
        encoder_output = self.transformer_encoder(x)

        # Prepare decoder queries (replicate for batch)
        decoder_input = self.decoder_queries.unsqueeze(0).repeat(batch_size, 1, 1)

        # Add positional encoding to decoder input
        decoder_input = self.decoder_pos_encoding(decoder_input)

        # Decode to get hazard rates
        decoder_output = self.transformer_decoder(decoder_input, encoder_output)

        # Apply dropout
        decoder_output = self.dropout(decoder_output)

        # Map to hazard rates
        # Shape: (batch_size, pred_horizon, 1)
        hazard_logits = self.output_layer(decoder_output)

        # Squeeze last dimension
        # Shape: (batch_size, pred_horizon)
        hazard_logits = hazard_logits.squeeze(-1)

        return hazard_logits


# ==================== ProbSparse Attention (Informer) ====================

class ProbSparseAttention(nn.Module):
    """
    ProbSparse Self-Attention mechanism from Informer paper (AAAI 2021)

    Key idea: Only compute attention for top-k queries with highest sparsity scores,
    reducing complexity from O(L^2) to O(L log L)
    """

    def __init__(self, d_model, nhead, factor=5, dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.factor = factor

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Compute sparsity measurement M(q_i, K)

        Args:
            Q: Query tensor (batch, nhead, seq_len, d_k)
            K: Key tensor (batch, nhead, seq_len, d_k)
            sample_k: Number of keys to sample
            n_top: Number of top queries to select
        """
        batch, nhead, L_Q, d_k = Q.shape
        _, _, L_K, _ = K.shape

        # Sample keys for approximating M
        K_expand = K.unsqueeze(-3).expand(batch, nhead, L_Q, L_K, d_k)

        # Sample random keys
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # Q * K^T for sampled keys
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Compute sparsity measurement
        # M = max(Q*K) - mean(Q*K)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)

        # Select top-n_top queries
        M_top = M.topk(n_top, sorted=False)[1]

        return M_top

    def _get_initial_context(self, V, L_Q):
        """Get initial context using mean of values"""
        batch, nhead, L_V, d_v = V.shape

        # Use cumsum for efficient mean computation
        V_sum = V.mean(dim=-2)
        context = V_sum.unsqueeze(-2).expand(batch, nhead, L_Q, d_v).clone()

        return context

    def forward(self, queries, keys, values, attn_mask=None):
        """
        Forward pass for ProbSparse attention

        Args:
            queries: (batch, L_Q, d_model)
            keys: (batch, L_K, d_model)
            values: (batch, L_V, d_model)
            attn_mask: Optional attention mask

        Returns:
            output: (batch, L_Q, d_model)
        """
        batch, L_Q, _ = queries.shape
        _, L_K, _ = keys.shape
        _, L_V, _ = values.shape

        # Linear projections and reshape for multi-head
        Q = self.W_Q(queries).view(batch, L_Q, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_K(keys).view(batch, L_K, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_V(values).view(batch, L_V, self.nhead, self.d_k).transpose(1, 2)

        # Compute number of samples and top queries
        U = self.factor * int(np.ceil(np.log(L_K + 1)))  # c * ln(L_K)
        u = self.factor * int(np.ceil(np.log(L_Q + 1)))  # c * ln(L_Q)

        U = min(U, L_K)
        u = min(u, L_Q)

        # Get initial context (mean of values)
        context = self._get_initial_context(V, L_Q)

        # Get top queries indices
        if L_Q > u:
            M_top = self._prob_QK(Q, K, sample_k=U, n_top=u)

            # Compute attention only for top queries
            Q_reduce = torch.gather(
                Q, 2,
                M_top.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
            )

            # Full attention for selected queries
            scores = torch.matmul(Q_reduce, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e9)

            attn = self.dropout(F.softmax(scores, dim=-1))
            context_in = torch.matmul(attn, V)

            # Update context at selected positions
            context = context.scatter(
                2,
                M_top.unsqueeze(-1).expand(-1, -1, -1, self.d_k),
                context_in
            )
        else:
            # Fall back to full attention for short sequences
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e9)

            attn = self.dropout(F.softmax(scores, dim=-1))
            context = torch.matmul(attn, V)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch, L_Q, self.d_model)
        output = self.W_O(context)

        return output


class ProbSparseEncoderLayer(nn.Module):
    """Encoder layer with ProbSparse attention"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='gelu', factor=5):
        super(ProbSparseEncoderLayer, self).__init__()

        self.self_attn = ProbSparseAttention(d_model, nhead, factor, dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        if activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = F.relu

    def forward(self, src, src_mask=None):
        # Self-attention with residual
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward with residual
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class ConvLayer(nn.Module):
    """Distilling layer to reduce sequence length"""

    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)  # (batch, seq_len//2, d_model)
        return x


class DDRSA_ProbSparse(nn.Module):
    """
    Dynamic Deep Recurrent Survival Analysis with ProbSparse (Informer) Encoder

    Architecture:
    - ProbSparse Encoder: Efficient attention with O(L log L) complexity
    - Distilling: Conv layers to reduce sequence length between encoder layers
    - RNN Decoder: Produces conditional hazard rates for future time steps
    """

    def __init__(self, input_dim, d_model=512, nhead=8, num_encoder_layers=2,
                 decoder_hidden_dim=512, decoder_layers=1, dim_feedforward=2048,
                 pred_horizon=100, dropout=0.05, activation='gelu', factor=5):
        """
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            decoder_hidden_dim: Hidden dimension for decoder RNN
            decoder_layers: Number of decoder RNN layers
            dim_feedforward: Dimension of feedforward network
            pred_horizon: Maximum prediction horizon (L_max)
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            factor: ProbSparse attention factor (controls sparsity)
        """
        super(DDRSA_ProbSparse, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_horizon = pred_horizon

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # ProbSparse encoder layers with distilling
        self.encoder_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        for i in range(num_encoder_layers):
            self.encoder_layers.append(
                ProbSparseEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    factor=factor
                )
            )
            # Add distilling layer (except for last layer)
            if i < num_encoder_layers - 1:
                self.conv_layers.append(ConvLayer(d_model))

        # Global average pooling to get fixed-size encoding
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Decoder RNN
        self.decoder = nn.LSTM(
            d_model,
            decoder_hidden_dim,
            decoder_layers,
            batch_first=True,
            dropout=dropout if decoder_layers > 1 else 0
        )

        # Output layer
        self.output_layer = nn.Linear(decoder_hidden_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, lookback_window, input_dim)

        Returns:
            hazard_logits: Tensor of shape (batch_size, pred_horizon)
        """
        batch_size = x.size(0)

        # Embed input
        x = self.input_embedding(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through encoder layers with distilling
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x)
            if i < len(self.conv_layers):
                x = self.conv_layers[i](x)

        # Global pooling to get encoder output
        # x: (batch, seq_len, d_model) -> (batch, d_model)
        encoder_output = self.pool(x.transpose(1, 2)).squeeze(-1)

        # Replicate for decoder input
        decoder_input = encoder_output.unsqueeze(1).repeat(1, self.pred_horizon, 1)

        # Decode to get hazard rates
        decoder_output, _ = self.decoder(decoder_input)

        # Apply dropout
        decoder_output = self.dropout(decoder_output)

        # Map to hazard rates
        hazard_logits = self.output_layer(decoder_output)

        # Squeeze last dimension
        hazard_logits = hazard_logits.squeeze(-1)

        return hazard_logits


# ==================== Model Factory ====================

def create_ddrsa_model(model_type='rnn', input_dim=24, **kwargs):
    """
    Factory function to create DDRSA models

    Args:
        model_type: 'rnn', 'transformer', or 'probsparse'
        input_dim: Input feature dimension
        **kwargs: Additional model-specific arguments

    Returns:
        DDRSA model instance
    """
    if model_type.lower() == 'rnn':
        return DDRSA_RNN(input_dim=input_dim, **kwargs)
    elif model_type.lower() == 'transformer':
        return DDRSA_Transformer(input_dim=input_dim, **kwargs)
    elif model_type.lower() == 'probsparse':
        return DDRSA_ProbSparse(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'rnn', 'transformer', or 'probsparse'")


if __name__ == '__main__':
    # Test models
    batch_size = 8
    lookback_window = 128
    input_dim = 24
    pred_horizon = 100

    # Create dummy input
    x = torch.randn(batch_size, lookback_window, input_dim)

    # Test RNN model
    print("Testing DDRSA-RNN...")
    model_rnn = create_ddrsa_model('rnn', input_dim=input_dim, pred_horizon=pred_horizon)
    output_rnn = model_rnn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_rnn.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model_rnn.parameters()):,}")

    # Test Transformer model
    print("\nTesting DDRSA-Transformer...")
    model_transformer = create_ddrsa_model('transformer', input_dim=input_dim, pred_horizon=pred_horizon)
    output_transformer = model_transformer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_transformer.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model_transformer.parameters()):,}")

    # Test ProbSparse model
    print("\nTesting DDRSA-ProbSparse (Informer)...")
    model_probsparse = create_ddrsa_model('probsparse', input_dim=input_dim, pred_horizon=pred_horizon,
                                          d_model=128, nhead=4, num_encoder_layers=2,
                                          decoder_hidden_dim=128, factor=5)
    output_probsparse = model_probsparse(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_probsparse.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model_probsparse.parameters()):,}")
