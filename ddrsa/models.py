"""
DDRSA (Dynamic Deep Recurrent Survival Analysis) Model Architecture
Implements both RNN-based and Transformer-based variants as described in the paper:
"When to Intervene: Learning Optimal Intervention Policies for Critical Events" (NeurIPS 2022)
"""

import torch
import torch.nn as nn
import math


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
    ProbSparse Self-Attention from Informer architecture
    Achieves O(N log N) complexity instead of O(N^2)

    Reference: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
    """

    def __init__(self, d_model, nhead, factor=5, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            factor: Sampling factor for ProbSparse attention (controls sparsity)
            dropout: Dropout rate
        """
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
        Compute the probability sparse attention

        Args:
            Q: Query tensor (batch, heads, seq_len, d_k)
            K: Key tensor (batch, heads, seq_len, d_k)
            sample_k: Number of keys to sample
            n_top: Number of top queries to select
        """
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Calculate sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)

        # Sample keys
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # Compute Q_K_sample
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Find the top_k queries with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def forward(self, x, attn_mask=None):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attn_mask: Optional attention mask

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        B, L, _ = x.shape

        # Linear projections
        Q = self.W_Q(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)

        # Compute sampling parameters
        U_part = self.factor * int(math.ceil(math.log(L + 1)))  # c * ln(L_Q)
        u = self.factor * int(math.ceil(math.log(L + 1)))  # c * ln(L_K)

        U_part = min(U_part, L)
        u = min(u, L)

        # ProbSparse attention
        scores_top, index = self._prob_QK(Q, K, sample_k=u, n_top=U_part)

        # Scale
        scale = 1.0 / math.sqrt(self.d_k)
        scores_top = scores_top * scale

        # Softmax
        attn = torch.softmax(scores_top, dim=-1)
        attn = self.dropout(attn)

        # Compute context for top queries
        context = torch.matmul(attn, V)

        # Initialize output with mean of V (for non-selected queries)
        V_mean = V.mean(dim=2, keepdim=True).expand(B, self.nhead, L, self.d_k)
        output = V_mean.clone()

        # Fill in the top query results
        output[torch.arange(B)[:, None, None],
               torch.arange(self.nhead)[None, :, None],
               index, :] = context

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.W_O(output)

        return output


class ProbSparseEncoderLayer(nn.Module):
    """Encoder layer with ProbSparse Attention"""

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1,
                 activation='gelu', factor=5):
        super(ProbSparseEncoderLayer, self).__init__()

        self.self_attn = ProbSparseAttention(d_model, nhead, factor, dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, src, src_mask=None):
        # Self attention with ProbSparse
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class DDRSA_ProbSparse(nn.Module):
    """
    Modified DDRSA with ProbSparse Attention Encoder

    Architecture (from poster):
    - ProbSparse Attention Encoder: O(N log N) complexity
    - LSTM Decoder: Same as original DDRSA
    - Models hazard rate functions
    """

    def __init__(self, input_dim, d_model=64, nhead=4, num_encoder_layers=2,
                 decoder_hidden_dim=64, decoder_layers=2, dim_feedforward=256,
                 pred_horizon=100, dropout=0.1, activation='gelu', factor=5):
        """
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of transformer embeddings
            nhead: Number of attention heads
            num_encoder_layers: Number of ProbSparse encoder layers
            decoder_hidden_dim: Hidden dimension for LSTM decoder
            decoder_layers: Number of LSTM decoder layers
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

        # ProbSparse Encoder layers
        self.encoder_layers = nn.ModuleList([
            ProbSparseEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, factor)
            for _ in range(num_encoder_layers)
        ])

        # Context vector projection (from encoder output to decoder input)
        self.context_projection = nn.Linear(d_model, decoder_hidden_dim)

        # LSTM Decoder (same as original DDRSA)
        self.decoder = nn.LSTM(
            decoder_hidden_dim,  # Input is context vector
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

        # Embed input
        x = self.input_embedding(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through ProbSparse encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Get context vector Z_j (last position or mean pooling)
        z_j = x[:, -1, :]  # Use last position as context

        # Project to decoder dimension
        z_j = self.context_projection(z_j)

        # Replicate Z_j for each decoder time step
        decoder_input = z_j.unsqueeze(1).repeat(1, self.pred_horizon, 1)

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
