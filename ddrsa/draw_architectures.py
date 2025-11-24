"""
Draw Publication-Quality Architectural Diagrams for DDRSA Models
Creates professional diagrams similar to those in research papers
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os


# Professional color scheme
COLORS = {
    'input': '#E8F4F8',      # Light blue
    'encoder': '#B8E6F0',    # Sky blue
    'attention': '#FFE8CC',  # Light orange
    'decoder': '#D4E8D4',    # Light green
    'output': '#FFD4D4',     # Light red
    'embedding': '#E8D4F8',  # Light purple
    'dense': '#F0E68C',      # Khaki
    'arrow': '#555555',      # Dark gray
    'text': '#000000'        # Black
}


def draw_box(ax, x, y, width, height, text, color, fontsize=9):
    """Draw a fancy box with text"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2),
        width, height,
        boxstyle="round,pad=0.05",
        linewidth=1.5,
        edgecolor='black',
        facecolor=color,
        alpha=0.8,
        zorder=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, style='->'):
    """Draw a fancy arrow"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=COLORS['arrow'],
        linewidth=2,
        mutation_scale=20,
        zorder=1
    )
    ax.add_patch(arrow)


def draw_multi_arrow(ax, x1, y1, x2, y2, n=3, spacing=0.1):
    """Draw multiple parallel arrows to indicate data flow"""
    for i in range(n):
        offset = (i - n/2 + 0.5) * spacing
        arrow = FancyArrowPatch(
            (x1 + offset, y1), (x2 + offset, y2),
            arrowstyle='->',
            color=COLORS['arrow'],
            linewidth=1.5,
            mutation_scale=15,
            alpha=0.6,
            zorder=1
        )
        ax.add_patch(arrow)


def draw_rnn_architecture(output_path='figures/architecture_rnn.png'):
    """Draw DDRSA-RNN Architecture (LSTM/GRU)"""

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'DDRSA-RNN Architecture (LSTM/GRU)',
            ha='center', fontsize=16, fontweight='bold')

    # Input
    draw_box(ax, 2, 8, 1.5, 0.6, 'Input\nSequence\n(B, L, D)', COLORS['input'])

    # Input projection
    draw_arrow(ax, 2, 7.7, 2, 7.2)
    draw_box(ax, 2, 6.8, 1.5, 0.5, 'Linear\nProjection', COLORS['embedding'])

    # Encoder
    draw_arrow(ax, 2, 6.55, 2, 6.0)
    draw_box(ax, 2, 5.0, 1.8, 1.6, 'LSTM/GRU\nEncoder\n\nLayers: N_enc\nHidden: H_enc',
             COLORS['encoder'])

    # Show encoder internal structure
    ax.text(0.5, 5.5, 'LSTM Cell:', fontsize=8, style='italic')
    ax.text(0.5, 5.2, 'i = σ(W_i·x + U_i·h)', fontsize=7, family='monospace')
    ax.text(0.5, 4.9, 'f = σ(W_f·x + U_f·h)', fontsize=7, family='monospace')
    ax.text(0.5, 4.6, 'o = σ(W_o·x + U_o·h)', fontsize=7, family='monospace')
    ax.text(0.5, 4.3, 'c̃ = tanh(W_c·x + U_c·h)', fontsize=7, family='monospace')

    # Hidden state
    draw_arrow(ax, 2, 4.2, 2, 3.7)
    draw_box(ax, 2, 3.4, 1.5, 0.5, 'Hidden State\nh_enc', COLORS['encoder'])

    # Replicate to decoder length
    draw_arrow(ax, 2.75, 3.4, 4.5, 3.4)
    draw_box(ax, 5.5, 3.4, 1.5, 0.5, 'Replicate\nto T steps', COLORS['embedding'])
    draw_arrow(ax, 6.25, 3.4, 8, 3.4)

    # Decoder input
    draw_box(ax, 9, 8, 1.5, 0.6, 'Decoder\nInput\n(zeros)', COLORS['input'])
    draw_arrow(ax, 9, 7.7, 9, 7.2)
    draw_box(ax, 9, 6.8, 1.5, 0.5, 'Linear\nProjection', COLORS['embedding'])

    # Combine encoder output and decoder input
    draw_arrow(ax, 9, 6.55, 9, 6.0)
    draw_box(ax, 9, 5.5, 1.5, 0.6, 'Concatenate\n+ h_enc', COLORS['embedding'])
    draw_arrow(ax, 8.5, 3.7, 8.5, 5.2)
    draw_arrow(ax, 8.5, 5.2, 8.4, 5.5)

    # Decoder
    draw_arrow(ax, 9, 5.2, 9, 4.7)
    draw_box(ax, 9, 3.6, 1.8, 1.6, 'LSTM/GRU\nDecoder\n\nLayers: N_dec\nHidden: H_dec',
             COLORS['decoder'])

    # Decoder outputs
    draw_arrow(ax, 9, 2.8, 9, 2.3)

    # Three output heads
    y_out = 1.5

    # Hazard head
    draw_box(ax, 6, y_out, 1.3, 0.5, 'Linear\n(H→1)', COLORS['dense'])
    draw_arrow(ax, 8.1, 2.0, 6.65, 1.75)
    draw_arrow(ax, 6, 1.25, 6, 0.8)
    draw_box(ax, 6, 0.5, 1.3, 0.4, 'Sigmoid\nλ(t)', COLORS['output'])

    # Survival head
    draw_box(ax, 9, y_out, 1.3, 0.5, 'Linear\n(H→1)', COLORS['dense'])
    draw_arrow(ax, 9, 2.1, 9, 1.75)
    draw_arrow(ax, 9, 1.25, 9, 0.8)
    draw_box(ax, 9, 0.5, 1.3, 0.4, 'Sigmoid\nS(t)', COLORS['output'])

    # TTE head
    draw_box(ax, 12, y_out, 1.3, 0.5, 'Linear\n(H→1)', COLORS['dense'])
    draw_arrow(ax, 9.9, 2.0, 12.35, 1.75)
    draw_arrow(ax, 12, 1.25, 12, 0.8)
    draw_box(ax, 12, 0.5, 1.3, 0.4, 'ReLU\nE[TTE]', COLORS['output'])

    # Add legend for dimensions
    ax.text(0.3, 1.0, 'Dimensions:', fontsize=9, fontweight='bold')
    ax.text(0.3, 0.7, 'B = Batch size', fontsize=8)
    ax.text(0.3, 0.5, 'L = Sequence length (lookback)', fontsize=8)
    ax.text(0.3, 0.3, 'D = Input features', fontsize=8)
    ax.text(2.5, 0.7, 'T = Prediction horizon', fontsize=8)
    ax.text(2.5, 0.5, 'H_enc = Encoder hidden dim', fontsize=8)
    ax.text(2.5, 0.3, 'H_dec = Decoder hidden dim', fontsize=8)

    # Configuration examples box
    config_text = """Model Variants:
• Paper Exact: H=16, N=1
• Basic: H=128, N=2
• Deep: H=256, N=4
• Wide: H=256, N=2
• Complex: H=512, N=4"""

    ax.text(11.5, 5.5, config_text, fontsize=7,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            verticalalignment='top', family='monospace')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved RNN architecture to: {output_path}")
    plt.close()


def draw_transformer_architecture(output_path='figures/architecture_transformer.png'):
    """Draw DDRSA-Transformer Architecture"""

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(8, 11.5, 'DDRSA-Transformer Architecture',
            ha='center', fontsize=16, fontweight='bold')

    # ===== ENCODER SIDE =====
    enc_x = 3.5

    # Input
    draw_box(ax, enc_x, 10.5, 1.5, 0.5, 'Input\nSequence', COLORS['input'])
    draw_arrow(ax, enc_x, 10.25, enc_x, 9.8)

    # Input embedding
    draw_box(ax, enc_x, 9.5, 1.5, 0.4, 'Linear\nEmbedding', COLORS['embedding'])
    draw_arrow(ax, enc_x, 9.3, enc_x, 9.0)

    # Positional encoding
    draw_box(ax, enc_x, 8.7, 1.5, 0.4, 'Positional\nEncoding', COLORS['embedding'])
    draw_arrow(ax, enc_x, 8.5, enc_x, 8.1)

    # Encoder layers (show 2 in detail)
    for i, layer_y in enumerate([7.3, 5.8]):
        # Multi-head self-attention
        draw_box(ax, enc_x, layer_y + 0.4, 1.8, 0.5,
                'Multi-Head\nSelf-Attention', COLORS['attention'])

        # Add & Norm
        draw_box(ax, enc_x, layer_y, 1.5, 0.3, 'Add & Norm', COLORS['embedding'])
        draw_arrow(ax, enc_x, layer_y + 0.15, enc_x, layer_y - 0.15)

        # Feed-forward
        draw_box(ax, enc_x, layer_y - 0.5, 1.8, 0.5,
                'Feed-Forward\nFFN(x)', COLORS['decoder'])
        draw_arrow(ax, enc_x, layer_y - 0.75, enc_x, layer_y - 1.05)

        # Add & Norm
        draw_box(ax, enc_x, layer_y - 1.25, 1.5, 0.3, 'Add & Norm', COLORS['embedding'])

        if i == 0:
            draw_arrow(ax, enc_x, layer_y - 1.4, enc_x, layer_y - 1.65)

    # More layers indicator
    ax.text(enc_x, 4.9, '⋮\nN_enc layers\n⋮', ha='center', fontsize=10)

    # Encoder output
    draw_arrow(ax, enc_x, 4.5, enc_x, 4.1)
    draw_box(ax, enc_x, 3.8, 1.5, 0.4, 'Encoder\nOutput', COLORS['encoder'])

    # ===== DECODER SIDE =====
    dec_x = 12

    # Decoder input
    draw_box(ax, dec_x, 10.5, 1.5, 0.5, 'Decoder\nInput', COLORS['input'])
    draw_arrow(ax, dec_x, 10.25, dec_x, 9.8)

    # Input embedding
    draw_box(ax, dec_x, 9.5, 1.5, 0.4, 'Linear\nEmbedding', COLORS['embedding'])
    draw_arrow(ax, dec_x, 9.3, dec_x, 9.0)

    # Positional encoding
    draw_box(ax, dec_x, 8.7, 1.5, 0.4, 'Positional\nEncoding', COLORS['embedding'])
    draw_arrow(ax, dec_x, 8.5, dec_x, 8.1)

    # Decoder layers (show 2 in detail)
    for i, layer_y in enumerate([7.3, 5.0]):
        # Masked self-attention
        draw_box(ax, dec_x, layer_y + 0.4, 1.8, 0.5,
                'Masked\nSelf-Attention', COLORS['attention'])

        # Add & Norm
        draw_box(ax, dec_x, layer_y, 1.5, 0.3, 'Add & Norm', COLORS['embedding'])
        draw_arrow(ax, dec_x, layer_y + 0.15, dec_x, layer_y - 0.15)

        # Cross-attention (connects to encoder)
        draw_box(ax, dec_x, layer_y - 0.5, 1.8, 0.5,
                'Cross-Attention\nQ from Dec\nK,V from Enc', COLORS['attention'])

        # Arrow from encoder
        draw_arrow(ax, enc_x + 0.75, 3.8, dec_x - 0.9, layer_y - 0.25, style='->')

        draw_arrow(ax, dec_x, layer_y - 0.75, dec_x, layer_y - 1.05)

        # Add & Norm
        draw_box(ax, dec_x, layer_y - 1.25, 1.5, 0.3, 'Add & Norm', COLORS['embedding'])
        draw_arrow(ax, dec_x, layer_y - 1.4, dec_x, layer_y - 1.6)

        # Feed-forward
        draw_box(ax, dec_x, layer_y - 1.9, 1.8, 0.5,
                'Feed-Forward\nFFN(x)', COLORS['decoder'])
        draw_arrow(ax, dec_x, layer_y - 2.15, dec_x, layer_y - 2.45)

        # Add & Norm
        draw_box(ax, dec_x, layer_y - 2.65, 1.5, 0.3, 'Add & Norm', COLORS['embedding'])

        if i == 0:
            draw_arrow(ax, dec_x, layer_y - 2.8, dec_x, layer_y - 3.05)

    # More layers indicator
    ax.text(dec_x, 3.1, '⋮\nN_dec layers\n⋮', ha='center', fontsize=10)

    # Decoder output
    draw_arrow(ax, dec_x, 2.0, dec_x, 1.6)
    draw_box(ax, dec_x, 1.3, 1.5, 0.4, 'Decoder\nOutput', COLORS['decoder'])

    # ===== OUTPUT HEADS =====
    draw_arrow(ax, dec_x, 1.1, dec_x, 0.7)

    # Three outputs side by side
    for idx, (out_x, name, activation) in enumerate([
        (9.5, 'λ(t)\nHazard', 'Sigmoid'),
        (12, 'S(t)\nSurvival', 'Sigmoid'),
        (14.5, 'E[TTE]\nExpected', 'ReLU')
    ]):
        if idx == 0:
            draw_arrow(ax, dec_x - 0.5, 0.5, out_x + 0.5, 0.5)
        elif idx == 2:
            draw_arrow(ax, dec_x + 0.5, 0.5, out_x - 0.5, 0.5)

        draw_box(ax, out_x, 0.5, 1.2, 0.5, name + f'\n{activation}', COLORS['output'])

    # ===== ATTENTION MECHANISM DETAIL =====
    detail_x = 7.5
    detail_y = 10

    ax.text(detail_x, detail_y + 0.5, 'Multi-Head Attention Detail',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # QKV boxes
    for i, (qkv_x, label) in enumerate([(detail_x - 1.5, 'Q'),
                                          (detail_x, 'K'),
                                          (detail_x + 1.5, 'V')]):
        draw_box(ax, qkv_x, detail_y - 0.5, 0.6, 0.3, label, COLORS['attention'])

    # Attention formula
    formula_text = r"Attention(Q,K,V) = softmax(QK^T/√d_k)V"
    ax.text(detail_x, detail_y - 1.0, formula_text, ha='center', fontsize=9,
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Multi-head
    ax.text(detail_x, detail_y - 1.5, '↓ Split into h heads ↓', ha='center', fontsize=8)

    for i in range(4):
        head_x = detail_x - 1.5 + i * 1.0
        draw_box(ax, head_x, detail_y - 2.0, 0.7, 0.25, f'Head {i+1}', COLORS['attention'])

    ax.text(detail_x, detail_y - 2.5, '↓ Concatenate ↓', ha='center', fontsize=8)
    draw_box(ax, detail_x, detail_y - 2.9, 1.5, 0.3, 'Linear', COLORS['dense'])

    # Configuration box
    config_text = """Model Variants:
• Basic: d=64, h=4, N_enc=2, N_dec=2
• Deep: d=128, h=8, N_enc=6, N_dec=4
• Wide: d=256, h=8, N_enc=4, N_dec=4
• GELU: d=128, h=8, N_enc=4, N_dec=4
• Complex: d=256, h=16, N_enc=8, N_dec=6"""

    ax.text(0.5, 3.5, config_text, fontsize=7,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            verticalalignment='top', family='monospace')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Transformer architecture to: {output_path}")
    plt.close()


def draw_probsparse_architecture(output_path='figures/architecture_probsparse.png'):
    """Draw DDRSA-ProbSparse (Informer) Architecture"""

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(8, 11.5, 'DDRSA-ProbSparse (Informer) Architecture',
            ha='center', fontsize=16, fontweight='bold')

    # ===== ENCODER SIDE (ProbSparse) =====
    enc_x = 3.5

    # Input
    draw_box(ax, enc_x, 10.5, 1.5, 0.5, 'Input\nSequence\n(L, D)', COLORS['input'])
    draw_arrow(ax, enc_x, 10.25, enc_x, 9.8)

    # Input embedding
    draw_box(ax, enc_x, 9.5, 1.5, 0.4, 'Linear\nEmbedding', COLORS['embedding'])
    draw_arrow(ax, enc_x, 9.3, enc_x, 9.0)

    # Positional encoding
    draw_box(ax, enc_x, 8.7, 1.5, 0.4, 'Positional\nEncoding', COLORS['embedding'])
    draw_arrow(ax, enc_x, 8.5, enc_x, 8.1)

    # ProbSparse attention layers
    layer_positions = [7.5, 5.8, 4.1]
    seq_lengths = ['L', 'L/2', 'L/4']

    for i, (layer_y, seq_len) in enumerate(zip(layer_positions, seq_lengths)):
        # ProbSparse self-attention
        draw_box(ax, enc_x, layer_y + 0.3, 2.2, 0.5,
                f'ProbSparse\nSelf-Attention\n({seq_len})', COLORS['attention'])

        draw_arrow(ax, enc_x, layer_y + 0.05, enc_x, layer_y - 0.25)

        # Feed-forward
        draw_box(ax, enc_x, layer_y - 0.5, 1.8, 0.4,
                'Feed-Forward', COLORS['decoder'])

        if i < len(layer_positions) - 1:
            draw_arrow(ax, enc_x, layer_y - 0.7, enc_x, layer_y - 1.0)

            # Distilling operation
            draw_box(ax, enc_x, layer_y - 1.2, 1.5, 0.3,
                    f'Distilling\nConv1D', COLORS['embedding'])

            ax.text(enc_x + 1.5, layer_y - 1.05, f'{seq_len}→{seq_lengths[i+1]}',
                    fontsize=7, style='italic')

            draw_arrow(ax, enc_x, layer_y - 1.35, enc_x, layer_y - 1.6)

    # Encoder output
    draw_arrow(ax, enc_x, 3.3, enc_x, 2.9)
    draw_box(ax, enc_x, 2.6, 1.5, 0.4, 'Encoder\nOutput\n(L/4, d)', COLORS['encoder'])

    # ===== DECODER SIDE (Standard LSTM) =====
    dec_x = 12

    # Decoder input
    draw_box(ax, dec_x, 10.5, 1.5, 0.5, 'Decoder\nInput\n(zeros)', COLORS['input'])
    draw_arrow(ax, dec_x, 10.25, dec_x, 9.8)

    # LSTM decoder
    draw_box(ax, dec_x, 8.5, 2.0, 1.8,
            'LSTM\nDecoder\n\nLayers: N_dec\nHidden: H_dec\n+ Encoder Context',
            COLORS['decoder'])

    # Connection from encoder
    draw_arrow(ax, enc_x + 0.75, 2.6, dec_x - 1.0, 7.8, style='->')
    ax.text(7.5, 5.5, 'Cross-Attention\nto Encoder', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Decoder output
    draw_arrow(ax, dec_x, 7.6, dec_x, 7.2)
    draw_box(ax, dec_x, 6.9, 1.5, 0.4, 'Decoder\nOutput', COLORS['decoder'])

    # ===== OUTPUT HEADS =====
    draw_arrow(ax, dec_x, 6.7, dec_x, 6.3)

    # Three outputs
    for idx, (dy, name, activation) in enumerate([
        (5.8, 'λ(t)\nHazard', 'Sigmoid'),
        (5.0, 'S(t)\nSurvival', 'Sigmoid'),
        (4.2, 'E[TTE]', 'ReLU')
    ]):
        draw_box(ax, dec_x - 1.5, dy, 1.2, 0.4, name, COLORS['output'])
        draw_box(ax, dec_x + 1.5, dy, 1.0, 0.4, activation, COLORS['dense'])
        draw_arrow(ax, dec_x - 0.5, 6.2 if idx == 0 else dy + 0.6, dec_x - 2.1, dy + 0.2)
        draw_arrow(ax, dec_x - 0.9, dy, dec_x + 1.0, dy)

    # ===== PROBSPARSE ATTENTION DETAIL =====
    detail_x = 8
    detail_y = 10.5

    ax.text(detail_x, detail_y, 'ProbSparse Attention Mechanism',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Sparsity measurement
    sparsity_text = """1. Sparsity Measurement:
   M(q_i, K) = max(q_i·K^T) - (1/L)∑(q_i·K^T)

2. Select Top-u queries with highest M(q_i)
   u = c·ln(L)  →  O(L log L) complexity

3. Compute attention only for selected queries:
   Attention = softmax(Q̃K^T/√d_k)V
   where Q̃ contains only top-u queries"""

    ax.text(detail_x, detail_y - 0.5, sparsity_text, fontsize=7,
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Complexity comparison
    ax.text(detail_x, detail_y - 2.8, 'Complexity Comparison',
            ha='center', fontsize=10, fontweight='bold')

    complexity_text = """Standard Attention:    O(L²)
ProbSparse Attention:  O(L log L)

For L=512:  L² = 262,144  vs  L·log(L) ≈ 4,608
Speedup: ~57×"""

    ax.text(detail_x, detail_y - 3.2, complexity_text, fontsize=8,
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Distilling operation detail
    ax.text(detail_x, detail_y - 5.0, 'Distilling Operation',
            ha='center', fontsize=10, fontweight='bold')

    distill_text = """Purpose: Reduce sequence length between layers
Method:  Conv1D with stride=2 + MaxPool
Effect:  L → L/2 → L/4 → ...
Benefit: Further reduces computation"""

    ax.text(detail_x, detail_y - 5.4, distill_text, fontsize=8,
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Configuration box
    config_text = """Model Variants:
• Basic:
  - Encoder: d=512, h=8, N_enc=2
  - Decoder: LSTM H=128, N=2

• Deep:
  - Encoder: d=512, h=8, N_enc=4
  - Decoder: LSTM H=256, N=4

• Factor: c=5 (controls sparsity)"""

    ax.text(0.5, 2.5, config_text, fontsize=7,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            verticalalignment='top', family='monospace')

    # Key innovation box
    innovation_text = """Key Innovations:
✓ O(L log L) attention
✓ Distilling for sequence reduction
✓ Handles long sequences efficiently
✓ Best for: Long-range dependencies"""

    ax.text(15.5, 2.5, innovation_text, fontsize=7,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3),
            verticalalignment='top', ha='right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ProbSparse architecture to: {output_path}")
    plt.close()


def draw_loss_function_diagram(output_path='figures/architecture_loss.png'):
    """Draw Loss Function Architecture"""

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'DDRSA Loss Function Architecture',
            ha='center', fontsize=16, fontweight='bold')

    # Model outputs
    draw_box(ax, 7, 8.5, 2.5, 0.6, 'Model Outputs\nλ(t), S(t), E[TTE]', COLORS['output'])

    draw_arrow(ax, 5.5, 8.2, 3.5, 7.5)
    draw_arrow(ax, 8.5, 8.2, 10.5, 7.5)

    # Two loss components
    # Survival loss
    draw_box(ax, 3, 7.0, 2.5, 1.0,
            'Survival Loss\nL_survival\n\nNegative Log-Likelihood\nof Survival Function',
            COLORS['encoder'])

    survival_formula = r"""L_survival = -∑[
  δ_i · log(1 - S(t_i)) +
  (1-δ_i) · log(S(t_i))
]"""
    ax.text(3, 5.5, survival_formula, ha='center', fontsize=8,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Hazard loss
    draw_box(ax, 11, 7.0, 2.5, 1.0,
            'Hazard Loss\nL_hazard\n\nMSE between\nPredicted & True Hazards',
            COLORS['decoder'])

    hazard_formula = r"""L_hazard = MSE(
  λ_pred(t),
  λ_true(t)
)"""
    ax.text(11, 5.5, hazard_formula, ha='center', fontsize=8,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Combine with lambda
    draw_arrow(ax, 3, 5.2, 5, 4.5)
    draw_arrow(ax, 11, 5.2, 9, 4.5)

    draw_box(ax, 7, 4.0, 3.0, 0.7,
            'Combined DDRSA Loss\nL = λ·L_survival + (1-λ)·L_hazard',
            COLORS['dense'])

    # Lambda parameter
    ax.text(2, 4.0, 'λ', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.5))
    ax.text(1, 3.5, 'Balance\nparameter\n(0.5-0.9)', fontsize=7, ha='center')

    ax.text(12, 4.0, '(1-λ)', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.5))

    # Optional NASA loss
    draw_arrow(ax, 7, 3.65, 7, 3.2)

    draw_box(ax, 7, 2.7, 2.5, 0.7,
            'Optional:\nNASA Scoring Loss',
            COLORS['attention'])

    nasa_formula = r"""NASA_score = ∑[
  e^(-error/13) if error < 0 (early)
  e^(error/10)  if error ≥ 0 (late)
]

L_total = L + w_NASA · NASA_score"""

    ax.text(7, 1.5, nasa_formula, ha='center', fontsize=8,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # NASA weight
    ax.text(10.5, 2.7, 'w_NASA', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='orange', alpha=0.5))
    ax.text(10.5, 2.2, 'Weight\n(0.0-0.2)', fontsize=7, ha='center')

    # Backpropagation
    draw_arrow(ax, 7, 0.9, 7, 0.5)
    draw_box(ax, 7, 0.2, 2.0, 0.3, 'Backpropagation', COLORS['input'])

    # Explanation boxes
    explanation_1 = """Survival Loss:
• Penalizes incorrect survival probabilities
• Uses event indicator δ_i:
  - δ_i=1: Event occurred
  - δ_i=0: Censored (no event yet)"""

    ax.text(0.5, 8.5, explanation_1, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    explanation_2 = """Hazard Loss:
• Ensures hazard rates are realistic
• Prevents trivial solutions
• Improves interpretability"""

    ax.text(13.5, 8.5, explanation_2, fontsize=7, verticalalignment='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    explanation_3 = """NASA Loss:
• Asymmetric penalty
• Late predictions penalized MORE
• Early predictions penalized LESS
• Competition metric from PHM08"""

    ax.text(13.5, 3.5, explanation_3, fontsize=7, verticalalignment='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    # Lambda selection guide
    lambda_guide = """λ Selection Guide:
λ = 0.5:  Equal weight
λ = 0.75: Prefer survival (common)
λ = 0.9:  Strong survival focus
λ = 1.0:  Survival only"""

    ax.text(0.5, 3.5, lambda_guide, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Loss function diagram to: {output_path}")
    plt.close()


def draw_all_architectures():
    """Draw all architectural diagrams"""

    print("\n" + "="*80)
    print("DRAWING PUBLICATION-QUALITY ARCHITECTURAL DIAGRAMS")
    print("="*80 + "\n")

    # Create output directory
    os.makedirs('figures', exist_ok=True)

    # Draw each architecture
    print("Drawing architectures...\n")
    draw_rnn_architecture()
    draw_transformer_architecture()
    draw_probsparse_architecture()
    draw_loss_function_diagram()

    print("\n" + "="*80)
    print("✓ ALL DIAGRAMS COMPLETE")
    print("="*80)
    print(f"\nDiagrams saved to: {os.path.abspath('figures')}/")
    print("  - architecture_rnn.png         (DDRSA-RNN/LSTM/GRU)")
    print("  - architecture_transformer.png (DDRSA-Transformer)")
    print("  - architecture_probsparse.png  (DDRSA-ProbSparse/Informer)")
    print("  - architecture_loss.png        (Loss Function)")
    print("\nAll diagrams are publication-ready at 300 DPI")
    print("="*80 + "\n")


if __name__ == '__main__':
    draw_all_architectures()
