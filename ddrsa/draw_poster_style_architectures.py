"""
Draw Poster-Style Architectural Diagrams for DDRSA Models
Matches the style from the research poster
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os


# Poster color scheme (matching the original)
COLORS = {
    'lstm': '#2E86AB',        # Blue for LSTM
    'attention': '#A23B72',   # Purple for attention
    'ffn': '#F18F01',         # Orange for FFN
    'decoder': '#C73E1D',     # Red for decoder
    'light_blue': '#AED9E0',  # Light blue background
    'light_green': '#B8E6B8', # Light green
    'light_orange': '#FFE5CC',# Light orange
    'white': '#FFFFFF',       # White
    'arrow': '#333333',       # Dark gray arrows
    'text': '#000000'         # Black text
}


def draw_rounded_box(ax, x, y, width, height, text, color, textsize=10, bold=True):
    """Draw rounded rectangle box with text (poster style)"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2),
        width, height,
        boxstyle="round,pad=0.08",
        linewidth=2,
        edgecolor='#333333',
        facecolor=color,
        alpha=0.9,
        zorder=2
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center',
            fontsize=textsize, fontweight=weight, zorder=3,
            color='white' if color in [COLORS['lstm'], COLORS['attention'], COLORS['decoder']] else 'black')


def draw_thick_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw thick arrow (poster style)"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color=COLORS['arrow'],
        linewidth=2.5,
        mutation_scale=25,
        zorder=1
    )
    ax.add_patch(arrow)

    # Add label if provided
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def draw_ddrsa_original(output_path='figures/poster_ddrsa_original.png'):
    """Draw original DDRSA architecture (poster style)"""

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'DDRSA Architecture', fontsize=18, fontweight='bold', ha='center')

    # ===== ENCODER SECTION =====
    ax.text(3, 6.8, 'Encoder', fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.5))

    # Input sequence
    for i, x_pos in enumerate([1.5, 2.5, 4.5]):
        label = f'$X_{i+1}$' if i < 2 else '$X_j$'
        draw_rounded_box(ax, x_pos, 6, 0.6, 0.4, label, COLORS['light_blue'], textsize=11)

    # Dots between X2 and Xj
    ax.text(3.5, 6, '...', fontsize=16, ha='center', va='center')

    # LSTM encoder cells
    for i, x_pos in enumerate([1.5, 2.5, 4.5]):
        draw_rounded_box(ax, x_pos, 5, 0.8, 0.5, 'LSTM', COLORS['lstm'], textsize=10)
        # Arrow from input to LSTM
        draw_thick_arrow(ax, x_pos, 5.8, x_pos, 5.25)

    # Connections between LSTM cells
    draw_thick_arrow(ax, 1.9, 5, 2.1, 5)
    ax.text(2.8, 5, '...', fontsize=16, ha='center', va='center')
    draw_thick_arrow(ax, 3.3, 5, 4.1, 5)

    # Hidden state Zj
    draw_thick_arrow(ax, 4.5, 4.75, 4.5, 4.3)
    draw_rounded_box(ax, 4.5, 3.9, 1.2, 0.5, 'Hidden State\n$Z_j$', COLORS['light_green'], textsize=10)

    # ===== DECODER SECTION =====
    ax.text(10, 6.8, 'Decoder', fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['light_orange'], alpha=0.5))

    # Context vector replication
    draw_thick_arrow(ax, 5.1, 3.9, 7, 3.9)
    draw_rounded_box(ax, 7.5, 3.9, 0.6, 0.4, '$Z_j$', COLORS['light_green'], textsize=10)

    # LSTM decoder cells (3 shown)
    for i, x_pos in enumerate([8.5, 9.5, 10.5]):
        # Context to each decoder cell
        draw_thick_arrow(ax, 7.5, 3.65, x_pos, 3.3)

        # LSTM decoder
        draw_rounded_box(ax, x_pos, 2.8, 0.8, 0.5, 'LSTM', COLORS['lstm'], textsize=10)

        # Connections
        if i > 0:
            draw_thick_arrow(ax, x_pos - 0.6, 2.8, x_pos - 0.4, 2.8)

        # Output layers
        draw_thick_arrow(ax, x_pos, 2.55, x_pos, 2.15)

        # Q_j outputs
        draw_rounded_box(ax, x_pos, 1.8, 0.7, 0.4, f'$Q_j({i})$', COLORS['light_blue'], textsize=9)

        # FFN
        draw_thick_arrow(ax, x_pos, 1.6, x_pos, 1.25)
        draw_rounded_box(ax, x_pos, 1.0, 0.7, 0.3, 'FFN', COLORS['ffn'], textsize=9)

        # Sigmoid
        draw_thick_arrow(ax, x_pos, 0.85, x_pos, 0.55)
        draw_rounded_box(ax, x_pos, 0.3, 0.8, 0.35, f'$h_j({i})$\nσ', COLORS['decoder'], textsize=8)

    # L_max notation
    draw_rounded_box(ax, 11.5, 2.8, 0.8, 0.5, 'LSTM', COLORS['lstm'], textsize=10)
    draw_rounded_box(ax, 11.5, 1.8, 0.7, 0.4, f'$Q_j(L_{{max}})$', COLORS['light_blue'], textsize=9)
    draw_rounded_box(ax, 11.5, 1.0, 0.7, 0.3, 'FFN', COLORS['ffn'], textsize=9)
    draw_rounded_box(ax, 11.5, 0.3, 0.8, 0.35, f'$h_j(L_{{max}})$\nσ', COLORS['decoder'], textsize=8)

    ax.text(11, 2.8, '...', fontsize=16, ha='center', va='center')

    # Legend
    legend_text = """Loss: Same as DRSA (Ren et al)
Models Hazard Rate functions $h_j(·)$"""
    ax.text(0.5, 1.5, legend_text, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved original DDRSA (poster style) to: {output_path}")
    plt.close()


def draw_ddrsa_probsparse(output_path='figures/poster_ddrsa_probsparse.png'):
    """Draw DDRSA with ProbSparse attention (poster style)"""

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Modified DDRSA with ProbSparse Attention', fontsize=18, fontweight='bold', ha='center')

    # ===== ENCODER (PROBSPARSE) =====
    ax.text(3, 8.8, 'Encoder', fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.5))

    # Input sequence
    enc_x = 3
    draw_rounded_box(ax, enc_x, 8.2, 2.5, 0.5, '$X_1, X_2, ..., X_j$', COLORS['light_blue'], textsize=11)

    draw_thick_arrow(ax, enc_x, 7.95, enc_x, 7.5)

    # Context Vector
    draw_rounded_box(ax, enc_x, 7.2, 1.8, 0.4, 'Context Vector\n$Z$', COLORS['light_green'], textsize=10)

    draw_thick_arrow(ax, enc_x, 7.0, enc_x, 6.6)

    # ProbSparse Attention layers
    layer_y = [6.2, 4.8, 3.4]
    for i, y in enumerate(layer_y):
        # ProbSparse Attention
        draw_rounded_box(ax, enc_x, y, 2.2, 0.5, 'Prob-Sparse Attention', COLORS['attention'], textsize=9)

        draw_thick_arrow(ax, enc_x, y - 0.25, enc_x, y - 0.55)

        # Feed Forward
        draw_rounded_box(ax, enc_x, y - 0.85, 1.8, 0.4, 'Feed Forward', COLORS['ffn'], textsize=9)

        if i < len(layer_y) - 1:
            draw_thick_arrow(ax, enc_x, y - 1.05, enc_x, y - 1.35)

    # Output to decoder
    draw_thick_arrow(ax, enc_x, 2.55, enc_x, 2.2)
    draw_rounded_box(ax, enc_x, 1.9, 1.2, 0.4, 'Encoder\nOutput', COLORS['light_green'], textsize=9)

    # ===== DECODER (LSTM) =====
    dec_x = 10
    ax.text(dec_x, 8.8, 'Decoder', fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['light_orange'], alpha=0.5))

    # LSTM decoder stack
    draw_rounded_box(ax, dec_x, 5.5, 2.5, 2.5, 'LSTM\nDecoder\n\nLayers: $N_{dec}$\nHidden: $H_{dec}$',
                    COLORS['lstm'], textsize=10)

    # Connection from encoder
    draw_thick_arrow(ax, 3.6, 1.9, 9, 4.5, label='Context')

    # Outputs
    draw_thick_arrow(ax, dec_x, 4.25, dec_x, 3.5)

    # Three output branches
    output_positions = [8.5, 10, 11.5]
    output_labels = ['$h_j(0)$', '$h_j(1)$', '$h_j(L_{max})$']

    for pos, label in zip(output_positions, output_labels):
        draw_thick_arrow(ax, dec_x, 3.2, pos, 2.8)
        draw_rounded_box(ax, pos, 2.5, 1.0, 0.4, label + '\nσ', COLORS['decoder'], textsize=9)

    # Key features box
    features_text = """Key Features:
• O(N logN) computation in encoder
• ProbSparse attention mechanism
• Same decoder as original DDRSA
• Same loss function"""

    ax.text(0.5, 5, features_text, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Advantages
    advantages_text = """Advantages over RNN:
✓ No forgetting over long sequences
✓ Attention weights for covariates
✓ Efficient O(N logN) complexity"""

    ax.text(13.5, 5, advantages_text, fontsize=9, verticalalignment='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved ProbSparse DDRSA (poster style) to: {output_path}")
    plt.close()


def draw_probsparse_mechanism(output_path='figures/poster_probsparse_mechanism.png'):
    """Draw ProbSparse 
     mechanism detail (poster style)"""

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(8, 5.5, 'ProbSparse Attention Mechanism', fontsize=16, fontweight='bold', ha='center')

    # Step 1: Input
    step_x = 1
    draw_rounded_box(ax, step_x, 4.5, 1.5, 0.5, 'Query Q\n$L_Q$ length', COLORS['light_blue'], textsize=9)
    draw_rounded_box(ax, step_x, 3.5, 1.5, 0.5, 'Key K\n$L_K$ length', COLORS['light_blue'], textsize=9)
    draw_rounded_box(ax, step_x, 2.5, 1.5, 0.5, 'Value V', COLORS['light_blue'], textsize=9)

    ax.text(step_x, 1.8, 'Input', fontsize=10, ha='center', fontweight='bold')

    # Arrow
    draw_thick_arrow(ax, step_x + 0.75, 4, step_x + 1.5, 4)

    # Step 2: Sparsity measurement
    step_x = 4
    draw_rounded_box(ax, step_x, 4.5, 2.0, 0.8,
                    'Sparsity\nMeasurement\n$M(q_i, K)$',
                    COLORS['attention'], textsize=9)

    formula = r"$\dot{L}_Q = f \cdot \log(L_Q)$"
    ax.text(step_x, 3.3, formula, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(step_x, 1.8, 'Sample Top-$\dot{L}_Q$', fontsize=10, ha='center', fontweight='bold')

    # Arrow
    draw_thick_arrow(ax, step_x + 1.0, 4.5, step_x + 1.8, 4.5)

    # Step 3: Query selection
    step_x = 7.5
    draw_rounded_box(ax, step_x, 4.5, 1.8, 0.8,
                    'Top $\dot{L}_Q$\nQuery\nSelection',
                    COLORS['ffn'], textsize=9)

    draw_rounded_box(ax, step_x, 3.0, 1.5, 0.5, '$\widetilde{Q}$\n(Selected)', COLORS['light_orange'], textsize=9)

    ax.text(step_x, 1.8, 'Selected Queries', fontsize=10, ha='center', fontweight='bold')

    # Arrow
    draw_thick_arrow(ax, step_x + 0.9, 4.5, step_x + 1.7, 4.5)

    # Step 4: Attention computation
    step_x = 11
    draw_rounded_box(ax, step_x, 4.5, 2.2, 0.8,
                    'Attention\nComputation\nOnly on $\widetilde{Q}$',
                    COLORS['attention'], textsize=9)

    formula2 = r"$\text{softmax}\left(\frac{\widetilde{Q}K^T}{\sqrt{d_k}}\right)V$"
    ax.text(step_x, 3.0, formula2, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(step_x, 1.8, 'Sparse Attention', fontsize=10, ha='center', fontweight='bold')

    # Arrow
    draw_thick_arrow(ax, step_x + 1.1, 4.5, step_x + 1.9, 4.5)

    # Step 5: Output
    step_x = 14.5
    draw_rounded_box(ax, step_x, 4.5, 1.5, 0.8,
                    'Output\nVectors',
                    COLORS['light_green'], textsize=9)

    ax.text(step_x, 3.0, '$A$ (Attention)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax.text(step_x, 1.8, 'Result', fontsize=10, ha='center', fontweight='bold')

    # Complexity note
    complexity_text = """Complexity: O(N logN)
vs Standard O(N²)

For L=512:
• Standard: 262,144 ops
• ProbSparse: ~4,608 ops
• Speedup: 57×"""

    ax.text(8, 0.8, complexity_text, fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved ProbSparse mechanism (poster style) to: {output_path}")
    plt.close()


def draw_all_poster_diagrams():
    """Draw all poster-style diagrams"""

    print("\n" + "="*80)
    print("DRAWING POSTER-STYLE ARCHITECTURAL DIAGRAMS")
    print("="*80 + "\n")

    os.makedirs('figures', exist_ok=True)

    print("Drawing diagrams in poster style...\n")
    draw_ddrsa_original()
    draw_ddrsa_probsparse()
    draw_probsparse_mechanism()

    print("\n" + "="*80)
    print("✓ ALL POSTER-STYLE DIAGRAMS COMPLETE")
    print("="*80)
    print(f"\nDiagrams saved to: {os.path.abspath('figures')}/")
    print("  - poster_ddrsa_original.png      (Original DDRSA)")
    print("  - poster_ddrsa_probsparse.png    (Modified DDRSA)")
    print("  - poster_probsparse_mechanism.png (Attention Detail)")
    print("\nAll diagrams match the poster style!")
    print("="*80 + "\n")


if __name__ == '__main__':
    draw_all_poster_diagrams()
