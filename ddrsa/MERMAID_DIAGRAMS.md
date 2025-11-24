# Mermaid Diagrams for DDRSA Architectures

This file contains Mermaid diagram code for all DDRSA architectures. You can render these in:
- GitHub (native support)
- Mermaid Live Editor: https://mermaid.live/
- Documentation sites (GitBook, MkDocs, etc.)
- VS Code with Mermaid extension

---

## 1. Original DDRSA Architecture

```mermaid
graph TB
    subgraph Encoder
        X1[X₁] --> LSTM1[LSTM]
        X2[X₂] --> LSTM2[LSTM]
        Xj[Xⱼ] --> LSTMj[LSTM]

        LSTM1 --> LSTM2
        LSTM2 -.-> LSTMj

        LSTMj --> Z[Hidden State Zⱼ]
    end

    subgraph Decoder
        Z --> Z1[Zⱼ]
        Z --> Z2[Zⱼ]
        Z --> Z3[Zⱼ]

        Z1 --> DLSTM1[LSTM]
        Z2 --> DLSTM2[LSTM]
        Z3 --> DLSTM3[LSTM]

        DLSTM1 --> DLSTM2
        DLSTM2 --> DLSTM3

        DLSTM1 --> Q0[Qⱼ0]
        DLSTM2 --> Q1[Qⱼ1]
        DLSTM3 --> QL[QⱼLmax]

        Q0 --> FFN0[FFN]
        Q1 --> FFN1[FFN]
        QL --> FFNL[FFN]

        FFN0 --> H0[hⱼ0<br/>Sigmoid]
        FFN1 --> H1[hⱼ1<br/>Sigmoid]
        FFNL --> HL[hⱼLmax<br/>Sigmoid]
    end

    style X1 fill:#AED9E0
    style X2 fill:#AED9E0
    style Xj fill:#AED9E0
    style LSTM1 fill:#2E86AB,color:#fff
    style LSTM2 fill:#2E86AB,color:#fff
    style LSTMj fill:#2E86AB,color:#fff
    style Z fill:#B8E6B8
    style Z1 fill:#B8E6B8
    style Z2 fill:#B8E6B8
    style Z3 fill:#B8E6B8
    style DLSTM1 fill:#2E86AB,color:#fff
    style DLSTM2 fill:#2E86AB,color:#fff
    style DLSTM3 fill:#2E86AB,color:#fff
    style FFN0 fill:#F18F01,color:#fff
    style FFN1 fill:#F18F01,color:#fff
    style FFNL fill:#F18F01,color:#fff
    style H0 fill:#C73E1D,color:#fff
    style H1 fill:#C73E1D,color:#fff
    style HL fill:#C73E1D,color:#fff
```

---

## 2. Modified DDRSA with ProbSparse Attention

```mermaid
graph TB
    subgraph "Encoder (ProbSparse)"
        Input[X₁, X₂, ..., Xⱼ] --> Context[Context Vector Z]
        Context --> PSA1[ProbSparse<br/>Attention]
        PSA1 --> FF1[Feed Forward]
        FF1 --> PSA2[ProbSparse<br/>Attention]
        PSA2 --> FF2[Feed Forward]
        FF2 --> PSA3[ProbSparse<br/>Attention]
        PSA3 --> FF3[Feed Forward]
        FF3 --> EncOut[Encoder Output]
    end

    subgraph "Decoder (LSTM)"
        EncOut --> DLSTM[LSTM Decoder<br/><br/>Layers: Ndec<br/>Hidden: Hdec]
        DLSTM --> Out1[hⱼ0<br/>Sigmoid]
        DLSTM --> Out2[hⱼ1<br/>Sigmoid]
        DLSTM --> Out3[hⱼLmax<br/>Sigmoid]
    end

    style Input fill:#AED9E0
    style Context fill:#B8E6B8
    style PSA1 fill:#A23B72,color:#fff
    style PSA2 fill:#A23B72,color:#fff
    style PSA3 fill:#A23B72,color:#fff
    style FF1 fill:#F18F01,color:#fff
    style FF2 fill:#F18F01,color:#fff
    style FF3 fill:#F18F01,color:#fff
    style EncOut fill:#B8E6B8
    style DLSTM fill:#2E86AB,color:#fff
    style Out1 fill:#C73E1D,color:#fff
    style Out2 fill:#C73E1D,color:#fff
    style Out3 fill:#C73E1D,color:#fff
```

---

## 3. ProbSparse Attention Mechanism

```mermaid
graph LR
    subgraph Input
        Q[Query Q<br/>LQ length]
        K[Key K<br/>LK length]
        V[Value V]
    end

    subgraph "Sparsity Measurement"
        Q --> SM[Sparsity<br/>Measurement<br/>Mqi K]
        SM --> Sample[Sample Top-L̇Q<br/>L̇Q = f·logLQ]
    end

    subgraph "Query Selection"
        Sample --> QS[Top L̇Q<br/>Query<br/>Selection]
        QS --> QTilde[Q̃<br/>Selected Queries]
    end

    subgraph "Attention Computation"
        QTilde --> Attn[Attention<br/>softmaxQ̃KT/√dk·V]
        K --> Attn
        V --> Attn
    end

    subgraph Output
        Attn --> Out[Output<br/>Vectors A]
    end

    style Q fill:#AED9E0
    style K fill:#AED9E0
    style V fill:#AED9E0
    style SM fill:#A23B72,color:#fff
    style Sample fill:#A23B72,color:#fff
    style QS fill:#F18F01,color:#fff
    style QTilde fill:#FFE5CC
    style Attn fill:#A23B72,color:#fff
    style Out fill:#B8E6B8
```

---

## 4. Complete DDRSA Flow (Simplified)

```mermaid
flowchart TD
    Start([Time Series Input<br/>X₁, X₂, ..., Xⱼ]) --> Enc{Model Type?}

    Enc -->|RNN| RNNEnc[LSTM/GRU<br/>Encoder]
    Enc -->|Transformer| TransEnc[ProbSparse<br/>Encoder]

    RNNEnc --> Hidden[Hidden State Zⱼ]
    TransEnc --> Hidden

    Hidden --> Dec{Decoder Type}

    Dec -->|RNN| RNNDec[LSTM/GRU<br/>Decoder]
    Dec -->|Both| TransDec[LSTM<br/>Decoder]

    RNNDec --> Outputs
    TransDec --> Outputs

    Outputs[Output Heads] --> H0[hⱼ0<br/>Hazard]
    Outputs --> H1[hⱼ1<br/>Hazard]
    Outputs --> HL[hⱼLmax<br/>Hazard]

    H0 --> Loss[DDRSA Loss]
    H1 --> Loss
    HL --> Loss

    Loss --> Train{Training}
    Train -->|Backprop| Start
    Train -->|Done| Policy[Optimal<br/>Intervention<br/>Policy φ*]

    style Start fill:#AED9E0
    style RNNEnc fill:#2E86AB,color:#fff
    style TransEnc fill:#A23B72,color:#fff
    style Hidden fill:#B8E6B8
    style RNNDec fill:#2E86AB,color:#fff
    style TransDec fill:#2E86AB,color:#fff
    style H0 fill:#C73E1D,color:#fff
    style H1 fill:#C73E1D,color:#fff
    style HL fill:#C73E1D,color:#fff
    style Loss fill:#F18F01,color:#fff
    style Policy fill:#B8E6B8
```

---

## 5. Loss Function Architecture

```mermaid
graph TB
    Model[Model Outputs<br/>hⱼt, Sⱼt] --> SurvLoss[Survival Loss<br/>Lsurvival]
    Model --> HazLoss[Hazard Loss<br/>Lhazard]

    SurvLoss --> Combine[Combined Loss<br/>L = λ·Lsurvival + 1-λ·Lhazard]
    HazLoss --> Combine

    Lambda[λ<br/>Balance Parameter<br/>0.5 - 0.9] -.-> Combine

    Combine --> Optional{Add NASA Loss?}

    Optional -->|Yes| NASA[NASA Scoring Loss<br/>Asymmetric Penalty]
    Optional -->|No| Final

    NASA --> Final[Total Loss<br/>L + wNASA·NASAscore]

    Final --> BP[Backpropagation]

    style Model fill:#FFD4D4
    style SurvLoss fill:#B8E6F0
    style HazLoss fill:#D4E8D4
    style Combine fill:#F0E68C
    style Lambda fill:#FFE5CC
    style NASA fill:#FFE8CC
    style Final fill:#F18F01,color:#fff
    style BP fill:#AED9E0
```

---

## 6. Training Pipeline

```mermaid
flowchart LR
    subgraph Data
        Raw[Raw Time Series<br/>Turbofan/Azure PM] --> Preprocess[Preprocessing<br/>Scaling, Windowing]
        Preprocess --> Split[Train/Val/Test<br/>Split]
    end

    subgraph Model
        Split --> Init[Initialize<br/>DDRSA Model]
        Init --> Forward[Forward Pass<br/>Compute Hazards]
        Forward --> Loss[Compute Loss<br/>DDRSA + NASA]
        Loss --> Backward[Backward Pass<br/>Update Weights]
        Backward --> Check{Early<br/>Stopping?}
        Check -->|No| Forward
        Check -->|Yes| Best[Best Model<br/>Checkpoint]
    end

    subgraph Evaluation
        Best --> Test[Test Set<br/>Evaluation]
        Test --> Metrics[Metrics<br/>MAE, C-Index<br/>NASA Score]
        Metrics --> Policy[Compute<br/>Optimal Policy φ*]
    end

    style Raw fill:#AED9E0
    style Init fill:#2E86AB,color:#fff
    style Forward fill:#A23B72,color:#fff
    style Loss fill:#F18F01,color:#fff
    style Best fill:#B8E6B8
    style Metrics fill:#FFE5CC
    style Policy fill:#C73E1D,color:#fff
```

---

## 7. Optimal Intervention Policy

```mermaid
graph TD
    Start([Monitoring<br/>Covariates Xⱼ]) --> Compute[Compute<br/>hⱼt using DDRSA]

    Compute --> CalcT[Calculate Tⱼ<br/>Expected Residual Time]

    CalcT --> CalcV[Calculate V'ⱼXⱼ<br/>Min Expected Cost]

    CalcV --> Compare{Tⱼ ≤ V'ⱼXⱼ?}

    Compare -->|Yes| Intervene[Intervene Now<br/>φⱼXⱼ = 1]
    Compare -->|No| Continue[Continue Monitoring<br/>φⱼXⱼ = 0]

    Continue --> Next[Next Time Step<br/>j ← j+1]
    Next --> Compute

    Intervene --> End([Intervention<br/>Applied])

    style Start fill:#AED9E0
    style Compute fill:#2E86AB,color:#fff
    style CalcT fill:#A23B72,color:#fff
    style CalcV fill:#F18F01,color:#fff
    style Intervene fill:#C73E1D,color:#fff
    style Continue fill:#B8E6B8
    style End fill:#FFE5CC
```

---

## 8. Transformer Encoder-Decoder (Detailed)

```mermaid
graph TB
    subgraph Encoder
        E_Input[Input Sequence] --> E_Embed[Linear<br/>Embedding]
        E_Embed --> E_Pos[Positional<br/>Encoding]
        E_Pos --> E_Attn1[Multi-Head<br/>Self-Attention]
        E_Attn1 --> E_Norm1[Add & Norm]
        E_Norm1 --> E_FF1[Feed-Forward<br/>Network]
        E_FF1 --> E_Norm2[Add & Norm]
        E_Norm2 -.-> E_AttnN[N encoder layers]
        E_AttnN --> E_Out[Encoder<br/>Output]
    end

    subgraph Decoder
        D_Input[Decoder Input] --> D_Embed[Linear<br/>Embedding]
        D_Embed --> D_Pos[Positional<br/>Encoding]
        D_Pos --> D_Attn1[Masked<br/>Self-Attention]
        D_Attn1 --> D_Norm1[Add & Norm]
        D_Norm1 --> D_Cross[Cross-Attention<br/>Q from Dec, K V from Enc]
        E_Out --> D_Cross
        D_Cross --> D_Norm2[Add & Norm]
        D_Norm2 --> D_FF[Feed-Forward<br/>Network]
        D_FF --> D_Norm3[Add & Norm]
        D_Norm3 -.-> D_AttnN[N decoder layers]
        D_AttnN --> D_Out[Decoder<br/>Output]
    end

    D_Out --> Out1[λt<br/>Hazard]
    D_Out --> Out2[St<br/>Survival]
    D_Out --> Out3[E[TTE]<br/>Expected]

    style E_Input fill:#AED9E0
    style E_Embed fill:#E8D4F8
    style E_Pos fill:#E8D4F8
    style E_Attn1 fill:#FFE8CC
    style E_FF1 fill:#F18F01,color:#fff
    style E_Out fill:#B8E6B8
    style D_Input fill:#AED9E0
    style D_Embed fill:#E8D4F8
    style D_Pos fill:#E8D4F8
    style D_Attn1 fill:#FFE8CC
    style D_Cross fill:#A23B72,color:#fff
    style D_FF fill:#F18F01,color:#fff
    style D_Out fill:#B8E6B8
    style Out1 fill:#C73E1D,color:#fff
    style Out2 fill:#C73E1D,color:#fff
    style Out3 fill:#C73E1D,color:#fff
```

---

## Usage Instructions

### In GitHub README

Simply paste the code blocks into your `README.md`:

```markdown
## Architecture Diagram

```mermaid
graph TB
    ...
```
```

### In Mermaid Live Editor

1. Go to https://mermaid.live/
2. Paste any diagram code
3. Export as PNG/SVG/PDF

### In VS Code

1. Install "Markdown Preview Mermaid Support" extension
2. Open this file
3. Click "Preview" to see diagrams

### In Documentation Sites

Most modern documentation frameworks (MkDocs, GitBook, Docusaurus) support Mermaid natively.

---

## Customization

### Change Colors

Modify the `style` lines at the end of each diagram:

```mermaid
style NodeName fill:#ColorCode,color:#TextColor
```

### Add More Nodes

Add new nodes following the pattern:

```mermaid
NodeID[Display Text] --> NextNode[Next Step]
```

### Change Layout Direction

- `graph TB` - Top to Bottom
- `graph LR` - Left to Right
- `graph BT` - Bottom to Top
- `graph RL` - Right to Left

---

## Tips for Best Results

1. **Keep it Simple**: Mermaid works best with clear, simple diagrams
2. **Use Subgraphs**: Group related components together
3. **Color Code**: Use consistent colors for component types
4. **Add Legends**: Include style definitions for clarity
5. **Test Rendering**: Preview in Mermaid Live before committing

---

## Exporting

### To PNG/SVG (Mermaid Live)
1. Open diagram in https://mermaid.live/
2. Click "Actions" → "PNG/SVG"
3. Download

### To PDF (Mermaid CLI)
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i MERMAID_DIAGRAMS.md -o output.pdf
```

### To HTML
```bash
pandoc MERMAID_DIAGRAMS.md -o diagrams.html --filter mermaid-filter
```

---

## Comparison: Mermaid vs Python

| Feature | Mermaid | Python (Matplotlib) |
|---------|---------|---------------------|
| **Ease of use** | ✅ Very easy | ⚠️ Moderate |
| **Customization** | ⚠️ Limited | ✅ Unlimited |
| **Resolution** | ⚠️ Depends on export | ✅ 300+ DPI |
| **Interactivity** | ✅ Can be interactive | ❌ Static |
| **Version control** | ✅ Text-based | ❌ Binary images |
| **Auto-update** | ✅ Edit text, auto-renders | ❌ Re-run script |

**Recommendation**:
- Use **Mermaid** for documentation, GitHub, quick drafts
- Use **Python** for publications, posters, high-quality figures

---

## All Diagrams Combined

For a complete overview, you can render all diagrams in sequence or combine them into a single comprehensive diagram showing the entire DDRSA system from data input to optimal policy output.
