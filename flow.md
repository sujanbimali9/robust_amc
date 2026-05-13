# Project Flow: SNR-Adaptive AMC using Mixture of Experts

> A complete start-to-end walkthrough of how this project works — from raw I/Q signals to a final modulation prediction.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Project Structure](#2-project-structure)
3. [Configuration (`config/config.py`)](#3-configuration)
4. [Data Pipeline](#4-data-pipeline)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline (`train.py`)](#6-training-pipeline)
7. [Evaluation Pipeline (`evaluate.py`)](#7-evaluation-pipeline)
8. [Inference & Testing](#8-inference--testing)
9. [Web API (`app/main.py`)](#9-web-api)
10. [Defense Plot Generation](#10-defense-plot-generation)
11. [End-to-End Data Flow Diagram](#11-end-to-end-data-flow-diagram)

---

## 1. High-Level Overview

**Goal:** Automatically classify the modulation scheme (e.g. BPSK, QPSK, QAM16…) of a received radio signal, even under varying noise conditions.

**Key Insight:** A single classifier struggles across all SNR levels. This project uses a **Mixture of Experts (MoE)** — three specialized classifiers, each trained for a different SNR regime (low/mid/high), combined via a learned gating network.

```
Raw I/Q Signal
     │
     ▼
┌─────────────────┐
│  SNR Estimator   │──── Predicts: is this low, mid, or high SNR?
└────────┬────────┘
         │ SNR logits
         ▼
┌─────────────────┐        ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Gating Network  │───────▶│ Expert 0 │ │ Expert 1 │ │ Expert 2 │
│ (+ raw signal)   │        │  (Low)   │ │  (Mid)   │ │  (High)  │
└─────────────────┘        └────┬─────┘ └────┬─────┘ └────┬─────┘
                                │             │             │
                                ▼             ▼             ▼
                          ┌─────────────────────────────────────┐
                          │   Weighted Sum of Expert Outputs     │
                          └──────────────┬──────────────────────┘
                                         │
                                         ▼
                                  Final Prediction
                              (e.g. "QPSK", 94% conf)
```

---

## 2. Project Structure

```
robust_amc/
├── config/
│   └── config.py              # All hyperparameters, paths, SNR ranges
├── data/
│   ├── generator.py           # Synthetic I/Q signal generation (BPSK, QPSK, etc.)
│   ├── dataset.py             # PyTorch Dataset classes + RML2016.10a loader
│   └── sdr_output.dat         # Real SDR capture data
├── models/
│   ├── snr_estimator.py       # CNN that classifies SNR into bins
│   ├── expert_networks.py     # ExpertCNN with residual blocks + SE attention
│   ├── gating_network.py      # Soft/hard/signal-aware gating networks
│   └── moe_amc.py             # Full MoE system combining all components
├── utils/
│   ├── dataset.py             # KaggleHub dataset downloader
│   ├── metrics.py             # Accuracy, F1, confusion matrix helpers
│   └── visualization.py       # Plotting utilities
├── train.py                   # 3-stage training pipeline
├── evaluate.py                # Test set evaluation + plot generation
├── test.py                    # Interactive single-signal testing
├── test_sdr.py                # Classify real SDR capture data
├── generate_defense_plots.py  # Publication-quality defense plots
├── app/
│   ├── main.py                # FastAPI web server
│   └── templates/index.html   # Web UI
├── checkpoints/               # Saved model weights (.pth files)
└── results/                   # Generated plots (PNG)
```

---

## 3. Configuration

**File:** `config/config.py` → class `Config`

This is the single source of truth for the entire project. Key parameters:

| Parameter | Value | Purpose |
|---|---|---|
| `DATA_SOURCE` | `'rml'` or `'generated'` | Toggle between RadioML dataset and synthetic data |
| `MODULATIONS` | 11 classes (RML) or 8 classes (generated) | Target modulation schemes |
| `SAMPLE_LENGTH` | 128 (RML) or 1024 (generated) | Number of I/Q samples per signal |
| `SNR_RANGE` | (-20, 18) dB | Full SNR range |
| `SNR_LOW` | (-20, 2) dB | Low-SNR expert regime |
| `SNR_MID` | (-2, 12) dB | Mid-SNR expert regime (overlaps!) |
| `SNR_HIGH` | (8, 20) dB | High-SNR expert regime (overlaps!) |
| `NUM_EXPERTS` | 3 | Number of expert networks |
| `BATCH_SIZE` | 512 | Training batch size |

**Important:** The SNR ranges intentionally **overlap** (e.g., SNR=5 dB falls in both low and mid). This helps experts generalize at SNR boundaries.

**`get_snr_bin(snr)`** — Converts a continuous SNR value to a discrete bin index (0=low, 1=mid, 2=high) using hard thresholds at 2 dB and 12 dB.

---

## 4. Data Pipeline

### 4.1 Data Sources

#### Option A: RadioML 2016.10a (`DATA_SOURCE = 'rml'`)

**File:** `data/dataset.py` → `load_rml_data()`

1. Loads `RML2016.10a_dict.pkl` — a pickle file keyed by `(modulation_name, snr_value)`.
2. Each key maps to a `(1000, 2, 128)` array — 1000 samples, 2 channels (I and Q), 128 time steps.
3. Converts from 2-channel real to **complex representation**: `X_complex = I + j*Q` → shape `(1000, 128)`.
4. Builds label arrays by mapping modulation names to integer indices.
5. Returns: `X` (complex signals), `labels` (int), `snr_vals` (float), `mods` (list of names).

**Total:** 220,000 samples (11 mods × 20 SNR values × 1000 samples each).

#### Option B: Synthetic Data (`DATA_SOURCE = 'generated'`)

**File:** `data/generator.py` → class `SignalGenerator`

Generates I/Q signals from scratch for 8 modulation types:

| Modulation | Method |
|---|---|
| BPSK | Binary symbols {-1, +1}, pulse-shaped |
| QPSK | 4-point constellation on unit circle |
| 8PSK | 8-point constellation on unit circle |
| QAM16 | 4×4 grid, normalized by √10 |
| QAM64 | 8×8 grid, normalized by √42 |
| GFSK | Gaussian-filtered FSK with BT=0.5 |
| CPFSK | Continuous-phase FSK, mod_index=0.5 |
| PAM4 | 4-level pulse amplitude modulation |

**Channel effects** are applied in sequence:
1. **Fading** (optional): Rayleigh (flat fading, random complex gain per sample) or Rician (LOS + scattered component).
2. **AWGN**: Additive White Gaussian Noise calibrated to the target SNR in dB.
3. **Normalization**: Signal divided by its max absolute value.

#### Dataset Download

**File:** `utils/dataset.py` — Uses `kagglehub` to download the RadioML dataset and symlinks it into `data/`.

### 4.2 PyTorch Dataset Classes

**File:** `data/dataset.py`

#### `IQDataset(Dataset)`

The core dataset class. On `__getitem__`:

1. **Copy** the signal (avoid mutating original).
2. **Augment** (if training):
   - 50% chance: random phase rotation (multiply by `e^{jθ}`)
   - 30% chance: small circular time shift
   - 20% chance: tiny additive noise (-20 dB)
   - 20% chance: amplitude scaling (0.9–1.1×)
3. **Normalize** to unit power: `signal / √(mean(|signal|²))`.
4. **Convert** complex → 2-channel real: `[I_channel, Q_channel]` → tensor shape `(2, L)`.
5. Return: `(signal_tensor, label, snr)`.

#### `SNRStratifiedDataset(IQDataset)`

Subclass that filters samples to only include those within a specified SNR range. Used to train individual experts on their assigned SNR regime.

---

## 5. Model Architecture

### 5.1 SNR Estimator

**File:** `models/snr_estimator.py` → class `SNREstimator`

**Purpose:** Given a raw I/Q signal, predict which SNR bin (low/mid/high) it belongs to.

**Architecture:**

```
Input (batch, 2, L)
    │
    ├──▶ CNN Branch:
    │      Conv1d(2→32, k=7) → BN → ReLU → MaxPool(2)
    │      Conv1d(32→64, k=5) → BN → ReLU → MaxPool(2)
    │      Conv1d(64→128, k=3) → BN → ReLU → MaxPool(2)
    │      AdaptiveAvgPool1d(1) → flatten → 128-dim vector
    │
    ├──▶ Statistical Branch:
    │      Computes 6 hand-crafted features:
    │      [mean_power, PAPR, kurtosis_I, kurtosis_Q, variance_I, variance_Q]
    │
    └──▶ Concatenate (128 + 6 = 134)
              │
              FC(134→128) → BN → ReLU → Dropout(0.2)
              FC(128→64)  → BN → ReLU → Dropout(0.2)
              FC(64→3)    → raw logits (no softmax)
```

**Why statistical features?** Power, PAPR, and kurtosis are classic SNR indicators in signal processing — they give the network a strong prior alongside learned CNN features.

### 5.2 Expert Networks

**File:** `models/expert_networks.py` → class `ExpertCNN`

**Purpose:** Classify modulation type. Three instances created — one per SNR regime.

**Key building blocks:**

#### `SEBlock` (Squeeze-and-Excitation)
Channel attention: `AvgPool → FC(C→C/4) → ReLU → FC(C/4→C) → Sigmoid → scale channels`

#### `ResidualBlock`
```
Input → Conv1d → BN → ReLU → Dropout → Conv1d → BN → SE → Add(residual) → ReLU
  └──── Shortcut (1x1 conv if dimensions change) ──────────────────┘
```

#### Full ExpertCNN architecture:

```
Input (batch, 2, L)
    │
    ├─ Stem: two parallel Conv1d branches (k=3 and k=7) → concat → BN → ReLU
    │        → 64 channels (multi-scale feature extraction)
    │
    ├─ Stage 1: 2× ResidualBlock(64→64, k=7 and k=5)
    ├─ Stage 2: 2× ResidualBlock(64→128, stride=2, k=5 and k=3)  ← downsamples 2×
    ├─ Stage 3: 2× ResidualBlock(128→256, stride=2, k=3 and k=3) ← downsamples 2×
    │
    ├─ Global Average Pool + Global Max Pool → concat → 512-dim
    │
    └─ Classifier: FC(512→256) → BN → ReLU → Dropout(0.4)
                   FC(256→128) → BN → ReLU → Dropout(0.3)
                   FC(128→num_classes) → raw logits
```

### 5.3 Gating Network

**File:** `models/gating_network.py`

Four variants are implemented:

| Variant | Input | Description |
|---|---|---|
| `GatingNetwork` | SNR logits only | Simplest: softmax(SNR logits) → FC → softmax → weights |
| `SignalAwareGatingNetwork` | SNR logits + raw signal | CNN extracts signal features, concatenates with SNR probs, FC → weights. **Used by default.** |
| `AdaptiveGatingNetwork` | Raw signal only | Learns gating purely from signal features (no explicit SNR) |
| `HardGatingNetwork` | SNR logits | argmax → one-hot selection of single expert |

The **SignalAwareGatingNetwork** (default) flow:
```
Raw signal → Conv1d(2→32) → Pool → Conv1d(32→64) → Pool → GAP → 64-dim
SNR logits → softmax → 3-dim probabilities
    │                          │
    └────── concatenate ───────┘ → 67-dim
                    │
          FC(67→128) → ReLU → Dropout
          FC(128→64) → ReLU → Dropout
          FC(64→3) → softmax → expert weights [w₀, w₁, w₂]
```

### 5.4 Full MoE System

**File:** `models/moe_amc.py` → class `MoEAMC`

Combines everything:

```python
def forward(self, x):
    snr_logits = self.snr_estimator(x)          # (B, 3)
    gating_weights = self.gating(x, snr_logits) # (B, 3) — soft weights

    expert_outputs = [expert(x) for expert in self.experts]  # list of (B, C)
    expert_outputs = torch.stack(expert_outputs, dim=1)      # (B, 3, C)

    # Weighted combination
    weights = gating_weights.unsqueeze(2)     # (B, 3, 1)
    final = (expert_outputs * weights).sum(1) # (B, C)
    return final
```

**Auxiliary losses** (used during MoE fine-tuning):

1. **Load-Balance Loss:** KL divergence of average gating weights from uniform distribution → prevents one expert from dominating (expert collapse).
2. **Diversity Loss:** Average pairwise cosine similarity of expert softmax outputs → encourages experts to learn different representations.

---

## 6. Training Pipeline

**File:** `train.py`

**Entry point:** `python train.py`

### 6.0 Setup

```python
config = Config()
np.random.seed(42); torch.manual_seed(42)
```

1. Load data (RML pickle or synthetic generation).
2. Random permutation split: **70% train / 15% val / 15% test**.
3. Create `IQDataset` with `augment=True` for training, `augment=False` for validation.
4. Create `DataLoader` with batch_size=512, shuffle=True (train), pin_memory=True.

### 6.1 Stage 1: Train SNR Estimator

**Function:** `train_snr_estimator()`

- **Target:** Classify each signal's SNR into one of 3 bins (low/mid/high) using `Config.get_snr_bin()`.
- **Loss:** CrossEntropy with label_smoothing=0.05.
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4.
- **Scheduler:** CosineAnnealingWarmRestarts (T₀=20, T_mult=2).
- **Early stopping:** patience=20, saves best to `checkpoints/snr_estimator_best.pth`.
- **Gradient clipping:** max_norm=5.0.
- **Epochs:** up to 100.

### 6.2 Stage 2: Train Expert Networks (×3)

**Function:** `train_expert(expert_id, ...)`

For each SNR regime (low, mid, high):

1. Create `SNRStratifiedDataset` that filters training data to only samples within that expert's SNR range.
2. Train an independent `ExpertCNN`.
3. **Mixup augmentation** applied 50% of the time (α=0.2 Beta distribution).
4. **Loss:** CrossEntropy with label_smoothing=0.1.
5. **Optimizer:** AdamW, lr=1e-3.
6. **Scheduler:** CosineAnnealingWarmRestarts (T₀=25, T_mult=2).
7. **Early stopping:** patience=25.
8. **Epochs:** up to 150.
9. Saves best to `checkpoints/expert_{i}_best.pth`.

### 6.3 Stage 3: End-to-End MoE Fine-Tuning

**Function:** `train_moe_system()`

1. Create full `MoEAMC` model.
2. **Load pre-trained weights** from Stages 1 & 2 into corresponding components.
3. **Differential learning rates:**
   - SNR estimator: `MOE_LR × 0.1` (nearly frozen, already good)
   - Expert networks: `MOE_LR × 0.5` (slow updates, preserve specialization)
   - Gating network: `MOE_LR × 2.0` (fast learning, new component)
4. **Combined loss** per batch:
   ```
   total_loss = cls_loss
              + 0.3  × snr_aux_loss      (keeps SNR estimator accurate)
              + 0.1  × load_balance_loss  (prevents expert collapse)
              + 0.05 × diversity_loss     (encourages expert specialization)
   ```
5. Train on **full dataset** (all SNR ranges), up to 200 epochs.
6. Saves best to `checkpoints/moe_amc_best.pth`.

### Training Flow Summary

```
Stage 1                     Stage 2                         Stage 3
───────                     ───────                         ───────
Train SNR Estimator    ──▶  Train Expert_Low (SNR<2)   ──▶  Load all pre-trained
(all data, 100 epochs)      Train Expert_Mid (-2<SNR<12)    weights into MoEAMC.
                            Train Expert_High (SNR>8)       Fine-tune end-to-end
                            (filtered data, 150 epochs)     with auxiliary losses.
                                                            (all data, 200 epochs)
```

---

## 7. Evaluation Pipeline

**File:** `evaluate.py`

**Entry point:** `python evaluate.py`

### Flow:

1. **Reload data** with same seed (42) and same split → extracts the **test set** (15%).
2. **Load trained model** from `checkpoints/moe_amc_best.pth`.
3. **Run inference** on all test samples → collect predictions, labels, SNRs.
4. **Compute overall accuracy**.
5. **Generate plots:**
   - Accuracy vs. SNR curve.
   - Confusion matrices at SNR = 0, 10, 18 dB.
   - Overall normalized confusion matrix.
   - Classification report (precision, recall, F1 per class).
6. Save plots to `results/` and report to `results/classification_report.txt`.

---

## 8. Inference & Testing

### 8.1 Interactive Testing (`test.py`)

**Entry point:** `python test.py`

1. Loads the MoE model.
2. Prompts user for modulation type and SNR.
3. **Generates** a synthetic signal using `SignalGenerator`.
4. Adds AWGN at the specified SNR.
5. Normalizes, converts to tensor, runs through model.
6. Prints: predicted modulation, confidence, top-3 probabilities, correct/wrong.

### 8.2 SDR Testing (`test_sdr.py`)

**Entry point:** `python test_sdr.py`

Tests the model on **real captured radio data**:

1. Reads `data/sdr_output.dat` — raw complex64 binary file from a Software-Defined Radio.
2. Slices into non-overlapping windows of `SAMPLE_LENGTH` samples.
3. Normalizes each window to unit power.
4. Runs through MoE model → collects predictions.
5. Prints prediction distribution (e.g., "QPSK: 72%, BPSK: 15%...").
6. Saves bar chart to `results/sdr_predictions_distribution.png`.

---

## 9. Web API

**File:** `app/main.py` — FastAPI application

**Entry point:** `python -m uvicorn app.main:app --reload` or `python app/main.py`

### Startup

On server start (`@app.on_event("startup")`):
1. Initialize `MoEAMC` model with config params.
2. Load trained weights from `checkpoints/moe_amc_best.pth`.
3. Set model to eval mode.
4. Initialize `SignalGenerator` for synthetic signal creation.
5. If using RML data source, load the full RML dataset into memory for sampling.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Renders the web UI (`templates/index.html`) with available modulations and result plots |
| `/predict` | POST | Accepts I/Q data as JSON, returns prediction + confidence + expert used + SNR estimate |
| `/generate` | POST | Generates (or samples from RML) a signal with given modulation and SNR, returns I/Q data |

### `/predict` Flow:

```
JSON { iq_data: [[I₁,Q₁], [I₂,Q₂], ...], snr: optional }
  │
  ├─ Pad/truncate to SAMPLE_LENGTH
  ├─ Normalize to unit power
  ├─ Convert to tensor (1, 2, L)
  │
  ├─ model.forward(x, return_expert_outputs=True)
  │     → final_output, expert_outputs, gating_weights, snr_logits
  │
  ├─ prediction = argmax(softmax(final_output))
  ├─ confidence = max(softmax(final_output))
  ├─ expert_used = SNR_BINS[argmax(gating_weights)]
  ├─ snr_estimate = weighted sum of SNR bin centers using snr_logits probs
  │
  └─ Return { prediction: "QPSK", confidence: 0.94, expert_used: "mid", snr_estimate: 5.2 }
```

### `/generate` Flow:

- If `DATA_SOURCE == 'rml'`: finds closest SNR match in RML data, randomly samples one signal.
- If `DATA_SOURCE == 'generated'`: creates signal with `SignalGenerator`, adds AWGN.
- Returns I/Q pairs as JSON.

---

## 10. Defense Plot Generation

**File:** `generate_defense_plots.py`

**Entry point:** `python generate_defense_plots.py`

Generates 5 publication-quality plots for project defense:

1. **Accuracy vs. SNR:** Line plot showing overall accuracy at each SNR level.
2. **Expert Gating Activation Map:** Heatmap showing average expert weights per SNR — visually demonstrates how the system switches experts as SNR changes.
3. **Overall Confusion Matrix:** Normalized, all SNR levels combined.
4. **High-SNR Confusion Matrix:** At SNR=18 dB, showing peak performance.
5. **Top-6 Modulations Accuracy:** Per-class accuracy vs SNR for the 6 best-performing modulation types.

---

## 11. End-to-End Data Flow Diagram

### Training Phase

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LOADING                             │
│                                                                 │
│  RML2016.10a_dict.pkl ──▶ load_rml_data() ──▶ (signals, labels, │
│                              snrs, mod_names)                   │
│  OR                                                             │
│  SignalGenerator.generate_dataset() ──▶ (signals, labels, snrs) │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SPLITTING & LOADING                    │
│                                                                 │
│  np.random.permutation(seed=42)                                 │
│  ├─ 70% ──▶ IQDataset(augment=True)  ──▶ train_loader          │
│  ├─ 15% ──▶ IQDataset(augment=False) ──▶ val_loader            │
│  └─ 15% ──▶ (reserved for test)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────────┐
            ▼                ▼                    ▼
    ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐
    │   STAGE 1    │  │   STAGE 2   │  │     STAGE 3      │
    │ SNR Estimator│  │  3 Experts  │  │ MoE Fine-Tuning  │
    │              │  │  (filtered  │  │ (load pretrained, │
    │ all data     │  │   by SNR)   │  │  differential LR, │
    │ 100 epochs   │  │ 150 epochs  │  │  auxiliary losses) │
    │              │  │  each       │  │  200 epochs       │
    └──────┬───────┘  └──────┬──────┘  └────────┬─────────┘
           │                 │                   │
           ▼                 ▼                   ▼
    snr_estimator_     expert_0_best.pth    moe_amc_best.pth
    best.pth           expert_1_best.pth    (FINAL MODEL)
                       expert_2_best.pth
```

### Inference Phase

```
   Raw I/Q Signal (2, 128)
          │
          │  Normalize to unit power
          ▼
   ┌──────────────────┐
   │   SNR Estimator   │
   │  (CNN + Stats)    │──────────┐
   └──────┬───────────┘          │
          │ logits (3,)           │ logits
          ▼                      ▼
   ┌──────────────────┐   ┌───────────┐  ┌───────────┐  ┌───────────┐
   │  Signal-Aware    │   │ Expert 0  │  │ Expert 1  │  │ Expert 2  │
   │  Gating Network  │   │ (Low SNR) │  │ (Mid SNR) │  │(High SNR) │
   │  (CNN + SNR)     │   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
   └──────┬───────────┘         │              │              │
          │ weights              │              │              │
          │ [w₀, w₁, w₂]        │ logits(C,)   │ logits(C,)   │ logits(C,)
          │                      ▼              ▼              ▼
          │              ┌──────────────────────────────────────────┐
          └─────────────▶│  output = w₀·E₀ + w₁·E₁ + w₂·E₂       │
                         └──────────────────┬───────────────────────┘
                                            │
                                            ▼
                                    argmax → Predicted Modulation
                                    softmax → Confidence Score
```

---

## Key Technical Details

### Why Overlapping SNR Ranges?
Expert SNR ranges overlap (e.g., Mid goes from -2 to 12 while Low goes up to 2). This means signals near boundaries are seen by multiple experts during training, preventing hard performance drops at transition points.

### Why Differential Learning Rates in Stage 3?
- **SNR estimator** (×0.1): Already well-trained in Stage 1, just needs minor adjustments.
- **Experts** (×0.5): Preserve their specialization while allowing adaptation.
- **Gating network** (×2.0): Brand new component, needs to learn fast how to route signals.

### Why Auxiliary Losses?
- **SNR auxiliary loss** (weight 0.3): Without this, the SNR estimator's accuracy could degrade during end-to-end training as gradients from classification dominate.
- **Load-balance loss** (weight 0.1): Without this, the gating network might learn to always pick one expert.
- **Diversity loss** (weight 0.05): Without this, all three experts might converge to the same solution, defeating the purpose of MoE.

### Data Augmentation Strategy
Augmentations are deliberately conservative because aggressive transforms (like large time shifts or noise) could destroy modulation-specific features:
- **Phase rotation** (safe for all mods) — most frequent at 50%.
- **Small time shift** — only 30% chance, max shift L/8.
- **Tiny noise** — only 20% chance, -20 dB below signal.
- **Amplitude scaling** — only 20%, very narrow range (0.9–1.1).

### Mixup Augmentation (Expert Training)
During Stage 2, 50% of batches use **Mixup** — linear interpolation of two random training samples and their labels (λ from Beta(0.2, 0.2)). This smooths decision boundaries and improves generalization, especially important for experts that see limited data subsets.
