# Lévy Area Lead-Lag Trading Strategies with Graph Neural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated quantitative trading framework that combines **Lévy Area analysis** for lead-lag relationship detection with **Temporal Graph Neural Networks (GNNs)** for cross-asset return prediction. This project explores novel approaches to systematic macro trading by leveraging mathematical signatures from rough path theory and modern deep learning architectures.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Innovations](#-key-innovations)
- [Methodologies](#-methodologies)
  - [Lévy Area Lead-Lag Detection](#1-lévy-area-lead-lag-detection)
  - [Temporal GNN Trading System](#2-temporal-gnn-trading-system)
  - [Trading Strategies](#3-trading-strategies)
- [Asset Universe](#-asset-universe)
- [Installation](#-installation)

---

## 🎯 Overview

This project presents a multi-strategy quantitative trading system designed for macro instruments. We investigate whether **lead-lag relationships** between assets—detected via Lévy Area (a mathematical measure from stochastic calculus)—can be exploited to generate alpha. Additionally, we develop a **Temporal Graph Neural Network** that models the entire asset universe as a dynamic graph, learning complex inter-asset dependencies for return prediction.

### Core Research Questions

1. Can Lévy Area effectively identify which assets "lead" and which "follow" in cross-sectional returns?
2. Do followers exhibit predictable behavior based on leader signals?
3. Can GNNs learn meaningful asset relationships that translate into profitable trading signals?

---

## 🚀 Key Innovations

| Innovation | Description |
|------------|-------------|
| **Lévy Area Scoring** | Novel application of path signatures to rank assets by their lead-lag tendency |
| **Antisymmetric Signal Extraction** | Exploiting the mathematical property that $L_{ij} = -L_{ji}$ for directional relationships |
| **Dynamic Graph Construction** | Fully-connected asset graphs with learned signed edge weights |
| **Ranking-Based Loss Function** | Differentiable Spearman correlation loss to avoid model collapse |
| **Ensemble Architecture** | Multiple GNN models with different seeds for robust predictions |
| **Z-Score Signal Thresholding** | Adaptive confidence filtering based on rolling signal distributions |

---

## 🔬 Methodologies

### 1. Lévy Area Lead-Lag Detection

The **Lévy Area** between two time series $X$ and $Y$ is defined as:

$$L_{XY} = \frac{1}{2} \int_0^T \left( X_t \, dY_t - Y_t \, dX_t \right)$$

This antisymmetric quantity captures the "area" enclosed by the parametric path $(X_t, Y_t)$. A positive value suggests $X$ tends to lead $Y$, while a negative value suggests the opposite.

#### Discrete Approximation

For standardized returns $x_t$ and $y_t$:

$$\hat{L}_{XY} = \frac{1}{2(T-1)} \sum_{t=1}^{T-1} \left( x_{t-1} \Delta y_t - y_{t-1} \Delta x_t \right)$$

#### Scoring Mechanism

For each asset $i$, we compute a **lead score** as the row-sum of positive Lévy entries:

$$\text{Score}_i = \sum_{j \neq i} \max(0, L_{ij})$$

- **Leaders**: Assets with highest scores (top 20-30%)
- **Followers**: Assets with lowest scores (bottom 20-30%)

---

### 2. Temporal GNN Trading System

Our GNN architecture combines temporal and graph-based learning:

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: 60 Days × N Assets × F Features   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Bidirectional LSTM (per asset)             │
│                  • 2 layers, 128 hidden units               │
│                  • Captures temporal patterns               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Signed Edge Network                         │
│                 • Learns directed relationships             │
│                 • Edge weights ∈ [-1, 1]                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              GATv2 Layers (2 layers, 4 heads)               │
│              • Message passing with edge features           │
│              • Jumping Knowledge aggregation                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Prediction Head (MLP)                    │
│                    • Outputs ranking score per asset        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 OUTPUT: Asset Rankings → Portfolio          │
└─────────────────────────────────────────────────────────────┘
```

#### Features Extracted

| Feature | Window | Description |
|---------|--------|-------------|
| 1-day return | 1 | Daily log return |
| 5-day return | 5 | Weekly momentum |
| 20-day return | 20 | Monthly momentum |
| 20-day volatility | 20 | Realized annualized vol |
| RSI | 14 | Mean-reversion indicator |
| Lévy Area | 20 | Pairwise lead-lag scores |

#### Loss Function

We use a differentiable approximation to Spearman rank correlation:

$$\mathcal{L}_{\text{rank}} = -\frac{1}{B} \sum_{b=1}^{B} \frac{\sum_{i \neq j} \tanh\left(\frac{p_i - p_j}{\tau}\right) \cdot \tanh\left(\frac{r_i - r_j}{\tau}\right)}{N(N-1)}$$

where $p$ are predictions, $r$ are actual returns, and $\tau$ is a temperature parameter.

---

### 3. Trading Strategies

#### Strategy 1: Lévy Leader-Follower (Long-Only)

**Parameters:**
- Lookback window: 30-50 trading days
- Selection percentile: 20% leaders, 20% followers
- Holding period: $k \in \{5, 7, 10\}$ days (non-overlapping)

**Logic:**
1. Compute rolling Lévy matrices and score assets
2. Identify leaders (high score) and followers (low score)
3. At rebalance date $t$:
   - If leaders' average return on day $t > 0$: **Long followers**
   - Else: **Long equal-weight market**
4. Hold for $k$ days, compound returns, repeat

#### Strategy 2: Asset-Specific Targeting

**Hypothesis:** Certain assets (e.g., SPY, EFA) may benefit more from leader signals.

**Logic:**
1. If leaders' return > 0 AND target asset is in follower set:
   - Long target asset(s) only
2. Else:
   - Hold equal-weight market

#### Strategy 3: GNN Ensemble

**Parameters:**
- Ensemble size: 3-5 models with different seeds
- Signal threshold: 1.5 standard deviations (dynamic)
- Prediction horizon: 5 days

**Portfolio Construction:**
- Long top 30% by predicted score
- Short bottom 30% by predicted score
- Equal-weight within legs

---

## 📊 Asset Universe

### Base Tradable Assets (13)

| Category | Ticker | Description |
|----------|--------|-------------|
| **Equities** | SPY | S&P 500 |
| | IWM | Russell 2000 |
| | EFA | Developed Markets ex-US |
| | EEM | Emerging Markets |
| **Fixed Income** | SHY | 1-3 Year Treasury |
| | IEF | 7-10 Year Treasury |
| | LQD | Investment Grade Corporate |
| | HYG | High Yield Corporate |
| | TIP | Treasury Inflation-Protected |
| **Commodities** | USO | Crude Oil |
| | GLD | Gold |
| | DBA | Agriculture |
| **Currency** | UUP | US Dollar Index |

### Derived Ratios (4)

| Ratio | Components | Interpretation |
|-------|------------|----------------|
| XLY/XLP | Consumer Cyclicals / Defensives | Economic strength |
| LQD/IEF | IG Corp / Treasury | Credit spread |
| HYG/IEI | HY Corp / Short Treasury | Risk appetite |
| TIP/IEF | TIPS / Treasury | Inflation expectations |

### Signal Modifiers (Non-Tradable)

- **^VIX**: Equity volatility index
- **^MOVE**: Bond volatility index

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/JacobNiyazov/ML-Capstone.git
cd ML-Capstone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.21.0
pandas>=1.3.0
torch>=2.0.0
torch-geometric>=2.3.0
yfinance>=0.2.0
matplotlib>=3.5.0
seaborn>=0.12.0
scipy>=1.9.0
scikit-learn>=1.0.0
esig>=0.9.0  # For path signatures
tqdm>=4.64.0
openpyxl>=3.0.0  # For Excel export
```
