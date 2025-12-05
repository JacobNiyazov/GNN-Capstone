"""
Temporal GNN Trading System - ENHANCED VERSION
Key improvements:
1. Ensemble of 5 models with different seeds
2. Xavier weight initialization
3. Validation increased to 20% with rolling IC average for early stopping
4. Differentiable rank correlation loss (aligned with IC metric)
5. Reduced model complexity: 2 GAT layers, hidden_dim=64
6. Batch normalization in predictor network
7. Signal confidence thresholds to filter weak signals
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
import seaborn as sns
import networkx as nx
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_TICKERS = ['SPY', 'IWM', 'EFA', 'EEM', 'SHY', 'IEF', 'LQD', 'HYG',
                'TIP', 'USO', 'GLD', 'DBA', 'UUP']

RATIO_TICKERS = ['XLY', 'XLP', 'IEI']

DERIVED_RATIOS = {
    'XLY_XLP': ('XLY', 'XLP', 'Cyclicals/Defensives'),
    'LQD_IEF': ('LQD', 'IEF', 'IG Spread'),
    'HYG_IEI': ('HYG', 'IEI', 'HY Spread'),
    'TIP_IEF': ('TIP', 'IEF', 'Inflation BE')
}

SIGNAL_MODIFIERS = ['^VIX', '^MOVE']

MODIFIER_RELATIONSHIPS = {
    '^VIX': {
        'SPY': -1.9, 'IWM': -1.9, 'EFA': -1.9, 'EEM': -1.9,
        'HYG': -1.9, 'USO': -1.9,
        'SHY': +1.9, 'GLD': +1.9, 'IEF': +1.9, 'UUP': +1.9,
    },
    '^MOVE': {
        'IEF': -1.9, 'LQD': -1.9, 'HYG': -1.9, 'TIP': -1.9,
        'SHY': +1.9, 'SPY': +1.9,
    }
}

LOOKBACK = 60
PREDICTION_HORIZON = 5
TRAIN_RATIO = 0.7
VAL_RATIO = 0.20  # Increased from 0.15
BATCH_SIZE = 32
INITIAL_CAPITAL = 1_000_000
LEVY_WINDOW = 20

# New parameters
ENSEMBLE_SEEDS = [0, 43, 44] #, 45, 46]
# --- REPLACED STATIC THRESHOLD WITH DYNAMIC CONFIG ---
# SIGNAL_THRESHOLD = 0.1  <-- REMOVED
SIGNAL_Z_THRESHOLD = 1.5      # Require signal to be 1.25 std devs from mean 
SIGNAL_ROLLING_WINDOW = 60     # Lookback days to establish "normal" signal distribution
MIN_HISTORY_FOR_Z = 20         # Minimum days before applying Z-score (use fallback before)
# -----------------------------------------------------
ROLLING_IC_WINDOW = 10  # Epochs to average for early stopping


class DataLoader:
    """Download and prepare price data"""
    def __init__(self, start_date='2010-01-01', end_date='2024-12-31'):
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        """Download price data and compute derived ratios"""
        all_tickers = list(set(BASE_TICKERS + RATIO_TICKERS + SIGNAL_MODIFIERS))
        
        print(f"Downloading {len(all_tickers)} tickers...")
        data = yf.download(all_tickers, start=self.start_date, end=self.end_date,
                          progress=False)['Close']
        
        if len(all_tickers) == 1:
            data = pd.DataFrame(data, columns=[all_tickers[0]])
        
        data = data.ffill().dropna()
        
        print(f"Downloaded data: {data.shape}")
        
        # Compute derived ratios
        print("Computing derived ratios...")
        ratio_data = {}
        
        for ratio_name, (num, denom, desc) in DERIVED_RATIOS.items():
            if num in data.columns and denom in data.columns:
                ratio_data[ratio_name] = data[num] / data[denom]
                print(f"  {ratio_name}: {desc}")
        
        # Combine base tickers + modifiers + ratios
        result = data[BASE_TICKERS + SIGNAL_MODIFIERS].copy()
        for ratio_name, ratio_series in ratio_data.items():
            result[ratio_name] = ratio_series
        
        print(f"\nFinal dataset: {result.shape}")
        print(f"  Base tradable: {len(BASE_TICKERS)}")
        print(f"  Ratio features (tradable via legs): {len(DERIVED_RATIOS)}")
        print(f"  Signal modifiers (non-tradable): {len(SIGNAL_MODIFIERS)}")
        
        return result


class FeatureExtractor:
    """Extract features with Lévy area lead-lag relationships"""
    def __init__(self, lookback=20, rsi_period=14, levy_window=20):
        self.lookback = lookback
        self.rsi_period = rsi_period
        self.levy_window = levy_window

    def compute_rsi(self, prices, period=14):
        """Compute RSI"""
        n = len(prices)
        rsi = np.zeros(n)
        
        for i in range(period, n):
            price_window = prices[i-period:i+1]
            deltas = np.diff(price_window)
            
            gains = deltas.copy()
            losses = deltas.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = np.abs(losses)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi

    def compute_levy_area(self, series_x, series_y):
        """Compute Lévy area between two time series"""
        if len(series_x) < 2 or len(series_y) < 2:
            return 0.0
        
        std_x = np.std(series_x)
        std_y = np.std(series_y)
        
        if std_x < 1e-10 or std_y < 1e-10:
            return 0.0
        
        x = (series_x - np.mean(series_x)) / std_x
        y = (series_y - np.mean(series_y)) / std_y
        
        dx = np.diff(x)
        dy = np.diff(y)
        
        levy = np.sum(x[:-1] * dy - y[:-1] * dx)
        levy = levy / (len(x) - 1)
        
        return levy

    def compute_features(self, prices):
        """Compute features for all assets"""
        n_assets = prices.shape[1]
        n_time = prices.shape[0]
        
        print(f"\nComputing features with Lévy area lead-lag...")
        print(f"  Lévy window: {self.levy_window} days")
        
        basic_features_list = []
        
        for col_idx, col in enumerate(prices.columns):
            if col_idx % 5 == 0:
                print(f"  Processing asset {col_idx+1}/{n_assets}: {col}")
            
            price = prices[col].values
            
            ret_1d = np.zeros(n_time)
            ret_5d = np.zeros(n_time)
            ret_20d = np.zeros(n_time)
            vol_20d = np.zeros(n_time)
            rsi = self.compute_rsi(price, period=self.rsi_period)
            
            for i in range(n_time):
                if i > 0:
                    ret_1d[i] = np.log(price[i] / price[i-1])
                
                if i >= 5:
                    ret_5d[i] = np.log(price[i] / price[i-5])
                
                if i >= 20:
                    ret_20d[i] = np.log(price[i] / price[i-20])
                    recent_returns = np.diff(np.log(price[i-20:i+1]))
                    vol_20d[i] = np.std(recent_returns) * np.sqrt(252)
            
            rsi_normalized = (rsi - 50) / 50
            
            feats = np.stack([ret_1d, ret_5d, ret_20d, vol_20d, rsi_normalized], axis=1)
            basic_features_list.append(feats)
        
        basic_features = np.stack(basic_features_list, axis=1)
        
        print(f"\nComputing Lévy area features...")
        levy_features_list = []
        
        for i in range(n_assets):
            if i % 5 == 0:
                print(f"  Computing lead-lag for asset {i+1}/{n_assets}")
            
            levy_with_others = np.zeros((n_time, n_assets - 1))
            
            other_idx = 0
            for j in range(n_assets):
                if i == j:
                    continue
                
                series_i = prices.iloc[:, i].values
                series_j = prices.iloc[:, j].values
                
                for t in range(self.levy_window, n_time):
                    window_i = series_i[t-self.levy_window:t]
                    window_j = series_j[t-self.levy_window:t]
                    
                    levy_with_others[t, other_idx] = self.compute_levy_area(window_i, window_j)
                
                other_idx += 1
            
            levy_features_list.append(levy_with_others)
        
        levy_features = np.stack(levy_features_list, axis=1)
        
        features = np.concatenate([basic_features, levy_features], axis=2)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"\nFeature extraction complete:")
        print(f"  Basic features: {basic_features.shape[2]}")
        print(f"  Lévy features: {levy_features.shape[2]}")
        print(f"  Total features: {features.shape[2]}")
        
        return features

class SignedEdgeNetwork(nn.Module):
    """Explicitly learn SIGNED edge weights"""
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.edge_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
    
    def forward(self, h, edge_index):
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        edge_features = torch.cat([h[src_nodes], h[dst_nodes]], dim=1)
        edge_weights = self.edge_net(edge_features).squeeze(-1)
        
        return edge_weights


class TemporalGNN(nn.Module):
    """
    Temporal GNN with FUNCTIONAL signed edge weights.
    
    Key change: Signed edges are now computed BEFORE GAT layers and passed
    as edge_attr to influence message passing.
    """
    def __init__(self, n_features, hidden_dim=64, lstm_layers=2, gat_layers=2):
        super().__init__()
        
        # 1. Learnable Feature Weights (L1 Regularization Target)
        self.feature_weights = nn.Parameter(torch.ones(n_features))
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Project LSTM (128) -> Hidden (64)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Signed edge network - NOW USED BEFORE GAT layers
        self.signed_edge_net = SignedEdgeNetwork(hidden_dim)
        
        # GAT layers WITH edge_dim to accept signed edge weights
        # edge_dim=1 means each edge has a scalar weight in [-1, 1]
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                hidden_dim, 
                hidden_dim // 4, 
                heads=4, 
                concat=True, 
                dropout=0.2,
                edge_dim=1  # <-- KEY CHANGE: Accept edge features
            )
            for _ in range(gat_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(gat_layers)
        ])
        
        # JK PROJECTION (Standard)
        self.jk_projection = nn.Sequential(
            nn.Linear(hidden_dim * gat_layers, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # PREDICTOR (Standard Size)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, edge_index):
        """
        Forward pass for single sample inference.
        
        Args:
            x: [N, L, F] - N assets, L lookback, F features
            edge_index: [2, E] - Edge connectivity
            
        Returns:
            pred: [N] - Predictions per asset
            h_final: [N, hidden_dim] - Final embeddings
            feature_weights: [F] - Learned feature importance
            signed_edges: [E] - Final signed edge weights
        """
        x_weighted = x * self.feature_weights
        lstm_out, _ = self.lstm(x_weighted)
        h = self.proj(lstm_out[:, -1, :])
        
        # Compute initial signed edges from LSTM embeddings
        signed_edges = self.signed_edge_net(h, edge_index)
        edge_attr = signed_edges.unsqueeze(-1)  # [E] -> [E, 1] for GATv2Conv
        
        # Store outputs from all layers for JK aggregation
        jk_outputs = []
        
        for gat, norm in zip(self.gat_layers, self.norms):
            # KEY CHANGE: Pass edge_attr to GAT layer
            h_new = gat(h, edge_index, edge_attr=edge_attr)
            h = norm(h + h_new)
            jk_outputs.append(h)
            
            # Update edge weights based on refined embeddings
            signed_edges = self.signed_edge_net(h, edge_index)
            edge_attr = signed_edges.unsqueeze(-1)
        
        # Concatenate: [N, 64] + [N, 64] -> [N, 128]
        h_concat = torch.cat(jk_outputs, dim=-1)
        
        # Project back to 64: [N, 128] -> [N, 64]
        h_final = self.jk_projection(h_concat)
        
        pred = self.predictor(h_final).squeeze(-1)
        
        return pred, h_final, self.feature_weights, signed_edges

    def forward_batch(self, x_b, edge_index):
        """
        Batched forward pass for training.
        
        Args:
            x_b: [B, N, L, F] - Batch of samples
            edge_index: [2, E] - Edge connectivity (same for all samples)
            
        Returns:
            preds: [B, N] - Predictions
            H_final: [B, N, hidden_dim] - Final embeddings
            feature_weights: [F] - Learned feature importance
            signed_edges_batch: [B, E] - Signed edges per sample
        """
        B, N, L, F = x_b.shape
        
        x_b_weighted = x_b * self.feature_weights
        x_flat = x_b_weighted.reshape(B * N, L, F)
        
        lstm_out, _ = self.lstm(x_flat)
        h = self.proj(lstm_out[:, -1, :])
        h = h.view(B, N, -1)
        
        # Storage for layer outputs and final edges
        layer_outputs = [[] for _ in range(len(self.gat_layers))]
        signed_edges_list = []
        
        for b in range(B):
            hb = h[b]  # [N, hidden_dim]
            
            # Compute initial signed edges for this batch element
            signed_edges = self.signed_edge_net(hb, edge_index)
            edge_attr = signed_edges.unsqueeze(-1)  # [E, 1]
            
            for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
                # KEY CHANGE: Pass edge_attr to GAT layer
                hb_new = gat(hb, edge_index, edge_attr=edge_attr)
                hb = norm(hb + hb_new)
                layer_outputs[i].append(hb)
                
                # Update edge weights with refined embeddings
                signed_edges = self.signed_edge_net(hb, edge_index)
                edge_attr = signed_edges.unsqueeze(-1)
            
            signed_edges_list.append(signed_edges)
        
        # Stack batches: List of [B, N, 64]
        H_layers = [torch.stack(l, dim=0) for l in layer_outputs]
        
        # Concatenate features: [B, N, 128]
        H_concat = torch.cat(H_layers, dim=-1)
        
        # Flatten: [B*N, 128]
        H_concat_flat = H_concat.view(B * N, -1)
        
        # Project: [B*N, 128] -> [B*N, 64]
        H_final_flat = self.jk_projection(H_concat_flat)
        
        # Predict: [B*N, 64] -> [B*N, 1]
        preds = self.predictor(H_final_flat).view(B, N)
        
        H_final = H_final_flat.view(B, N, -1)
        signed_edges_batch = torch.stack(signed_edges_list, dim=0)
        
        return preds, H_final, self.feature_weights, signed_edges_batch

def differentiable_rank_correlation_loss(pred, target):
    """
    Differentiable approximation of Spearman rank correlation
    Uses pairwise ordering agreement
    Returns negative correlation (for minimization)
    """
    B, N = pred.shape
    
    # Center predictions and targets
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    
    # Add small epsilon to avoid division by zero in case of constant predictions
    pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8
    target_std = target_centered.std(dim=1, keepdim=True) + 1e-8
    
    # Normalize for stability
    pred_normalized = pred_centered / pred_std
    target_normalized = target_centered / target_std
    
    # Compute pairwise differences
    pred_diff = pred_normalized.unsqueeze(2) - pred_normalized.unsqueeze(1)  # [B, N, N]
    target_diff = target_normalized.unsqueeze(2) - target_normalized.unsqueeze(1)  # [B, N, N]
    
    # Soft sign using tanh (differentiable)
    temperature = 0.1
    pred_sign = torch.tanh(pred_diff / temperature)
    target_sign = torch.tanh(target_diff / temperature)
    
    # Agreement on pairwise orderings
    agreement = pred_sign * target_sign
    
    # Mask out diagonal (comparing with self)
    mask = 1 - torch.eye(N, device=pred.device).unsqueeze(0)
    
    # Average agreement (negative for loss)
    correlation = (agreement * mask).sum() / (mask.sum() * B + 1e-8)
    
    return -correlation  # Negative because we minimize loss


class TradingSystem:
    """Single model trading system"""
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.feature_means = None
        self.feature_stds = None
        self.signed_edge_history = []
        self.rolling_ic_history = deque(maxlen=ROLLING_IC_WINDOW)

    def create_graph(self, n_assets):
        """Create fully connected graph"""
        edges = []
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).t().to(self.device)

    def normalize_features(self, features, train_idx=None):
        """Normalize using training data statistics"""
        if train_idx is not None:
            train_features = features[train_idx]
            self.feature_means = np.mean(train_features, axis=(0, 1), keepdims=True)
            self.feature_stds = np.std(train_features, axis=(0, 1), keepdims=True) + 1e-8
        
        normalized = (features - self.feature_means) / self.feature_stds
        return normalized

    def prepare_minibatch(self, features_norm, future_returns, indices):
        """Build minibatch"""
        valid = [i for i in indices if i >= LOOKBACK]
        if not valid:
            return None, None
        
        x_list = [features_norm[i-LOOKBACK:i].transpose(1, 0, 2) for i in valid]
        x_b = torch.from_numpy(np.stack(x_list, axis=0)).float().to(self.device)
        
        if future_returns is None:
            y_b = None
        else:
            y_b = torch.from_numpy(future_returns[valid]).float().to(self.device)
        
        return x_b, y_b

    def embedding_diversity_loss(self, embeddings):
        """Encourage diverse embeddings"""
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        correlation = torch.mm(embeddings_norm, embeddings_norm.t())
        
        mask = 1 - torch.eye(embeddings.shape[0], device=embeddings.device)
        off_diagonal_corr = (correlation * mask).sum() / mask.sum()
        
        return off_diagonal_corr.abs()

    def train_model(self, features, future_returns, train_idx, val_idx):
        """
        Train with differentiable rank correlation loss, rolling IC early stopping,
        and L1 regularization on feature weights.
        """
        print(f"\n{'='*60}")
        print(f"Training model with RANK LOSS + L1 FEATURE REGULARIZATION")
        print(f"{'='*60}")
        print(f"Prediction horizon: {PREDICTION_HORIZON} day(s)")
        print(f"Validation set: {VAL_RATIO*100:.0f}%")
        print(f"Rolling IC window: {ROLLING_IC_WINDOW} epochs")
        
        features_norm = self.normalize_features(features, train_idx=train_idx)
        
        n_assets = features.shape[1]
        edge_index = self.create_graph(n_assets)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        
        def lr_lambda(epoch):
            if epoch < 20:
                return (epoch + 1) / 20
            else:
                return 1.0
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        best_rolling_ic = -float('inf')
        patience = 40
        patience_counter = 0
        
        # L1 Regularization strength (Hyperparameter)
        # Higher = more sparse features (more weights driven to 0)
        l1_lambda = 5e-4
        
        for epoch in range(300):
            # Training
            self.model.train()
            train_losses = []
            l1_losses = []  # Track L1 loss specifically
            
            shuffled_train = train_idx.copy()
            np.random.shuffle(shuffled_train)
            
            for batch_start in range(0, len(shuffled_train), BATCH_SIZE):
                batch_indices = shuffled_train[batch_start:batch_start + BATCH_SIZE]
                
                xb, yb = self.prepare_minibatch(features_norm, future_returns, batch_indices)
                if xb is None:
                    continue
                
                optimizer.zero_grad()
                
                # Capture feature_weights from the forward pass
                preds_b, H_b, feat_weights, signed_edges_b = self.model.forward_batch(xb, edge_index)
                
                # Normalize targets per sample
                y_mean = yb.mean(dim=1, keepdim=True)
                y_std = yb.std(dim=1, keepdim=True) + 1e-8
                y_norm = (yb - y_mean) / y_std
                
                # 1. Rank Correlation Loss
                loss_rank = differentiable_rank_correlation_loss(preds_b, y_norm)
                
                # 2. Embedding Diversity Loss
                div_terms = []
                for b in range(H_b.shape[0]):
                    div_terms.append(self.embedding_diversity_loss(H_b[b]))
                loss_diversity = torch.stack(div_terms).mean()
                
                # 3. Edge Diversity Loss
                edge_mean = signed_edges_b.mean()
                edge_diversity = F.mse_loss(edge_mean, torch.tensor(0.0, device=self.device))
                
                # 4. L1 Feature Regularization (LASSO)
                # Penalize the sum of absolute values of the feature weights
                loss_l1 = torch.norm(feat_weights, p=1)
                
                total_loss = (
                    1.0 * loss_rank +
                    0.03 * loss_diversity + 
                    0.05 * edge_diversity +
                    l1_lambda * loss_l1  # Add L1 term
                )
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                optimizer.step()
                
                train_losses.append(total_loss.item())
                l1_losses.append(loss_l1.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                val_sample_idx = [idx for idx in val_idx[::5] if idx >= LOOKBACK]
                
                for b0 in range(0, len(val_sample_idx), BATCH_SIZE):
                    chunk = val_sample_idx[b0:b0 + BATCH_SIZE]
                    xb, yb = self.prepare_minibatch(features_norm, future_returns, chunk)
                    if xb is None:
                        continue
                    
                    # Note: We ignore feat_weights during validation
                    preds_b, _, _, _ = self.model.forward_batch(xb, edge_index)
                    
                    y_mean = yb.mean(dim=1, keepdim=True)
                    y_std = yb.std(dim=1, keepdim=True) + 1e-8
                    yb_z = (yb - y_mean) / y_std
                    
                    loss = differentiable_rank_correlation_loss(preds_b, yb_z)
                    val_losses.append(loss.item())
                    
                    val_preds.append(preds_b.cpu().numpy())
                    val_targets.append(yb.cpu().numpy())
            
            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            avg_l1 = float(np.mean(l1_losses)) if l1_losses else 0.0
            
            if val_preds and len(val_preds) > 0:
                all_val_preds = np.concatenate(val_preds, axis=0).ravel()
                all_val_targets = np.concatenate(val_targets, axis=0).ravel()
                
                if len(all_val_preds) > 1 and np.std(all_val_preds) > 1e-6 and np.std(all_val_targets) > 1e-6:
                    ic_val, _ = spearmanr(all_val_preds, all_val_targets)
                    ic = 0.0 if (ic_val is None or np.isnan(ic_val)) else float(ic_val)
                else:
                    ic = 0.0
                pred_std = float(np.std(all_val_preds))
                
                self.rolling_ic_history.append(ic)
                rolling_ic = np.mean(self.rolling_ic_history)
            else:
                ic = 0.0
                pred_std = 0.0
                rolling_ic = 0.0
            
            if epoch < 20:
                warmup_scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={train_loss:.4f} (L1={avg_l1:.2f}), Val={val_loss:.4f}, "
                      f"IC={ic:.4f}, RollingIC={rolling_ic:.4f}")
            
            if rolling_ic > best_rolling_ic:
                best_rolling_ic = rolling_ic
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        print(f"Training complete. Best rolling validation IC: {best_rolling_ic:.4f}")
        
        return best_rolling_ic

    def apply_signal_modifiers(self, signals, prices_current, asset_names):
        """Apply VIX/MOVE signal modulation"""
        modified_signals = signals.copy()
        
        for modifier in SIGNAL_MODIFIERS:
            if modifier not in asset_names:
                continue
            
            mod_idx = asset_names.index(modifier)
            modifier_level = prices_current[mod_idx]
            
            relationships = MODIFIER_RELATIONSHIPS.get(modifier, {})
            
            for asset, sensitivity in relationships.items():
                if asset not in asset_names:
                    continue
                
                asset_idx = asset_names.index(asset)
                
                if modifier == '^VIX':
                    modifier_shock = (modifier_level - 15) / 15
                elif modifier == '^MOVE':
                    modifier_shock = (modifier_level - 80) / 80
                else:
                    modifier_shock = 0
                
                adjustment = sensitivity * modifier_shock
                
                if signals[asset_idx] > 0:
                    modified_signals[asset_idx] *= (1 + adjustment)
                else:
                    modified_signals[asset_idx] *= (1 - adjustment)
        
        return modified_signals

    def decompose_ratio_signals(self, signals, asset_names):
        """Decompose ratio signals into constituent leg positions"""
        leg_signals = {}
        
        for ratio_name, (num_ticker, denom_ticker, _) in DERIVED_RATIOS.items():
            if ratio_name not in asset_names:
                continue
            
            ratio_idx = asset_names.index(ratio_name)
            ratio_signal = signals[ratio_idx]
            
            if num_ticker in asset_names:
                num_idx = asset_names.index(num_ticker)
                if num_idx not in leg_signals:
                    leg_signals[num_idx] = 0
                leg_signals[num_idx] += ratio_signal
            
            if denom_ticker in asset_names:
                denom_idx = asset_names.index(denom_ticker)
                if denom_idx not in leg_signals:
                    leg_signals[denom_idx] = 0
                leg_signals[denom_idx] -= ratio_signal
        
        return leg_signals

    def generate_signals(self, features, idx, prices_current, asset_names):
        """Generate trading signals"""
        self.model.eval()
        
        features_norm = self.normalize_features(features)
        
        if idx < LOOKBACK:
            return np.zeros(features.shape[1]), None, None
        
        x = features_norm[idx-LOOKBACK:idx].transpose(1, 0, 2)
        x = torch.FloatTensor(x).to(self.device)
        
        n_assets = features.shape[1]
        edge_index = self.create_graph(n_assets)
        
        with torch.no_grad():
            # attn is now the 1D feature_weights vector [n_features]
            pred, _, attn, signed_edges = self.model(x, edge_index)
        
        raw_signals = pred.cpu().numpy()
        modified_signals = self.apply_signal_modifiers(raw_signals, prices_current, asset_names)
        
        # FIXED: Added .detach()
        # Since 'attn' is a model parameter, it has requires_grad=True.
        # We must detach it before converting to numpy.
        avg_attn = attn.detach().cpu().numpy()
        
        signed_edges_np = signed_edges.cpu().numpy()
        
        self.signed_edge_history.append(signed_edges_np)
        
        return modified_signals, avg_attn, signed_edges_np
    
    def collect_analytics_snapshot(self, features, idx, prices_current, asset_names):
        """Collect all analytics data for dashboard"""
        self.model.eval()
        
        features_norm = self.normalize_features(features)
        
        if idx < LOOKBACK:
            return None
        
        x = features_norm[idx-LOOKBACK:idx].transpose(1, 0, 2)
        x = torch.FloatTensor(x).to(self.device)
        
        n_assets = features.shape[1]
        edge_index = self.create_graph(n_assets)
        
        with torch.no_grad():
            pred, embeddings, feature_weights, signed_edges = self.model(x, edge_index)
        
        raw_signals = pred.cpu().numpy()
        modified_signals = self.apply_signal_modifiers(raw_signals, prices_current, asset_names)
        
        # Get feature values for current timestep
        current_features = features_norm[idx-1, :, :]  # Last day in lookback
        
        return {
            'predictions': raw_signals,
            'signals': modified_signals,
            'embeddings': embeddings.cpu().numpy(),
            'feature_weights': feature_weights.detach().cpu().numpy(),
            'signed_edges': signed_edges.cpu().numpy(),
            'edge_index': edge_index.cpu().numpy(),
            'features': current_features,
            'asset_names': asset_names
        }


class EnsembleTradingSystem:
    """
    ENHANCED: Ensemble of 5 models with DYNAMIC Z-SCORE thresholding
    """
    def __init__(self, n_features, device='cpu'):
        self.device = device
        self.n_features = n_features
        self.models = []
        self.trading_systems = []
        
        # --- NEW: Signal History Buffer ---
        # We use a deque to efficiently track the rolling window of signal distributions
        self.signal_history = deque(maxlen=SIGNAL_ROLLING_WINDOW)
        # ----------------------------------
        
        print(f"\n{'='*60}")
        print(f"INITIALIZING ENSEMBLE OF {len(ENSEMBLE_SEEDS)} MODELS")
        print(f"{'='*60}")
        
        for seed in ENSEMBLE_SEEDS:
            # Create model with reduced complexity
            model = TemporalGNN(n_features=n_features, hidden_dim=64, gat_layers=2)
            trading_sys = TradingSystem(model, device=device)
            
            self.models.append(model)
            self.trading_systems.append(trading_sys)
        
        print(f"Ensemble initialized with {len(self.models)} models")

    def train_ensemble(self, features, future_returns, train_idx, val_idx):
        """Train all models in the ensemble with different seeds"""
        print(f"\n{'='*60}")
        print(f"TRAINING ENSEMBLE")
        print(f"{'='*60}")
        
        ensemble_ics = []
        
        for i, (seed, trading_sys) in enumerate(zip(ENSEMBLE_SEEDS, self.trading_systems)):
            print(f"\n{'='*60}")
            print(f"Training Model {i+1}/{len(ENSEMBLE_SEEDS)} (Seed={seed})")
            print(f"{'='*60}")
            
            # Set seed for this model
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            best_ic = trading_sys.train_model(features, future_returns, train_idx, val_idx)
            ensemble_ics.append(best_ic)
        
        print(f"\n{'='*60}")
        print(f"ENSEMBLE TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Individual model best ICs: {[f'{ic:.4f}' for ic in ensemble_ics]}")
        print(f"Mean IC: {np.mean(ensemble_ics):.4f}")
        print(f"Std IC:  {np.std(ensemble_ics):.4f}")

    def generate_ensemble_signals(self, features, idx, prices_current, asset_names):
        """
        Generate signals from all models and average them.
        Apply DYNAMIC Z-SCORE threshold based on rolling signal history.
        """
        all_signals = []
        
        # Generate signals from each model
        for trading_sys in self.trading_systems:
            signals, _, _ = trading_sys.generate_signals(features, idx, prices_current, asset_names)
            all_signals.append(signals)
        
        # Vectorized averaging
        all_signals = np.stack(all_signals, axis=0)  # [n_models, n_assets]
        ensemble_signals = np.mean(all_signals, axis=0)  # [n_assets]
        
        # Handle potential NaN/Inf from ensemble
        ensemble_signals = np.nan_to_num(ensemble_signals, nan=0.0, posinf=0.0, neginf=0.0)
        
        # --- NEW: Dynamic Z-Score Logic ---
        self.signal_history.append(ensemble_signals)
        
        # Default mask (allow everything if not enough history)
        signal_mask = np.ones_like(ensemble_signals, dtype=bool)
        
        if len(self.signal_history) >= MIN_HISTORY_FOR_Z:
            # Convert history to array: [Time, Assets]
            history_arr = np.array(self.signal_history)
            
            # Calculate stats over the entire history window (Time * Assets)
            # This normalizes based on the general volatility of the model's recent output
            hist_mean = np.mean(history_arr)
            hist_std = np.std(history_arr) + 1e-8  # Avoid div/0
            
            # Calculate Z-score for TODAY'S signals
            z_scores = (ensemble_signals - hist_mean) / hist_std
            
            # Apply Threshold
            signal_mask = np.abs(z_scores) >= SIGNAL_Z_THRESHOLD
        else:
            # Fallback during warm-up (use a small hardcoded safety floor)
            signal_mask = np.abs(ensemble_signals) >= 0.05
            
        filtered_signals = ensemble_signals * signal_mask
        # ----------------------------------
        
        return filtered_signals, all_signals

    def backtest_ensemble(self, features, prices, test_idx, asset_names):
        """
        Run backtest with ensemble signals and Z-SCORE thresholding
        """
        print("\n" + "="*60)
        print("ENSEMBLE BACKTEST WITH DYNAMIC Z-SCORE THRESHOLDING")
        print("="*60)
        print(f"Initial capital: ${INITIAL_CAPITAL:,.0f}")
        print(f"Z-Score Threshold: {SIGNAL_Z_THRESHOLD} sigma")
        print(f"Rolling Window: {SIGNAL_ROLLING_WINDOW} days")
        print(f"Daily allocation: ${INITIAL_CAPITAL/PREDICTION_HORIZON:,.0f}")
        
        # Identify tradable assets
        tradable_assets = BASE_TICKERS + [leg for ratio in DERIVED_RATIOS.values() 
                                          for leg in [ratio[0], ratio[1]]]
        tradable_assets = list(set(tradable_assets))
        
        tradable_mask = np.array([name in tradable_assets for name in asset_names])
        tradable_indices = np.where(tradable_mask)[0]
        
        print(f"\nAsset breakdown:")
        print(f"  Total assets: {len(asset_names)}")
        print(f"  Tradable: {len(tradable_indices)}")
        print(f"  Modifiers: {len(SIGNAL_MODIFIERS)}")
        
        buckets = {i: None for i in range(PREDICTION_HORIZON)}
        
        portfolio_value_history = []
        dates = []
        cash_balance = INITIAL_CAPITAL
        daily_allocation = INITIAL_CAPITAL / PREDICTION_HORIZON
        
        trades_opened = 0
        trades_closed = 0
        signals_filtered_count = 0
        
        for day_num, idx in enumerate(test_idx):
            if idx < LOOKBACK:
                continue
            
            if idx >= len(prices) - PREDICTION_HORIZON:
                break
            
            current_date = prices.index[idx]
            bucket_idx = day_num % PREDICTION_HORIZON
            
            # Close positions
            freed_capital = 0
            if buckets[bucket_idx] is not None:
                old_position = buckets[bucket_idx]
                
                entry_prices = old_position['entry_prices']
                exit_prices = prices.iloc[idx].values
                position_weights = old_position['positions']
                capital_used = old_position['capital']
                
                tradable_returns = (exit_prices[tradable_indices] / entry_prices[tradable_indices] - 1)
                tradable_weights = position_weights[tradable_indices]
                
                portfolio_return = np.sum(tradable_weights * tradable_returns)
                pnl = capital_used * portfolio_return
                freed_capital = capital_used + pnl
                
                buckets[bucket_idx] = None
                trades_closed += 1
            else:
                if cash_balance >= daily_allocation:
                    freed_capital = daily_allocation
                    cash_balance -= daily_allocation
                else:
                    freed_capital = cash_balance
                    cash_balance = 0
            
            # Generate ensemble signals
            prices_prev = prices.iloc[idx-1].values  # Yesterday's Close
            signals, all_model_signals = self.generate_ensemble_signals(
                features, idx, prices_prev, asset_names
            )
            
            # Decompose ratio signals
            leg_signals = self.trading_systems[0].decompose_ratio_signals(signals, asset_names)
            
            # Merge signals - ADD ratio-derived signals to base signals
            combined_signals = signals.copy()
            for leg_idx, leg_signal in leg_signals.items():
                combined_signals[leg_idx] += leg_signal
            
            # Zero out non-tradable
            for i, name in enumerate(asset_names):
                if name in SIGNAL_MODIFIERS or name in DERIVED_RATIOS:
                    combined_signals[i] = 0
            
            # Count how many signals passed threshold
            tradable_signals = combined_signals[tradable_indices]
            n_strong_signals = (np.abs(tradable_signals) > 0).sum()
            n_total_tradable = len(tradable_indices)
            signals_filtered_count += (n_total_tradable - n_strong_signals)
            
            # Position sizing - only for strong signals
            if n_strong_signals > 0:
                # Adaptive position sizing based on signal strength
                n_tradable = len(tradable_indices)
                n_long = max(1, min(int(0.3 * n_strong_signals), n_strong_signals // 2 + 1))
                n_short = max(1, min(int(0.3 * n_strong_signals), n_strong_signals // 2 + 1))
                
                n_long = min(n_long, n_strong_signals)
                n_short = min(n_short, n_strong_signals)
                
                # Strong mask is now determined by the Z-score logic inside generate_ensemble_signals
                # We just check which ones are non-zero
                strong_signal_mask = np.abs(tradable_signals) > 1e-6
                strong_indices = tradable_indices[strong_signal_mask]
                strong_signals = tradable_signals[strong_signal_mask]
                
                if len(strong_signals) == 0:
                    positions = np.zeros(len(asset_names))
                elif len(strong_signals) == 1:
                    positions = np.zeros(len(asset_names))
                    positions[strong_indices[0]] = 1.0 if strong_signals[0] > 0 else -1.0
                else:
                    top_strong_idx = np.argsort(strong_signals)[-n_long:]
                    bottom_strong_idx = np.argsort(strong_signals)[:n_short]
                    
                    top_idx = strong_indices[top_strong_idx]
                    bottom_idx = strong_indices[bottom_strong_idx]
                    
                    positions = np.zeros(len(asset_names))
                    if len(top_idx) > 0:
                        positions[top_idx] = 0.5 / len(top_idx)
                    if len(bottom_idx) > 0:
                        positions[bottom_idx] = -0.5/ len(bottom_idx)

            else:
                positions = np.zeros(len(asset_names))
            
            entry_prices = prices.iloc[idx].values
            
            buckets[bucket_idx] = {
                'positions': positions,
                'entry_prices': entry_prices,
                'capital': freed_capital,
                'entry_date': current_date,
                'signals': combined_signals
            }
            
            trades_opened += 1
            
            # Calculate portfolio value
            invested_value = 0
            for b_idx, bucket in buckets.items():
                if bucket is not None:
                    current_prices = prices.iloc[idx].values
                    
                    bucket_positions = bucket['positions']
                    bucket_entry_prices = bucket['entry_prices']
                    
                    tradable_returns = (current_prices[tradable_indices] / bucket_entry_prices[tradable_indices] - 1)
                    tradable_positions = bucket_positions[tradable_indices]
                    
                    bucket_value = bucket['capital'] * (1 + np.sum(tradable_positions * tradable_returns))
                    invested_value += bucket_value
            
            total_value = invested_value + cash_balance
            
            portfolio_value_history.append(total_value)
            dates.append(current_date)
        
        portfolio_values = np.array(portfolio_value_history)
        portfolio_returns = (portfolio_values / INITIAL_CAPITAL - 1) * 100
        
        print(f"\nTrading statistics:")
        print(f"  Trades opened: {trades_opened}")
        print(f"  Trades closed: {trades_closed}")
        print(f"  Signals filtered (below Z-threshold): {signals_filtered_count}")
        print(f"  Average % of signals filtered: {signals_filtered_count/(trades_opened*len(tradable_indices))*100:.1f}%")
        
        return dates, portfolio_returns, portfolio_values
    
    def generate_analytics_dashboard(self, features, prices, test_idx, asset_names, save_path='analytics_dashboard.png'):
        """
        Generate comprehensive analytics dashboard - FIXED VERSION
        
        Fixes:
        1. All asset names shown on network graph (including modifiers and ratios)
        2. Direct 2D projection of embeddings instead of t-SNE
        3. Proper feature names instead of "feat_n"
        """
        print("\n" + "="*60)
        print("GENERATING ANALYTICS DASHBOARD")
        print("="*60)
        
        # Use a recent snapshot from test period (middle of test set)
        snapshot_idx = test_idx[len(test_idx)//2]
        prices_prev = prices.iloc[snapshot_idx-1].values
        
        # Collect analytics from first model (representative)
        analytics = self.trading_systems[0].collect_analytics_snapshot(
            features, snapshot_idx, prices_prev, asset_names
        )
        
        if analytics is None:
            print("Cannot generate dashboard: insufficient data")
            return
        
        # Extract data
        predictions = analytics['predictions']
        signals = analytics['signals']
        embeddings = analytics['embeddings']
        feature_weights = analytics['feature_weights']
        signed_edges = analytics['signed_edges']
        edge_index = analytics['edge_index']
        current_features = analytics['features']
        
        # Identify tradable assets
        tradable_assets = BASE_TICKERS + [leg for ratio in DERIVED_RATIOS.values() 
                                        for leg in [ratio[0], ratio[1]]]
        tradable_assets = list(set(tradable_assets))
        tradable_mask = np.array([name in tradable_assets for name in asset_names])
        
        # ----- BUILD PROPER FEATURE NAMES -----
        n_assets = len(asset_names)
        basic_feature_names = ['ret_1d', 'ret_5d', 'ret_20d', 'vol_20d', 'rsi']
        n_basic = len(basic_feature_names)
        
        # Lévy features: for asset i, we have n_assets-1 levy features with other assets
        # Build full feature name list
        all_feature_names = basic_feature_names.copy()
        
        # Add Lévy feature names (these are relative to asset 0, but represent general structure)
        for j, name in enumerate(asset_names):
            if j < n_assets - 1:  # n_assets - 1 levy features
                all_feature_names.append(f'levy_{name}')
        
        # Truncate or pad to match actual feature count
        n_features = current_features.shape[1]
        if len(all_feature_names) < n_features:
            # Pad with generic names for any extras
            for i in range(len(all_feature_names), n_features):
                all_feature_names.append(f'feat_{i}')
        else:
            all_feature_names = all_feature_names[:n_features]
        # --------------------------------------
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # ===== 1. Asset Network Structure =====
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Create networkx graph
        G = nx.DiGraph()
        for i, name in enumerate(asset_names):
            G.add_node(i, name=name)
        
        # Add edges with weights
        edge_weights = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            weight = signed_edges[i]
            if abs(weight) > 0.2:  # Only show strong connections
                G.add_edge(src, dst, weight=weight)
                edge_weights.append(weight)
        
        # Node colors based on signals
        node_colors = []
        node_sizes = []
        for i in range(len(asset_names)):
            if signals[i] > 0.01:
                node_colors.append('#2E7D32')  # Green for long
            elif signals[i] < -0.01:
                node_colors.append('#C62828')  # Red for short
            else:
                node_colors.append('#BDBDBD')  # Gray for neutral
            node_sizes.append(300 + abs(predictions[i]) * 1000)
        
        # Draw network
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                            alpha=0.9, ax=ax1)
        
        # Draw edges with colors
        if len(edge_weights) > 0:
            edges = G.edges()
            edge_colors = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.RdYlGn,
                                width=1.5, alpha=0.6, arrows=False, ax=ax1,
                                edge_vmin=-1, edge_vmax=1)
        
        # FIX #1: Draw labels for ALL assets (not just tradable)
        labels = {i: asset_names[i] for i in range(len(asset_names))}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax1)
        
        ax1.set_title('Asset Network Structure\n(Node size = prediction strength, Green = Long, Red = Short)', 
                    fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # ===== 2. Asset Embedding Space (Direct 2D Projection) =====
        ax2 = fig.add_subplot(gs[0, 1])
        
        # FIX #2: Direct 2D projection instead of t-SNE
        # Just take first 2 dimensions of the embedding space
        if embeddings.shape[1] >= 2:
            embeddings_2d = embeddings[:, :2]
        else:
            # Fallback: pad with zeros if less than 2 dimensions
            embeddings_2d = np.zeros((embeddings.shape[0], 2))
            embeddings_2d[:, :embeddings.shape[1]] = embeddings
        
        # Color by prediction score
        scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=predictions, cmap='RdYlGn', s=150, alpha=0.7,
                            edgecolors='black', linewidth=0.5, vmin=-0.1, vmax=0.1)
        
        # Label ALL assets
        for i, name in enumerate(asset_names):
            ax2.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=7, ha='center', va='bottom')
        
        plt.colorbar(scatter, ax=ax2, label='Prediction Score')
        ax2.set_xlabel('Embedding Dimension 1', fontsize=10)
        ax2.set_ylabel('Embedding Dimension 2', fontsize=10)
        ax2.set_title('Asset Embedding Space (First 2 Dims)\n(Color = prediction score)', 
                    fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ===== 3. Asset Similarity Matrix =====
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Compute cosine similarity (can be negative)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = embeddings_norm @ embeddings_norm.T
        
        # Plot heatmap - ensure we use diverging colormap centered at 0
        im = ax3.imshow(similarity_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(asset_names)))
        ax3.set_yticks(range(len(asset_names)))
        ax3.set_xticklabels(asset_names, rotation=90, fontsize=7)
        ax3.set_yticklabels(asset_names, fontsize=7)
        plt.colorbar(im, ax=ax3, label='Cosine Similarity')
        ax3.set_title('Asset Similarity Matrix\n(from learned embeddings)', 
                    fontsize=11, fontweight='bold')
        
        # ===== 4. Feature Values (Top Long + Top Short) =====
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Get top 3 long and short positions
        tradable_signals = signals.copy()
        tradable_signals[~tradable_mask] = 0
        
        if np.any(np.abs(tradable_signals) > 1e-6):
            n_long = min(3, (tradable_signals > 0.01).sum())
            n_short = min(3, (tradable_signals < -0.01).sum())
            
            top_long_idx = np.argsort(tradable_signals)[-n_long:] if n_long > 0 else []
            top_short_idx = np.argsort(tradable_signals)[:n_short] if n_short > 0 else []
            
            selected_idx = list(top_long_idx) + list(top_short_idx)
            
            if len(selected_idx) > 0:
                # FIX #4: Select top features by importance and use proper names
                top_features_idx = np.argsort(np.abs(feature_weights))[-5:]
                
                feature_data = current_features[selected_idx][:, top_features_idx]
                
                # Normalize for visualization
                feat_std = feature_data.std(axis=0)
                feat_std[feat_std < 1e-8] = 1.0  # Avoid division by zero
                feature_data_norm = (feature_data - feature_data.mean(axis=0)) / feat_std
                
                im = ax4.imshow(feature_data_norm.T, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
                ax4.set_yticks(range(len(top_features_idx)))
                # Use actual feature names
                ax4.set_yticklabels([all_feature_names[i] for i in top_features_idx], fontsize=9)
                ax4.set_xticks(range(len(selected_idx)))
                ax4.set_xticklabels([asset_names[i] for i in selected_idx], rotation=45, 
                                ha='right', fontsize=9)
                plt.colorbar(im, ax=ax4, label='Normalized Value')
                ax4.set_title('Feature Values (Top 3 Long + Top 3 Short)', 
                            fontsize=11, fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'No positions', ha='center', va='center', fontsize=12)
                ax4.axis('off')
        else:
            ax4.text(0.5, 0.5, 'No tradable signals', ha='center', va='center', fontsize=12)
            ax4.axis('off')
        
        # ===== 5. Ranked Prediction Scores =====
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Sort by prediction score
        sorted_idx = np.argsort(predictions)
        sorted_predictions = predictions[sorted_idx]
        sorted_names = [asset_names[i] for i in sorted_idx]
        
        # Color bars
        colors = ['#C62828' if p < 0 else '#2E7D32' for p in sorted_predictions]
        
        bars = ax5.barh(range(len(sorted_predictions)), sorted_predictions, color=colors, alpha=0.7)
        ax5.set_yticks(range(len(sorted_names)))
        ax5.set_yticklabels(sorted_names, fontsize=8)
        ax5.set_xlabel('Prediction Score', fontsize=10)
        ax5.set_title('Ranked Prediction Scores\n(Green = Long Signal, Red = Short Signal)', 
                    fontsize=11, fontweight='bold')
        ax5.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax5.grid(True, alpha=0.3, axis='x')
        
        # ===== 6. Distribution of Edge Weights =====
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Filter out very weak edges for clarity
        significant_edges = signed_edges[np.abs(signed_edges) > 0.01]
        
        if len(significant_edges) > 0:
            ax6.hist(significant_edges, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax6.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero')
            ax6.set_xlabel('Edge Strength (Correlation)', fontsize=10)
            ax6.set_ylabel('Frequency', fontsize=10)
            ax6.set_title('Distribution of Edge Weights\n(Asset Relationship Strengths)', 
                        fontsize=11, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            mean_edge = np.mean(significant_edges)
            std_edge = np.std(significant_edges)
            ax6.text(0.02, 0.98, f'Mean: {mean_edge:.3f}\nStd: {std_edge:.3f}',
                    transform=ax6.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax6.text(0.5, 0.5, 'No significant edges', ha='center', va='center', fontsize=12)
            ax6.axis('off')
        
        # Overall title
        fig.suptitle('GNN Trading System Analytics Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to: {save_path}")
        plt.close()

def generate_supplementary_analysis(ensemble, features, prices, test_idx, all_assets):
    """
    Generate additional analytical plots - FIXED VERSION

    Fix: Added check for identical x values before calling linregress
    """

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ----- BUILD PROPER FEATURE NAMES -----
    n_assets = len(all_assets)
    basic_feature_names = ['ret_1d', 'ret_5d', 'ret_20d', 'vol_20d', 'rsi']
    n_features = features.shape[2]

    all_feature_names = basic_feature_names.copy()
    for j, name in enumerate(all_assets):
        if len(all_feature_names) < n_features:
            all_feature_names.append(f'levy_{name}')

    # Pad if needed
    while len(all_feature_names) < n_features:
        all_feature_names.append(f'feat_{len(all_feature_names)}')
    all_feature_names = all_feature_names[:n_features]
    # --------------------------------------

    # 1. Feature Importance Over Time
    ax = axes[0, 0]

    # Collect feature weights from all models
    all_feature_weights = []
    for trading_sys in ensemble.trading_systems:
        fw = trading_sys.model.feature_weights.detach().cpu().numpy()
        all_feature_weights.append(fw)

    mean_weights = np.mean(all_feature_weights, axis=0)
    std_weights = np.std(all_feature_weights, axis=0)

    # Plot top 10 features
    top_k = min(10, len(mean_weights))
    top_indices = np.argsort(np.abs(mean_weights))[-top_k:]

    y_pos = np.arange(top_k)
    ax.barh(y_pos, mean_weights[top_indices], xerr=std_weights[top_indices],
            color='steelblue', alpha=0.7, capsize=5)
    ax.set_yticks(y_pos)
    # Use actual feature names
    ax.set_yticklabels([all_feature_names[i] for i in top_indices])
    ax.set_xlabel('Feature Weight', fontsize=11)
    ax.set_title('Top 10 Feature Importance\n(Mean ± Std across ensemble)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 2. Signal Stability (ensemble agreement)
    ax = axes[0, 1]

    # Sample multiple points in test period
    sample_indices = test_idx[::len(test_idx)//20][:20]
    signal_std_over_time = []

    for idx in sample_indices:
        if idx < LOOKBACK:
            continue
        prices_prev = prices.iloc[idx-1].values
        _, all_model_signals = ensemble.generate_ensemble_signals(
            features, idx, prices_prev, all_assets
        )
        # Calculate std across models for each asset
        signal_std = np.std(all_model_signals, axis=0)
        signal_std_over_time.append(np.mean(signal_std))

    if signal_std_over_time:
        ax.plot(signal_std_over_time, marker='o', linestyle='-', 
            color='coral', linewidth=2, markersize=6)
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Mean Signal Std Dev', fontsize=11)
        ax.set_title('Signal Stability Across Ensemble\n(Lower = more agreement)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 3. Correlation Network Strength Over Time
    ax = axes[1, 0]

    # Get edge weights history from first model
    if ensemble.trading_systems[0].signed_edge_history:
        edge_history = ensemble.trading_systems[0].signed_edge_history[-100:]  # Last 100 days
        
        mean_abs_edges = [np.mean(np.abs(edges)) for edges in edge_history]
        max_abs_edges = [np.max(np.abs(edges)) for edges in edge_history]
        
        ax.plot(mean_abs_edges, label='Mean |Edge Weight|', 
            color='steelblue', linewidth=2)
        ax.plot(max_abs_edges, label='Max |Edge Weight|', 
            color='coral', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_ylabel('Edge Weight Magnitude', fontsize=11)
        ax.set_title('Network Connection Strength Over Time', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Prediction vs Actual Returns (Scatter)
    ax = axes[1, 1]

    # Collect predictions and actual returns for sample period
    sample_idx = test_idx[len(test_idx)//2]
    if sample_idx >= LOOKBACK and sample_idx < len(prices) - PREDICTION_HORIZON:
        
        prices_prev = prices.iloc[sample_idx-1].values
        ensemble_signals, _ = ensemble.generate_ensemble_signals(
            features, sample_idx, prices_prev, all_assets
        )
        
        # Get actual returns
        future_prices = prices.iloc[sample_idx + PREDICTION_HORIZON].values
        current_prices = prices.iloc[sample_idx].values
        actual_returns = (future_prices / current_prices - 1) * 100
        
        # Filter tradable assets
        tradable_assets = BASE_TICKERS + [leg for ratio in DERIVED_RATIOS.values() 
                                        for leg in [ratio[0], ratio[1]]]
        tradable_assets = list(set(tradable_assets))
        tradable_mask = np.array([name in tradable_assets for name in all_assets])
        
        # Create valid mask (tradable + no NaN)
        valid_mask = (tradable_mask & 
                        ~np.isnan(actual_returns) & 
                        ~np.isnan(ensemble_signals) &
                        ~np.isinf(actual_returns) &
                        ~np.isinf(ensemble_signals))
        
        if valid_mask.sum() > 0:
            scatter = ax.scatter(ensemble_signals[valid_mask], 
                            actual_returns[valid_mask],
                            c=actual_returns[valid_mask], 
                            cmap='RdYlGn', s=100, alpha=0.6, 
                            edgecolors='black', linewidth=0.5)
            
            # FIX: Check if x values have sufficient variance before linregress
            x_vals = ensemble_signals[valid_mask]
            y_vals = actual_returns[valid_mask]
            
            if valid_mask.sum() > 2 and np.std(x_vals) > 1e-8:
                # Safe to compute linear regression
                slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2,
                    label=f'$R^2$ = {r_value**2:.3f}')
                ax.legend()
            elif valid_mask.sum() > 2:
                # All x values are identical (e.g., all zeros after filtering)
                ax.text(0.5, 0.95, 'Signals constant (no regression)', 
                        transform=ax.transAxes, fontsize=9, ha='center',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            ax.set_xlabel('Ensemble Signal', fontsize=11)
            ax.set_ylabel(f'Actual Return (%) @ {PREDICTION_HORIZON}d', fontsize=11)
            ax.set_title('Prediction vs Actual Returns\n(Sample snapshot)', 
                        fontsize=12, fontweight='bold')
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Actual Return (%)')
        else:
            ax.text(0.5, 0.5, 'No valid data points', ha='center', va='center', fontsize=12)
            ax.set_title('Prediction vs Actual Returns\n(Sample snapshot)', 
                        fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        ax.set_title('Prediction vs Actual Returns\n(Sample snapshot)', 
                    fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('supplementary_analysis.png', dpi=300, bbox_inches='tight')
    print("Supplementary analysis saved to: supplementary_analysis.png")
    plt.close()


def main():
    """Main execution with ensemble"""
    print("=" * 60)
    print("ENHANCED GNN TRADING SYSTEM")
    print("=" * 60)
    print("ENHANCEMENTS:")
    print("1. Ensemble of 5 models (seeds 42-46)")
    print("2. Xavier weight initialization")
    print("3. Validation 20% with rolling IC early stopping")
    print("4. Differentiable rank correlation loss")
    print("5. Reduced complexity (2 GAT layers, hidden_dim=64)")
    print("6. Batch normalization in predictor")
    print("7. Signal confidence thresholding")
    print("=" * 60)
    
    # Set initial seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    loader = DataLoader()
    prices = loader.download_data()
    
    all_assets = list(prices.columns)
    
    feature_extractor = FeatureExtractor(levy_window=LEVY_WINDOW)
    features = feature_extractor.compute_features(prices)
    
    print(f"\nFinal feature shape: {features.shape}")
    
    # Compute future returns
    future_returns = np.zeros((len(prices), len(all_assets)))
    for i in range(len(prices) - PREDICTION_HORIZON):
        future_returns[i] = (
            prices.iloc[i + PREDICTION_HORIZON].values / prices.iloc[i].values - 1
        )
    
    n = len(prices)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    # fix split look ahead
    train_stop = max(LOOKBACK, train_end - PREDICTION_HORIZON)
    val_start = max(train_end, LOOKBACK)
    val_stop = max(val_start, val_end - PREDICTION_HORIZON)
    test_start = max(val_end, LOOKBACK)
    test_stop = n - PREDICTION_HORIZON

    train_idx = np.arange(LOOKBACK, train_stop)
    val_idx = np.arange(val_start, val_stop)
    test_idx = np.arange(test_start, test_stop)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_idx)} samples ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val:   {len(val_idx)} samples ({VAL_RATIO*100:.0f}%)")
    print(f"  Test:  {len(test_idx)} samples ({(1-TRAIN_RATIO-VAL_RATIO)*100:.0f}%)")
    
    n_features = features.shape[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create and train ensemble
    ensemble = EnsembleTradingSystem(n_features=n_features, device=device)
    ensemble.train_ensemble(features, future_returns, train_idx, val_idx)
    
    # Backtest
    dates, strategy_returns, portfolio_values = ensemble.backtest_ensemble(
        features, prices, test_idx, all_assets
    )
    
    # Create benchmarks
    base_prices = prices[BASE_TICKERS].loc[dates]
    benchmark_values = np.zeros(len(dates))
    benchmark_capital = INITIAL_CAPITAL
    
    for i, date in enumerate(dates):
        if i == 0:
            benchmark_values[i] = benchmark_capital
        else:
            prev_prices = base_prices.iloc[i-1].values
            curr_prices = base_prices.iloc[i].values
            asset_returns = (curr_prices / prev_prices - 1)
            portfolio_return = np.mean(asset_returns)
            benchmark_capital = benchmark_capital * (1 + portfolio_return)
            benchmark_values[i] = benchmark_capital
    
    benchmark_returns = (benchmark_values / INITIAL_CAPITAL - 1) * 100
    
    spy_prices = prices['SPY'].loc[dates]
    spy_values = INITIAL_CAPITAL * (spy_prices / spy_prices.iloc[0])
    spy_returns = (spy_values / INITIAL_CAPITAL - 1) * 100
    
    # Log returns for plotting
    strategy_log_returns = np.log(portfolio_values / INITIAL_CAPITAL) * 100
    benchmark_log_returns = np.log(benchmark_values / INITIAL_CAPITAL) * 100
    spy_log_returns = np.log(spy_values / INITIAL_CAPITAL) * 100
    
    plt.figure(figsize=(14, 7))
    plt.plot(dates, strategy_log_returns, label='Enhanced GNN Ensemble',
             linewidth=2.5, color='#2E86AB')
    plt.plot(dates, benchmark_log_returns, label='Equal-Weighted BASE',
             linewidth=2, alpha=0.7, color='#A23B72')
    plt.plot(dates, spy_log_returns, label='SPY',
             linewidth=2, alpha=0.7, color='#F18F01', linestyle='--')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (Log %)', fontsize=12)
    plt.title('Enhanced GNN Strategy: Ensemble + Rank Loss + Signal Thresholding',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('strategy_performance_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Performance metrics
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    strategy_total = strategy_returns[-1]
    benchmark_total = benchmark_returns[-1]
    spy_total = spy_returns.iloc[-1]
    
    n_years = len(dates) / 252
    strategy_annual = ((portfolio_values[-1] / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100
    benchmark_annual = ((benchmark_values[-1] / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100
    spy_annual = ((spy_values.iloc[-1] / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100
    
    strategy_daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    benchmark_daily_returns = np.diff(benchmark_values) / benchmark_values[:-1]
    spy_daily_returns = np.diff(spy_values.values) / spy_values.values[:-1]
    
    # strategy_sharpe = np.mean(strategy_daily_returns) / (np.std(strategy_daily_returns) + 1e-6) * np.sqrt(252)
    H = PREDICTION_HORIZON  
    values_sub = portfolio_values[::H] # sample every five days
    H_returns = values_sub[1:] / values_sub[:-1] - 1
    strategy_sharpe = np.mean(H_returns) / (np.std(H_returns) + 1e-6) * np.sqrt(252 / H)


    benchmark_sharpe = np.mean(benchmark_daily_returns) / (np.std(benchmark_daily_returns) + 1e-6) * np.sqrt(252)
    spy_sharpe = np.mean(spy_daily_returns) / (np.std(spy_daily_returns) + 1e-6) * np.sqrt(252)
    
    def max_drawdown(values):
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        return np.min(drawdown) * 100
    
    strategy_dd = max_drawdown(portfolio_values)
    benchmark_dd = max_drawdown(benchmark_values)
    spy_dd = max_drawdown(spy_values.values)
    
    win_rate = (strategy_daily_returns > 0).sum() / len(strategy_daily_returns) * 100
    
    print(f"\nEnhanced GNN Ensemble:")
    print(f"  Total Return:      {strategy_total:>12.2f}%")
    print(f"  Annualized Return: {strategy_annual:>12.2f}%")
    print(f"  Sharpe Ratio:      {strategy_sharpe:>12.2f}")
    print(f"  Max Drawdown:      {strategy_dd:>12.2f}%")
    print(f"  Win Rate:          {win_rate:>12.2f}%")
    
    print(f"\nBenchmark:")
    print(f"  Total Return:      {benchmark_total:>12.2f}%")
    print(f"  Annualized Return: {benchmark_annual:>12.2f}%")
    print(f"  Sharpe Ratio:      {benchmark_sharpe:>12.2f}")
    print(f"  Max Drawdown:      {benchmark_dd:>12.2f}%")
    
    print(f"\nSPY:")
    print(f"  Total Return:      {spy_total:>12.2f}%")
    print(f"  Annualized Return: {spy_annual:>12.2f}%")
    print(f"  Sharpe Ratio:      {spy_sharpe:>12.2f}")
    print(f"  Max Drawdown:      {spy_dd:>12.2f}%")
    
    print("\n" + "=" * 60)
    print("IMPROVEMENTS")
    print("=" * 60)
    print(f"Strategy vs Benchmark: {strategy_total - benchmark_total:>+.2f}%")
    print(f"Strategy vs SPY:       {strategy_total - spy_total:>+.2f}%")
    print(f"Sharpe improvement:    {strategy_sharpe - benchmark_sharpe:>+.2f}")
    print("=" * 60)

    print("\nGenerating analytics dashboard...")
    ensemble.generate_analytics_dashboard(
        features=features,
        prices=prices,
        test_idx=test_idx,
        asset_names=all_assets,
        save_path='analytics_dashboard.png'
    )

    print("\nGenerating supplementary analysis...")
    generate_supplementary_analysis(
        ensemble=ensemble,
        features=features,
        prices=prices,
        test_idx=test_idx,
        all_assets=all_assets
    )


if __name__ == '__main__':
    main()
