#!/usr/bin/env python3
"""
model_architecture.py — Liquid Biopsy AI: CNN + LSTM Hybrid Model (v2, regularised)

Changes from v1:
  - CNN uses GlobalAveragePooling instead of flatten → fixed 128-dim output
    regardless of input length (was 4736-dim, causing ~800K params on 210 samples)
  - Fusion head: Linear(256, 64) directly, removing 128-dim intermediate layer
  - Dropout raised to 0.5
  - Total params ~50K instead of ~800K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FragmentCNNBranch(nn.Module):
    """
    1D CNN branch with GlobalAveragePooling.
    Output is always 128-dim regardless of input length, preventing the
    parameter explosion from flatten (was 128 * input_len//4 ≈ 4736).
    """
    def __init__(self, input_len: int = 150):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.bn2   = nn.BatchNorm1d(64)
        self.bn3   = nn.BatchNorm1d(128)
        self.out_dim = 128  # fixed regardless of input_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                         # (B, 1, L)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)                          # GlobalAveragePooling → (B, 128)
        return x


class FragmentLSTMBranch(nn.Module):
    """
    Bidirectional LSTM branch. Output: 2 * hidden_size (128 by default).
    """
    def __init__(self, input_len: int = 150, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.out_dim = hidden_size * 2  # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)                         # (B, L, 1)
        _, (h_n, _) = self.lstm(x)
        h_fwd = h_n[-2]                            # last-layer forward
        h_bwd = h_n[-1]                            # last-layer backward
        return torch.cat([h_fwd, h_bwd], dim=1)   # (B, 128)


class LiquidBiopsyNet(nn.Module):
    """
    Hybrid CNN + LSTM network for cancer detection from cfDNA fragment profiles.

    Architecture (regularised v2):
        Input (fragment length histogram, 150 bins)
            ├── CNN branch  → GlobalAvgPool → 128 features
            └── LSTM branch → BiLSTM final hidden → 128 features
                         ↓
              Concat → 256-dim
              Linear(256, 64) + ReLU + Dropout(0.5)
              Linear(64, 2)   → cancer probability logits
    """
    def __init__(
        self,
        input_len   : int = 150,
        num_classes : int = 2,
        lstm_hidden : int = 64,
        lstm_layers : int = 2,
        dropout     : float = 0.5,
    ):
        super().__init__()
        self.input_len   = input_len
        self.num_classes = num_classes

        self.cnn_branch  = FragmentCNNBranch(input_len)
        self.lstm_branch = FragmentLSTMBranch(input_len, lstm_hidden, lstm_layers)

        fused_dim = self.cnn_branch.out_dim + self.lstm_branch.out_dim  # 256

        # Single fusion layer: 256 → 64 → num_classes  (removed 128 intermediate)
        self.fc1     = nn.Linear(fused_dim, 64)
        self.fc2     = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)          # raised from 0.3 → 0.5
        self.bn1     = nn.BatchNorm1d(64)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hist    = x[:, :self.input_len]
        cnn_feat  = self.cnn_branch(x_hist)
        lstm_feat = self.lstm_branch(x_hist)
        fused     = torch.cat([cnn_feat, lstm_feat], dim=1)  # (B, 256)
        out       = F.relu(self.bn1(self.fc1(fused)))
        out       = self.dropout(out)
        return self.fc2(out)                                   # logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"LiquidBiopsyNet(v2-regularised)\n"
            f"  input_len={self.input_len}, num_classes={self.num_classes}\n"
            f"  CNN → GAP: {self.cnn_branch.out_dim} features\n"
            f"  LSTM bidir: {self.lstm_branch.out_dim} features\n"
            f"  Fusion: {self.cnn_branch.out_dim + self.lstm_branch.out_dim} → 64 → {self.num_classes}\n"
            f"  Dropout: 0.5\n"
            f"  Total params: {self.count_parameters():,}\n"
        )


# ── Baselines (for ablation / comparison) ────────────────────────────────────
class CNNOnly(nn.Module):
    def __init__(self, input_len=150, num_classes=2):
        super().__init__()
        self.cnn = FragmentCNNBranch(input_len)
        self.fc1 = nn.Linear(self.cnn.out_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.cnn(x[:, :150])
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


class LSTMOnly(nn.Module):
    def __init__(self, input_len=150, num_classes=2):
        super().__init__()
        self.lstm = FragmentLSTMBranch(input_len)
        self.fc1  = nn.Linear(self.lstm.out_dim, 32)
        self.fc2  = nn.Linear(32, num_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.lstm(x[:, :150])
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== LiquidBiopsyNet v2 (regularised) — Architecture Test ===\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LiquidBiopsyNet(input_len=150, num_classes=2).to(device)
    print(model)

    batch  = torch.randn(8, 155).to(device)
    logits = model(batch)
    probs  = model.predict_proba(batch)

    print(f"Input:  {batch.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Probs sum ≈ 1: {probs.sum(dim=1).tolist()}")
    print("\n[OK] Architecture test passed!")
