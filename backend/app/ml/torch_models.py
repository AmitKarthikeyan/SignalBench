from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (B, hidden)
        logits = self.head(last)
        return logits

@dataclass
class TorchBundle:
    state_dict: dict
    n_features: int
    lookback: int
    feature_cols: list[str]

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    lookback: int,
    feature_cols: list[str],
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> TorchBundle:
    device = torch.device("cpu")
    model = LSTMClassifier(n_features=n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = -1.0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_acc = correct / max(total, 1)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    return TorchBundle(state_dict=best_state, n_features=n_features, lookback=lookback, feature_cols=feature_cols)

def predict_lstm(bundle: TorchBundle, X: np.ndarray) -> np.ndarray:
    device = torch.device("cpu")
    model = LSTMClassifier(n_features=bundle.n_features).to(device)
    model.load_state_dict(bundle.state_dict)
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs
