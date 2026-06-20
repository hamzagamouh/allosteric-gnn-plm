import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.classifier(x)


def bce_loss(logits, y, eps=1e-8):
    pos = y.sum()
    neg = y.numel() - pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=neg / (pos + eps))
    return criterion(logits, y.float())


def train_and_evaluate(
    model, train_feats, train_y, val_feats, val_y,
    num_epochs=1000, lr=1e-3, device=None, evaluate_each=10,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_y_t = torch.tensor(train_y).float().to(device)
    val_y_t = torch.tensor(val_y).float().to(device)
    N = train_feats.size(0)
    batch_size = 256
    idx = list(range(N))
    train_mccs, val_mccs = [], []

    for epoch in range(num_epochs):
        model.train()
        random.shuffle(idx)
        idx_t = torch.tensor(idx).to(device)
        epoch_loss = 0.0

        for i in range(0, N, batch_size):
            optimizer.zero_grad()
            b_idx = idx_t[i : i + batch_size]
            loss = bce_loss(model(train_feats[b_idx]), train_y_t[b_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b_idx)

        if epoch % evaluate_each == 0:
            model.eval()
            with torch.no_grad():
                tr_pred = (torch.sigmoid(model(train_feats)).flatten().cpu().numpy() > 0.5).astype(int)
                va_pred = (torch.sigmoid(model(val_feats)).flatten().cpu().numpy() > 0.5).astype(int)
            tm = matthews_corrcoef(train_y.ravel(), tr_pred)
            vm = matthews_corrcoef(val_y.ravel(), va_pred)
            train_mccs.append(tm)
            val_mccs.append(vm)
            print(f"Epoch {epoch:4d}: loss={epoch_loss/N:.4f}  train_mcc={tm:.4f}  val_mcc={vm:.4f}")

    return train_mccs, val_mccs
