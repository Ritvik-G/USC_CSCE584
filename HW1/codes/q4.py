#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file GNN overview: train GCN/GAT/GraphSAGE on Cora/CiteSeer/PubMed (Planetoid).
Usage examples:
  python gnn_overview_single.py --model gcn --dataset Cora
  python gnn_overview_single.py --model gat --dataset Cora --epochs 300
  python gnn_overview_single.py --model sage --dataset PubMed --hidden 128

Requires: torch, torch_geometric (and its deps). If torch_geometric is missing,
this script prints a friendly install hint.
"""

from __future__ import annotations
import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
except Exception as e:
    print("\n[Import Error] You need PyTorch Geometric to run this script.")
    print("Quick start (CPU example):")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("  pip install torch-geometric")
    print("\nFor CUDA builds, follow the official install matrix:")
    print("  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html\n")
    raise

# --------------------------
# Utilities
# --------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@dataclass
class TrainConfig:
    model: str = "gcn"              # gcn | gat | sage
    dataset: str = "Cora"           # Cora | CiteSeer | PubMed
    hidden: int = 64
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.5
    heads: int = 8                  # for GAT first layer
    out_heads: int = 1              # for GAT last layer
    patience: int = 50              # early stopping
    runs: int = 1                   # repeat runs for mean/std
    seed: int = 42
    verbose: bool = True

# --------------------------
# Models
# --------------------------

class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden, out_channels, cached=True, normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float, heads: int, out_heads: int):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        # Concatenation happens over heads -> hidden*heads
        self.gat2 = GATConv(hidden*heads, out_channels, heads=out_heads, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden)
        self.sage2 = SAGEConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        return x

def build_model(name: str, in_dim: int, hidden: int, out_dim: int, cfg: TrainConfig) -> nn.Module:
    name = name.lower()
    if name == "gcn":
        return GCN(in_dim, hidden, out_dim, cfg.dropout)
    if name == "gat":
        return GAT(in_dim, hidden, out_dim, cfg.dropout, cfg.heads, cfg.out_heads)
    if name == "sage":
        return GraphSAGE(in_dim, hidden, out_dim, cfg.dropout)
    raise ValueError(f"Unknown model: {name}")

# --------------------------
# Data
# --------------------------

def load_data(name: str, data_dir: str = "./data") -> Tuple[Planetoid, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = Planetoid(root=os.path.join(data_dir, name), name=name, transform=NormalizeFeatures())
    data = dataset[0]
    return dataset, data

# --------------------------
# Train & Eval
# --------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == y).sum().item()
    return correct / y.numel()

def train_one(model: nn.Module, data, optimizer: torch.optim.Optimizer, weight_decay: float) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    # L2 weight decay: Adam already supports weight_decay, but we show manual penalty if needed.
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model: nn.Module, data) -> Tuple[float, float, float]:
    model.eval()
    logits = model(data.x, data.edge_index)
    train_acc = accuracy(logits[data.train_mask], data.y[data.train_mask])
    val_acc = accuracy(logits[data.val_mask], data.y[data.val_mask])
    test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
    return train_acc, val_acc, test_acc

def run_one(cfg: TrainConfig, device: torch.device) -> Tuple[float, float, float]:
    dataset, data = load_data(cfg.dataset)
    data = data.to(device)

    # Reasonable defaults: slightly lower WD for GAT/SAGE than classic GCN recipe
    wd = cfg.weight_decay
    if cfg.model.lower() in {"gat", "sage"}:
        wd = 5e-4  # still fine; change if you like via CLI

    model = build_model(cfg.model, dataset.num_node_features, cfg.hidden, dataset.num_classes, cfg).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=wd)

    best_val = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one(model, data, optimizer, wd)
        train_acc, val_acc, test_acc = evaluate(model, data)

        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if cfg.verbose and (epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs):
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Train {train_acc*100:5.2f}% | "
                  f"Val {val_acc*100:5.2f}% | Test {test_acc*100:5.2f}%")

        if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
            if cfg.verbose:
                print(f"Early stopping at epoch {epoch} (no val improvement for {cfg.patience} epochs).")
            break

    # Load best model (by validation accuracy) for final test report:
    if best_state is not None:
        model.load_state_dict(best_state)

    train_acc, val_acc, test_acc = evaluate(model, data)
    return train_acc, val_acc, test_acc

# --------------------------
# CLI
# --------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Single-file GNN training on Planetoid datasets (Cora/CiteSeer/PubMed).")
    p.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "sage"], help="GNN architecture")
    p.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"], help="Dataset name")
    p.add_argument("--hidden", type=int, default=64, help="Hidden channels")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs (upper bound with early stopping)")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=5e-4, help="L2 weight decay")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout prob")
    p.add_argument("--heads", type=int, default=8, help="GAT heads for first layer")
    p.add_argument("--out_heads", type=int, default=1, help="GAT heads for output layer")
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience (0 disables)")
    p.add_argument("--runs", type=int, default=1, help="Repeat training runs and report mean/std")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--quiet", action="store_true", help="Less logging")
    args = p.parse_args()

    cfg = TrainConfig(
        model=args.model,
        dataset=args.dataset,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        heads=args.heads,
        out_heads=args.out_heads,
        patience=args.patience,
        runs=args.runs,
        seed=args.seed,
        verbose=not args.quiet,
    )
    return cfg

def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device()
    if cfg.verbose:
        print(f"Device: {device}")
        print(f"Config: {cfg}")

    results = []
    for r in range(cfg.runs):
        if cfg.runs > 1 and cfg.verbose:
            print(f"\n=== Run {r+1}/{cfg.runs} ===")
        set_seed(cfg.seed + r)  # vary seed across runs if repeating
        train_acc, val_acc, test_acc = run_one(cfg, device)
        results.append((train_acc, val_acc, test_acc))
        if cfg.verbose:
            print(f"[Run {r+1}] Final: Train {train_acc*100:.2f}% | Val {val_acc*100:.2f}% | Test {test_acc*100:.2f}%")

    if cfg.runs > 1:
        import statistics as stats
        train_list, val_list, test_list = zip(*results)
        print("\n==== Summary over runs ====")
        print(f"Train: mean {100*stats.mean(train_list):.2f}% ± {100*stats.pstdev(train_list):.2f}%")
        print(f"Val:   mean {100*stats.mean(val_list):.2f}% ± {100*stats.pstdev(val_list):.2f}%")
        print(f"Test:  mean {100*stats.mean(test_list):.2f}% ± {100*stats.pstdev(test_list):.2f}%")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
