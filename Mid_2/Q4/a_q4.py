import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# -------------------------
# Config
# -------------------------
DATASET_NAME = "Citeseer"

HIDDEN_DIM = 64      
DROPOUT = 0.6        
LR = 0.01
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 300     # max epochs; early stopping implemented
PATIENCE = 50        # epochs without val improvement before stopping

OUTPUT_DIR = "./q4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -------------------------
# Load dataset
# -------------------------
dataset = Planetoid(root="./data/Citeseer", name=DATASET_NAME)
data = dataset[0].to(DEVICE)

print("Dataset:", DATASET_NAME)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Node feature dimension:", dataset.num_node_features)
print("Number of classes:", dataset.num_classes)

# -------------------------
# Simple 2-layer GCN model
# -------------------------
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # logits

model = GCN(
    in_dim=dataset.num_node_features,
    hidden_dim=HIDDEN_DIM,
    out_dim=dataset.num_classes,
    dropout=DROPOUT,
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
criterion = nn.CrossEntropyLoss()

# -------------------------
# Training / evaluation helpers
# -------------------------
def train_one_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate():
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1)

    accs = {}
    for split_name, mask in [
        ("train", data.train_mask),
        ("val", data.val_mask),
        ("test", data.test_mask),
    ]:
        correct = (preds[mask] == data.y[mask]).sum().item()
        total = int(mask.sum())
        accs[split_name] = correct / total
    return accs

# -------------------------
# Training loop with early stopping
# -------------------------
train_losses = []
train_accs = []
val_accs = []
test_accs = []

best_val_acc = 0.0
best_test_at_val = 0.0
best_epoch = 0
patience_counter = 0

best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")

for epoch in range(1, NUM_EPOCHS + 1):
    loss = train_one_epoch()
    accs = evaluate()

    train_losses.append(loss)
    train_accs.append(accs["train"])
    val_accs.append(accs["val"])
    test_accs.append(accs["test"])

    # track best val
    if accs["val"] > best_val_acc:
        best_val_acc = accs["val"]
        best_test_at_val = accs["test"]
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Train Acc: {accs['train']:.4f} | "
            f"Val Acc: {accs['val']:.4f} | "
            f"Test Acc: {accs['test']:.4f}"
        )

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
        break

print("\nTraining finished.")
print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
print(f"Test accuracy at that epoch: {best_test_at_val:.4f}")

# Load best model for saving / plotting
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

# -------------------------
# Save metrics & model
# -------------------------
model_path = os.path.join(OUTPUT_DIR, "model_final.pt")
torch.save(model.state_dict(), model_path)
print("Saved best model to:", model_path)

metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    for epoch in range(len(train_losses)):
        f.write(
            f"Epoch {epoch+1}: "
            f"loss={train_losses[epoch]:.4f}, "
            f"train_acc={train_accs[epoch]:.4f}, "
            f"val_acc={val_accs[epoch]:.4f}, "
            f"test_acc={test_accs[epoch]:.4f}\n"
        )
    f.write(
        f"\nBest val_acc={best_val_acc:.4f} at epoch={best_epoch}, "
        f"test_acc_at_best_val={best_test_at_val:.4f}\n"
    )
print("Saved metrics to:", metrics_path)

# -------------------------
# Save plots
# -------------------------
epochs_range = range(1, len(train_losses) + 1)

plt.figure()
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GCN Training Loss (Citeseer, tuned)")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = os.path.join(OUTPUT_DIR, "loss.png")
plt.savefig(loss_plot_path)
plt.close()
print("Saved loss plot to:", loss_plot_path)

plt.figure()
plt.plot(epochs_range, train_accs, label="Train Acc")
plt.plot(epochs_range, val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("GCN Accuracy (Citeseer, tuned)")
plt.legend()
plt.grid(True)
plt.tight_layout()
acc_plot_path = os.path.join(OUTPUT_DIR, "accuracy.png")
plt.savefig(acc_plot_path)
plt.close()
print("Saved accuracy plot to:", acc_plot_path)

print("\nAll done. Results are in:", os.path.abspath(OUTPUT_DIR))
