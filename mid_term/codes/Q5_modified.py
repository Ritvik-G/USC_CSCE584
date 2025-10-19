import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ----------------------------
# Load CIFAR-10 batches
# ----------------------------
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def load_cifar10(data_dir="../cifar-10-batches-py"):
    # Load all 5 training batches
    X_train, y_train = [], []
    for i in range(1, 6):
        batch = unpickle(f"{data_dir}/data_batch_{i}")
        X_train.append(batch["data"])
        y_train += batch["labels"]
    X_train = np.concatenate(X_train)
    y_train = np.array(y_train)

    # Load test batch
    test_batch = unpickle(f"{data_dir}/test_batch")
    X_test = test_batch["data"]
    y_test = np.array(test_batch["labels"])

    # Reshape and scale to [0,1]
    X_train = X_train.reshape(-1, 3, 32, 32).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 3, 32, 32).astype("float32") / 255.0
    return X_train, y_train, X_test, y_test


class CIFARDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------
# Three-layer CNN model (from Question 3)
# ----------------------------
class ThreeLayerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

        # Updated over question 3 : two FC hidden layers before the final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),                # (N, 128, 1, 1) -> (N, 128)
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),         # CIFAR-10 logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)                 # -> (N, 128, 1, 1)
        x = self.classifier(x)          # -> (N, 10)
        return x


# ----------------------------
# Training and evaluation
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = criterion(outputs, y)
        loss_sum += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), correct / total

# ----------------------------
# Plots
# ----------------------------
def plot_curves(train_vals, val_vals, ylabel, title, filename):
    # One chart per figure, no custom colors/styles (kept simple for students)
    plt.figure()
    plt.plot(train_vals, label="Train")
    plt.plot(val_vals, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    try:
        plt.show()
    except Exception:
        # In some environments (like headless servers), .show() might fail
        pass

# ----------------------------
# Main 
# ----------------------------
def main():
    # Load data
    X_train, y_train, X_test, y_test = load_cifar10("../cifar-10-batches-py")

    # Build full training dataset, then split into train/val (e.g., 45k/5k)
    full_train_ds = CIFARDataset(X_train, y_train)
    val_size = 5000
    train_size = len(full_train_ds) - val_size
    train_ds, val_ds = random_split(
        full_train_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Build test dataset
    test_ds = CIFARDataset(X_test, y_test)

    # Dataloaders
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    # Create model, loss, optimizer
    model = ThreeLayerCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop with validation tracking
    epochs = 30 
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d}: "
              f"Train loss={tr_loss:.3f}, acc={tr_acc*100:.2f}% | "
              f"Val loss={va_loss:.3f}, acc={va_acc*100:.2f}%")

    # Plot curves
    os.makedirs("figs", exist_ok=True)
    plot_curves(train_losses, val_losses, ylabel="Loss",     title="Loss (Train vs Validation)", filename="figs/loss_curve.png")
    plot_curves(train_accs,   val_accs,   ylabel="Accuracy", title="Accuracy (Train vs Validation)", filename="figs/accuracy_curve.png")
    print("Saved plots to figs/loss_curve.png and figs/accuracy_curve.png")

    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (Val acc = {best_val_acc*100:.2f}%).")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test: loss={test_loss:.3f}, acc={test_acc*100:.2f}%")

    # Save model (optional)
    torch.save(model.state_dict(), "three_layer_cnn_cifar10.pth")
    print("Model saved as 'three_layer_cnn_cifar10.pth'.")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    main()
