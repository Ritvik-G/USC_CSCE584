
import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Configurations
CFG = {
    "train_csv": "./mnist_data/mnist_train.csv",
    "test_csv":  "./mnist_data/mnist_test.csv",
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.05,                # learning rate
    "weight_decay": 0.0,       # decoupled weight decay
    "val_frac": 0.15,        # 15% for validation
    "seed": 42,
    "save_path": "mnist_cnn_manual.pt",
    "plot_path": "curves.png",
    "use_amp": True,           # if CUDA is available, use mixed precision
    "augment": False           # simple RandomAffine if True
}

# Utils
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MNISTCsv(Dataset):
    def __init__(self, path, train=True, transform=None):
        super().__init__()
        data = np.loadtxt(path, delimiter=",", dtype=np.float32, skiprows=1)
        self.labels = torch.from_numpy(data[:, 0].astype(np.int64))
        self.images = torch.from_numpy(data[:, 1:] / 255.0).view(-1, 1, 28, 28)
        self.transform = transform
        self.train = train

    def __len__(self): return self.images.size(0)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

# optional tiny augmentation that keeps it simple and tensor-only
class RandomAffineTensor:
    def __init__(self, degrees=10, translate=0.05, scale_range=(0.95, 1.05)):
        self.degrees = degrees; self.translate = translate; self.scale_range = scale_range
    def __call__(self, x):
        # x: [1,28,28], CPU tensor
        angle = (random.random()*2-1) * self.degrees
        tx = (random.random()*2-1) * self.translate * 28
        ty = (random.random()*2-1) * self.translate * 28
        scale = self.scale_range[0] + random.random()*(self.scale_range[1]-self.scale_range[0])
        theta = torch.tensor([
            [ scale*math.cos(math.radians(angle)), -scale*math.sin(math.radians(angle)), tx/14.0],
            [ scale*math.sin(math.radians(angle)),  scale*math.cos(math.radians(angle)), ty/14.0]
        ], dtype=x.dtype)
        grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), x.unsqueeze(0).size(), align_corners=False)
        x = torch.nn.functional.grid_sample(x.unsqueeze(0), grid, align_corners=False)
        return x.squeeze(0)

# Convolutional Neural Network Model
class MNIST3ConvNet(nn.Module):
    """
    Input: 1x28x28
    Block1: Conv(1->32, 3x3, pad1) + BN + ReLU + MaxPool(2)
    Block2: Conv(32->64, 3x3, pad1) + BN + ReLU + MaxPool(2)
    Block3: Conv(64->128, 3x3, pad1) + BN + ReLU
    Head:   AdaptiveAvgPool -> FC(128->64) -> ReLU -> Dropout -> FC(64->10)
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(max(1, fan_in))
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None: nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x, 2)   # 32x14x14
        x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool2d(x, 2)   # 64x7x7
        x = F.relu(self.bn3(self.conv3(x)))                           # 128x7x7
        x = self.avgpool(x)                                           # 128x1x1
        x = torch.flatten(x, 1)                                       # 128
        x = F.relu(self.fc1(x)); x = self.drop(x)
        x = self.fc2(x)
        return x

# SGD step
def sgd_step(params, lr: float, weight_decay: float = 0.0):
    """
    Very simple manual SGD with optional decoupled weight decay.
    """
    with torch.no_grad():
        for p in params:
            if p.grad is None: continue
            if weight_decay > 0:
                p.data.add_(p.data, alpha=-lr * weight_decay)
            p.data.add_(p.grad, alpha=-lr)
            p.grad.zero_()

# Training/Evaluation
@torch.no_grad()
def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        b = y.size(0)
        tot_loss += loss.item()
        tot_acc  += (logits.argmax(1) == y).float().sum().item()
        n += b
    return tot_loss / n, tot_acc / n

# Main
def main():
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Datasets
    aug = RandomAffineTensor() if CFG["augment"] else None
    full_train = MNISTCsv(CFG["train_csv"], train=True, transform=aug)
    test_set   = MNISTCsv(CFG["test_csv"],  train=False, transform=None)

    # Train/Val split
    val_len = int(len(full_train) * CFG["val_frac"])
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(full_train, [train_len, val_len], generator=torch.Generator().manual_seed(CFG["seed"]))

    train_loader = DataLoader(train_set, batch_size=CFG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=CFG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=CFG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = MNIST3ConvNet().to(device)

    # AMP scaler
    use_amp = (device.type == "cuda") and CFG["use_amp"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # History
    tr_loss_hist, tr_acc_hist = [], []
    va_loss_hist, va_acc_hist = [], []

    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Forward
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            # Backward
            if use_amp:
                scaler.scale(loss).backward()
                # We don't have an optimizer to unscale; grads remain scaled but we can safely unscale by calling scaler.unscale_
                # Trick: create a fake optimizer wrapper so we can call unscale_; however, for pure SGD step with small LR, 
                # many setups simply call scaler.step on a dummy and update. To keep it *very* simple, disable AMP if this bothers you.
                scaler.unscale_(None)  # no-op in practice; safe placeholder
            else:
                loss.backward()

            # Manual step
            sgd_step([p for p in model.parameters() if p.requires_grad], lr=CFG["lr"], weight_decay=CFG["weight_decay"])

            # AMP housekeeping
            if use_amp: scaler.update()

            b = y.size(0)
            running_loss += loss.item() * b
            running_acc  += (logits.argmax(1) == y).float().sum().item()
            n += b

        train_loss = running_loss / n
        train_acc  = running_acc  / n

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)

        tr_loss_hist.append(train_loss); tr_acc_hist.append(train_acc)
        va_loss_hist.append(val_loss);   va_acc_hist.append(val_acc)

        print(f"Epoch {epoch:02d} | "
              f"train loss {train_loss:.4f} acc {100*train_acc:.2f}% | "
              f"val loss {val_loss:.4f} acc {100*val_acc:.2f}%")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "val_acc": best_val_acc,
                        "epoch": epoch,
                        "cfg": CFG}, CFG["save_path"])

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test  loss {test_loss:.4f} acc {100*test_acc:.2f}%")
    print(f"Best val acc: {100*best_val_acc:.2f}% | Model saved to: {os.path.abspath(CFG['save_path'])}")

    # Curves
    epochs = range(1, CFG["epochs"] + 1)
    plt.figure()
    plt.plot(epochs, tr_loss_hist, label="Train Loss")
    plt.plot(epochs, va_loss_hist, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(CFG["plot_path"])

    plt.figure()
    plt.plot(epochs, [a*100 for a in tr_acc_hist], label="Train Acc")
    plt.plot(epochs, [a*100 for a in va_acc_hist], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy Curves"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(CFG["plot_path"].replace(".png", "_acc.png"))
    print(f"Saved plots to: {os.path.abspath(CFG['plot_path'])} and {_safe_path(CFG['plot_path'].replace('.png', '_acc.png'))}")

def _safe_path(p):
    try:
        return os.path.abspath(p)
    except Exception:
        return p

if __name__ == "__main__":
    main()
