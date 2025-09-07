"""
Perceptron output surfaces for different activations and sample sizes.

Model:
    y = σ(z),  where z = -4.79*x1 + 5.90*x2 - 0.93

Activations:
    (a) Sigmoid:      σ(z) = 1 / (1 + exp(-z))
    (b) Hard limit:   σ(z) = 1[z >= 0]
    (c) RBF (Gaussian on z): σ(z) = exp(-z^2)

This script generates 9 figures total (3 activations × 3 sample sizes).
Figures are shown on screen and saved under ./perceptron_surfaces/.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- Domain
XMIN, XMAX = -2.0,  2.0
YMIN, YMAX = -2.0,  2.0

# --- Linear pre-activation
def z_fn(x1, x2):
    return -4.79 * x1 + 5.90 * x2 - 0.93

# --- Activations
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def hard_limit(z):
    return (z >= 0).astype(float)

def rbf_gaussian(z):
    # Assumption for "radial basis function" with scalar input: Gaussian φ(z) = exp(-z^2)
    return np.exp(-(z**2))

# --- Build a grid with (approximately) the requested number of samples
def grid_for_n(n_points):
    """
    Returns (X1, X2, actual_n) for a uniform grid over the domain
    with ~n_points samples. Ensures exactly 10x10, 50x100, 100x100 for 100/5000/10000.
    """
    if n_points == 100:
        nx, ny = 10, 10
    elif n_points == 5000:
        nx, ny = 50, 100
    elif n_points == 10000:
        nx, ny = 100, 100
    else:
        # Fallback: make a roughly square grid
        nx = int(np.floor(np.sqrt(n_points)))
        ny = int(np.ceil(n_points / nx))
    x = np.linspace(XMIN, XMAX, nx)
    y = np.linspace(YMIN, YMAX, ny)
    X1, X2 = np.meshgrid(x, y, indexing="xy")
    return X1, X2, nx * ny

# --- Plot helper
def plot_surface(X1, X2, Y, title, outpath):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, Y, linewidth=0, antialiased=False)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def main():
    out_dir = Path("perceptron_surfaces")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_sizes = [100, 5000, 10000]
    activations = [
        ("Sigmoid", sigmoid),
        ("Hard limit", hard_limit),
        ("Radial basis (exp(-z^2))", rbf_gaussian),
    ]

    for n in sample_sizes:
        X1, X2, actual_n = grid_for_n(n)
        Z = z_fn(X1, X2)

        for name, act in activations:
            Y = act(Z)
            fname = f"surface_{name.split()[0].lower()}_{actual_n}_points.png"
            outpath = out_dir / fname
            title = f"{name} — {actual_n} points"
            plot_surface(X1, X2, Y, title, outpath)

if __name__ == "__main__":
    main()
