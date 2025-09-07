
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# -----------------------------
# Network parameters (given)
# -----------------------------
V_T = np.array([[-2.69, -2.80],
                [-3.39, -4.56]])  # shape (2, 2)
b_v = np.array([-2.21,  4.76])    # shape (2,)
W   = np.array([-4.91,  4.95])    # shape (2,)
b_w = -2.28                        # scalar

# -----------------------------
# Activations
# -----------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def hard_limit(z):
    return (z >= 0).astype(float)

def rbf(z):
    # element-wise Gaussian φ(z) = exp(-z^2)
    return np.exp(-(z ** 2))

ACTS = [
    ("Sigmoid", sigmoid),
    ("Hard limit", hard_limit),
    ("Radial basis", rbf),
]

SAMPLES = [100, 5000, 10000]

# -----------------------------
# Sampling grid helper
# -----------------------------
def make_grid(n_pts, x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0):
    rows = int(np.sqrt(n_pts))
    while n_pts % rows != 0:
        rows -= 1
    cols = n_pts // rows
    x = np.linspace(x_min, x_max, cols)
    y = np.linspace(y_min, y_max, rows)
    X, Y = np.meshgrid(x, y)
    return X, Y

# -----------------------------
# Forward pass
# -----------------------------
def nn_forward(X1, X2, act_fn):
    x = np.stack([X1.ravel(), X2.ravel()], axis=0)  # (2, N)
    z = V_T @ x + b_v[:, None]                      # (2, N)
    h = act_fn(z)                                   # (2, N)
    y = W @ h + b_w                                 # (N,)
    return y.reshape(X1.shape)

# -----------------------------
# Plot one surface to file
# -----------------------------
def plot_surface_to_file(X, Y, Z, title, path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)  # default colors
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Text measurement (Pillow 10+ safe)
# -----------------------------
def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """
    Returns (width, height) of text using textbbox() if available,
    otherwise falls back to textsize() for older Pillow.
    """
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    # Pillow < 10
    return draw.textsize(text, font=font)

# -----------------------------
# Main
# -----------------------------
def main():
    out_dir = Path("./nn_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_paths = []  # (act_name, n_samples, path)

    # Generate and save the 9 individual panels
    for act_name, act_fn in ACTS:
        for n in SAMPLES:
            X, Y = make_grid(n)
            Z = nn_forward(X, Y, act_fn)
            title = f"{act_name} — {n} samples"
            img_path = out_dir / f"panel_{act_name.replace(' ', '_').lower()}_{n}.png"
            plot_surface_to_file(X, Y, Z, title, img_path)
            panel_paths.append((act_name, n, img_path))

    # Load panels and stitch into a single labeled PNG (3 rows × 3 cols)
    imgs = [Image.open(str(p)) for _, _, p in panel_paths]
    pw, ph = imgs[0].size

    rows, cols = 3, 3
    margin = 30
    pad = 20
    col_header_h = 60
    row_header_w = 180

    canvas_w = margin + row_header_w + cols * pw + (cols - 1) * pad + margin
    canvas_h = margin + col_header_h + rows * ph + (rows - 1) * pad + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Fonts (fallback if unavailable)
    try:
        font_header = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        font_label = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font_header = ImageFont.load_default()
        font_label = ImageFont.load_default()

    # Column headers: sample sizes
    for c, n in enumerate(SAMPLES):
        text = f"{n} samples"
        tx = margin + row_header_w + c * (pw + pad) + pw // 2
        ty = margin + col_header_h // 2
        w, h = text_size(draw, text, font_header)
        draw.text((tx - w // 2, ty - h // 2), text, font=font_header, fill=(0, 0, 0))

    # Row headers: activation names
    for r, (act_name, _) in enumerate(ACTS):
        text = act_name
        tx = margin + row_header_w // 2
        ty = margin + col_header_h + r * (ph + pad) + ph // 2
        w, h = text_size(draw, text, font_header)
        draw.text((tx - w // 2, ty - h // 2), text, font=font_header, fill=(0, 0, 0))

    # Paste panels
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x = margin + row_header_w + c * (pw + pad)
            y = margin + col_header_h + r * (ph + pad)
            canvas.paste(imgs[idx], (x, y))
            idx += 1

    final_path = out_dir / "nn_output_surfaces_grid.png"
    canvas.save(final_path, format="PNG")
    print(f"Saved: {final_path.resolve()}")

if __name__ == "__main__":
    main()
