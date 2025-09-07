# mnist_width_sweep.py
# Sweeps hidden-layer size for the SAME 1-hidden-layer architecture and plots accuracy.
# Run: python mnist_width_sweep.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# 0) Reproducibility helper
# --------------------------
np.random.seed(0)

# --------------------------
# 1) Load data (same format as your code)
# --------------------------
train_df = pd.read_csv('./archive/mnist_train.csv')
test_df  = pd.read_csv('./archive/mnist_test.csv')

y_train_int = np.array(train_df.iloc[:, 0])
X_train = np.array(train_df.iloc[:, 1:]) / 255.0

y_test_int = np.array(test_df.iloc[:, 0])
X_test = np.array(test_df.iloc[:, 1:]) / 255.0

# One-hot encoder (10 classes)
def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes), dtype=float)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

Y_train = one_hot(y_train_int, 10)
Y_test  = one_hot(y_test_int, 10)  # not needed for accuracy, but kept for symmetry

# --------------------------
# 2) Your DNN class (unchanged behavior)
# --------------------------
class DNN():
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        for i in range(len(layers)-1):
            layers_weights = np.random.rand(layers[i+1], layers[i] + 1)  # +1 for bias
            self.weights.append(layers_weights)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-0.01 * x))  # same scaled sigmoid

    def predict(self, data):
        x_s = [data]
        for i in range(len(self.layers)-1):
            # add bias
            x_s[-1] = np.concatenate((x_s[-1], [1]))
            z = np.dot(self.weights[i], x_s[i])
            x_s.append(self.sigmoid(z))
        return x_s[-1]

    def train(self, data, y_true):
        x_s = [data]
        # forward (store activations)
        for i in range(len(self.layers)-1):
            x_s[-1] = np.concatenate((x_s[-1], [1]))  # add bias
            z = np.dot(self.weights[i], x_s[i])
            x_s.append(self.sigmoid(z))

        # output layer error signal (squared error + sigmoid')
        psi = []
        for i in range(len(y_true)):
            output = x_s[-1][i]
            psi.append(-2*(y_true[i] - output) * (output * (1 - output)))
        psi = np.array(psi).reshape((-1, 1))

        # gradients list: last layer first
        gradients = []
        gradients.append(psi * x_s[-2])  # includes bias of previous layer

        # backprop hidden layers
        for i in range(len(self.layers) - 2, 0, -1):
            w = self.weights[i][:, :-1]      # drop bias column
            x = x_s[i][:-1]                  # drop bias entry
            term = w * x * (1 - x)           # broadcasting elementwise
            term = term.T
            psi = np.dot(term, psi).reshape((-1, 1))
            gradients.append(psi * x_s[i-1]) # includes bias of previous prev layer

        # SGD update (lr = 0.1, same as your code)
        for i in range(len(gradients)):
            self.weights[i] -= 0.1 * gradients[-(i+1)]

        # return per-sample squared error
        return np.sum((y_true - x_s[-1])**2)

# --------------------------
# 3) Helpers to train & evaluate
# --------------------------
def train_steps(model, X, Y, steps=10000, report_every=2000):
    n = X.shape[0]
    running = 0.0
    for t in range(1, steps+1):
        idx = np.random.randint(0, n)
        running += model.train(X[idx], Y[idx])
        if (t % report_every) == 0:
            print(f"  step {t}: avg loss over last {report_every} ≈ {running/report_every:.4f}")
            running = 0.0

def accuracy(model, X, y_int, max_samples=None):
    # Evaluate classification accuracy
    if max_samples is None or max_samples > X.shape[0]:
        max_samples = X.shape[0]
    correct = 0
    for i in range(max_samples):
        pred = np.argmax(model.predict(X[i]))
        if pred == y_int[i]:
            correct += 1
    return 100.0 * correct / max_samples

# --------------------------
# 4) Sweep hidden sizes (same depth: 1 hidden layer)
# --------------------------
HIDDEN_SIZES = [50, 100, 200, 400, 800, 1250]
TRAIN_STEPS = 10000          # adjust up for better accuracy, down for speed
TRAIN_EVAL_SAMPLES = 5000    # to estimate train accuracy faster
TEST_EVAL_SAMPLES = None     # None = full test set (10,000 rows)

train_accs = []
test_accs = []

for h in HIDDEN_SIZES:
    print(f"\n=== Training model with hidden size = {h} ===")
    model = DNN([784, h, 10])  # same architecture, only hidden width changes

    train_steps(model, X_train, Y_train, steps=TRAIN_STEPS, report_every=2000)

    tr_acc = accuracy(model, X_train, y_train_int, max_samples=TRAIN_EVAL_SAMPLES)
    te_acc = accuracy(model, X_test,  y_test_int,  max_samples=TEST_EVAL_SAMPLES)
    print(f"  Train accuracy (@{TRAIN_EVAL_SAMPLES} samples): {tr_acc:.2f}%")
    print(f"  Test  accuracy: {te_acc:.2f}%")

    train_accs.append(tr_acc)
    test_accs.append(te_acc)

# --------------------------
# 5) Plot and save the figure
# --------------------------
plt.figure(figsize=(10, 7), dpi=150)
plt.plot(HIDDEN_SIZES, train_accs, marker='o', linewidth=2, label='Train accuracy')
plt.plot(HIDDEN_SIZES, test_accs,  marker='o', linewidth=2, label='Test accuracy')
plt.xlabel("Hidden layer size (neurons)")
plt.ylabel("Accuracy (%)")
plt.title("Test accuracy vs hidden-layer size (same depth)")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.savefig("hidden_size_vs_accuracy.png")
print("\nSaved figure: hidden_size_vs_accuracy.png")
