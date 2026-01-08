#!/usr/bin/env python3
"""
Train tiny stroke classifiers (dx, dy) and export symmetric int8 weights for the systolic array.

Modes:
- binary (default): horizontal vs vertical (dx > dy ? class 0 : class 1)
- dir4: 4-way up/down/left/right classifier

Export format (JSON):
  {
    "weights": [[...], ...]  # int8, shape n_out x 2
    "bias": [...],            # int32, length n_out (scaled)
    "scale": float,           # weight scale
    "input_scale": float,     # 1/127 for dx/dy in [0,1]
    "zero_point": 0,
    "classes": ["up","down","left","right"] or ["horiz","vert"]
  }

Dependencies: pip install torch
Usage:
  python gesture_train.py --epochs 200 --out gesture_model_int8.json
  python gesture_train.py --mode dir4 --epochs 200 --out gesture_dir4_int8.json
"""

import argparse
import json
import os
import random
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

model_dir = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _ in range(n_samples):
        dx = random.uniform(0, 1)
        dy = random.uniform(0, 1)
        X.append([dx, dy])
        y.append(0 if dx > dy else 1)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GestureNet(nn.Module):
    def __init__(self, out_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(2, out_dim, bias=True)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(mode: str = "binary", epochs: int = 200, verbose: bool = True) -> GestureNet:
    if mode == "binary":
        X, y = generate_data(5000)
        out_dim = 2
    elif mode == "dir4":
        # Generate 4-way data: up/down/left/right based on dx, dy signs
        X_list = []
        y_list = []
        for _ in range(6000):
            dx = random.uniform(-1, 1)
            dy = random.uniform(-1, 1)
            # Normalize to [0,1] input space used elsewhere by shifting/scaling later
            X_list.append([0.5 + dx * 0.5, 0.5 + dy * 0.5])
            if abs(dx) > abs(dy):
                y_list.append(2 if dx < 0 else 3)  # left/right
            else:
                y_list.append(0 if dy < 0 else 1)  # up/down
        X = torch.tensor(X_list, dtype=torch.float32).numpy()
        y = torch.tensor(y_list, dtype=torch.int64).numpy()
        out_dim = 4
    else:
        raise ValueError("mode must be 'binary' or 'dir4'")

    split = int(0.8 * len(X))
    X_train = torch.from_numpy(X[:split])
    y_train = torch.from_numpy(y[:split])
    X_val = torch.from_numpy(X[split:])
    y_val = torch.from_numpy(y[split:])

    model = GestureNet(out_dim=out_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    best_acc = 0.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, val_acc={val_acc:.1%}")

    if best_weights:
        model.load_state_dict(best_weights)
    if verbose:
        print(f"Best val accuracy: {best_acc:.1%}")
    return model


def evaluate(model: GestureNet) -> None:
    tests = [
        ([0.9, 0.1], 0),
        ([0.1, 0.9], 1),
        ([0.6, 0.4], 0),
        ([0.4, 0.6], 1),
        ([0.55, 0.45], 0),
        ([0.45, 0.55], 1),
    ]
    ok = 0
    model.eval()
    for xy, exp in tests:
        with torch.no_grad():
            pred = model(torch.tensor([xy])).argmax().item()
        ok += int(pred == exp)
    print(f"Eval: {ok}/{len(tests)} correct")


# ---------------------------------------------------------------------------
# Export (symmetric int8, zp=0)
# ---------------------------------------------------------------------------

def save_int8(model: GestureNet, path: str, mode: str) -> None:
    w = model.fc.weight.detach().numpy()  # shape out x 2
    b = model.fc.bias.detach().numpy()    # shape out
    w_max = max(abs(w.min()), abs(w.max()))
    scale_w = 127.0 / (w_max + 1e-8)
    w_q = np.clip(np.round(w * scale_w), -128, 127).astype(np.int8)
    # Input scale assumes dx, dy in [0,1], mapped by q = round(x * 127)
    scale_in = 1.0 / 127.0
    # Bias in int32: bias_float / (scale_in * scale_w)
    b_q = np.round(b / (scale_in * scale_w)).astype(np.int32)
    classes = ["horiz", "vert"] if w.shape[0] == 2 else ["up", "down", "left", "right"]

    data = {
        "weights": w_q.tolist(),
        "bias": b_q.tolist(),
        "scale": float(scale_w),
        "input_scale": float(scale_in),
        "zero_point": 0,
        "classes": classes,
        "description": f"Gesture classifier int8 (zp=0), mode={mode}",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved int8 model to {path}")
    print(f"Weights (int8):\n{w_q}")
    print(f"Bias (int32): {b_q}")
    print(f"scale_w={scale_w:.6f}, input_scale={scale_in:.6f}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train gesture classifier and export int8 JSON")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", type=str, default="gesture_model_int8.json")
    ap.add_argument("--mode", type=str, choices=["binary", "dir4"], default="binary",
                    help="binary: horiz/vert; dir4: up/down/left/right")
    args = ap.parse_args()

    model = train_model(mode=args.mode, epochs=args.epochs, verbose=True)
    evaluate(model)
    out_path = os.path.join(model_dir, args.out)
    save_int8(model, out_path, mode=args.mode)


if __name__ == "__main__":
    main()
