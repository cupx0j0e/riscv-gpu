#!/usr/bin/env python3
"""
Train a tiny MNIST MLP (8x8 grayscale by default) and export symmetric int8 weights.

Default architecture: 14x14 -> 128 -> 10 with ReLU (resize and hidden configurable).
Export: JSON with per-layer int8 weights, int32 biases, input/weight/act scales.

Dependencies:
  pip install torch torchvision

Usage:
  python train_mini_mnist.py --epochs 15 --batch-size 256 --hidden 128 --resize 14 --out mnist_int8.json
"""

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MiniMLP(nn.Module):
    def __init__(self, input_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Quantization helpers (symmetric int8, zp=0)
# ---------------------------------------------------------------------------


def symmetric_scale(t: torch.Tensor) -> float:
    max_abs = t.abs().max().item()
    return (max_abs / 127.0) if max_abs > 0 else 1.0


def quantize_tensor(t: torch.Tensor, scale: float) -> torch.Tensor:
    q = torch.round(t / scale).clamp(-128, 127)
    return q.to(torch.int8)


# ---------------------------------------------------------------------------
# Export structure
# ---------------------------------------------------------------------------


@dataclass
class LinearLayerExport:
    w: List[List[int]]  # int8
    b: List[int]        # int32
    scale_in: float
    scale_w: float
    scale_out: float
    zp_in: int = 0
    zp_w: int = 0
    zp_out: int = 0


@dataclass
class ModelExport:
    input_scale: float
    input_zp: int
    layers: List[LinearLayerExport]
    description: str


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def load_data(batch_size: int, resize: int = 8) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),  # scales to [0,1]
    ])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train(model: nn.Module, loader, device, epochs: int, lr: float) -> None:
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            total += loss.item() * imgs.size(0)
        avg = total / len(loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}: loss {avg:.4f}")


def evaluate(model: nn.Module, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    acc = correct / total
    print(f"Test accuracy: {acc*100:.2f}%")
    return acc


# ---------------------------------------------------------------------------
# Quantize & export
# ---------------------------------------------------------------------------


def collect_act_max(model: nn.Module, loader, device) -> Tuple[float, float]:
    """Collect max abs activations for layer1 pre-ReLU and layer2 input."""
    model.eval()
    max1 = 0.0
    max2 = 0.0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            x = imgs.view(imgs.size(0), -1)
            a1 = model.net[1](x)  # first linear
            max1 = max(max1, a1.abs().max().item())
            h1 = torch.relu(a1)
            max2 = max(max2, h1.abs().max().item())
    return max1, max2


def export_int8(model: MiniMLP, calib_loader, device, input_scale: float) -> ModelExport:
    # Collect activation ranges for better scale_out choices
    act1_max, act1_relu_max = collect_act_max(model, calib_loader, device)

    layers_export: List[LinearLayerExport] = []

    # Layer 1
    lin1: nn.Linear = model.net[1]
    w1 = lin1.weight.detach().cpu()
    b1 = lin1.bias.detach().cpu()
    w1_scale = symmetric_scale(w1)
    w1_q = quantize_tensor(w1, w1_scale)
    # Bias in int32 domain: bias * (input_scale * w_scale)
    b1_q = torch.round(b1 / (input_scale * w1_scale)).to(torch.int32)
    # Activation scale after ReLU: use observed max
    act1_out_scale = max(act1_relu_max / 127.0, 1e-6)
    layers_export.append(LinearLayerExport(
        w=w1_q.tolist(),
        b=b1_q.tolist(),
        scale_in=input_scale,
        scale_w=w1_scale,
        scale_out=act1_out_scale,
    ))

    # Layer 2
    lin2: nn.Linear = model.net[3]
    w2 = lin2.weight.detach().cpu()
    b2 = lin2.bias.detach().cpu()
    w2_scale = symmetric_scale(w2)
    w2_q = quantize_tensor(w2, w2_scale)
    b2_q = torch.round(b2 / (act1_out_scale * w2_scale)).to(torch.int32)
    # Output logits scale: keep float in host; set to 1.0 so host treats int32 as logits
    layers_export.append(LinearLayerExport(
        w=w2_q.tolist(),
        b=b2_q.tolist(),
        scale_in=act1_out_scale,
        scale_w=w2_scale,
        scale_out=1.0,
    ))

    return ModelExport(
        input_scale=input_scale,
        input_zp=0,
        layers=layers_export,
        description="MNIST int8 MLP symmetric quantization (zp=0)",
    )


def save_export(export: ModelExport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(export), f, indent=2)
    print(f"Saved int8 model to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train and export tiny MNIST MLP (int8)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="mnist_int8.json")
    parser.add_argument("--hidden", type=int, default=128, help="hidden size (default 128)")
    parser.add_argument("--resize", type=int, default=14, help="input resize (default 14x14)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = load_data(args.batch_size, resize=args.resize)

    input_dim = args.resize * args.resize
    model = MiniMLP(input_dim=input_dim, hidden=args.hidden)
    train(model, train_loader, device, epochs=args.epochs, lr=args.lr)
    evaluate(model, test_loader, device)

    # Input scale: images are in [0,1], keep scale so max ~127 -> scale_in ~1/127
    input_scale = 1.0 / 127.0
    export = export_int8(model, test_loader, device, input_scale)
    save_export(export, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
