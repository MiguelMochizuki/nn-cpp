#!/usr/bin/env python3
"""PyTorch (CPU, eager) counterpart of the nn-cpp bench workloads.

Builds the same six training workloads as apps/bench/workloads.cpp (same shapes,
batch sizes, optimizers and learning rates) and times a step the same way as
runner.cpp: forward + loss, then zero_grad + backward, then optimizer step.
Reports the median over runs and writes the same CSV columns as `bench`, with
config = "torch", so compare_csv.py and plain diffs work on both.

Usage: bench_torch.py --out torch.csv [--runs 50] [--warmup 5] [--seed 1]
Threads follow NN_NUM_THREADS (default: all cores), as in `bench`.
"""
import argparse
import csv
import math
import os
import time
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

WORKLOADS = ["mlp_wide", "cnn_mnist", "cnn_large", "mlp_adam", "cnn_fer", "transformer_tiny"]


def mlp():
    """784-1024-1024-10 MLP shared by mlp_wide and mlp_adam."""
    return nn.Sequential(nn.Linear(784, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 10))


def cnn(channels, side, hidden):
    """Conv(3x3, pad 1) -> ReLU -> MaxPool(2) per channel entry, then the classifier head.

    Args:
        channels: Channel counts, input first.
        side: Input height and width.
        hidden: Hidden Linear width, 0 for none.

    Returns:
        The nn.Sequential model.
    """
    layers = []
    for i in range(len(channels) - 1):
        layers += [nn.Conv2d(channels[i], channels[i + 1], 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)]
        side //= 2
    flat = channels[-1] * side * side
    layers.append(nn.Flatten())
    if hidden:
        layers += [nn.Linear(flat, hidden), nn.ReLU()]
        flat = hidden
    layers.append(nn.Linear(flat, 10))
    return nn.Sequential(*layers)


def cnn_fer():
    """Conv-BN-ReLU stacks (16, 32, 64), global average pooling, 7-class head."""
    layers, cin = [], 1
    for c in (16, 32, 64):
        layers += [nn.Conv2d(cin, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU()]
        if c != 64:
            layers.append(nn.MaxPool2d(2, 2))
        cin = c
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 7))


class TinyTransformer(nn.Module):
    """Decoder-only pre-norm Transformer matching nn::Transformer (sinusoidal positions, tanh GELU)."""

    def __init__(self, vocab=64, dim=64, heads=4, layers=2, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.pow(10000.0, -torch.arange(0, dim, 2) / dim)
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(max_len))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True,
                                       activation=partial(F.gelu, approximate="tanh"))
            for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids):
        h = self.embed(ids) + self.pe[: ids.shape[1]]
        for b in self.blocks:
            h = b(h, src_mask=self.mask[: ids.shape[1], : ids.shape[1]], is_causal=True)
        return self.head(self.ln_f(h)).reshape(-1, self.head.out_features)


def make(name, seed):
    """Build one workload.

    Args:
        name: Workload name from WORKLOADS.
        seed: Seed for parameters and inputs.

    Returns:
        (model, optimizer, x, y) with random inputs and random integer targets.
    """
    torch.manual_seed(seed)
    if name in ("mlp_wide", "mlp_adam"):
        m, shape, classes = mlp(), (256, 784), 10
        opt = torch.optim.SGD(m.parameters(), lr=0.05) if name == "mlp_wide" else torch.optim.Adam(m.parameters(), lr=1e-3)
    elif name == "cnn_mnist":
        m, shape, classes = cnn([1, 8, 16], 28, 128), (32, 1, 28, 28), 10
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
    elif name == "cnn_large":
        m, shape, classes = cnn([3, 32, 64], 64, 0), (64, 3, 64, 64), 10
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
    elif name == "cnn_fer":
        m, shape, classes = cnn_fer(), (64, 1, 48, 48), 7
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    else:
        m, classes = TinyTransformer(), 64
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        return m, opt, torch.randint(0, 64, (16, 64)), torch.randint(0, 64, (16 * 64,))
    return m, opt, torch.rand(shape), torch.randint(0, classes, (shape[0],))


def step(m, opt, x, y):
    """One timed training step; returns (forward_ms, backward_ms, optim_ms)."""
    t0 = time.perf_counter()
    loss = F.cross_entropy(m(x), y)
    t1 = time.perf_counter()
    opt.zero_grad()
    loss.backward()
    t2 = time.perf_counter()
    opt.step()
    t3 = time.perf_counter()
    return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3


def quantile(s, q):
    """Linear-interpolated quantile of a sorted list (same as stats.hpp)."""
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label", default="pinned", choices=["pinned", "unpinned"])
    a = ap.parse_args()
    threads = int(os.environ.get("NN_NUM_THREADS", os.cpu_count()))
    torch.set_num_threads(threads)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("config,label,threads,benchmark,size,forward_ms,backward_ms,optim_ms,total_ms,gflops,spread,runs".split(","))
        for name in WORKLOADS:
            m, opt, x, y = make(name, a.seed)
            for _ in range(a.warmup):
                step(m, opt, x, y)
            rows = [step(m, opt, x, y) for _ in range(a.runs)]
            tot = sorted(sum(r) for r in rows)
            med = [quantile(sorted(r[i] for r in rows), 0.5) for i in range(3)]
            spread = (quantile(tot, 0.75) - quantile(tot, 0.25)) / quantile(tot, 0.5)
            w.writerow(["torch", a.label, threads, name, "", *(f"{v:.6g}" for v in med),
                        f"{quantile(tot, 0.5):.6g}", "", f"{spread:.6g}", a.runs])
            f.flush()


if __name__ == "__main__":
    main()
