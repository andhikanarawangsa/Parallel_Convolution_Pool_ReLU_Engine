#!/usr/bin/env python3
"""Golden model for Convolution (3 kernels) + Maxpool 2x2 + ReLU.

Reads tb/vectors/image.mem and tb/vectors/kernel.mem (hex, one value per line),
prints every stage in the same layout as the Verilog testbench so the two logs
can be compared directly. Pure Python, no third-party packages.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_W = IMG_H = 16
K = 3
NUM_K = 3


def read_mem(path, signed=False):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                v = int(line, 16)
                if signed and v >= 128:
                    v -= 256
                vals.append(v)
    return vals


def conv(img, ker):
    cw, ch = IMG_W - K + 1, IMG_H - K + 1
    out = []
    for r in range(ch):
        row = []
        for c in range(cw):
            acc = 0
            for i in range(K):
                for j in range(K):
                    acc += img[(r + i) * IMG_W + c + j] * ker[i * K + j]
            row.append(acc)
        out.append(row)
    return out


def pool(m):
    return [[max(m[2 * r][2 * c], m[2 * r][2 * c + 1],
                 m[2 * r + 1][2 * c], m[2 * r + 1][2 * c + 1])
             for c in range(len(m[0]) // 2)] for r in range(len(m) // 2)]


def relu(m):
    return [[max(0, v) for v in row] for row in m]


def sat(m, bits=8):
    top = (1 << bits) - 1
    return [[min(v, top) for v in row] for row in m]


def show(title, m):
    print(f"\n=== {title} ===")
    for row in m:
        print("  [" + "".join(f"{v:7d}" for v in row) + " ]")


def main():
    img = read_mem(os.path.join(ROOT, "tb/vectors/image.mem"))
    ker = read_mem(os.path.join(ROOT, "tb/vectors/kernel.mem"), signed=True)
    show(f"INPUT IMAGE ({IMG_W}x{IMG_H})",
         [img[r * IMG_W:(r + 1) * IMG_W] for r in range(IMG_H)])
    convs = [conv(img, ker[k * K * K:(k + 1) * K * K]) for k in range(NUM_K)]
    pools = [pool(c) for c in convs]
    relus = [relu(p) for p in pools]
    outs = [sat(r) for r in relus]
    for name, data in (("CONVOLUTION", convs), ("MAXPOOL 2x2", pools),
                       ("RELU", relus), ("FINAL OUTPUT 8-bit unsigned (saturated)", outs)):
        for k, m in enumerate(data):
            show(f"{name}, kernel/channel {k + 1} ({len(m[0])}x{len(m)})", m)


if __name__ == "__main__":
    sys.exit(main())
