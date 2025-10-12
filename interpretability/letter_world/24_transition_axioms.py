#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path


def shift_matrix(H: int, W: int, action: int) -> np.ndarray:
    # actions: 0=RIGHT,1=LEFT,2=DOWN,3=UP
    S = H * W
    T = np.zeros((S, S), dtype=float)
    for i in range(H):
        for j in range(W):
            s = i * W + j
            if action == 0:
                ni, nj = i, (j + 1) % W
            elif action == 1:
                ni, nj = i, (j - 1) % W
            elif action == 2:
                ni, nj = (i + 1) % H, j
            else:
                ni, nj = (i - 1) % H, j
            sp = ni * W + nj
            T[sp, s] = 1.0
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=int, default=7)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--trials', type=int, default=200)
    args = ap.parse_args()

    H = W = args.grid
    Ts = [shift_matrix(H, W, a) for a in range(4)]

    rng = np.random.RandomState(0)
    hamm = []
    tv = []
    for _ in range(args.trials):
        m0 = np.zeros(H * W); m0[rng.randint(0, H * W)] = 1.0
        actions = rng.randint(0, 4, size=args.k)
        m = m0.copy()
        for a in actions:
            m = Ts[a] @ m
        # Ground truth using same operators (identity here)
        m_gt = m0.copy()
        for a in actions:
            m_gt = Ts[a] @ m_gt
        hamm.append(float((m == m_gt).mean()))
        tv.append(0.5 * np.abs(m - m_gt).sum())
    print(f"Hamming acc mean: {np.mean(hamm):.3f} @ k={args.k}")
    print(f"TV mean: {np.mean(tv):.3f}")


if __name__ == '__main__':
    main()


