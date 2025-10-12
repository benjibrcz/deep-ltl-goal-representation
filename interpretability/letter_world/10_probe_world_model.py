import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='path to npz saved by 03_log_rollouts_letterworld.py')
    p.add_argument('--feature', type=str, default='actor_in', choices=['actor_in', 'obs_emb', 'e_sigma'])
    p.add_argument('--target', type=str, default='pos', choices=['pos', 'next_pos'])
    p.add_argument('--split', type=float, default=0.8)
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.data, allow_pickle=True)

    X_all = data[args.feature]
    # Keep only array entries
    mask = np.array([isinstance(x, np.ndarray) for x in X_all])
    X = [x for x in X_all if isinstance(x, np.ndarray)]
    if len(X) < 2:
        print(f"Not enough samples for {args.feature}: {len(X)}")
        return
    X = np.stack(X, axis=0)

    Y_all = data[args.target]
    Y = Y_all[mask]
    try:
        Y = np.stack(Y, axis=0)
    except Exception:
        Y = np.array(Y)

    # Train/test split
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(args.split * n)
    tr, te = idx[:split], idx[split:]
    Xtr, Xte = X[tr], X[te]
    Ytr, Yte = Y[tr], Y[te]

    # Regression for positions
    reg = LinearRegression()
    reg.fit(Xtr, Ytr)
    pred = reg.predict(Xte)
    r2 = r2_score(Yte, pred, multioutput='variance_weighted')
    print(f'R2({args.feature} -> {args.target}): {r2:.3f}')


if __name__ == '__main__':
    main()


