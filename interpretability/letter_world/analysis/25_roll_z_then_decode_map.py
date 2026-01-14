#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def one_hot(a, K):
    M = np.zeros((len(a), K)); M[np.arange(len(a)), a] = 1.0; return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='branched dataset with z, next maps')
    ap.add_argument('--z_feature', default='actor_prelogits')
    ap.add_argument('--lds', required=True, help='npz with A,B,b (scaled space)')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', default='interpretability/letter_world/results/roll_decode_metrics.txt')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    Z = np.stack([z for z in d[args.z_feature] if isinstance(z, np.ndarray)])
    A = np.asarray(d['branched_action']).astype(int)[:len(Z)]
    ORn = np.stack([o for o in d['next_obs_raw'] if isinstance(o, np.ndarray)])[:len(Z)]
    # Train map decoder y(cell,letter) <- z
    N, H, W, C = ORn.shape
    ch_sums = ORn.reshape(N, -1, C).sum(axis=1).mean(axis=0)
    agent_ch = int(np.argmin(ch_sums)); letter_chs = [c for c in range(C) if c != agent_ch]
    Ys = []
    for i in range(H):
        for j in range(W):
            for c in letter_chs:
                Ys.append((ORn[:, i, j, c] > 0.5).astype(int))
    Y = np.stack(Ys, axis=1)
    dec = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000))
    dec.fit(Z, Y)

    # Load LDS and roll
    l = np.load(args.lds, allow_pickle=True)
    A1, B, b = l['A1'], l['B'], l['b']
    K = B.shape[1]
    z = Z.copy()
    aurocs = []
    for step in range(1, args.k + 1):
        u = one_hot(A, K)
        z = (z @ A1.T) + (u @ B.T) + b
        p = dec.predict_proba(z)  # returns list of arrays; simplify by stacking first column
        # crude: evaluate AUROC per target then average
        P = np.column_stack([c[:, 1] for c in p])
        au = []
        for k in range(Y.shape[1]):
            try:
                au.append(roc_auc_score(Y[:, k], P[:, k]))
            except ValueError:
                pass
        aurocs.append(float(np.mean(au)))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        f.write('AUROC per step: ' + ','.join(f"{x:.3f}" for x in aurocs) + '\n')
    print('Saved', out)


if __name__ == '__main__':
    main()


