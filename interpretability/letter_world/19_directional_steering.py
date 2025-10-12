#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def power_pca_first(Z: np.ndarray, iters: int = 10) -> np.ndarray:
    Zc = Z - Z.mean(0, keepdims=True)
    v = np.random.randn(Z.shape[1]); v /= (np.linalg.norm(v) + 1e-8)
    for _ in range(iters):
        v = (Zc.T @ (Zc @ v))
        v /= (np.linalg.norm(v) + 1e-8)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='actor_prelogits')
    ap.add_argument('--lds', type=str, default=None, help='Path to LDS npz for B columns')
    ap.add_argument('--alpha_grid', type=str, default='0,0.25,0.5,1.0')
    ap.add_argument('--out', type=str, default='interpretability/letter_world/results/steer_grid.csv')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    Z_all = d[args.feature]
    A_all = d['action']
    E_all = d['episode'] if 'episode' in d.files else np.zeros(len(A_all))
    mask = np.array([isinstance(z, np.ndarray) for z in Z_all])
    Z = np.stack(Z_all[mask]); A = np.asarray(A_all[mask], int); E = E_all[mask]

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    tr, te = next(gss.split(Z, groups=E))

    clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000))
    clf.fit(Z[tr], A[tr])

    dirs = []
    # PC1 direction
    dirs.append(('PC1', power_pca_first(Z[tr])))
    # Random orthogonal set (size 4)
    np.random.seed(0)
    D = Z.shape[1]
    Q, _ = np.linalg.qr(np.random.randn(D, 4))
    for i in range(4):
        dirs.append((f'rand{i+1}', Q[:, i]))
    # B columns if provided
    if args.lds is not None:
        npz = np.load(args.lds, allow_pickle=True)
        if 'B' in npz.files:
            B = npz['B']  # shape [D_out, K]
            for a in range(B.shape[1]):
                dirs.append((f'Bcol{a}', B[:, a]))

    alphas = [float(x) for x in args.alpha_grid.split(',') if x.strip()]
    rows = []
    for name, v in dirs:
        v = v / (np.linalg.norm(v) + 1e-8)
        p_base = clf.predict_proba(Z[te])
        ce_base = log_loss(A[te], p_base, labels=clf[-1].classes_)
        for a in alphas:
            Zp = Z[te] + a * v
            p_patch = clf.predict_proba(Zp)
            ce_patch = log_loss(A[te], p_patch, labels=clf[-1].classes_)
            flips = (np.argmax(p_base, -1) != np.argmax(p_patch, -1)).mean()
            rows.append(dict(feature=args.feature, dir=name, alpha=a, dce=float(ce_patch - ce_base), flip_rate=float(flips)))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved steering grid to {out}")


if __name__ == '__main__':
    main()


