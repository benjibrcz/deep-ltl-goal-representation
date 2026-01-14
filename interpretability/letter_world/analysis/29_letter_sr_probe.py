#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def build_letter_targets(letters: np.ndarray, episodes: np.ndarray, H: int, gamma: float, L: int) -> np.ndarray:
    N = len(letters)
    V = np.zeros((N, L), dtype=np.float32)
    for e in np.unique(episodes):
        idx = np.where(episodes == e)[0]
        T = len(idx)
        for t in range(T):
            tglob = idx[t]
            for k in range(1, H + 1):
                if t + k >= T:
                    break
                li = int(letters[idx[t + k]])
                if li >= 0 and li < L:
                    V[tglob, li] += (gamma ** (k - 1))
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='feature_t')
    ap.add_argument('--letters', type=int, default=12, help='Number of letter channels/IDs to predict')
    ap.add_argument('--horizon', type=int, default=12)
    ap.add_argument('--gamma', type=float, default=0.9)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--out_csv', default='interpretability/letter_world/results/letter_sr.csv')
    ap.add_argument('--permute', action='store_true', help='Permute features across time (control)')
    ap.add_argument('--action_only', action='store_true', help='Use one-hot(a_t) instead of feature_t')
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    A = np.asarray(D['action']).astype(int)
    E = np.asarray(D['episode']) if 'episode' in D.files else np.zeros(len(A))
    LID = np.asarray(D['letter_id']) if 'letter_id' in D.files else None
    if LID is None:
        raise SystemExit('letter_id not found in dataset; regenerate sequential rollouts with letter logging')

    # Build X
    if args.action_only:
        K = int(A.max()) + 1
        Z = np.eye(K, dtype=np.float32)[A]
    else:
        Z = D[args.feature]
        if isinstance(Z, np.ndarray) and Z.dtype == object:
            Z = np.vstack([np.asarray(z).ravel() for z in Z]).astype(np.float32)
        else:
            Z = np.asarray(Z)
            if Z.ndim > 2:
                Z = Z.reshape(Z.shape[0], -1)
            Z = Z.astype(np.float32)

    V = build_letter_targets(LID, E, H=args.horizon, gamma=args.gamma, L=args.letters)

    # split by episode
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(gss.split(np.arange(len(Z)), groups=E))

    if args.permute:
        rng = np.random.RandomState(0)
        Z_tr, Z_te = Z[tr], Z[te]
        perm = rng.permutation(len(Z_te))
        Z_te = Z_te[perm]
    else:
        Z_tr, Z_te = Z[tr], Z[te]

    # Standardize targets per-dimension on train
    y_tr, y_te = V[tr], V[te]
    mu, std = y_tr.mean(axis=0, keepdims=True), y_tr.std(axis=0, keepdims=True) + 1e-6
    yz_tr, yz_te = (y_tr - mu)/std, (y_te - mu)/std

    probe = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=args.alpha))
    probe.fit(Z_tr, yz_tr)
    Yhat = probe.predict(Z_te) * std + mu

    # metrics: per-letter R2 and cosine macro
    eps = 1e-12
    var = y_te.var(axis=0)
    keep = var > 1e-8
    yt, yp = y_te[:, keep], Yhat[:, keep]
    ss_res = ((yt - yp)**2).sum(axis=0)
    ss_tot = ((yt - yt.mean(axis=0, keepdims=True))**2).sum(axis=0) + eps
    r2_macro = float(np.mean(1.0 - ss_res/ss_tot)) if keep.any() else float('nan')
    # cosine macro
    a = y_te / (np.linalg.norm(y_te, axis=1, keepdims=True) + eps)
    b = Yhat / (np.linalg.norm(Yhat, axis=1, keepdims=True) + eps)
    cos_macro = float(np.mean(np.sum(a*b, axis=1)))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['variant','r2_macro','cosine_macro'])
        w.writeheader()
        w.writerow(dict(variant=('action_only' if args.action_only else ('permute' if args.permute else 'base')), r2_macro=r2_macro, cosine_macro=cos_macro))
    print('Saved letter-SR probe to', out)


if __name__ == '__main__':
    main()


