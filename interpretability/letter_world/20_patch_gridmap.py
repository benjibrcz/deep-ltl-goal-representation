#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='actor_prelogits')
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--out', type=str, default='interpretability/letter_world/results/patch_gridmap.npz')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    Z_all = d[args.feature]
    A_all = d['action']
    P_all = d['pos']
    E_all = d['episode'] if 'episode' in d.files else np.zeros(len(A_all))
    mask = np.array([isinstance(z, np.ndarray) and isinstance(p, np.ndarray) for z, p in zip(Z_all, P_all)])
    Z = np.stack(Z_all[mask]); A = np.asarray(A_all[mask], int); P = np.stack(P_all[mask]); E = E_all[mask]

    H = int(P[:, 0].max()) + 1; W = int(P[:, 1].max()) + 1

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    tr, te = next(gss.split(Z, groups=E))

    clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000))
    clf.fit(Z[tr], A[tr])

    # Simple steering dir: PC1 on train
    Z_tr = Z[tr]
    mu = Z_tr.mean(0, keepdims=True)
    Zc = Z_tr - mu
    v = np.random.randn(Zc.shape[1]); v /= (np.linalg.norm(v) + 1e-8)
    for _ in range(10):
        v = (Zc.T @ (Zc @ v))
        v /= (np.linalg.norm(v) + 1e-8)

    p_base = clf.predict_proba(Z[te])
    ce_base = -np.log(p_base[np.arange(len(p_base)), A[te]] + 1e-12)
    Zp = Z[te] + args.alpha * v
    p_patch = clf.predict_proba(Zp)
    ce_patch = -np.log(p_patch[np.arange(len(p_patch)), A[te]] + 1e-12)
    flips = (np.argmax(p_base, -1) != np.argmax(p_patch, -1)).astype(np.float32)

    flip_map = np.zeros((H, W), dtype=np.float32)
    dce_map = np.zeros((H, W), dtype=np.float32)
    cnt_map = np.zeros((H, W), dtype=np.float32)
    P_te = P[te].astype(int)
    for i, (r, c) in enumerate(P_te):
        flip_map[r, c] += flips[i]
        dce_map[r, c] += (ce_patch[i] - ce_base[i])
        cnt_map[r, c] += 1.0
    cnt_map[cnt_map == 0] = 1.0
    flip_map /= cnt_map
    dce_map /= cnt_map

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, flip_map=flip_map, dce_map=dce_map, counts=cnt_map)
    print(f"Saved grid patch maps to {out}")


if __name__ == '__main__':
    main()


