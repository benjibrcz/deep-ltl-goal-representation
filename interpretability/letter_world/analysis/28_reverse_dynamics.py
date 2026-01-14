#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_seq_npz(path: str, feature_key: str):
    D = np.load(path, allow_pickle=True)
    Z = D[feature_key]
    if isinstance(Z, np.ndarray) and Z.dtype == object:
        Z = np.vstack([np.asarray(z).ravel() for z in Z]).astype(np.float32)
    else:
        Z = np.asarray(Z)
        if Z.ndim > 2:
            Z = Z.reshape(Z.shape[0], -1)
        Z = Z.astype(np.float32)
    A = np.asarray(D['action']).astype(int)
    E = D['episode'] if 'episode' in D.files else np.zeros(len(A))
    return Z, A, np.asarray(E)


def build_pairs(Z: np.ndarray, A: np.ndarray, E: np.ndarray):
    idx_t = []
    idx_tp = []
    for e in np.unique(E):
        ids = np.where(E == e)[0]
        if len(ids) < 2:
            continue
        idx_t.append(ids[:-1])
        idx_tp.append(ids[1:])
    if not idx_t:
        raise SystemExit('Not enough sequential steps to build pairs')
    t = np.concatenate(idx_t)
    tp = np.concatenate(idx_tp)
    return Z[t], Z[tp], A[t]


def eval_variant(X: np.ndarray, y: np.ndarray, stratify=True, seed=0):
    strat = y if stratify else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
    clf = LogisticRegression(max_iter=200, multi_class='multinomial')
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    acc = accuracy_score(yte, yhat)
    # per-action accuracy
    acc_by_a = {}
    for a in np.unique(yte):
        m = (yte == a)
        if m.any():
            acc_by_a[int(a)] = float(accuracy_score(yte[m], yhat[m]))
    return acc, acc_by_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='feature_t')
    ap.add_argument('--out_csv', default='interpretability/letter_world/results/rev_dyn.csv')
    ap.add_argument('--permute', action='store_true', help='Shuffle pairing across time for control')
    args = ap.parse_args()

    Z, A, E = load_seq_npz(args.data, args.feature)
    Zt, Ztp, y = build_pairs(Z, A, E)

    # Build design matrices
    X_pair = np.concatenate([Zt, Ztp], axis=1)
    X_diff = Ztp - Zt
    X_zt = Zt
    X_ztp = Ztp

    # Permutation control for X_pair
    if args.permute:
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(Ztp))
        X_pair = np.concatenate([Zt, Ztp[perm]], axis=1)

    rows = []
    for name, X in [('pair', X_pair), ('diff', X_diff), ('zt', X_zt), ('ztp', X_ztp)]:
        acc, acc_by_a = eval_variant(X, y, stratify=True)
        row = dict(variant=name, acc=acc)
        for a, v in acc_by_a.items():
            row[f'acc_a{a}'] = v
        rows.append(row)

    # write CSV
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open('w', newline='') as f:
        keys = sorted(set().union(*[r.keys() for r in rows]))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print('Saved reverse dynamics results to', out)


if __name__ == '__main__':
    main()


