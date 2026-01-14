#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def build_labels(obs_raw: np.ndarray) -> tuple[np.ndarray, list[str], int, list[int]]:
    # obs_raw: [N, H, W, C] one-hot-ish channels per letter + agent
    N, H, W, C = obs_raw.shape
    # Robust agent detection: ~one active pixel per frame
    per_frame_counts = (obs_raw > 0.5).reshape(N, -1, C).sum(axis=1)  # [N, C]
    oneish = np.median(per_frame_counts, axis=0)
    agent_ch = int(np.argmin(np.abs(oneish - 1.0)))
    # Letter-like channels: typically small active pixel count
    letter_chs = []
    for c in range(C):
        if c == agent_ch:
            continue
        occ = (obs_raw[..., c] > 0.5).reshape(N, -1).sum(1)
        med = float(np.median(occ))
        if 0.5 <= med <= 2.5:  # tuneable threshold
            letter_chs.append(c)
    if not letter_chs:
        # fallback: treat all non-agent as letters
        letter_chs = [c for c in range(C) if c != agent_ch]
    labels = []
    names = []
    for i in range(H):
        for j in range(W):
            for c in letter_chs:
                y = (obs_raw[:, i, j, c] > 0.5).astype(int)
                labels.append(y)
                names.append(f"cell({i},{j})_ch{c}")
    Y = np.stack(labels, axis=1)  # [N, H*W*L]
    return Y, names, agent_ch, letter_chs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='obs_conv', help='env activation like obs_conv or obs_local_flat')
    ap.add_argument('--out_csv', default='interpretability/letter_world/results/cell_letter_auc.csv')
    ap.add_argument('--permute_labels', action='store_true')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X_all = d[args.feature]
    E_all = d['episode'] if 'episode' in d.files else np.zeros(len(X_all))
    OR = d['obs_raw'] if 'obs_raw' in d.files else None
    mask = np.array([isinstance(x, np.ndarray) for x in X_all])
    X = np.stack([X_all[i] for i in np.where(mask)[0]])
    E = np.asarray(E_all)[mask]
    ORm = np.stack([OR[i] for i in np.where(mask)[0]]) if OR is not None else None
    assert ORm is not None, 'obs_raw required in dataset for labels'

    Y, names, agent_ch, letter_chs = build_labels(ORm)
    # quick diagnostic: agent & letter channels
    print(f"Detected agent channel: {agent_ch}; letter-like channels (first 5): {letter_chs[:5]}")
    # Egocentric heuristic: check if agent tends to be central
    center = (ORm.shape[1]//2, ORm.shape[2]//2)
    agent_mask0 = ORm[0, ..., agent_ch] > 0.5
    coords0 = np.argwhere(agent_mask0)
    if coords0.size:
        print(f"Agent sample coord (frame0): {coords0[0].tolist()} (center {center})")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    tr, te = next(gss.split(X, groups=E))

    pipe = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, class_weight='balanced'))
    rows = []
    # Permutation control
    Y_te_perm = Y[te].copy()
    rng = np.random.RandomState(0)
    rng.shuffle(Y_te_perm)
    included = 0; aurocs = []; auprcs = []
    for k, name in enumerate(names):
        y = Y[:, k]
        # minimum positives/negatives for stability
        valid = (y.sum() >= 20) and ((len(y) - y.sum()) >= 20)
        if not valid:
            continue
        included += 1
        if args.permute_labels:
            y_te = Y_te_perm[:, k]
        else:
            y_te = y[te]
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        try:
            auroc = roc_auc_score(y_te, p)
            auprc = average_precision_score(y_te, p)
        except ValueError:
            continue
        aurocs.append(auroc); auprcs.append(auprc)
        # simple ECE via 10-bin reliability
        bins = np.linspace(0, 1, 11)
        ece = 0.0; n = len(y_te)
        for i in range(10):
            m = (p >= bins[i]) & (p < bins[i+1])
            if m.any():
                ece += np.abs(p[m].mean() - y_te[m].mean()) * m.sum()
        ece /= n
        rows.append(dict(target=name, auroc=float(auroc), auprc=float(auprc), ece=float(ece)))

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved per-(cell,letter) AUCs to {out}")
    if aurocs:
        print(f"Summary: N_targets={included} | Macro AUROC={np.mean(aurocs):.3f} | Macro AUPRC={np.mean(auprcs):.3f}")


if __name__ == '__main__':
    main()


