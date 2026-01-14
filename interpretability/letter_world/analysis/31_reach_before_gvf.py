#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


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
    E = np.asarray(D['episode']) if 'episode' in D.files else np.zeros(len(A))
    LID = np.asarray(D['letter_id']) if 'letter_id' in D.files else None
    if LID is None:
        raise SystemExit("letter_id missing; regenerate with 03c_log_rollouts_seq.py after recent changes")
    return Z, A, E, LID


def build_reach_before_targets(letter_id: np.ndarray, episodes: np.ndarray, A_id: int, B_id: int, H: int):
    y = np.full((len(letter_id),), None, dtype=object)
    for e in np.unique(episodes):
        idx = np.where(episodes == e)[0]
        T = len(idx)
        for t_local in range(T):
            t = idx[t_local]
            # scan future
            winner = None
            for k in range(1, H + 1):
                if t_local + k >= T:
                    break
                li = int(letter_id[idx[t_local + k]])
                if li == A_id:
                    winner = 1; break
                if li == B_id:
                    winner = 0; break
            y[t] = winner
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='feature_t')
    ap.add_argument('--pair', type=str, default='0,1', help='letter ids as A,B (e.g., 0,1)')
    ap.add_argument('--horizon', type=int, default=12)
    ap.add_argument('--action_only', action='store_true')
    ap.add_argument('--permute', action='store_true')
    ap.add_argument('--out_csv', default='interpretability/letter_world/results/reach_before_gvf.csv')
    args = ap.parse_args()

    Z, A, E, LID = load_seq_npz(args.data, args.feature)
    A_id, B_id = [int(x) for x in args.pair.split(',')]

    y_all = build_reach_before_targets(LID, E, A_id, B_id, H=args.horizon)
    mask = np.array([v is not None for v in y_all])
    if mask.sum() < 50:
        raise SystemExit('Too few labeled examples; increase horizon or episodes')

    # build X
    if args.action_only:
        K = int(A.max()) + 1
        X = np.eye(K, dtype=np.float32)[A]
    else:
        X = Z

    X, y, groups = X[mask], np.asarray(y_all[mask], dtype=int), E[mask]

    # split by episode, ensure both classes in train and test
    idx_all = np.arange(len(X))
    tr = te = None
    for seed in range(20):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_cand, te_cand = next(gss.split(idx_all, groups=groups))
        yt, yv = y[tr_cand], y[te_cand]
        if np.unique(yt).size >= 2 and np.unique(yv).size >= 2:
            tr, te = tr_cand, te_cand
            break
    if tr is None:
        raise SystemExit('Could not form a split with both classes in train/test; increase horizon or episodes, or choose a different A,B pair.')

    if args.permute:
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(te))
        X_te = X[te][perm]
    else:
        X_te = X[te]

    # train base logistic
    clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=1000, class_weight='balanced'))
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X_te)[:, 1]
    try:
        auroc = float(roc_auc_score(y[te], p))
    except Exception:
        auroc = float('nan')
    brier = float(brier_score_loss(y[te], p))

    # swap A<->B targets (counterfactual sensitivity on the same model)
    y_swap = 1 - y[te]
    try:
        auroc_swap = float(roc_auc_score(y_swap, p))
        delta_swap = float(auroc - auroc_swap) if not np.isnan(auroc) else float('nan')
    except Exception:
        auroc_swap = float('nan')
        delta_swap = float('nan')

    # write CSV
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['variant','pair','horizon','n_test','auroc','brier','auroc_swap','delta_swap'])
        w.writeheader()
        w.writerow(dict(variant=('action_only' if args.action_only else ('permute' if args.permute else 'base')),
                        pair=f"{A_id},{B_id}", horizon=args.horizon, n_test=int(len(te)),
                        auroc=auroc, brier=brier, auroc_swap=auroc_swap, delta_swap=delta_swap))
    print('Saved reach-before GVF to', out)


if __name__ == '__main__':
    main()


