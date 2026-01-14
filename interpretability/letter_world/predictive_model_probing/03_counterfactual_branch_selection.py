#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import csv
from typing import Dict, List, Tuple, Optional
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
# Robust import for both direct and module execution
try:
    from interpretability.letter_world.predictive_model_probing.bfs_oracle import detect_agent_channel, extract_targets_and_blocked, bfs_shortest
except Exception:
    try:
        from predictive_model_probing.bfs_oracle import detect_agent_channel, extract_targets_and_blocked, bfs_shortest
    except Exception:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.append(str(_Path(__file__).resolve().parent))
        from bfs_oracle import detect_agent_channel, extract_targets_and_blocked, bfs_shortest


def label_feasible(next_obs: np.ndarray, agent_pos_next: Optional[np.ndarray], target_ch: int, avoid_ch: Optional[int], H: int) -> np.ndarray:
    """
    next_obs: [N, H, W, C]
    agent_pos_next: [N, 2] optional; if None, infer next pos from agent channel argmax per-frame
    returns y in {0,1} of shape [N]
    """
    N, Hh, Ww, C = next_obs.shape
    y = np.zeros(N, dtype=int)
    for i in range(N):
        obs = next_obs[i]
        a_ch = detect_agent_channel(obs)
        if agent_pos_next is not None and len(agent_pos_next) == N:
            pos = tuple(int(v) for v in agent_pos_next[i])
        else:
            flat = obs[..., a_ch].reshape(-1)
            pos_idx = int(np.argmax(flat))
            pos = (pos_idx // Ww, pos_idx % Ww)
        tset, bset = extract_targets_and_blocked(obs, target_ch, avoid_ch)
        dist, _ = bfs_shortest(Hh, Ww, pos, tset, bset, H, wrap=True)
        y[i] = int(dist is not None and dist <= H)
    return y


def top1_accuracy_for_groups(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """
    scores: [N] predicted prob for y=1 per candidate
    labels: [N] true in {0,1}
    groups: [N] base_id indicating 4 candidates per group
    """
    acc = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if idx.size == 0:
            continue
        # skip groups with no positive
        if labels[idx].sum() == 0:
            continue
        pick = idx[np.argmax(scores[idx])]
        acc.append(int(labels[pick] == 1))
    return float(np.mean(acc)) if acc else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Branched CLEAN NPZ with feature_t, base_id, obs_next_raw')
    ap.add_argument('--out_csv', type=str, required=True)
    ap.add_argument('--target_ch', type=int, required=True)
    ap.add_argument('--avoid_ch', type=int, default=None)
    ap.add_argument('--horizon', type=int, default=10)
    ap.add_argument('--permute', action='store_true', help='Permutation control: shuffle labels across bases')
    ap.add_argument('--retry_splits', type=int, default=30, help='Retry group splits with different seeds until both classes appear in train/test')
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    for k in ['feature_t', 'obs_next_raw', 'base_id']:
        if k not in D.files:
            raise KeyError(f"Dataset missing '{k}'.")
    X = np.asarray(D['feature_t'])
    ON = np.asarray(D['obs_next_raw'])
    G = np.asarray(D['base_id']).astype(int)
    A = np.asarray(D['action']) if 'action' in D.files else None
    Pnext = np.asarray(D['agent_pos_next']) if 'agent_pos_next' in D.files else None

    # ensure consistent lengths
    N = min(len(X), len(ON), len(G))
    X, ON, G = X[:N], ON[:N], G[:N]
    if A is not None and len(A) != N:
        A = None
    if Pnext is not None and len(Pnext) != N:
        Pnext = None

    # label feasibility y per candidate
    y = label_feasible(ON, Pnext, args.target_ch, args.avoid_ch, args.horizon)
    if args.permute:
        rng = np.random.RandomState(0)
        y = y.copy()
        rng.shuffle(y)

    # require groups with both classes present (at least 1 positive and 1 negative)
    valid = []
    for g in np.unique(G):
        idx = np.where(G == g)[0]
        if idx.size >= 2:
            pos = int(y[idx].sum())
            if pos >= 1 and pos < idx.size:
                valid.append(g)
    m = np.isin(G, valid)
    X, ON, G, y = X[m], ON[m], G[m], y[m]
    if A is not None:
        A = A[m]

    # diagnostics and robust split
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    print(f"[branch-selection] candidates={len(y)} pos={n_pos} neg={n_neg} groups={np.unique(G).size}")
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("All labels are the same after filtering. Adjust --horizon/--target_ch/--avoid_ch or collect more data.")
    tr_idx = te_idx = None
    for seed in range(args.retry_splits):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tri, tei = next(gss.split(np.arange(len(X)), groups=G))
        # Require both classes in TRAIN so classifier can fit; TEST may be imbalanced
        if np.unique(y[tri]).size >= 2:
            tr_idx, te_idx = tri, tei
            print(f"[branch-selection] split seed={seed} train_pos={int(y[tri].sum())}/{len(tri)} test_pos={int(y[tei].sum())}/{len(tei)}")
            break
    if tr_idx is None or te_idx is None:
        raise SystemExit("Could not form a split with both classes in TRAIN after retries. Tweak horizon/avoid or dataset.")

    # build design matrices
    K = 4
    if A is None:
        # infer from counts per group (0..3 repeated); fall back to zeros
        A = np.zeros(len(X), dtype=int)
    Aoh = np.eye(K, dtype=np.float32)[np.clip(A, 0, K - 1)]
    X_plus = np.concatenate([X, Aoh], axis=1)
    X_act = Aoh
    X_base = X

    # classifier
    def fit_and_score(Z):
        clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, class_weight='balanced'))
        clf.fit(Z[tr_idx], y[tr_idx])
        p = clf.predict_proba(Z)[:, 1]
        return p, clf

    p_base, _ = fit_and_score(X_base)
    p_plus, _ = fit_and_score(X_plus)
    p_act, _ = fit_and_score(X_act)

    # top-1 accuracies on test groups (already filtered to both-classes)
    acc_base = top1_accuracy_for_groups(p_base[te_idx], y[te_idx], G[te_idx])
    acc_plus = top1_accuracy_for_groups(p_plus[te_idx], y[te_idx], G[te_idx])
    acc_act  = top1_accuracy_for_groups(p_act[te_idx],  y[te_idx], G[te_idx])

    outp = Path(args.out_csv); outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['variant','metric','value','n_groups_test'])
        w.writeheader()
        n_groups = int(np.unique(G[te_idx]).size)
        w.writerow(dict(variant='features_only', metric='top1', value=acc_base, n_groups_test=n_groups))
        w.writerow(dict(variant='features_plus_action', metric='top1', value=acc_plus, n_groups_test=n_groups))
        w.writerow(dict(variant='action_only', metric='top1', value=acc_act, n_groups_test=n_groups))
    print(f"[branch-selection] wrote results to {outp}")


if __name__ == '__main__':
    main()


