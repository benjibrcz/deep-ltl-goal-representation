#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import csv
from typing import Tuple, Optional
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

try:
    from interpretability.letter_world.predictive_model_probing.bfs_oracle import letter_cells, bfs_shortest
except Exception:
    try:
        from predictive_model_probing.bfs_oracle import letter_cells, bfs_shortest
    except Exception:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.append(str(_Path(__file__).resolve().parent))
        from bfs_oracle import letter_cells, bfs_shortest

OFFSETS = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}  # R,L,D,U

def step_pos(H: int, W: int, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    di, dj = OFFSETS.get(int(action), (0, 0))
    return ((pos[0] + di) % H, (pos[1] + dj) % W)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Sequential NPZ with obs_raw, agent_pos, feature_t')
    ap.add_argument('--candidates', type=str, required=True, help='NPZ from 05_mine_scenarios_letter.py')
    ap.add_argument('--feature_key', type=str, default='feature_t', help='Feature array to use (e.g., feature_t or a hook key)')
    ap.add_argument('--out_csv', type=str, required=True)
    ap.add_argument('--horizon', type=int, default=10)
    # Channels for safety and lookahead
    ap.add_argument('--A_ch', type=int, default=0)
    ap.add_argument('--B_ch', type=int, default=1)
    ap.add_argument('--C_ch', type=int, default=2)
    ap.add_argument('--X_ch', type=int, default=0)
    ap.add_argument('--Y_ch', type=int, default=1)
    ap.add_argument('--permute', action='store_true')
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    C = np.load(args.candidates, allow_pickle=True)
    for k in [args.feature_key, 'obs_raw', 'agent_pos']:
        if k not in D.files:
            raise SystemExit(f"Dataset missing '{k}'.")
    Xall = np.asarray(D[args.feature_key])
    OBS  = np.asarray(D['obs_raw'])
    POS  = np.asarray(D['agent_pos'])

    idxs = np.asarray(C['indices'])
    kinds = np.asarray(C['kind'])
    oracle = np.asarray(C['oracle'])

    # Build per-candidate four-action labels using oracle
    feats = []
    labels = []
    groups = []
    actions = []
    for gid, (i, k, oc) in enumerate(zip(idxs, kinds, oracle)):
        obs = OBS[i]; pos = tuple(int(v) for v in POS[i])
        H, W, _ = obs.shape
        # feature for this decision point (same for 4 actions)
        z = Xall[i]
        if z.ndim > 1:
            z = z.reshape(-1)
        for a in range(4):
            pos_next = step_pos(H, W, pos, a)
            y = 0
            if k == 'safety_detour':
                Aset = letter_cells(obs, args.A_ch)
                Bset = letter_cells(obs, args.B_ch)
                Cset = letter_cells(obs, args.C_ch)
                target_set = Aset if oc == 'A' else Bset
                d_now, _  = bfs_shortest(H, W, pos, target_set, blocked=Cset, Hmax=args.horizon, wrap=True)
                d_next, _ = bfs_shortest(H, W, pos_next, target_set, blocked=Cset, Hmax=args.horizon, wrap=True)
                feasible = d_next is not None
                progress = (d_now is not None and d_next is not None and d_next < d_now)
                y = int(feasible and progress)
            elif k == 'lookahead':
                Xs = letter_cells(obs, args.X_ch)
                # Load x* from meta if present
                meta = C['meta'][list(idxs).index(i)]
                x_best = (int(meta['x_best_i']), int(meta['x_best_j'])) if isinstance(meta, dict) and 'x_best_i' in meta else None
                Xstar = {x_best} if x_best is not None else Xs
                d_now, _  = bfs_shortest(H, W, pos, Xstar, blocked=set(), Hmax=args.horizon, wrap=True)
                d_next, _ = bfs_shortest(H, W, pos_next, Xstar, blocked=set(), Hmax=args.horizon, wrap=True)
                y = int(d_now is not None and d_next is not None and d_next < d_now)
            feats.append(z)
            labels.append(y)
            groups.append(gid)
            actions.append(a)

    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    G = np.asarray(groups, dtype=int)
    A = np.asarray(actions, dtype=int)
    # Optionally permute labels
    if args.permute:
        rng = np.random.RandomState(0)
        perm = np.arange(len(y)); rng.shuffle(perm)
        y = y[perm]; G = G[perm]; A = A[perm]

    # Keep groups with both classes
    keep = []
    for g in np.unique(G):
        idx = np.where(G == g)[0]
        if idx.size >= 2:
            pos = int(y[idx].sum())
            if pos >= 1 and pos < idx.size:
                keep.extend(list(idx))
    keep = np.asarray(keep, dtype=int)
    X, y, G, A = X[keep], y[keep], G[keep], A[keep]
    if len(X) == 0:
        raise SystemExit("No mixed groups for probe; relax mining or adjust horizon.")

    # Train/test split by group
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr_idx, te_idx = next(gss.split(np.arange(len(X)), groups=G))
    # Build plus and baselines
    Aoh = np.eye(4, dtype=np.float32)[np.clip(A, 0, 3)]
    X_plus = np.concatenate([X, Aoh], axis=1)
    X_act  = Aoh
    X_base = X

    def fit_and_prob(Z):
        clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, class_weight='balanced'))
        clf.fit(Z[tr_idx], y[tr_idx])
        return clf.predict_proba(Z)[:, 1]

    p_base = fit_and_prob(X_base)
    p_plus = fit_and_prob(X_plus)
    p_act  = fit_and_prob(X_act)

    # Evaluate top-1 per mixed group on test split
    def top1(scores):
        acc = []
        for g in np.unique(G[te_idx]):
            idx = te_idx[np.where(G[te_idx] == g)[0]]
            pick = idx[np.argmax(scores[idx])]
            acc.append(int(y[pick] == 1))
        return float(np.mean(acc)) if acc else float('nan'), len(np.unique(G[te_idx]))

    acc_base, n_te = top1(p_base)
    acc_plus, _    = top1(p_plus)
    acc_act,  _    = top1(p_act)

    outp = Path(args.out_csv); outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['variant','metric','value','n_groups_test'])
        w.writeheader()
        w.writerow(dict(variant='features_only', metric='top1', value=acc_base, n_groups_test=n_te))
        w.writerow(dict(variant='features_plus_action', metric='top1', value=acc_plus, n_groups_test=n_te))
        w.writerow(dict(variant='action_only', metric='top1', value=acc_act, n_groups_test=n_te))
    print(f"[first_step_probe] wrote {outp}")

if __name__ == '__main__':
    main()


