#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from typing import Optional

# Reuse BFS utilities
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Branched CLEAN NPZ path')
    ap.add_argument('--target_ch', type=int, required=True)
    ap.add_argument('--avoid_ch', type=int, default=None)
    ap.add_argument('--horizon', type=int, default=10)
    ap.add_argument('--min_groups', type=int, default=50, help='Warn if fewer than this many groups')
    ap.add_argument('--min_pos_frac', type=float, default=0.05, help='Warn if positives fraction below this threshold')
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    req_keys = ['feature_t', 'obs_next_raw', 'base_id']
    missing = [k for k in req_keys if k not in D.files]
    if missing:
        raise SystemExit(f"[health] Missing required keys: {missing}")

    X = np.asarray(D['feature_t'])
    ON = np.asarray(D['obs_next_raw'])
    G  = np.asarray(D['base_id']).astype(int)
    A  = np.asarray(D['action']) if 'action' in D.files else None
    Pn = np.asarray(D['agent_pos_next']) if 'agent_pos_next' in D.files else None

    N = min(len(X), len(ON), len(G))
    X, ON, G = X[:N], ON[:N], G[:N]
    if A is not None and len(A) != N: A = None
    if Pn is not None and len(Pn) != N: Pn = None

    # shapes and dims
    print(f"[health] candidates={N}  feature_dim={X.shape[1] if X.ndim==2 else 'NA'}  next_obs_shape={ON.shape[1:]}")
    groups, counts = np.unique(G, return_counts=True)
    print(f"[health] groups={groups.size}  candidates_per_group: min={counts.min()} med={int(np.median(counts))} max={counts.max()}")
    if A is not None:
        for k in range(int(A.max())+1):
            print(f"[health] action {k}: {int((A==k).sum())} candidates")

    # oracle labels
    y = label_feasible(ON, Pn, args.target_ch, args.avoid_ch, args.horizon)
    n_pos, n_neg = int(y.sum()), int(len(y)-y.sum())
    print(f"[health] oracle labels: pos={n_pos} ({n_pos/len(y):.3f})  neg={n_neg} ({n_neg/len(y):.3f})")

    # group-level label presence
    with_pos = 0; with_neg = 0; with_both = 0
    for g in groups:
        idx = np.where(G==g)[0]
        yp = y[idx].sum(); yn = idx.size - yp
        with_pos += int(yp > 0)
        with_neg += int(yn > 0)
        with_both += int(yp > 0 and yn > 0)
    print(f"[health] groups with pos={with_pos}, with neg={with_neg}, with both={with_both}")

    # simple verdicts
    verdicts = []
    if groups.size < args.min_groups:
        verdicts.append(f"LOW_GROUPS<{args.min_groups}")
    if n_pos / max(1,len(y)) < args.min_pos_frac:
        verdicts.append(f"LOW_POS_FRAC<{args.min_pos_frac}")
    if with_both == 0:
        verdicts.append("NO_GROUP_WITH_BOTH_CLASSES")
    if verdicts:
        print(f"[health] WARN: {', '.join(verdicts)}")
        print("[health] Suggest: increase steps/branch density, lower --horizon, or choose a stronger avoid channel.")
    else:
        print("[health] OK: dataset has sufficient size and label balance.")


if __name__ == '__main__':
    main()


