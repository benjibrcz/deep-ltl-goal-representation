#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from typing import Tuple, Optional, List, Dict

# Robust import of BFS helpers
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


def build_sets(obs_hw_c: np.ndarray, ch: Optional[int]) -> set:
    if ch is None or ch < 0:
        return set()
    return letter_cells(obs_hw_c, int(ch))


def mine_safety_detour(obs: np.ndarray,
                       pos: Tuple[int, int],
                       ch_a: int, ch_b: int, ch_c_avoid: int,
                       require_conflict: bool = True) -> Optional[Dict]:
    """Mine a state where the safe target (A or B) differs from myopic nearest target in free space."""
    H, W, C = obs.shape
    A = build_sets(obs, ch_a)
    B = build_sets(obs, ch_b)
    Cset = build_sets(obs, ch_c_avoid)
    if not A or not B:
        return None
    dA_free, _ = bfs_shortest(H, W, pos, A, blocked=set(), Hmax=H * W, wrap=True)
    dB_free, _ = bfs_shortest(H, W, pos, B, blocked=set(), Hmax=H * W, wrap=True)
    dA_safe, _ = bfs_shortest(H, W, pos, A, blocked=Cset, Hmax=H * W, wrap=True)
    dB_safe, _ = bfs_shortest(H, W, pos, B, blocked=Cset, Hmax=H * W, wrap=True)
    # If neither target is reachable safely, skip
    if dA_safe is None and dB_safe is None:
        return None
    # Choose safe target
    INF = 10**9
    a_safe = dA_safe if dA_safe is not None else INF
    b_safe = dB_safe if dB_safe is not None else INF
    safe_choice = 'A' if a_safe < b_safe else 'B'
    # Myopic nearest in free space (ties break to A)
    a_free = dA_free if dA_free is not None else INF
    b_free = dB_free if dB_free is not None else INF
    myopic = 'A' if a_free <= b_free else 'B'
    if not require_conflict or (safe_choice != myopic):
        return dict(type='safety_detour',
                    safe_choice=safe_choice,
                    myopic=myopic,
                    dA_free=(a_free if a_free < INF else -1),
                    dB_free=(b_free if b_free < INF else -1),
                    dA_safe=(a_safe if a_safe < INF else -1),
                    dB_safe=(b_safe if b_safe < INF else -1))
    return None


def all_pairs_shortest(H: int, W: int, S: set, T: set) -> int:
    """Min over s in S of BFS(s, T). Returns steps or large if None."""
    best = None
    for s in S:
        d, _ = bfs_shortest(H, W, s, T, blocked=set(), Hmax=H * W, wrap=True)
        if d is None:
            continue
        best = d if best is None else min(best, d)
    return best if best is not None else 10**9


def mine_lookahead(obs: np.ndarray,
                   pos: Tuple[int, int],
                   ch_x: int, ch_y: int,
                   min_gap: int = 2) -> Optional[Dict]:
    """
    Two-stage: choose which X to hit first to minimize J = d(s,Xi) + min_y d(Xi, y).
    We treat each connected cell in X as a candidate; approximation via best distances.
    """
    H, W, C = obs.shape
    Xs = build_sets(obs, ch_x)
    Ys = build_sets(obs, ch_y)
    if not Xs or not Ys:
        return None
    # Score per Xi by splitting: d(s,X) + min_y d(X, y). Approximate d(X, y) by min over all X-cells to Y for this Xi ~ global min X->Y.
    # To keep it cheap, compute global min from each X-cell? We approximate by using the closest X to s for distance part and
    # for "choice matters", compare two proxies: nearest-X and second-nearest-X with J values using pairwise approximations.
    # More robust: evaluate J for each X-cell explicitly.
    J_list = []
    X_list = list(Xs)
    for xi in X_list:
        d1, _ = bfs_shortest(H, W, pos, {xi}, blocked=set(), Hmax=H * W, wrap=True)
        if d1 is None:
            J_list.append((10**9, xi))
            continue
        d2 = all_pairs_shortest(H, W, {xi}, Ys)
        J_list.append((d1 + d2, xi))
    J_sorted = sorted(J_list, key=lambda t: t[0])
    if len(J_sorted) < 2:
        return None
    best, x_best = J_sorted[0]
    second, x_second = J_sorted[1]
    if second - best >= max(1, int(min_gap)):
        return dict(type='lookahead',
                    x_best_i=x_best[0], x_best_j=x_best[1],
                    J_best=int(best), J_second=int(second))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Sequential NPZ with obs_raw, agent_pos, action')
    ap.add_argument('--out', type=str, required=True)
    # Channels for letters
    ap.add_argument('--A_ch', type=int, default=0)
    ap.add_argument('--B_ch', type=int, default=1)
    ap.add_argument('--C_ch', type=int, default=2, help='Avoid channel for safety-detour')
    ap.add_argument('--X_ch', type=int, default=0)
    ap.add_argument('--Y_ch', type=int, default=1)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--min_gap', type=int, default=2, help='Min J gap for lookahead mining')
    ap.add_argument('--require_conflict', action='store_true', help='Require safe vs myopic disagreement for safety-detour')
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    for k in ['obs_raw', 'agent_pos']:
        if k not in D.files:
            raise SystemExit(f"Dataset missing '{k}'. Re-run sequential logger with --save_obs.")
    OBS = np.asarray(D['obs_raw'])
    POS = np.asarray(D['agent_pos'])
    EPI = np.asarray(D['episode']) if 'episode' in D.files else np.zeros(len(POS), dtype=int)
    N = min(len(OBS), len(POS))
    OBS, POS, EPI = OBS[:N], POS[:N], EPI[:N]

    idxs = []
    kinds = []
    oracle = []
    extra = []
    for i in range(0, N, max(1, int(args.stride))):
        obs = OBS[i]
        pos = tuple(int(v) for v in POS[i])
        sd = mine_safety_detour(obs, pos, args.A_ch, args.B_ch, args.C_ch, require_conflict=args.require_conflict)
        if sd is not None:
            idxs.append(i); kinds.append('safety_detour'); oracle.append(sd['safe_choice']); extra.append(sd)
            continue
        lo = mine_lookahead(obs, pos, args.X_ch, args.Y_ch, min_gap=args.min_gap)
        if lo is not None:
            idxs.append(i); kinds.append('lookahead'); oracle.append('X'); extra.append(lo)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    # Pack extras as object arrays to preserve dicts
    np.savez_compressed(outp,
                        data=args.data,
                        indices=np.asarray(idxs, dtype=np.int64),
                        kind=np.asarray(kinds, dtype=object),
                        oracle=np.asarray(oracle, dtype=object),
                        meta=np.asarray(extra, dtype=object))
    print(f"[mine_scenarios] kept={len(idxs)}  out={outp}")


if __name__ == '__main__':
    main()


