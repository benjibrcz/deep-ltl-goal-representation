#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import csv
from typing import Tuple, Optional, Dict

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
    ap.add_argument('--data', type=str, required=True, help='Sequential NPZ with obs_raw, agent_pos, action')
    ap.add_argument('--candidates', type=str, required=True, help='NPZ from 05_mine_scenarios_letter.py')
    ap.add_argument('--out_csv', type=str, required=True)
    ap.add_argument('--A_ch', type=int, default=0)
    ap.add_argument('--B_ch', type=int, default=1)
    ap.add_argument('--C_ch', type=int, default=2)
    ap.add_argument('--X_ch', type=int, default=0)
    ap.add_argument('--Y_ch', type=int, default=1)
    ap.add_argument('--horizon', type=int, default=10)
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    C = np.load(args.candidates, allow_pickle=True)
    for k in ['obs_raw', 'agent_pos', 'action']:
        if k not in D.files:
            raise SystemExit("Sequential dataset missing required keys; re-run with --save_obs.")
    OBS = np.asarray(D['obs_raw'])
    POS = np.asarray(D['agent_pos'])
    ACT = np.asarray(D['action'])

    idxs = np.asarray(C['indices'])
    kinds = np.asarray(C['kind'])
    oracle = np.asarray(C['oracle'])

    outp = Path(args.out_csv); outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, k, oc in zip(idxs, kinds, oracle):
        obs = OBS[i]; pos = tuple(int(v) for v in POS[i]); a = int(ACT[i])
        H, W, _ = obs.shape
        pos_next = step_pos(H, W, pos, a)
        ok = 0
        if k == 'safety_detour':
            Aset = letter_cells(obs, args.A_ch)
            Bset = letter_cells(obs, args.B_ch)
            Cset = letter_cells(obs, args.C_ch)
            target_set = Aset if oc == 'A' else Bset
            d_now, _ = bfs_shortest(H, W, pos, target_set, blocked=Cset, Hmax=args.horizon, wrap=True)
            d_next, _ = bfs_shortest(H, W, pos_next, target_set, blocked=Cset, Hmax=args.horizon, wrap=True)
            if d_now is not None and d_next is not None and d_next < d_now:
                ok = 1
        elif k == 'lookahead':
            Xs = letter_cells(obs, args.X_ch)
            Ys = letter_cells(obs, args.Y_ch)
            # Use the x* stored in meta if present, else nearest-X heuristic
            # Distance to chosen X* should decrease
            meta = C['meta'][list(idxs).index(i)]
            x_best = (int(meta['x_best_i']), int(meta['x_best_j'])) if isinstance(meta, dict) and 'x_best_i' in meta else None
            Xstar = {x_best} if x_best is not None else Xs
            d_now, _ = bfs_shortest(H, W, pos, Xstar, blocked=set(), Hmax=args.horizon, wrap=True)
            d_next, _ = bfs_shortest(H, W, pos_next, Xstar, blocked=set(), Hmax=args.horizon, wrap=True)
            if d_now is not None and d_next is not None and d_next < d_now:
                ok = 1
        rows.append(dict(idx=int(i), kind=str(k), oracle=str(oc), action=int(a), correct=int(ok)))

    with outp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['idx','kind','oracle','action','correct'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[behavior_first_step] wrote {outp}  (N={len(rows)}; acc={np.mean([r['correct'] for r in rows]) if rows else float('nan'):.3f})")


if __name__ == '__main__':
    main()


