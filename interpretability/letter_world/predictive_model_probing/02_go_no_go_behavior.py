#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import csv
from typing import Tuple, Optional
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


def dist_to_target(obs, agent_pos: Tuple[int, int], target_ch: int, avoid_ch: Optional[int], Hmax: int) -> Optional[int]:
    H, W, C = obs.shape
    tset, bset = extract_targets_and_blocked(obs, target_ch, avoid_ch)
    d, _ = bfs_shortest(H, W, tuple(agent_pos), tset, bset, Hmax, wrap=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Sequential NPZ with obs_raw, agent_pos, episode, action')
    ap.add_argument('--out_csv', type=str, required=True)
    ap.add_argument('--horizon', type=int, default=10)
    ap.add_argument('--prefix', type=int, default=3, help='Commit window in steps')
    ap.add_argument('--target_ch_a', type=int, required=True)
    ap.add_argument('--target_ch_b', type=int, required=True)
    ap.add_argument('--avoid_ch', type=int, default=None)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    for k in ['obs_raw', 'agent_pos', 'action']:
        if k not in D.files:
            raise KeyError(f"Dataset missing '{k}'. Re-run sequential logger with --save_obs.")
    OBS = np.asarray(D['obs_raw'])
    POS = np.asarray(D['agent_pos'])
    ACT = np.asarray(D['action'])
    EPI = np.asarray(D['episode']) if 'episode' in D.files else np.zeros(len(ACT), dtype=int)
    N = min(len(OBS), len(POS), len(ACT))
    OBS, POS, ACT, EPI = OBS[:N], POS[:N], ACT[:N], EPI[:N]

    rows = []
    outp = Path(args.out_csv); outp.parent.mkdir(parents=True, exist_ok=True)

    # iterate per-episode, per starting index
    for ep in np.unique(EPI):
        idxs = np.where(EPI == ep)[0]
        if len(idxs) < args.prefix + 1:
            continue
        for s in range(0, len(idxs) - args.prefix - 1, max(1, int(args.stride))):
            i0 = idxs[s]
            obs0 = OBS[i0]; pos0 = tuple(POS[i0])
            # feasibility at t
            dA = dist_to_target(obs0, pos0, args.target_ch_a, args.avoid_ch, args.horizon)
            dB = dist_to_target(obs0, pos0, args.target_ch_b, args.avoid_ch, args.horizon)
            feasA = dA is not None and dA <= args.horizon
            feasB = dB is not None and dB <= args.horizon
            if feasA == feasB:
                # skip if both feasible or both impossible
                continue
            feasible_branch = 'A' if feasA else 'B'
            target_ch = args.target_ch_a if feasA else args.target_ch_b
            # commitment: within prefix steps, distances should decrease monotonically on average
            committed = False
            last_d = dA if feasA else dB
            ok = True
            for k in range(1, args.prefix + 1):
                ik = idxs[s + k] if (s + k) < len(idxs) else None
                if ik is None:
                    ok = False; break
                dk = dist_to_target(OBS[ik], tuple(POS[ik]), target_ch, args.avoid_ch, args.horizon)
                if dk is None or (last_d is not None and dk > last_d):
                    ok = False; break
                last_d = dk
            committed = ok
            # success: reach any target cell within horizon steps
            reached = False
            for k in range(1, min(args.horizon, len(idxs) - s - 1) + 1):
                ik = idxs[s + k]
                tset, _ = extract_targets_and_blocked(OBS[ik], target_ch, None)
                if tuple(POS[ik]) in tset:
                    reached = True
                    break
            rows.append(dict(episode=int(ep), idx=int(i0), feasible_branch=feasible_branch, committed=int(committed), reached=int(reached)))

    with outp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['episode','idx','feasible_branch','committed','reached'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[go-no-go] wrote {len(rows)} rows to {outp}")


if __name__ == '__main__':
    main()


