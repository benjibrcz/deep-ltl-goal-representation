#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import csv
from typing import Optional
# Robust import: works for both `python file.py` and `python -m ...`
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Sequential NPZ with obs_raw, agent_pos, episode')
    ap.add_argument('--out_csv', type=str, required=True)
    ap.add_argument('--horizon', type=int, default=10)
    ap.add_argument('--target_ch', type=int, default=None, help='Channel index of target letter (required unless dataset has goals)')
    ap.add_argument('--avoid_ch', type=int, default=None, help='Channel index of avoid letter (optional)')
    ap.add_argument('--every', type=int, default=1, help='Subsample timesteps for speed')
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    required = ['agent_pos']
    for k in required:
        if k not in D.files:
            raise KeyError(f"Dataset missing '{k}'.")
    if 'obs_raw' not in D.files:
        raise KeyError("Dataset missing 'obs_raw'. Re-run 03c_log_rollouts_seq.py with --save_obs.")

    P = np.asarray(D['agent_pos'])
    E = np.asarray(D['episode']) if 'episode' in D.files else np.zeros(len(P), dtype=int)
    OBS = np.asarray(D['obs_raw'])
    N = min(len(P), len(OBS))
    P = P[:N]; OBS = OBS[:N]; E = E[:N]

    outp = Path(args.out_csv); outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(0, N, max(1, int(args.every))):
        obs = OBS[idx]
        agent = tuple(P[idx])
        H, W, C = obs.shape
        a_ch = detect_agent_channel(obs)
        t_ch = args.target_ch
        av_ch = args.avoid_ch
        if t_ch is None:
            raise ValueError("target_ch is required unless you extend this script to read goals from dataset.")
        targets, blocked = extract_targets_and_blocked(obs, t_ch, av_ch)
        dist, first_action = bfs_shortest(H, W, agent, targets, blocked, args.horizon, wrap=True)
        feasible = int(dist is not None and dist <= args.horizon)
        rows.append(dict(idx=int(idx), episode=int(E[idx]), feasible=feasible, dist=(dist if dist is not None else -1), first_action=(first_action if first_action is not None else -1)))

    with outp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['idx','episode','feasible','dist','first_action'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[oracle] wrote {len(rows)} rows to {outp}")


if __name__ == '__main__':
    main()


