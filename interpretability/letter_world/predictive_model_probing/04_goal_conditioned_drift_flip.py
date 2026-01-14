#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import csv
from typing import Tuple, Optional, List
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# Robust import for both direct and module execution
try:
    from interpretability.letter_world.predictive_model_probing.bfs_oracle import detect_agent_channel, first_letter_channel_at_pos, extract_targets_and_blocked, bfs_shortest
except Exception:
    try:
        from predictive_model_probing.bfs_oracle import detect_agent_channel, first_letter_channel_at_pos, extract_targets_and_blocked, bfs_shortest
    except Exception:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.append(str(_Path(__file__).resolve().parent))
        from bfs_oracle import detect_agent_channel, first_letter_channel_at_pos, extract_targets_and_blocked, bfs_shortest


def find_A_B_channels(obs_seq: np.ndarray, pos_seq: np.ndarray) -> Tuple[Optional[int], Optional[int], int, int]:
    """
    Detect first letter channel encountered (A) and first distinct letter channel after A (B).
    Returns (A_ch, B_ch, tA, tB). If not found, channels are None and times -1.
    """
    A_ch = None; B_ch = None; tA = -1; tB = -1
    a_ch = detect_agent_channel(obs_seq[0])
    for t in range(len(obs_seq)):
        ch = first_letter_channel_at_pos(obs_seq[t], tuple(pos_seq[t]), a_ch)
        if ch is None:
            continue
        if A_ch is None:
            A_ch = ch; tA = t
        elif ch != A_ch:
            B_ch = ch; tB = t
            break
    return A_ch, B_ch, tA, tB


def oracle_progress_delta(obs_t, pos_t, obs_tp1, pos_tp1, target_ch: int, avoid_ch: Optional[int], Hmax: int) -> Optional[float]:
    H, W, C = obs_t.shape
    tset, bset = extract_targets_and_blocked(obs_t, target_ch, avoid_ch)
    d_t, _ = bfs_shortest(H, W, tuple(pos_t), tset, bset, Hmax, wrap=True)
    d_tp1, _ = bfs_shortest(H, W, tuple(pos_tp1), tset, bset, Hmax, wrap=True)
    if d_t is None or d_tp1 is None:
        return None
    return float(d_t - d_tp1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Sequential NPZ with obs_raw, agent_pos, feature_t, episode')
    ap.add_argument('--out_csv', type=str, required=True)
    ap.add_argument('--horizon', type=int, default=10)
    ap.add_argument('--avoid_ch', type=int, default=None)
    ap.add_argument('--min_pre', type=int, default=3)
    ap.add_argument('--min_post', type=int, default=3)
    ap.add_argument('--progress', action='store_true')
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    for k in ['obs_raw', 'agent_pos', 'feature_t', 'episode']:
        if k not in D.files:
            raise KeyError(f"Dataset missing '{k}'. Ensure sequential logger saved obs_raw and standard fields.")
    OBS = np.asarray(D['obs_raw'])
    POS = np.asarray(D['agent_pos'])
    Z   = np.asarray(D['feature_t'])
    EPI = np.asarray(D['episode'])
    N = min(len(OBS), len(POS), len(Z), len(EPI))
    OBS, POS, Z, EPI = OBS[:N], POS[:N], Z[:N], EPI[:N]

    rows = []
    outp = Path(args.out_csv); outp.parent.mkdir(parents=True, exist_ok=True)

    # collect pre/post samples across episodes
    X_pre: List[np.ndarray] = []; y_pre: List[float] = []
    X_post: List[np.ndarray] = []; y_post: List[float] = []
    for ep in np.unique(EPI):
        idxs = np.where(EPI == ep)[0]
        if idxs.size < 3:
            continue
        obs_seq = OBS[idxs]
        pos_seq = POS[idxs]
        z_seq   = Z[idxs]
        A_ch, B_ch, tA, tB = find_A_B_channels(obs_seq, pos_seq)
        if A_ch is None or B_ch is None or tA < 0 or tB < 0:
            continue
        # pre: [0, tA)
        for t in range(0, tA):
            dp = oracle_progress_delta(obs_seq[t], pos_seq[t], obs_seq[t+1], pos_seq[t+1], A_ch, args.avoid_ch, args.horizon)
            if dp is None:
                continue
            X_pre.append(z_seq[t]); y_pre.append(dp)
        # post: [tA, tB)
        for t in range(tA, min(tB, len(idxs) - 1)):
            dp = oracle_progress_delta(obs_seq[t], pos_seq[t], obs_seq[t+1], pos_seq[t+1], B_ch, args.avoid_ch, args.horizon)
            if dp is None:
                continue
            X_post.append(z_seq[t]); y_post.append(dp)

    if len(X_pre) < max(10, args.min_pre) or len(X_post) < max(10, args.min_post):
        raise RuntimeError(f"Insufficient pre/post samples: pre={len(X_pre)}, post={len(X_post)}. Increase episodes or relax mins.")

    X_pre = np.asarray(X_pre, dtype=np.float32); y_pre = np.asarray(y_pre, dtype=np.float32)
    X_post = np.asarray(X_post, dtype=np.float32); y_post = np.asarray(y_post, dtype=np.float32)

    def fit_eval(X, y, X_ref, y_ref, tag):
        model = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=1.0))
        model.fit(X, y)
        r2_in = float(model.score(X, y))
        r2_x  = float(model.score(X_ref, y_ref))
        # cosine between predictions and targets on in-domain
        yhat = model.predict(X)
        num = float(np.dot(yhat, y))
        den = float(np.linalg.norm(yhat) * np.linalg.norm(y) + 1e-8)
        cos = num / den if den > 0 else 0.0
        return dict(tag=tag, r2_in=r2_in, r2_cross=r2_x, cos_in=cos)

    r_pre = fit_eval(X_pre, y_pre, X_post, y_post, 'pre')
    r_post = fit_eval(X_post, y_post, X_pre, y_pre, 'post')

    with outp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tag','r2_in','r2_cross','cos_in','n_pre','n_post'])
        w.writeheader()
        r_pre.update(n_pre=int(len(X_pre))); r_pre.update(n_post=int(len(X_post)))
        r_post.update(n_pre=int(len(X_pre))); r_post.update(n_post=int(len(X_post)))
        w.writerow(r_pre); w.writerow(r_post)
    print(f"[drift-flip] wrote results to {outp}")


if __name__ == '__main__':
    main()


