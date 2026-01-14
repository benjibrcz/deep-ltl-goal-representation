#!/usr/bin/env python3
"""
3D PCA goal "sink" visualization and metrics.

Given a sweep NPZ (from 03e_log_rollouts_letter_sweep.py), this script:
  - selects episodes for a target goal letter under mode 'reach' (F a)
  - for each episode, finds the first hit of that letter
  - aggregates all pre-hit steps across episodes
  - fits PCA=3 on pooled pre-hit features (actor_mid by default)
  - plots 3D trajectories per episode in PCA space
  - marks a per-episode sink (mean of last N steps before hit)
  - computes monotonicity toward the sink and other simple metrics

Usage example:
  python interpretability/letter_world/35_pca_goal_sinks.py \
      --npz interpretability/letter_world/datasets/letter_sweep_all_layers_e1_s10.npz \
      --feature feature_t --goal_letter a --episodes_cap 60 \
      --out_dir interpretability/letter_world/results/sinks

Outputs (under out_dir):
  - pca_sinks_goal_A_base{B}.png            (3D trajectories with sinks)
  - sink_distance_goal_A_base{B}.png        (distance-to-sink over time)
  - sink_metrics_goal_A_base{B}.json        (summary metrics)
"""

import argparse
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, required for 3D projection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return x.astype(np.float32)


def letter_to_idx(letter: str, vocab: List[str]) -> int:
    if letter is None:
        return -1
    letter = str(letter).strip().lower()
    try:
        return int(vocab.index(letter))
    except Exception:
        return -1


def find_first_hit(seq_letters: np.ndarray, target_id: int) -> int:
    for i, v in enumerate(seq_letters):
        if int(v) == int(target_id):
            return int(i)
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=str, required=True)
    ap.add_argument('--feature', type=str, default='feature_t', help='e.g., feature_t (actor_mid)')
    ap.add_argument('--goal_letter', type=str, required=True, help="target letter like 'a'")
    ap.add_argument('--mode', type=str, default='reach', choices=['reach','any'], help='filter episodes by goal_mode')
    ap.add_argument('--episodes_cap', type=int, default=80, help='max episodes to plot')
    ap.add_argument('--n_last', type=int, default=5, help='steps before hit to average as sink')
    ap.add_argument('--out_dir', type=str, default='interpretability/letter_world/results/sinks')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    D = np.load(args.npz, allow_pickle=True)
    if args.feature not in D.files:
        raise SystemExit(f"Feature key '{args.feature}' not in NPZ. Available: {list(D.files)}")

    Z_all = ensure_2d(D[args.feature])
    E_all = np.asarray(D['episode']) if 'episode' in D.files else np.zeros(len(Z_all), dtype=int)
    L_all = np.asarray(D['letter_id']) if 'letter_id' in D.files else np.full(len(Z_all), -1, dtype=int)
    base_all = np.asarray(D['base_id']) if 'base_id' in D.files else np.zeros(len(Z_all), dtype=int)
    goal_mode = np.asarray(D['goal_mode']) if 'goal_mode' in D.files else np.array(['reach'] * len(Z_all), dtype=object)
    goal_reach = np.asarray(D['goal_reach']) if 'goal_reach' in D.files else np.full(len(Z_all), -1, dtype=int)
    vocab = list(D['letters_vocab']) if 'letters_vocab' in D.files else [chr(ord('a') + i) for i in range(12)]

    tgt_id = letter_to_idx(args.goal_letter, vocab)
    if tgt_id < 0:
        raise SystemExit(f"Could not resolve goal_letter '{args.goal_letter}' in vocab {vocab}")

    # Pre-filter steps to episodes where goal is reach and target letter matches
    mask_goal = (goal_reach == tgt_id)
    if args.mode == 'reach':
        mask_goal &= (goal_mode.astype(str) == 'reach')

    ep_ids = sorted(set(map(int, E_all[mask_goal].tolist())))
    if not ep_ids:
        raise SystemExit('No episodes with the requested goal found. Re-run logging with modes including reach and the target letter.')

    # Pick a dominant base map to get clearer geometry by fixing the map
    base_by_ep: Dict[int, int] = {}
    for e in ep_ids:
        idx = np.where(E_all == e)[0]
        if len(idx) == 0:
            continue
        # take the most frequent base_id inside the episode
        b = int(Counter(base_all[idx].tolist()).most_common(1)[0][0])
        base_by_ep[e] = b
    if not base_by_ep:
        raise SystemExit('Could not compute base_id per episode.')
    dominant_base = Counter(base_by_ep.values()).most_common(1)[0][0]

    # Select episodes on the dominant base
    ep_dom = [e for e in ep_ids if base_by_ep.get(e, None) == dominant_base]
    if not ep_dom:
        raise SystemExit('No episodes on a dominant base_id found.')
    if len(ep_dom) > args.episodes_cap:
        rng = np.random.default_rng(0)
        ep_dom = list(rng.choice(ep_dom, size=args.episodes_cap, replace=False))

    # Collect per-episode pre-hit trajectories
    episodes_data = []  # list of dict with keys: 'Z', 'Z3', 'idx', 'sink3', 'dist', 'epi'
    pooled_Z = []
    pooled_ep = []
    for e in ep_dom:
        idx = np.where(E_all == e)[0]
        if len(idx) < 3:
            continue
        # ensure this episode is the correct goal and base throughout
        if goal_reach[idx[0]] != tgt_id:
            continue
        if args.mode == 'reach' and str(goal_mode[idx[0]]) != 'reach':
            continue
        # find first hit of target letter inside episode
        tA_local = find_first_hit(L_all[idx], tgt_id)
        if tA_local < 1:  # need at least 1 pre-hit step
            continue
        # pre-hit segment (strictly before hit)
        seg = idx[:tA_local]
        Zseg = Z_all[seg]
        if len(Zseg) < 3:
            continue
        pooled_Z.append(Zseg)
        pooled_ep.extend([e] * len(Zseg))
        episodes_data.append(dict(ep=e, idx=seg, Z=Zseg))

    if not episodes_data:
        raise SystemExit('No qualifying pre-hit segments found. Try increasing steps_per_ep or using a different letter.')

    # Fit PCA=3 on pooled pre-hit features
    Z_pool = np.vstack(pooled_Z)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(Z_pool)
    Zs_pool = scaler.transform(Z_pool)
    pca = PCA(n_components=3, random_state=0).fit(Zs_pool)

    # Transform each episode to 3D and compute sink and metrics
    for item in episodes_data:
        Zs = scaler.transform(item['Z'])
        Z3 = pca.transform(Zs)
        item['Z3'] = Z3
        # sink = mean of last n_last points
        n_last = max(1, min(args.n_last, len(Z3)))
        sink3 = Z3[-n_last:, :].mean(axis=0)
        item['sink3'] = sink3
        d = np.linalg.norm(Z3 - sink3[None, :], axis=1)
        item['dist'] = d

    # Metrics
    def monotone_frac(d: np.ndarray) -> float:
        if len(d) < 2:
            return float('nan')
        return float(np.mean(d[1:] < d[:-1]))

    monos = [monotone_frac(it['dist']) for it in episodes_data]
    dist_drop = [float(it['dist'][0] - it['dist'][-1]) for it in episodes_data]
    avg_final_dist = float(np.mean([it['dist'][-1] for it in episodes_data]))

    # Plots: 3D trajectories
    goal_tag = str(args.goal_letter).upper()
    base_tag = f"{dominant_base}"
    fig = plt.figure(figsize=(8, 6), dpi=130)
    ax = fig.add_subplot(111, projection='3d')
    for it in episodes_data:
        Z3 = it['Z3']
        tnorm = np.linspace(0.0, 1.0, len(Z3))
        ax.plot(Z3[:, 0], Z3[:, 1], Z3[:, 2], '-', alpha=0.6, color=(0.2, 0.4, 0.8, 0.5))
        # mark start and sink
        ax.scatter([Z3[0, 0]], [Z3[0, 1]], [Z3[0, 2]], c='green', s=20)
        ax.scatter([it['sink3'][0]], [it['sink3'][1]], [it['sink3'][2]], c='red', s=60, marker='*')
    ax.set_title(f"3D PCA trajectories to sink | goal={goal_tag} | base={base_tag} | n={len(episodes_data)}")
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, f"pca_sinks_goal_{goal_tag}_base{base_tag}.png")
    plt.savefig(out_png)
    plt.close(fig)

    # Plot distances to sink over time (per episode)
    plt.figure(figsize=(7, 5), dpi=130)
    for it in episodes_data:
        d = it['dist']
        x = np.arange(len(d))
        plt.plot(x, d, '-', alpha=0.4)
    plt.xlabel('step (pre-hit)')
    plt.ylabel('distance to sink (PCA3)')
    plt.title(f"Distance to sink | goal={goal_tag} | base={base_tag}")
    plt.tight_layout()
    out_png2 = os.path.join(args.out_dir, f"sink_distance_goal_{goal_tag}_base{base_tag}.png")
    plt.savefig(out_png2)
    plt.close()

    # Save metrics
    metrics = dict(
        goal_letter=goal_tag,
        base_id=int(dominant_base),
        episodes=int(len(episodes_data)),
        monotone_mean=float(np.nanmean(monos)),
        monotone_median=float(np.nanmedian(monos)),
        mean_initial_distance=float(np.mean([float(it['dist'][0]) for it in episodes_data])),
        mean_final_distance=avg_final_dist,
        mean_distance_drop=float(np.mean(dist_drop)),
        n_last=args.n_last,
    )
    out_json = os.path.join(args.out_dir, f"sink_metrics_goal_{goal_tag}_base{base_tag}.json")
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print('[done] wrote:', out_png)
    print('[done] wrote:', out_png2)
    print('[done] wrote:', out_json)


if __name__ == '__main__':
    main()

