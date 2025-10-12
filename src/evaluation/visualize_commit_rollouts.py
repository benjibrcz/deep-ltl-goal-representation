import sys
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


SRC = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC))

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from visualize.zones import draw_zones, setup_axis, FancyAxes


def reset_unpack(env, **kwargs):
    out = env.reset(**kwargs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def fetch_zone_positions(env_id: str, colour: str, seed: int) -> Dict[str, Tuple[float, float]]:
    env = make_env(env_id, FixedSampler.partial(f'FG {colour}'))
    reset_unpack(env, seed=seed)
    zp = getattr(env, 'zone_positions', {})
    env.close()
    return zp


def group_segments_by_label(xs: np.ndarray, ys: np.ndarray, labels: List[str]) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    segs: List[Tuple[np.ndarray, np.ndarray, str]] = []
    if len(xs) < 2:
        return segs
    curr_label = labels[0]
    curr_x = [xs[0]]
    curr_y = [ys[0]]
    for i in range(1, len(xs)):
        if labels[i] == curr_label:
            curr_x.append(xs[i]); curr_y.append(ys[i])
        else:
            if len(curr_x) >= 2:
                segs.append((np.array(curr_x), np.array(curr_y), curr_label))
            curr_label = labels[i]
            curr_x = [xs[i-1], xs[i]]
            curr_y = [ys[i-1], ys[i]]
    if len(curr_x) >= 2:
        segs.append((np.array(curr_x), np.array(curr_y), curr_label))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-id', type=str, default='PointLtl2-v0')
    ap.add_argument('--csv', type=str, default=str(Path(__file__).resolve().parents[2] / 'interpretability' / 'commit_labels.csv'))
    ap.add_argument('--episodes', type=int, default=8)
    ap.add_argument('--start-episode', type=int, default=0)
    ap.add_argument('--out', type=str, default=str(Path(__file__).resolve().parents[2] / 'interpretability' / 'audit_plots' / 'commit_viz_grid.png'))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Select a slice of episodes
    ep_ids = sorted(df['episode_id'].unique())
    ep_ids = ep_ids[args.start_episode:args.start_episode + args.episodes]

    n = len(ep_ids)
    if n == 0:
        print('No episodes to visualize')
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    for idx, ep in enumerate(ep_ids):
        sub = df[df['episode_id'] == ep].copy()
        sub = sub.sort_values('t')
        xs = sub['x'].to_numpy()
        ys = sub['y'].to_numpy()
        labels = sub['Y'].astype(str).tolist()
        colour = str(sub['colour'].iloc[0])
        seed = int(sub['seed'].iloc[0])

        # zones
        zone_pos = fetch_zone_positions(args.env_id, colour, seed)

        ax = fig.add_subplot(rows, cols, idx + 1, axes_class=FancyAxes, edgecolor='gray', linewidth=.5)
        setup_axis(ax)
        draw_zones(ax, zone_pos)

        # draw start
        if len(xs) > 0:
            ax.plot(xs[0], ys[0], marker='D', color='orange', markersize=5)

        # draw segments colored by label
        segs = group_segments_by_label(xs, ys, labels)
        for xseg, yseg, lab in segs:
            ax.plot(xseg, yseg, color=('green' if lab == 'Y' else 'red'), linewidth=3)

        ax.set_title(f"ep={ep} colour={colour} seed={seed}")

    fig.tight_layout(pad=4)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()


