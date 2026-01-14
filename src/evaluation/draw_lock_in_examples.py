import sys
from pathlib import Path
import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange


SRC = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC))

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search.exhaustive_search import ExhaustiveSearch
from utils.model_store.model_store import ModelStore
from visualize.zones import draw_trajectories


def reset_unpack(env, **kwargs):
    out = env.reset(**kwargs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def find_colour_zone_centers(zone_positions: dict, colour: str) -> List[np.ndarray]:
    centres: List[np.ndarray] = []
    for name, pos in zone_positions.items():
        if str(name).lower().startswith(colour.lower()):
            p = np.asarray(pos, dtype=float).ravel()
            if p.size >= 2:
                centres.append(p[:2].copy())
    return centres


def nearest_zone_index(p: np.ndarray, centres: List[np.ndarray]) -> int:
    ds = [float(np.linalg.norm(p - c)) for c in centres]
    return int(np.argmin(ds)) if ds else -1


def two_nearest_centres(p0: np.ndarray, centres: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if len(centres) < 2:
        return None, None
    ds = [(float(np.linalg.norm(p0 - c)), i) for i, c in enumerate(centres)]
    ds.sort(key=lambda x: x[0])
    a, b = centres[ds[0][1]], centres[ds[1][1]]
    return a, b


def cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    un = float(np.linalg.norm(u)); vn = float(np.linalg.norm(v))
    if un <= eps or vn <= eps:
        return 0.0
    return float(np.dot(u, v) / (un * vn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-id', type=str, default='PointLtl2-v0')
    ap.add_argument('--exp', type=str, default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--episodes-per-colour', type=int, default=4)
    ap.add_argument('--colours', type=str, default='blue,green,yellow,magenta')
    ap.add_argument('--max-steps', type=int, default=300)
    ap.add_argument('--out-dir', type=str, default=str(Path(__file__).resolve().parents[2] / 'interpretability' / 'audit_plots'))
    ap.add_argument('--num-loops', type=int, default=2)
    # detection thresholds
    ap.add_argument('--diverge_rel_thresh', type=float, default=0.02, help='relative divergence (|dd1|+|dd2|)/(d1+d2)')
    ap.add_argument('--cos_margin', type=float, default=0.15, help='margin between cosines to two centres')
    ap.add_argument('--cos_L', type=int, default=3, help='consecutive steps for cosine decision')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    colours = [c.strip() for c in args.colours.split(',') if c.strip()]

    env = make_env(args.env_id, FixedSampler.partial('FG blue'))
    config = model_configs[args.env_id]
    model_store = ModelStore(args.env_id, args.exp, args.seed)
    model_store.load_vocab()
    training_status = model_store.load_training_status(map_location='cpu')
    model = build_model(env, training_status, config)
    env.close()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_colour = args.episodes_per_colour
    pbar = trange(len(colours) * per_colour, desc='examples')
    k = 0
    for colour in colours:
        for j in range(per_colour):
            pbar.update(1)
            seed = args.seed + 1000 * k + j
            sampler = FixedSampler.partial(f'FG {colour}')
            env = make_env(args.env_id, sampler)
            props = set(env.get_propositions())
            agent = Agent(model, search=ExhaustiveSearch(model, props, num_loops=args.num_loops), propositions=props)
            obs = reset_unpack(env, seed=seed)
            agent.reset()

            zone_pos = getattr(env, 'zone_positions', {})
            centres = find_colour_zone_centers(zone_pos, colour)
            if len(centres) < 2:
                env.close();
                continue

            traj: List[np.ndarray] = []
            idx_hist: List[int] = []
            d_hist: List[Tuple[float, float]] = []  # distances to two chosen centres
            cos_hist: List[Tuple[float, float]] = []  # cosine to each centre
            zA, zB = None, None
            done = False
            steps = 0
            while not done and steps < args.max_steps:
                # position from env
                pos = getattr(env, 'agent_pos', None)
                if pos is not None:
                    p = np.asarray(pos, dtype=float).ravel()[:2]
                    traj.append(p.copy())
                    idx_hist.append(nearest_zone_index(p, centres))
                    if zA is None or zB is None:
                        zA, zB = two_nearest_centres(p, centres)
                    if zA is not None and zB is not None:
                        d_hist.append((float(np.linalg.norm(p - zA)), float(np.linalg.norm(p - zB))))
                        if len(traj) >= 2:
                            v = traj[-1] - traj[-2]
                            cosA = cosine_similarity(v, zA - p)
                            cosB = cosine_similarity(v, zB - p)
                            cos_hist.append((cosA, cosB))

                # step
                act = agent.get_action(obs, {}, deterministic=True)
                act = act.flatten()
                step_out = env.step(act)
                if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                    obs, _r, term, trunc, _info = step_out
                    done = bool(term or trunc)
                else:
                    obs, _r, done, _info = step_out
                steps += 1

            env.close()

            # detect mind changes: changes in nearest zone index over time
            changes: List[int] = []
            for t in range(1, len(idx_hist)):
                if idx_hist[t] != idx_hist[t-1] and idx_hist[t] != -1 and idx_hist[t-1] != -1:
                    changes.append(t)
            changed = len(changes) > 0

            # detect divergence signal (one distance dec, other inc)
            diverge_t = None
            diverge_towards = None  # 'A' or 'B'
            for t in range(1, len(d_hist)):
                d1_prev, d2_prev = d_hist[t-1]
                d1, d2 = d_hist[t]
                dd1 = d1 - d1_prev
                dd2 = d2 - d2_prev
                rel = (abs(dd1) + abs(dd2)) / (d1 + d2 + 1e-8)
                if dd1 < 0 and dd2 > 0 and rel >= args.diverge_rel_thresh:
                    diverge_t = t; diverge_towards = 'A'; break
                if dd2 < 0 and dd1 > 0 and rel >= args.diverge_rel_thresh:
                    diverge_t = t; diverge_towards = 'B'; break

            # detect cosine directional commitment
            dir_t = None
            dir_towards = None
            if cos_hist:
                margin = args.cos_margin
                L = args.cos_L
                for t in range(L, len(cos_hist)):
                    window = cos_hist[t-L+1:t+1]
                    dcos = [ca - cb for (ca, cb) in window]
                    if all(x > margin for x in dcos):
                        dir_t = t+1  # align to traj index
                        dir_towards = 'A'
                        break
                    if all(x < -margin for x in dcos):
                        dir_t = t+1
                        dir_towards = 'B'
                        break

            # build figure
            fig = draw_trajectories([zone_pos], [traj], num_cols=1, num_rows=1)
            ax = fig.axes[0]
            # mark change points
            for t in changes:
                if 0 <= t < len(traj):
                    ax.plot(traj[t][0], traj[t][1], marker='x', color='red', markersize=6)
            # mark divergence detection
            if diverge_t is not None and 0 <= diverge_t < len(traj):
                ax.plot(traj[diverge_t][0], traj[diverge_t][1], marker='*', color='purple', markersize=7)
            # mark cosine commitment
            if dir_t is not None and 0 <= dir_t < len(traj):
                ax.plot(traj[dir_t][0], traj[dir_t][1], marker='o', color='black', markersize=5)
            title = (
                f"FG {colour} | seed={seed} | steps={len(traj)} | "
                f"changed_mind={'yes' if changed else 'no'} | "
                f"diverge={'none' if diverge_t is None else diverge_towards}@{diverge_t} | "
                f"dir={'none' if dir_t is None else dir_towards}@{dir_t}"
            )
            fig.suptitle(title, fontsize=11)
            out_path = out_dir / f"lockin_{colour}_seed{seed}.png"
            fig.savefig(out_path, dpi=140, bbox_inches='tight')
            plt.close(fig)

        k += 1


if __name__ == '__main__':
    main()


