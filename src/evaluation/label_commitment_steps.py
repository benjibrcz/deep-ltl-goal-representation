import sys
from pathlib import Path
import argparse
import csv
from typing import List, Tuple

import numpy as np
import torch
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
    ap.add_argument('--episodes', type=int, default=80)
    ap.add_argument('--colours', type=str, default='blue,green,yellow,magenta')
    ap.add_argument('--max-steps', type=int, default=300)
    ap.add_argument('--num-loops', type=int, default=1)
    ap.add_argument('--out-csv', type=str, default=str(Path(__file__).resolve().parents[2] / 'interpretability' / 'zone_env' / 'results' / 'commit_labels.csv'))
    # thresholds
    ap.add_argument('--diverge_rel_thresh', type=float, default=0.02)
    ap.add_argument('--cos_margin', type=float, default=0.15)
    ap.add_argument('--cos_L', type=int, default=3)
    ap.add_argument('--stability_L', type=int, default=3, help='consecutive Ys required to confirm commitment')
    args = ap.parse_args()

    # model
    env0 = make_env(args.env_id, FixedSampler.partial('FG blue'))
    config = model_configs[args.env_id]
    ms = ModelStore(args.env_id, args.exp, args.seed)
    ms.load_vocab()
    ts = ms.load_training_status(map_location='cpu')
    model = build_model(env0, ts, config)
    env0.close()

    colours = [c.strip() for c in args.colours.split(',') if c.strip()]
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['episode_id', 'seed', 'colour', 't', 'x', 'y', 'dA', 'dB', 'cosA', 'cosB', 'Y'])

        ep_idx = 0
        pbar = trange(args.episodes, desc='label')
        for i in pbar:
            colour = colours[i % len(colours)]
            sampler = FixedSampler.partial(f'FG {colour}')
            env = make_env(args.env_id, sampler)
            props = set(env.get_propositions())
            agent = Agent(model, search=ExhaustiveSearch(model, props, num_loops=args.num_loops), propositions=props)
            obs = reset_unpack(env, seed=args.seed + i)
            agent.reset()

            zone_pos = getattr(env, 'zone_positions', {})
            centres = find_colour_zone_centers(zone_pos, colour)
            if len(centres) < 2:
                env.close();
                continue
            A, B = centres[0], centres[1]

            traj: List[np.ndarray] = []
            d_hist: List[Tuple[float, float]] = []
            cos_hist: List[Tuple[float, float]] = []
            labels: List[str] = []
            consec_Y = 0

            done = False
            t = 0
            while not done and t < args.max_steps:
                # pos and derived
                pos = getattr(env, 'agent_pos', None)
                if pos is not None:
                    p = np.asarray(pos, dtype=float).ravel()[:2]
                    traj.append(p.copy())
                    dA = float(np.linalg.norm(p - A)); dB = float(np.linalg.norm(p - B))
                    d_hist.append((dA, dB))
                    if len(traj) >= 2:
                        v = traj[-1] - traj[-2]
                        cosA = cosine_similarity(v, A - p)
                        cosB = cosine_similarity(v, B - p)
                        cos_hist.append((cosA, cosB))
                    else:
                        cos_hist.append((0.0, 0.0))

                    # commit signals
                    Y_diverge = False
                    if len(d_hist) >= 2:
                        dA_prev, dB_prev = d_hist[-2]
                        ddA = dA - dA_prev; ddB = dB - dB_prev
                        rel = (abs(ddA) + abs(ddB)) / (dA + dB + 1e-8)
                        Y_diverge = (ddA < 0 and ddB > 0 and rel >= args.diverge_rel_thresh) or (ddB < 0 and ddA > 0 and rel >= args.diverge_rel_thresh)

                    Y_cos = False
                    if len(cos_hist) >= args.cos_L:
                        window = cos_hist[-args.cos_L:]
                        dcos = [ca - cb for (ca, cb) in window]
                        Y_cos = all(x > args.cos_margin for x in dcos) or all(x < -args.cos_margin for x in dcos)

                    Y_now = (Y_diverge or Y_cos)
                    consec_Y = consec_Y + 1 if Y_now else 0
                    Y_label = 'Y' if consec_Y >= args.stability_L else 'N'
                    labels.append(Y_label)

                    # write row
                    wr.writerow([ep_idx, args.seed + i, colour, t, p[0], p[1], dA, dB, cos_hist[-1][0], cos_hist[-1][1], Y_label])

                # step
                a = agent.get_action(obs, {}, deterministic=True)
                a = a.flatten()
                step_out = env.step(a)
                if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                    obs, _r, term, trunc, _info = step_out
                    done = bool(term or trunc)
                else:
                    obs, _r, done, _info = step_out
                t += 1

            env.close()
            ep_idx += 1

    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()


