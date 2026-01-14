import sys
from pathlib import Path
import argparse
from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt

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


def lookahead_choice(env, agent, obs, centres: List[np.ndarray], steps: int = 20, enter_tol: float = 1.0) -> int:
    traj: List[np.ndarray] = []
    o = obs
    for _ in range(steps):
        a = agent.get_action(o, {}, deterministic=True)
        a = a.flatten()
        step_out = env.step(a)
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            o, _r, term, trunc, _info = step_out
            done = bool(term or trunc)
        else:
            o, _r, done, _info = step_out
        pos = getattr(env, 'agent_pos', None)
        if pos is not None:
            traj.append(np.asarray(pos, dtype=float).ravel()[:2])
        if done:
            break
    # first-hit
    for p in traj:
        d = [float(np.linalg.norm(p - c)) for c in centres]
        for j in range(2):
            if d[j] <= enter_tol:
                return j
    # fallback: min future distance
    if traj:
        mins = [min(float(np.linalg.norm(pp - c)) for pp in traj) for c in centres]
        return int(np.argmin(mins))
    return 0


def decompose_along_axis(p: np.ndarray, cA: np.ndarray, cB: np.ndarray) -> Tuple[float, np.ndarray, float]:
    u = cB - cA
    D = float(np.linalg.norm(u))
    if D == 0:
        return 0.0, p - cA, 0.0
    u = u / D
    v = p - cA
    alpha = float(np.dot(v, u))
    perp = v - alpha * u
    return alpha, perp, D


def reconstruct_from(alpha: float, perp: np.ndarray, cA: np.ndarray, cB: np.ndarray) -> np.ndarray:
    u = cB - cA
    D = float(np.linalg.norm(u))
    if D == 0:
        return cA.copy()
    u = u / D
    return cA + alpha * u + perp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-id', type=str, default='PointLtl2-v0')
    ap.add_argument('--exp', type=str, default='big_test')
    ap.add_argument('--seed', type=int, default=1002)
    ap.add_argument('--colour', type=str, default='green')
    ap.add_argument('--num-loops', type=int, default=1)
    ap.add_argument('--max-steps', type=int, default=300)
    ap.add_argument('--lookahead', type=int, default=20)
    ap.add_argument('--sweep-points', type=int, default=21)
    ap.add_argument('--sweep-span', type=float, default=0.6, help='fraction of inter-zone distance to sweep towards the other zone')
    ap.add_argument('--out-dir', type=str, default=str(Path(__file__).resolve().parents[2] / 'interpretability' / 'audit_plots'))
    args = ap.parse_args()

    # Build model once
    env0 = make_env(args.env_id, FixedSampler.partial(f'FG {args.colour}'))
    config = model_configs[args.env_id]
    ms = ModelStore(args.env_id, args.exp, 0)
    ms.load_vocab()
    ts = ms.load_training_status(map_location='cpu')
    model = build_model(env0, ts, config)
    env0.close()

    # Baseline world to extract centres and start
    sampler = FixedSampler.partial(f'FG {args.colour}')
    env = make_env(args.env_id, sampler)
    props = set(env.get_propositions())
    agent = Agent(model, search=ExhaustiveSearch(model, props, num_loops=args.num_loops), propositions=props)
    obs = reset_unpack(env, seed=args.seed)
    agent.reset()
    zone_pos = getattr(env, 'zone_positions', {})
    centres = find_colour_zone_centers(zone_pos, args.colour)
    if len(centres) < 2:
        print('Not enough centres for colour')
        return
    cA, cB = centres[0], centres[1]
    p_start = np.asarray(getattr(env, 'agent_pos', [0.0, 0.0]), dtype=float).ravel()[:2]
    env.close()

    # Determine baseline choice and set axis direction from chosen to other
    envb = make_env(args.env_id, sampler)
    propsb = set(envb.get_propositions())
    agentb = Agent(model, search=ExhaustiveSearch(model, propsb, num_loops=args.num_loops), propositions=propsb)
    obsb = reset_unpack(envb, seed=args.seed)
    agentb.reset()
    choice = lookahead_choice(envb, agentb, obsb, [cA, cB], steps=args.lookahead)
    envb.close()
    if choice == 0:
        c_chosen, c_other = cA, cB
    else:
        c_chosen, c_other = cB, cA

    alpha0, perp, D = decompose_along_axis(p_start, c_chosen, c_other)
    alpha_end = alpha0 + args.sweep_span * max(0.0, D - alpha0)
    alphas = np.linspace(alpha0, alpha_end, args.sweep_points)

    # Sweep
    decisions: List[int] = []
    for a in alphas:
        envs = make_env(args.env_id, sampler)
        propss = set(envs.get_propositions())
        agents = Agent(model, search=ExhaustiveSearch(model, propss, num_loops=args.num_loops), propositions=propss)
        obss = reset_unpack(envs, seed=args.seed)
        agents.reset()
        # teleport start along axis-parallel line
        p_new = reconstruct_from(a, perp, c_chosen, c_other)
        try:
            envs.agent_pos = p_new.copy()
        except Exception:
            pass
        dec = lookahead_choice(envs, agents, obss, [cA, cB], steps=args.lookahead)
        decisions.append(dec)
        envs.close()

    # Find flip point index
    flip_idx = None
    base_label = 0 if (c_chosen is cA) else 1
    for i, d in enumerate(decisions):
        if d != base_label:
            flip_idx = i
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot decision vs alpha
    xs = np.arange(len(alphas))
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.scatter(xs, decisions, c=['tab:blue' if d==0 else 'tab:red' for d in decisions], s=28)
    if flip_idx is not None:
        ax.axvline(flip_idx, color='black', linestyle='--', linewidth=1)
    ax.set_yticks([0,1]); ax.set_yticklabels(['zone0','zone1'])
    ax.set_xlabel('sweep index (along chosen→other axis)')
    ax.set_title(f"sweep {args.colour} seed={args.seed} | flip={'none' if flip_idx is None else flip_idx}")
    fig.tight_layout()
    fig.savefig(out_dir / f"sweep_{args.colour}_seed{args.seed}.png", dpi=140, bbox_inches='tight')
    plt.close(fig)

    # Distance-time plot from baseline
    envd = make_env(args.env_id, sampler)
    propsd = set(envd.get_propositions())
    agentd = Agent(model, search=ExhaustiveSearch(model, propsd, num_loops=args.num_loops), propositions=propsd)
    obsd = reset_unpack(envd, seed=args.seed)
    agentd.reset()
    d0_list: List[float] = []
    d1_list: List[float] = []
    done = False
    t = 0
    while not done and t < args.max_steps:
        pos = getattr(envd, 'agent_pos', None)
        if pos is not None:
            p = np.asarray(pos, dtype=float).ravel()[:2]
            d0_list.append(float(np.linalg.norm(p - cA)))
            d1_list.append(float(np.linalg.norm(p - cB)))
        act = agentd.get_action(obsd, {}, deterministic=True)
        act = act.flatten()
        step_out = envd.step(act)
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            obsd, _r, term, trunc, _info = step_out
            done = bool(term or trunc)
        else:
            obsd, _r, done, _info = step_out
        t += 1
    envd.close()

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 3))
    ax2.plot(d0_list, color='tab:blue', label='zone0')
    ax2.plot(d1_list, color='tab:red', label='zone1')
    ax2.set_xlabel('t')
    ax2.set_ylabel('distance')
    ax2.legend()
    ax2.set_title(f"distances over time | {args.colour} seed={args.seed}")
    fig2.tight_layout()
    fig2.savefig(out_dir / f"dists_{args.colour}_seed{args.seed}.png", dpi=140, bbox_inches='tight')
    plt.close(fig2)


if __name__ == '__main__':
    main()


