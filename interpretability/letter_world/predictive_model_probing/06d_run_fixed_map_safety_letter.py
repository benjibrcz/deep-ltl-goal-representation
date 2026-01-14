#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from typing import Tuple, Optional

# BFS helpers
try:
    from interpretability.letter_world.predictive_model_probing.bfs_oracle import bfs_shortest
except Exception:
    try:
        from predictive_model_probing.bfs_oracle import bfs_shortest
    except Exception:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.append(str(_Path(__file__).resolve().parent))
        from bfs_oracle import bfs_shortest

from utils.model_store.model_store import ModelStore
from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from config import model_configs
from preprocessing.preprocessing import preprocess_obss
from preprocessing.vocab import VOCAB
from ltl.logic.assignment import FrozenAssignment


# Match LetterEnv.action ordering:
# self.actions = [(-1,0), (1,0), (0,-1), (0,1)]  => 0:Up, 1:Down, 2:Left, 3:Right
OFFSETS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def step_pos(H: int, W: int, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    di, dj = OFFSETS.get(int(action), (0, 0))
    return ((pos[0] + di) % H, (pos[1] + dj) % W)


def resolve_token(letter: str) -> Optional[FrozenAssignment]:
    target = str(letter).strip().lower()
    # Try exact/substring matches on FrozenAssignment keys
    for k in VOCAB.keys():
        if isinstance(k, FrozenAssignment):
            s = str(k).strip()
            if s.lower() == target or s.upper() == target.upper():
                return k
            if s.lower().endswith(target) or s.lower().startswith(target):
                return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', type=str, default='LetterEnv-v0')
    ap.add_argument('--exp', type=str, default='test')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--map_pkl', type=str, required=True, help='Pickled map produced by LetterEnv.save_world_info')
    ap.add_argument('--A', type=str, default='a')
    ap.add_argument('--C', type=str, default='c')
    ap.add_argument('--horizon', type=int, default=10)
    ap.add_argument('--steps', type=int, default=12, help='Number of policy steps to roll for path rendering')
    ap.add_argument('--out_png', type=str, default='interpretability/letter_world/results/fixed_map_safety_path.png')
    args = ap.parse_args()

    # Load model/vocab
    store = ModelStore(args.env, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[args.env]

    # Build env and load fixed map
    env = make_env(args.env, FixedSampler.partial("F a"), render_mode=None)
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    base_env.load_world_info(args.map_pkl)

    # Build model
    model = build_model(env, status, cfg).eval()

    # Reset and construct goal sequence: reach {A,B}, avoid {C}
    out = env.reset(seed=0)
    # Center the agent after reset
    base_env.agent = (base_env.grid_size // 2, base_env.grid_size // 2)
    obs = out[0] if isinstance(out, (tuple, list)) else out
    obs = dict(obs)
    try:
        obs['features'] = base_env._get_observation()
    except Exception:
        pass
    # Resolve tokens by matching env propositions to VOCAB keys to avoid mismatches
    props = set(base_env.get_propositions())
    env_tokens = {}
    for p in props:
        pstr = str(p).strip().lower()
        found = None
        for k in VOCAB.keys():
            if not isinstance(k, FrozenAssignment):
                continue
            s = str(k).strip().lower()
            # exact or lenient match (common in VOCAB formatting)
            if s == pstr or s.endswith(pstr) or pstr in s:
                found = k
                break
        if found is not None:
            env_tokens[pstr] = found
    tok_a = env_tokens.get(args.A.lower(), None)
    tok_c = env_tokens.get(args.C.lower(), None)
    if tok_a is None or tok_c is None:
        raise RuntimeError(f"Could not resolve tokens for A='{args.A}' or C='{args.C}' from env propositions {sorted(list(props))}.")
    # Always: Reach A, Avoid C
    goal_seq = ((tuple([tok_a]), tuple([tok_c])),)
    dl = preprocess_obss([dict(obs, goal=goal_seq)], props)

    # Policy action
    with np.errstate(all='ignore'):
        dist, _ = model(dl)
    try:
        a_t = dist.mode()
    except Exception:
        a_t = dist.sample()
    act = int(a_t.flatten()[0].item())

    print(f"Map: {Path(args.map_pkl).name}")
    print(f"Goal: F {args.A} & G ! {args.C}")
    print(f"Agent pos: {tuple(base_env.agent)}  | first_action={act}")

    # Rollout for --steps and render path overlay
    actions_seq = []
    cur_obs = obs
    reached_goal = False
    reached_avoid = False
    goal_step = -1
    avoid_step = -1
    for t in range(int(max(0, args.steps))):
        dl_t = preprocess_obss([dict(cur_obs, goal=goal_seq)], props)
        with np.errstate(all='ignore'):
            dist_t, _ = model(dl_t)
        try:
            a_t = dist_t.mode()
        except Exception:
            a_t = dist_t.sample()
        a = int(a_t.flatten()[0].item())
        actions_seq.append(a)
        step_out = env.step(a)
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            cur_obs, _, term, trunc, _ = step_out
            done_now = bool(term or trunc)
        else:
            cur_obs, _, done, _ = step_out
            done_now = bool(done)
        # check letter under agent
        cur_pos = tuple(base_env.agent)
        if (cur_pos in base_env.map):
            letter_here = base_env.map[cur_pos]
            if (not reached_goal) and letter_here == args.A:
                reached_goal = True; goal_step = t + 1
            if (not reached_avoid) and letter_here == args.C:
                reached_avoid = True; avoid_step = t + 1
        # stop early once goal is reached
        if reached_goal:
            break
        if done_now:
            break

    print(f"reached_goal={reached_goal} at_step={goal_step if reached_goal else -1}")
    print(f"reached_avoid={reached_avoid} at_step={avoid_step if reached_avoid else -1}")

    # Build a fresh LetterEnv with renderer to draw the path
    try:
        from envs.letter_world.letter_env import LetterEnv as _LE
    except Exception:
        from src.envs.letter_world.letter_env import LetterEnv as _LE
    H = base_env.grid_size
    W = base_env.grid_size
    letters_str = ''.join(sorted(set(base_env.map.values())))
    vis_env = _LE(grid_size=H, letters=letters_str, use_fixed_map=True,
                  use_agent_centric_view=False, render_mode='rgb_array', map=base_env.map)
    vis_env.agent = (0, 0)
    # Base rgb and custom overlay from center
    base_rgb = vis_env.render()
    Hpix, Wpix = base_rgb.shape[0], base_rgb.shape[1]
    cell_w = Wpix // H
    cell_h = Hpix // H
    ci, cj = H // 2, W // 2
    pts = []
    pi, pj = ci, cj
    pts.append((pj * cell_w + cell_w / 2.0, pi * cell_h + cell_h / 2.0))
    for a in actions_seq:
        di, dj = OFFSETS.get(int(a), (0, 0))
        pi = (pi + di) % H
        pj = (pj + dj) % W
        pts.append((pj * cell_w + cell_w / 2.0, pi * cell_h + cell_h / 2.0))
    import matplotlib.pyplot as _plt
    _fig, _ax = _plt.subplots(figsize=(Wpix / 120.0, Hpix / 120.0), dpi=120)
    _ax.imshow(base_rgb)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    _ax.plot(xs, ys, color='teal', linewidth=2, alpha=0.9)
    _ax.scatter(xs[0], ys[0], c='teal', s=30, marker='o')
    _ax.axis('off')
    outp = Path(args.out_png); outp.parent.mkdir(parents=True, exist_ok=True)
    _plt.tight_layout(pad=0)
    _plt.savefig(outp, dpi=120, bbox_inches='tight', pad_inches=0)
    _plt.close(_fig)
    print(f"Saved path render to {outp}")


if __name__ == '__main__':
    main()


