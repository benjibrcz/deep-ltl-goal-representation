#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import numpy as np
import csv
import imageio.v2 as imageio
from typing import Tuple, Optional, List, Dict
import torch
import matplotlib.pyplot as plt
import shutil

# NOTE: This script evaluates behaviour off-policy with a safety interpretation:
# - The environment is created with a simple F a formula.
# - We condition the model on (reach, avoid) tokens via preprocess_obss.
# - Success = the agent reaches the chosen reach letter at any step.
# - Failure = the agent steps on the avoid letter at any step.
# The env’s reward/automaton is not changed to include avoid; we are measuring behaviour, not task return.

# Local helpers (duplicate of 05d to avoid importing a filename starting with digits)
def torus_l1(i: int, j: int, H: int, W: int) -> int:
    return min(i, H - i) + min(j, W - j)

def build_map_two_letters(grid_size: int,
                          reach: str,
                          avoid: str,
                          rng: random.Random,
                          center_candidates: List[Tuple[int, int]],
                          far_threshold: int,
                          surround_k: int = 3) -> Dict[Tuple[int, int], str]:
    H = grid_size
    W = grid_size
    # Agent will start at center
    ci, cj = H // 2, W // 2
    # Choose axis and distance so that two reach letters are mirrored at equal distance
    axis = rng.choice(['h', 'v'])
    max_d = max(1, min(ci, cj))  # stay within grid bounds
    # Enforce at least one cell gap from center to reach; if single-block avoid, also ensure avoid is not adjacent
    min_d = 2
    if surround_k == 1:
        # mid cell would be at distance (d-1); require >=2 -> d>=3
        min_d = 3
    min_d = min(min_d, max_d)
    d_choices = list(range(min_d, max_d + 1)) if max_d >= min_d else [max_d]
    d = rng.choice(d_choices)
    if axis == 'h':
        p1 = (ci, (cj + d) % W)
        p2 = (ci, (cj - d) % W)
    else:
        p1 = ((ci + d) % H, cj)
        p2 = ((ci - d) % H, cj)
    m: Dict[Tuple[int, int], str] = {}
    m[p1] = reach
    m[p2] = reach
    # Avoid placement: if surround_k==1, place exactly one avoid between center and p1.
    # Otherwise, place up to surround_k avoids among 4-neighbors of p1 (skipping center and p2).
    k = max(0, min(int(surround_k), 4))
    if k == 1:
        if axis == 'h':
            # choose step direction toward p1
            step = 1 if p1[1] == (cj + d) % W else -1
            # place avoid two cells from center to keep a gap between agent and avoid
            mid = (ci, (cj + 2 * step) % W)
        else:
            step = 1 if p1[0] == (ci + d) % H else -1
            mid = ((ci + 2 * step) % H, cj)
        # ensure avoid does not overwrite the mirrored reach or the center or the target reach
        if mid != p2 and mid != (ci, cj) and mid != p1:
            m[mid] = avoid
    else:
        nbrs = [((p1[0] - 1) % H, p1[1]), ((p1[0] + 1) % H, p1[1]), (p1[0], (p1[1] - 1) % W), (p1[0], (p1[1] + 1) % W)]
        rng.shuffle(nbrs)
        placed = 0
        for pos in nbrs:
            if placed >= k:
                break
            # do not overwrite the mirrored reach or the agent start
            if pos == p2 or pos == (ci, cj):
                continue
            # also avoid placing avoid adjacent to center to maintain a gap
            if pos in {((ci - 1) % H, cj), ((ci + 1) % H, cj), (ci, (cj - 1) % W), (ci, (cj + 1) % W)}:
                continue
            m[pos] = avoid
            placed += 1
    return m

from utils.model_store.model_store import ModelStore
from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from config import model_configs
from preprocessing.preprocessing import preprocess_obss
from preprocessing.vocab import VOCAB
from ltl.logic.assignment import FrozenAssignment

# Match LetterEnv.action ordering: 0:Up, 1:Down, 2:Left, 3:Right
OFFSETS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def step_pos(H: int, W: int, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    di, dj = OFFSETS.get(int(action), (0, 0))
    return ((pos[0] + di) % H, (pos[1] + dj) % W)


def resolve_env_tokens(VOCAB_map, props: set, reach: str, avoid: str) -> Tuple[FrozenAssignment, FrozenAssignment]:
    env_tokens = {}
    for p in props:
        pstr = str(p).strip().lower()
        found = None
        for k in VOCAB_map.keys():
            if not isinstance(k, FrozenAssignment):
                continue
            s = str(k).strip().lower()
            if s == pstr or s.endswith(pstr) or pstr in s:
                found = k
                break
        if found is not None:
            env_tokens[pstr] = found
    tok_a = env_tokens.get(reach.lower(), None)
    tok_c = env_tokens.get(avoid.lower(), None)
    return tok_a, tok_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=10, help='Number of maps to generate')
    ap.add_argument('--grid_size', type=int, default=7)
    ap.add_argument('--letters', type=str, default='abcdefghijklmnopqrstuvwxyz')
    ap.add_argument('--out_dir', type=str, default='interpretability/letter_world/results/two_letter_e2e')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--far_threshold', type=int, default=4)
    ap.add_argument('--center_candidates', type=str, default='(1,1),(1,5),(5,1),(5,5),(2,4),(4,2)')
    ap.add_argument('--surround_k', type=int, default=3, help='Number of avoid neighbors around the surrounded reach (0..4)')
    # rollout
    ap.add_argument('--env', type=str, default='LetterEnv-v0')
    ap.add_argument('--exp', type=str, default='test')
    ap.add_argument('--model_seed', type=int, default=1)
    ap.add_argument('--horizon', type=int, default=10)
    ap.add_argument('--steps', type=int, default=20)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_dir / 'maps'; maps_dir.mkdir(parents=True, exist_ok=True)
    paths_dir = out_dir / 'paths'; paths_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'rollout_summary.csv'

    # parse centers robustly using regex
    import re
    centers: List[Tuple[int, int]] = []
    for m in re.finditer(r'\((\d+),\s*(\d+)\)', args.center_candidates):
        ii, jj = m.groups()
        centers.append((int(ii), int(jj)))
    if not centers:
        centers = [(1, 1)]

    # build env and model once
    store = ModelStore(args.env, args.exp, args.model_seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[args.env]
    env = make_env(args.env, FixedSampler.partial("F a"), render_mode=None)
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    model = build_model(env, status, cfg).eval()

    rows = []
    # Prefer actual environment propositions if available to avoid token mismatches
    env_props = set(base_env.get_propositions()) if hasattr(base_env, 'get_propositions') else None
    if env_props:
        letters_pool = [str(p).strip().lower() for p in sorted(list(env_props))]
    else:
        letters_pool = [c for c in args.letters if c.isalpha()]
    for k in range(1, args.N + 1):
        reach = rng.choice(letters_pool)
        avoid = rng.choice([c for c in letters_pool if c != reach])
        # build map
        m = build_map_two_letters(args.grid_size, reach, avoid, rng, centers, args.far_threshold, surround_k=args.surround_k)
        letters_str = ''.join(sorted(set(m.values())))
        # load fixed map
        base_env.map = m
        base_env.use_fixed_map = True
        base_env.agent = (args.grid_size // 2, args.grid_size // 2)
        # render and save map image
        vis_env = None
        try:
            # Create a temp renderer env for image
            from envs.letter_world.letter_env import LetterEnv as _LE
        except Exception:
            from src.envs.letter_world.letter_env import LetterEnv as _LE
        vis_env = _LE(grid_size=args.grid_size, letters=letters_str, use_fixed_map=True,
                      use_agent_centric_view=False, render_mode='rgb_array', map=m)
        vis_env.agent = (args.grid_size // 2, args.grid_size // 2)
        png = maps_dir / f"map_{k:02d}_{reach}_{avoid}.png"
        imageio.imwrite(png, vis_env.render())
        # save pickled map
        pkl = maps_dir / f"map_{k:02d}_{reach}_{avoid}.pkl"
        vis_env.save_world_info(str(pkl))

        # set goal tokens
        out = env.reset(seed=k)  # vary seed per map to avoid accidental reuse
        # Force agent start at center after reset
        base_env.agent = (args.grid_size // 2, args.grid_size // 2)
        obs = out[0] if isinstance(out, (tuple, list)) else out
        # Ensure observation reflects centered agent
        obs = dict(obs)
        try:
            obs['features'] = base_env._get_observation()
        except Exception:
            pass
        props = set(base_env.get_propositions())
        tok_a, tok_c = resolve_env_tokens(VOCAB, props, reach, avoid)
        if tok_a is None or tok_c is None:
            rows.append(dict(map=str(pkl.name), reach=reach, avoid=avoid,
                             reached_goal=False, reached_avoid=False, note='token_resolve_failed'))
            continue
        goal_seq = ((tuple([tok_a]), tuple([tok_c])),)
        # rollout with early stop
        actions_seq = []
        cur_obs = obs
        reached_goal = False
        reached_avoid = False
        goal_step = -1
        avoid_step = -1
        for t in range(int(max(0, args.steps))):
            with torch.inference_mode():
                dist_t, _ = model(preprocess_obss([dict(cur_obs, goal=goal_seq)], props))
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
            cur_pos = tuple(base_env.agent)
            if cur_pos in base_env.map:
                letter_here = base_env.map[cur_pos]
                if (not reached_goal) and letter_here == reach:
                    reached_goal = True; goal_step = t + 1
                if (not reached_avoid) and letter_here == avoid:
                    reached_avoid = True; avoid_step = t + 1
            if reached_goal or done_now:
                break

        # save path image
        vis_env2 = _LE(grid_size=args.grid_size, letters=letters_str, use_fixed_map=True,
                       use_agent_centric_view=False, render_mode='rgb_array', map=m)
        # Base map image
        vis_env2.agent = (args.grid_size // 2, args.grid_size // 2)
        base_rgb = vis_env2.render()
        # Draw path overlay from center using matplotlib, scaled to pixel grid
        Hpix, Wpix = base_rgb.shape[0], base_rgb.shape[1]
        cell_w = Wpix // args.grid_size
        cell_h = Hpix // args.grid_size
        ci, cj = args.grid_size // 2, args.grid_size // 2
        # accumulate pixel centers
        pts = []
        pi, pj = ci, cj
        pts.append((pj * cell_w + cell_w / 2.0, pi * cell_h + cell_h / 2.0))
        for a in actions_seq:
            di, dj = OFFSETS.get(int(a), (0, 0))
            pi = (pi + di) % args.grid_size
            pj = (pj + dj) % args.grid_size
            pts.append((pj * cell_w + cell_w / 2.0, pi * cell_h + cell_h / 2.0))
        import matplotlib.pyplot as _plt
        _fig, _ax = _plt.subplots(figsize=(Wpix / 120.0, Hpix / 120.0), dpi=120)
        _ax.imshow(base_rgb)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        _ax.plot(xs, ys, color='teal', linewidth=2, alpha=0.9)
        _ax.scatter(xs[0], ys[0], c='teal', s=30, marker='o')
        _ax.axis('off')
        out_path = paths_dir / f"map_{k:02d}_{reach}_{avoid}_path.png"
        _plt.tight_layout(pad=0)
        _plt.savefig(out_path, dpi=120, bbox_inches='tight', pad_inches=0)
        _plt.close(_fig)

        rows.append(dict(map=str(pkl.name), reach=reach, avoid=avoid,
                         reached_goal=reached_goal, reached_avoid=reached_avoid,
                         goal_step=goal_step, avoid_step=avoid_step, note=''))

    # write summary CSV and simple stats
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['map','reach','avoid','reached_goal','reached_avoid','goal_step','avoid_step','note'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # stats
    rg = sum(1 for r in rows if r['reached_goal'])
    ra = sum(1 for r in rows if r['reached_avoid'])
    print(f"[e2e] maps={len(rows)}  reached_goal: {rg}/{len(rows)}  reached_avoid: {ra}/{len(rows)}")
    print(f"[e2e] outputs in {out_dir}")

    # Confusion matrix: rows=reached_goal False/True, cols=reached_avoid False/True
    cm = np.zeros((2, 2), dtype=int)
    for r in rows:
        i = 1 if r['reached_goal'] else 0
        j = 1 if r['reached_avoid'] else 0
        cm[i, j] += 1
    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=11)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['avoid=False','avoid=True'])
    ax.set_yticklabels(['goal=False','goal=True'])
    ax.set_title('Reached (goal vs avoid)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig_path = out_dir / 'confusion_matrix.png'
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()


