#!/usr/bin/env python3
"""
Sequencing Experiment: Testing Multi-Goal Navigation

This experiment tests whether the agent can handle sequential goals:
- F (blue & F green) - reach blue, then reach green
- F (blue & F (green & F yellow)) - reach blue, then green, then yellow

Key questions:
1. Does the agent visit zones in the correct order?
2. Does it plan an efficient path (visit closer zone first when order allows)?
3. How does performance degrade with sequence length?

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/sequencing_e2e.py
"""
import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Repo imports
SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(SRC))

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent
from visualize.zones import draw_zones, draw_diamond, draw_path, setup_axis

try:
    from gymnasium import spaces as gspaces
except Exception:
    from gym import spaces as gspaces

torch.set_grad_enabled(False)

ZONE_RADIUS = 0.4


def coerce_action(act, action_space):
    if isinstance(action_space, gspaces.Box):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        need = int(np.prod(action_space.shape))
        if a.size == 1 and need > 1:
            a = np.repeat(a, need)
        if a.size != need:
            raise ValueError(f"Action size {a.size} != {need}")
        a = np.clip(a, action_space.low, action_space.high)
        return a.reshape(action_space.shape)
    elif isinstance(action_space, gspaces.Discrete):
        a = int(np.asarray(act).ravel()[0]) if isinstance(act, (np.ndarray, list, tuple)) else int(act)
        return a
    return act


def extract_zone_info(env) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]]]:
    agent_pos = None
    zone_positions = {}

    try:
        task = env.unwrapped.task
        agent_pos = task.agent.pos[:2].copy()

        for geom_name, geom in task._geoms.items():
            if hasattr(geom, 'color_name') and hasattr(geom, 'num'):
                color = geom.color_name
                if color not in zone_positions:
                    zone_positions[color] = []
                for i in range(geom.num):
                    try:
                        body_name = f'{color}_zone{i}'
                        pos = task.data.body(body_name).xpos[:2].copy()
                        zone_positions[color].append(pos)
                    except:
                        pass
    except:
        pass

    return agent_pos, zone_positions


def run_sequencing_scenario(
    env_id: str,
    model,
    formula: str,
    target_sequence: List[str],  # e.g., ['blue', 'green']
    seed: int,
    max_steps: int = 300,
    deterministic: bool = True,
) -> Dict:
    """Run a sequencing scenario and track zone visits."""

    sampler_fn = FixedSampler.partial(formula)
    env = make_env(env_id, sampler_fn, sequence=False)
    props = set(env.get_propositions())

    planner = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, planner, propositions=props)

    reset_out = env.reset(seed=seed)
    obs, info = (reset_out, {}) if not isinstance(reset_out, tuple) else reset_out
    agent.reset()

    agent_start, zone_positions = extract_zone_info(env)

    if agent_start is None:
        env.close()
        return None

    # Track visits
    trajectory = []
    visited_zones = []  # List of (color, step) tuples
    visited_colors_order = []  # Just the colors in order visited

    for step in range(max_steps):
        with torch.no_grad():
            action = agent.get_action(obs, info, deterministic=deterministic)
        action = coerce_action(action, env.action_space)

        try:
            agent_pos = env.unwrapped.task.agent.pos[:2].copy()
        except:
            agent_pos = np.array([np.nan, np.nan])

        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret

        # Check zone entries
        for color, positions in zone_positions.items():
            for pos in positions:
                if np.linalg.norm(agent_pos - pos) < ZONE_RADIUS:
                    if color not in visited_colors_order:
                        visited_zones.append((color, step))
                        visited_colors_order.append(color)

        trajectory.append({
            'step': step,
            'pos_x': float(agent_pos[0]),
            'pos_y': float(agent_pos[1]),
        })

        if done:
            break

    env.close()

    # Check if sequence was completed correctly
    correct_order = True
    completed = True

    for i, target_color in enumerate(target_sequence):
        if i >= len(visited_colors_order):
            completed = False
            correct_order = False
            break
        if visited_colors_order[i] != target_color:
            correct_order = False

    # Check if all targets were visited (even if wrong order)
    all_visited = all(c in visited_colors_order for c in target_sequence)

    # Calculate optimal vs actual path efficiency
    # Optimal: visit zones in nearest-first order
    distances_from_start = {}
    for color in target_sequence:
        if color in zone_positions and zone_positions[color]:
            min_dist = min(np.linalg.norm(agent_start - pos) for pos in zone_positions[color])
            distances_from_start[color] = min_dist

    # What order would distance heuristic suggest?
    distance_order = sorted(target_sequence, key=lambda c: distances_from_start.get(c, float('inf')))

    # Did agent follow distance order?
    followed_distance_heuristic = (visited_colors_order[:len(target_sequence)] == distance_order)

    return {
        'seed': seed,
        'formula': formula,
        'target_sequence': target_sequence,
        'visited_order': visited_colors_order,
        'visited_zones': visited_zones,
        'correct_order': correct_order,
        'completed': completed,
        'all_visited': all_visited,
        'distance_order': distance_order,
        'followed_distance_heuristic': followed_distance_heuristic,
        'distances_from_start': distances_from_start,
        'trajectory': trajectory,
        'zone_positions': {k: [v.tolist() for v in vs] for k, vs in zone_positions.items()},
        'agent_start': agent_start.tolist() if agent_start is not None else None,
    }


def plot_sequencing(result: Dict, out_path: Path):
    """Plot sequencing scenario."""
    fig, ax = plt.subplots(figsize=(10, 10))
    setup_axis(ax)

    traj = result['trajectory']
    path = [(t['pos_x'], t['pos_y']) for t in traj]

    zone_pos = result['zone_positions']
    zone_positions_dict = {}
    for color, positions in zone_pos.items():
        for i, pos in enumerate(positions):
            zone_positions_dict[f'{color}_zone{i}'] = tuple(pos)

    draw_zones(ax, zone_positions_dict)

    if result['agent_start']:
        draw_diamond(ax, result['agent_start'], color='orange')

    draw_path(ax, path, color='darkgreen', linewidth=3)

    for i in range(20, len(path), 20):
        ax.plot(path[i][0], path[i][1], 's', color='red', markersize=6, zorder=5)
    if path:
        ax.plot(path[-1][0], path[-1][1], 's', color='red', markersize=8, zorder=5)

    # Label target sequence zones
    for i, color in enumerate(result['target_sequence']):
        if color in zone_pos and zone_pos[color]:
            pos = zone_pos[color][0]  # First zone of this color
            ax.annotate(f'{i+1}', xy=pos, ha='center', va='center',
                       fontsize=14, fontweight='bold', color='white',
                       bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', alpha=0.7))

    target = " → ".join(result['target_sequence'])
    visited = " → ".join(result['visited_order'][:len(result['target_sequence'])]) if result['visited_order'] else "none"
    status = "✓ CORRECT" if result['correct_order'] and result['completed'] else "✗ WRONG/INCOMPLETE"

    title = f"Seed {result['seed']}: {status}\n"
    title += f"Target: {target}\n"
    title += f"Visited: {visited}"
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_scenarios', type=int, default=30)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/sequencing')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Load model
    print(f"Loading model: {args.env_id} / {args.exp} / seed={args.seed}")
    dummy_env = make_env(args.env_id, FixedSampler.partial("F blue"), sequence=False)
    cfg = model_configs[args.env_id]
    store = ModelStore(args.env_id, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    model = build_model(dummy_env, status, cfg).eval()
    dummy_env.close()

    # Define test cases
    test_cases = [
        # 2-step sequences
        {'sequence': ['blue', 'green'], 'formula': 'F (blue & F green)'},
        {'sequence': ['green', 'blue'], 'formula': 'F (green & F blue)'},
        {'sequence': ['yellow', 'magenta'], 'formula': 'F (yellow & F magenta)'},
        {'sequence': ['magenta', 'yellow'], 'formula': 'F (magenta & F yellow)'},
        # 3-step sequences
        {'sequence': ['blue', 'green', 'yellow'], 'formula': 'F (blue & F (green & F yellow))'},
        {'sequence': ['green', 'yellow', 'blue'], 'formula': 'F (green & F (yellow & F blue))'},
    ]

    print(f"\nRunning {args.n_scenarios} scenarios per test case...")
    all_results = []

    for tc in test_cases:
        sequence = tc['sequence']
        formula = tc['formula']
        print(f"\nTesting: {' → '.join(sequence)}")

        for i in range(args.n_scenarios):
            seed = args.seed + i * 100

            result = run_sequencing_scenario(
                args.env_id, model, formula, sequence, seed,
                max_steps=args.max_steps,
            )

            if result is not None:
                all_results.append(result)

                # Save plot for first few
                if i < 5:
                    status = 'correct' if result['correct_order'] and result['completed'] else 'wrong'
                    plot_path = plots_dir / f"seq_{'_'.join(sequence)}_{seed}_{status}.png"
                    try:
                        plot_sequencing(result, plot_path)
                    except Exception as e:
                        print(f"  Plot failed: {e}")

    # Save results
    with open(out_dir / 'results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Create summary
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'seed': r['seed'],
            'sequence': '_'.join(r['target_sequence']),
            'sequence_length': len(r['target_sequence']),
            'correct_order': r['correct_order'],
            'completed': r['completed'],
            'all_visited': r['all_visited'],
            'followed_distance_heuristic': r['followed_distance_heuristic'],
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / 'summary.csv', index=False)

    # Print statistics
    print("\n" + "="*70)
    print("SEQUENCING EXPERIMENT RESULTS")
    print("="*70)

    for seq_name in df['sequence'].unique():
        subset = df[df['sequence'] == seq_name]
        total = len(subset)
        seq_len = subset['sequence_length'].iloc[0]

        print(f"\n{'='*50}")
        print(f"Sequence: {seq_name.replace('_', ' → ')} (length {seq_len})")
        print(f"{'='*50}")

        correct = subset['correct_order'].sum()
        completed = subset['completed'].sum()
        all_visited = subset['all_visited'].sum()
        followed_dist = subset['followed_distance_heuristic'].sum()

        print(f"  Completed correctly  : {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"  Completed (any order): {completed}/{total} ({100*completed/total:.1f}%)")
        print(f"  All zones visited    : {all_visited}/{total} ({100*all_visited/total:.1f}%)")
        print(f"  Followed distance heuristic: {followed_dist}/{total} ({100*followed_dist/total:.1f}%)")

    # Overall by sequence length
    print(f"\n{'='*50}")
    print("SUMMARY BY SEQUENCE LENGTH")
    print(f"{'='*50}")

    for length in sorted(df['sequence_length'].unique()):
        subset = df[df['sequence_length'] == length]
        total = len(subset)
        correct = subset['correct_order'].sum()
        followed_dist = subset['followed_distance_heuristic'].sum()

        print(f"\n  Length {length}:")
        print(f"    Correct order: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"    Distance heuristic: {followed_dist}/{total} ({100*followed_dist/total:.1f}%)")

    print(f"\n{'='*50}")
    print("KEY INSIGHT")
    print(f"{'='*50}")

    # Check if agent follows distance heuristic vs required order
    two_step = df[df['sequence_length'] == 2]
    if len(two_step) > 0:
        correct_2 = two_step['correct_order'].mean()
        dist_2 = two_step['followed_distance_heuristic'].mean()
        print(f"\n  For 2-step sequences:")
        print(f"    Followed required order: {100*correct_2:.1f}%")
        print(f"    Followed distance (nearest-first): {100*dist_2:.1f}%")

        if dist_2 > correct_2:
            print(f"\n  >>> Agent prefers NEAREST zone over REQUIRED order")
            print(f"  >>> This suggests distance heuristics, not goal sequencing")
        elif correct_2 > 0.8:
            print(f"\n  >>> Agent follows REQUIRED order correctly")
            print(f"  >>> This could indicate goal representation")

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
