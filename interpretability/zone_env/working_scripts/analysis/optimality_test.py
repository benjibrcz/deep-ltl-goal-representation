#!/usr/bin/env python3
"""
Optimality Test: Does the agent choose globally optimal paths?

From DeepLTL paper Figure 1(b):
- Task: F (blue & F green) - reach blue, then green
- Myopic approach: Go to nearest blue first (orange path)
- Optimal approach: Go to farther blue that's closer to green (green path)

This tests whether the agent plans ahead to minimize total path length,
or just greedily goes to the nearest subgoal.

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/optimality_test.py
"""
import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(SRC))

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent

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
        a = np.clip(a, action_space.low, action_space.high)
        return a.reshape(action_space.shape)
    elif isinstance(action_space, gspaces.Discrete):
        return int(np.asarray(act).ravel()[0]) if isinstance(act, (np.ndarray, list, tuple)) else int(act)
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


def compute_path_lengths(agent_start: np.ndarray, first_zones: List[np.ndarray],
                         second_zones: List[np.ndarray]) -> Dict:
    """Compute optimal and greedy path lengths for a two-goal task."""

    # Find nearest first goal (greedy choice)
    dists_to_first = [np.linalg.norm(agent_start - z) for z in first_zones]
    greedy_first_idx = np.argmin(dists_to_first)
    greedy_first = first_zones[greedy_first_idx]

    # From greedy first, find nearest second
    dists_greedy_to_second = [np.linalg.norm(greedy_first - z) for z in second_zones]
    greedy_second_idx = np.argmin(dists_greedy_to_second)
    greedy_path_length = dists_to_first[greedy_first_idx] + dists_greedy_to_second[greedy_second_idx]

    # Find globally optimal path
    best_path_length = float('inf')
    best_first_idx = None
    best_second_idx = None

    for i, fz in enumerate(first_zones):
        dist_to_first = np.linalg.norm(agent_start - fz)
        for j, sz in enumerate(second_zones):
            dist_first_to_second = np.linalg.norm(fz - sz)
            total = dist_to_first + dist_first_to_second
            if total < best_path_length:
                best_path_length = total
                best_first_idx = i
                best_second_idx = j

    optimal_first = first_zones[best_first_idx]

    return {
        'greedy_first_idx': greedy_first_idx,
        'greedy_first_pos': greedy_first,
        'greedy_path_length': greedy_path_length,
        'optimal_first_idx': best_first_idx,
        'optimal_first_pos': optimal_first,
        'optimal_path_length': best_path_length,
        'greedy_is_optimal': greedy_first_idx == best_first_idx,
        'path_length_difference': greedy_path_length - best_path_length,
    }


def run_optimality_scenario(
    env_id: str,
    model,
    first_color: str,
    second_color: str,
    seed: int,
    max_steps: int = 300,
) -> Optional[Dict]:
    """Run a scenario testing if agent chooses optimal vs greedy path."""

    # F (first & F second) - reach first, then second
    formula = f"F ({first_color} & F {second_color})"
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

    first_zones = zone_positions.get(first_color, [])
    second_zones = zone_positions.get(second_color, [])

    if len(first_zones) < 2 or not second_zones:
        # Need at least 2 first-color zones to have a choice
        env.close()
        return None

    # Compute optimal vs greedy paths
    path_info = compute_path_lengths(agent_start, first_zones, second_zones)

    # Skip if greedy is already optimal (no interesting test case)
    if path_info['greedy_is_optimal']:
        env.close()
        return None

    # Skip if difference is too small to matter
    if path_info['path_length_difference'] < 0.3:
        env.close()
        return None

    # Run the rollout
    trajectory = []
    visited_first = None
    visited_second = None
    visited_first_step = None
    visited_second_step = None

    for step in range(max_steps):
        with torch.no_grad():
            action = agent.get_action(obs, info, deterministic=True)
        action = coerce_action(action, env.action_space)

        try:
            agent_pos = env.unwrapped.task.agent.pos[:2].copy()
        except:
            agent_pos = np.array([np.nan, np.nan])

        trajectory.append({
            'step': step,
            'pos_x': float(agent_pos[0]),
            'pos_y': float(agent_pos[1]),
        })

        # Check zone visits
        if visited_first is None:
            for i, fz in enumerate(first_zones):
                if np.linalg.norm(agent_pos - fz) < ZONE_RADIUS:
                    visited_first = i
                    visited_first_step = step
                    break

        if visited_first is not None and visited_second is None:
            for i, sz in enumerate(second_zones):
                if np.linalg.norm(agent_pos - sz) < ZONE_RADIUS:
                    visited_second = i
                    visited_second_step = step
                    break

        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret

        if done:
            break

    env.close()

    # Determine what the agent chose
    if visited_first is None:
        choice = 'failed'
    elif visited_first == path_info['optimal_first_idx']:
        choice = 'optimal'
    elif visited_first == path_info['greedy_first_idx']:
        choice = 'greedy'
    else:
        choice = 'other'

    # Calculate actual path length
    actual_path_length = None
    if visited_first is not None and visited_second is not None:
        dist_to_first = np.linalg.norm(agent_start - first_zones[visited_first])
        dist_first_to_second = np.linalg.norm(first_zones[visited_first] - second_zones[visited_second])
        actual_path_length = dist_to_first + dist_first_to_second

    return {
        'seed': seed,
        'formula': formula,
        'first_color': first_color,
        'second_color': second_color,
        'agent_start': agent_start.tolist(),
        'first_zones': [z.tolist() for z in first_zones],
        'second_zones': [z.tolist() for z in second_zones],
        'greedy_first_idx': path_info['greedy_first_idx'],
        'optimal_first_idx': path_info['optimal_first_idx'],
        'greedy_path_length': path_info['greedy_path_length'],
        'optimal_path_length': path_info['optimal_path_length'],
        'path_length_difference': path_info['path_length_difference'],
        'visited_first_idx': visited_first,
        'visited_second_idx': visited_second,
        'choice': choice,
        'completed': visited_first is not None and visited_second is not None,
        'actual_path_length': actual_path_length,
        'trajectory': trajectory,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_scenarios', type=int, default=200)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/optimality_test')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.env_id} / {args.exp} / seed={args.seed}")
    dummy_env = make_env(args.env_id, FixedSampler.partial("F blue"), sequence=False)
    cfg = model_configs[args.env_id]
    store = ModelStore(args.env_id, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    model = build_model(dummy_env, status, cfg).eval()
    dummy_env.close()

    # Color pairs for two-goal tasks
    color_pairs = [
        ('blue', 'green'),
        ('green', 'blue'),
        ('yellow', 'magenta'),
        ('magenta', 'yellow'),
    ]

    print(f"\nSearching for optimality test scenarios (need greedy != optimal)...")
    results = []
    attempts = 0

    while len(results) < args.n_scenarios and attempts < args.n_scenarios * 20:
        seed = args.seed + attempts * 100
        first_color, second_color = color_pairs[attempts % len(color_pairs)]

        result = run_optimality_scenario(
            args.env_id, model, first_color, second_color, seed, args.max_steps
        )

        if result is not None:
            results.append(result)
            if len(results) % 10 == 0:
                print(f"  Found {len(results)} valid scenarios (tried {attempts})")

        attempts += 1

    print(f"\nCollected {len(results)} valid optimality test scenarios")

    if not results:
        print("No valid scenarios found!")
        return

    # Save results
    with open(out_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Analyze results
    df = pd.DataFrame([{
        'seed': r['seed'],
        'choice': r['choice'],
        'completed': r['completed'],
        'greedy_path': r['greedy_path_length'],
        'optimal_path': r['optimal_path_length'],
        'path_diff': r['path_length_difference'],
        'actual_path': r['actual_path_length'],
    } for r in results])

    df.to_csv(out_dir / 'summary.csv', index=False)

    # Print statistics
    print("\n" + "="*70)
    print("OPTIMALITY TEST RESULTS")
    print("="*70)

    total = len(df)
    optimal_count = (df['choice'] == 'optimal').sum()
    greedy_count = (df['choice'] == 'greedy').sum()
    other_count = (df['choice'] == 'other').sum()
    failed_count = (df['choice'] == 'failed').sum()

    print(f"\n  Total scenarios: {total}")
    print(f"  (All scenarios have greedy != optimal path)")
    print(f"\n  Agent chose:")
    print(f"    OPTIMAL path: {optimal_count} ({100*optimal_count/total:.1f}%)")
    print(f"    GREEDY path:  {greedy_count} ({100*greedy_count/total:.1f}%)")
    print(f"    Other:        {other_count} ({100*other_count/total:.1f}%)")
    print(f"    Failed:       {failed_count} ({100*failed_count/total:.1f}%)")

    print(f"\n  Average path length difference: {df['path_diff'].mean():.2f}")

    # Compare actual path lengths
    completed = df[df['completed'] == True]
    if len(completed) > 0:
        optimal_chosen = completed[completed['choice'] == 'optimal']
        greedy_chosen = completed[completed['choice'] == 'greedy']

        print(f"\n  Completed scenarios: {len(completed)}")
        if len(optimal_chosen) > 0:
            print(f"    When optimal chosen: avg path = {optimal_chosen['actual_path'].mean():.2f}")
        if len(greedy_chosen) > 0:
            print(f"    When greedy chosen:  avg path = {greedy_chosen['actual_path'].mean():.2f}")

    print(f"\n{'='*50}")
    print("INTERPRETATION")
    print(f"{'='*50}")

    if optimal_count > greedy_count:
        print(f"\n  >>> AGENT SHOWS OPTIMAL PLANNING")
        print(f"  >>> Chose optimal path {100*optimal_count/total:.1f}% of the time")
        print(f"  >>> This suggests planning beyond greedy/myopic behavior")
    elif greedy_count > optimal_count:
        print(f"\n  >>> AGENT IS GREEDY/MYOPIC")
        print(f"  >>> Chose greedy path {100*greedy_count/total:.1f}% of the time")
        print(f"  >>> No evidence of optimal planning")
    else:
        print(f"\n  >>> INCONCLUSIVE")
        print(f"  >>> Agent split between optimal and greedy")

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 6))

    choices = ['optimal', 'greedy', 'other', 'failed']
    counts = [optimal_count, greedy_count, other_count, failed_count]
    colors = ['green', 'orange', 'gray', 'red']

    ax.bar(choices, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Optimality Test: Agent Choice Distribution\n(Only scenarios where greedy ≠ optimal)')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (c, count) in enumerate(zip(choices, counts)):
        ax.text(i, count + 1, f'{100*count/total:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / 'optimality_results.png', dpi=150)
    plt.close()

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
