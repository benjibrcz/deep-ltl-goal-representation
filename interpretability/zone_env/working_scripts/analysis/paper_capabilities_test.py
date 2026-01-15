#!/usr/bin/env python3
"""
Paper Capabilities Test: Testing DeepLTL's claimed capabilities

From Figure 1 of the DeepLTL paper:
1. Infinite horizon (ω-regular): G F blue ∧ G F green
2. Optimality: F (blue & F green) - does agent find efficient paths?
3. Safety: (F green | F yellow) & G !blue - does agent avoid dangerous paths?

This tests these capabilities empirically without strict scenario filtering.

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/paper_capabilities_test.py
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


def run_generic_scenario(
    env_id: str,
    model,
    formula: str,
    seed: int,
    max_steps: int = 300,
) -> Optional[Dict]:
    """Run a scenario with any formula and collect comprehensive data."""

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
    zone_visits = []  # (color, step, position)
    total_reward = 0

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

        # Check zone contacts
        for color, positions in zone_positions.items():
            for pos in positions:
                if np.linalg.norm(agent_pos - pos) < ZONE_RADIUS:
                    zone_visits.append((color, step, pos.tolist()))

        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret

        total_reward += rew

        if done:
            break

    env.close()

    # Calculate path length
    path_length = 0
    for i in range(1, len(trajectory)):
        dx = trajectory[i]['pos_x'] - trajectory[i-1]['pos_x']
        dy = trajectory[i]['pos_y'] - trajectory[i-1]['pos_y']
        path_length += np.sqrt(dx*dx + dy*dy)

    return {
        'seed': seed,
        'formula': formula,
        'agent_start': agent_start.tolist(),
        'zone_positions': {k: [v.tolist() for v in vs] for k, vs in zone_positions.items()},
        'trajectory': trajectory,
        'zone_visits': zone_visits,
        'path_length': path_length,
        'total_reward': total_reward,
        'steps': len(trajectory),
        'success': total_reward > 0,
    }


def test_optimality(env_id: str, model, n_scenarios: int, seed: int, max_steps: int) -> pd.DataFrame:
    """Test optimality: F (blue & F green) - two-goal sequencing."""
    print("\n" + "="*60)
    print("TEST 1: OPTIMALITY - F (blue & F green)")
    print("="*60)

    formula = "F (blue & F green)"
    results = []

    for i in range(n_scenarios):
        s = seed + i * 100
        result = run_generic_scenario(env_id, model, formula, s, max_steps)
        if result:
            # Compute optimal path length for this scenario
            agent_start = np.array(result['agent_start'])
            blue_zones = [np.array(z) for z in result['zone_positions'].get('blue', [])]
            green_zones = [np.array(z) for z in result['zone_positions'].get('green', [])]

            if blue_zones and green_zones:
                # Find optimal path
                optimal_path = float('inf')
                for bz in blue_zones:
                    for gz in green_zones:
                        path = np.linalg.norm(agent_start - bz) + np.linalg.norm(bz - gz)
                        optimal_path = min(optimal_path, path)

                # Find greedy path (nearest blue first)
                nearest_blue_idx = np.argmin([np.linalg.norm(agent_start - bz) for bz in blue_zones])
                nearest_blue = blue_zones[nearest_blue_idx]
                nearest_green_from_blue = green_zones[np.argmin([np.linalg.norm(nearest_blue - gz) for gz in green_zones])]
                greedy_path = np.linalg.norm(agent_start - nearest_blue) + np.linalg.norm(nearest_blue - nearest_green_from_blue)

                # Check which blue the agent actually visited first
                blue_visits = [(v[1], v[2]) for v in result['zone_visits'] if v[0] == 'blue']
                if blue_visits:
                    first_blue = np.array(blue_visits[0][1])
                    # Did agent visit optimal blue or greedy blue?
                    dist_to_nearest = np.linalg.norm(first_blue - nearest_blue)
                    chose_greedy = dist_to_nearest < 0.1
                else:
                    chose_greedy = None

                results.append({
                    'seed': s,
                    'success': result['success'],
                    'path_length': result['path_length'],
                    'optimal_path': optimal_path,
                    'greedy_path': greedy_path,
                    'path_diff': greedy_path - optimal_path,
                    'efficiency': optimal_path / result['path_length'] if result['path_length'] > 0 else 0,
                    'chose_greedy': chose_greedy,
                })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_scenarios} completed")

    df = pd.DataFrame(results)

    print(f"\n  Success rate: {df['success'].mean()*100:.1f}%")
    print(f"  Average efficiency (optimal/actual): {df['efficiency'].mean():.2f}")
    print(f"  Scenarios where greedy != optimal: {(df['path_diff'] > 0.1).sum()}")

    greedy_diff = df[df['path_diff'] > 0.1]
    if len(greedy_diff) > 0:
        chose_greedy_count = greedy_diff['chose_greedy'].sum()
        print(f"  When greedy != optimal:")
        print(f"    Chose greedy first: {chose_greedy_count}/{len(greedy_diff)} ({100*chose_greedy_count/len(greedy_diff):.1f}%)")

    return df


def test_safety(env_id: str, model, n_scenarios: int, seed: int, max_steps: int) -> pd.DataFrame:
    """Test safety: (F green | F yellow) & G !blue - avoid while reaching."""
    print("\n" + "="*60)
    print("TEST 2: SAFETY - (F green | F yellow) & G !blue")
    print("="*60)

    formula = "(F green | F yellow) & G !blue"
    results = []

    for i in range(n_scenarios):
        s = seed + i * 100
        result = run_generic_scenario(env_id, model, formula, s, max_steps)
        if result:
            # Check if blue was ever visited (safety violation)
            blue_visits = [v for v in result['zone_visits'] if v[0] == 'blue']
            green_visits = [v for v in result['zone_visits'] if v[0] == 'green']
            yellow_visits = [v for v in result['zone_visits'] if v[0] == 'yellow']

            violated_safety = len(blue_visits) > 0
            reached_green = len(green_visits) > 0
            reached_yellow = len(yellow_visits) > 0

            # Which goal did agent go to first?
            goal_visits = [(v[0], v[1]) for v in result['zone_visits'] if v[0] in ['green', 'yellow']]
            first_goal = goal_visits[0][0] if goal_visits else None

            # Compute distances
            agent_start = np.array(result['agent_start'])
            green_zones = [np.array(z) for z in result['zone_positions'].get('green', [])]
            yellow_zones = [np.array(z) for z in result['zone_positions'].get('yellow', [])]

            dist_to_green = min([np.linalg.norm(agent_start - gz) for gz in green_zones]) if green_zones else float('inf')
            dist_to_yellow = min([np.linalg.norm(agent_start - yz) for yz in yellow_zones]) if yellow_zones else float('inf')

            results.append({
                'seed': s,
                'success': result['success'],
                'violated_safety': violated_safety,
                'reached_green': reached_green,
                'reached_yellow': reached_yellow,
                'first_goal': first_goal,
                'dist_to_green': dist_to_green,
                'dist_to_yellow': dist_to_yellow,
                'nearer_goal': 'green' if dist_to_green < dist_to_yellow else 'yellow',
            })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_scenarios} completed")

    df = pd.DataFrame(results)

    print(f"\n  Success rate: {df['success'].mean()*100:.1f}%")
    print(f"  Safety violations: {df['violated_safety'].sum()}/{len(df)} ({df['violated_safety'].mean()*100:.1f}%)")
    print(f"  Reached green: {df['reached_green'].sum()}/{len(df)}")
    print(f"  Reached yellow: {df['reached_yellow'].sum()}/{len(df)}")

    # Did agent choose nearer goal?
    valid = df[df['first_goal'].notna()]
    if len(valid) > 0:
        chose_nearer = (valid['first_goal'] == valid['nearer_goal']).sum()
        print(f"  Chose nearer goal: {chose_nearer}/{len(valid)} ({100*chose_nearer/len(valid):.1f}%)")

    return df


def test_infinite_horizon(env_id: str, model, n_scenarios: int, seed: int, max_steps: int) -> pd.DataFrame:
    """Test infinite horizon: G F blue & G F green - keep visiting both."""
    print("\n" + "="*60)
    print("TEST 3: INFINITE HORIZON - G F blue & G F green")
    print("="*60)

    formula = "G F blue & G F green"
    results = []

    for i in range(n_scenarios):
        s = seed + i * 100
        result = run_generic_scenario(env_id, model, formula, s, max_steps)
        if result:
            blue_visits = [v for v in result['zone_visits'] if v[0] == 'blue']
            green_visits = [v for v in result['zone_visits'] if v[0] == 'green']

            # Count unique visits (visits separated by at least 10 steps)
            def count_distinct_visits(visits):
                if not visits:
                    return 0
                distinct = 1
                last_step = visits[0][1]
                for v in visits[1:]:
                    if v[1] - last_step > 10:
                        distinct += 1
                        last_step = v[1]
                return distinct

            blue_count = count_distinct_visits(blue_visits)
            green_count = count_distinct_visits(green_visits)

            results.append({
                'seed': s,
                'success': result['success'],
                'blue_visits': blue_count,
                'green_visits': green_count,
                'total_visits': blue_count + green_count,
                'alternating': min(blue_count, green_count),
                'steps': result['steps'],
            })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_scenarios} completed")

    df = pd.DataFrame(results)

    print(f"\n  Success rate: {df['success'].mean()*100:.1f}%")
    print(f"  Average blue visits: {df['blue_visits'].mean():.1f}")
    print(f"  Average green visits: {df['green_visits'].mean():.1f}")
    print(f"  Average total visits: {df['total_visits'].mean():.1f}")
    print(f"  Average min(blue,green): {df['alternating'].mean():.1f}")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_scenarios', type=int, default=50)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/paper_capabilities')
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

    print(f"\nRunning {args.n_scenarios} scenarios per capability test...")

    # Run all three tests
    df_opt = test_optimality(args.env_id, model, args.n_scenarios, args.seed, args.max_steps)
    df_safe = test_safety(args.env_id, model, args.n_scenarios, args.seed + 10000, args.max_steps)
    df_inf = test_infinite_horizon(args.env_id, model, args.n_scenarios, args.seed + 20000, args.max_steps)

    # Save results
    df_opt.to_csv(out_dir / 'optimality.csv', index=False)
    df_safe.to_csv(out_dir / 'safety.csv', index=False)
    df_inf.to_csv(out_dir / 'infinite_horizon.csv', index=False)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: PAPER CAPABILITY TESTS")
    print("="*70)

    print(f"\n  1. OPTIMALITY (F (blue & F green)):")
    print(f"     Success: {df_opt['success'].mean()*100:.1f}%")
    print(f"     Efficiency: {df_opt['efficiency'].mean():.2f}")

    print(f"\n  2. SAFETY ((F green | F yellow) & G !blue):")
    print(f"     Success: {df_safe['success'].mean()*100:.1f}%")
    print(f"     Safety violations: {df_safe['violated_safety'].mean()*100:.1f}%")

    print(f"\n  3. INFINITE HORIZON (G F blue & G F green):")
    print(f"     Success: {df_inf['success'].mean()*100:.1f}%")
    print(f"     Avg visits per goal: {df_inf['alternating'].mean():.1f}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Optimality
    ax = axes[0]
    ax.bar(['Success', 'Greedy'], [df_opt['success'].mean(), df_opt['chose_greedy'].mean()], alpha=0.7)
    ax.set_ylabel('Rate')
    ax.set_title('Optimality Test')
    ax.set_ylim(0, 1)

    # Safety
    ax = axes[1]
    ax.bar(['Success', 'Violations'], [df_safe['success'].mean(), df_safe['violated_safety'].mean()],
           color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Rate')
    ax.set_title('Safety Test')
    ax.set_ylim(0, 1)

    # Infinite horizon
    ax = axes[2]
    ax.bar(['Blue', 'Green', 'Both'], [df_inf['blue_visits'].mean(), df_inf['green_visits'].mean(),
           df_inf['alternating'].mean()], alpha=0.7)
    ax.set_ylabel('Average Visits')
    ax.set_title('Infinite Horizon Test')

    plt.tight_layout()
    plt.savefig(out_dir / 'capabilities_summary.png', dpi=150)
    plt.close()

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
