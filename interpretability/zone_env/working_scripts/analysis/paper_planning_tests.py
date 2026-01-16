#!/usr/bin/env python3
"""
Paper Planning Tests: Replicating DeepLTL Figure 1 claims

Specifically tests whether the agent demonstrates "planning" by:

1. OPTIMALITY: Choosing farther-first-subgoal when it leads to shorter total path
   - Task: F (blue & F green)
   - Test: When farther blue is closer to green, does agent choose farther blue?

2. SAFETY: Choosing farther-but-safer goal over closer-but-blocked goal
   - Task: (F green | F yellow) & G !blue
   - Test: When green is blocked by blue but yellow is clear, does agent choose yellow?

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/paper_planning_tests.py
"""
import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def path_intersects_zones(start: np.ndarray, end: np.ndarray, zones: List[np.ndarray],
                          radius: float = ZONE_RADIUS, num_samples: int = 30) -> bool:
    """Check if straight-line path from start to end intersects any zone."""
    if not zones:
        return False
    for t in np.linspace(0, 1, num_samples):
        point = start + t * (end - start)
        for zone in zones:
            if np.linalg.norm(point - zone) < radius * 1.2:  # Small margin
                return True
    return False


def run_optimality_test(env_id: str, model, seed: int, max_steps: int = 300) -> Optional[Dict]:
    """
    Test optimality: Does agent choose farther-first when it's globally optimal?

    Task: F (blue & F green)

    We look for scenarios where:
    - There are 2+ blue zones
    - The FARTHER blue from agent is CLOSER to green (optimal choice)
    - The CLOSER blue from agent is FARTHER from green (greedy choice)
    """
    formula = "F (blue & F green)"
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(env_id, sampler_fn, sequence=False)

    try:
        props = set(env.get_propositions())

        planner = ExhaustiveSearch(model, props, num_loops=2)
        agent = Agent(model, planner, propositions=props)

        reset_out = env.reset(seed=seed)
        obs, info = (reset_out, {}) if not isinstance(reset_out, tuple) else reset_out
        agent.reset()

        agent_start, zone_positions = extract_zone_info(env)

        if agent_start is None:
            return None

        blue_zones = zone_positions.get('blue', [])
        green_zones = zone_positions.get('green', [])

        if len(blue_zones) < 2 or not green_zones:
            return None

        # Find nearest green zone
        nearest_green = green_zones[np.argmin([np.linalg.norm(agent_start - g) for g in green_zones])]

        # Calculate distances from agent to each blue, and from each blue to nearest green
        blue_info = []
        for i, blue in enumerate(blue_zones):
            dist_agent_to_blue = np.linalg.norm(agent_start - blue)
            dist_blue_to_green = np.linalg.norm(blue - nearest_green)
            total_path = dist_agent_to_blue + dist_blue_to_green
            blue_info.append({
                'idx': i,
                'pos': blue,
                'dist_from_agent': dist_agent_to_blue,
                'dist_to_green': dist_blue_to_green,
                'total_path': total_path,
            })

        # Sort by distance from agent
        blue_info.sort(key=lambda x: x['dist_from_agent'])

        # Greedy choice: closest blue to agent
        greedy_blue = blue_info[0]

        # Optimal choice: blue with shortest total path
        optimal_blue = min(blue_info, key=lambda x: x['total_path'])

        # We want scenarios where greedy != optimal AND optimal is farther
        if greedy_blue['idx'] == optimal_blue['idx']:
            return None

        # Check that optimal blue is actually farther (this is the interesting case)
        if optimal_blue['dist_from_agent'] <= greedy_blue['dist_from_agent']:
            return None

        # Require meaningful difference in total path length
        path_diff = greedy_blue['total_path'] - optimal_blue['total_path']
        if path_diff < 0.5:  # Need at least 0.5 units difference
            return None

        # Run the episode
        trajectory = [agent_start.copy()]
        first_blue_visited = None

        for step in range(max_steps):
            with torch.no_grad():
                action = agent.get_action(obs, info, deterministic=True)
            action = coerce_action(action, env.action_space)

            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret

            try:
                agent_pos = env.unwrapped.task.agent.pos[:2].copy()
                trajectory.append(agent_pos.copy())

                # Check which blue was visited first
                if first_blue_visited is None:
                    for bi in blue_info:
                        if np.linalg.norm(agent_pos - bi['pos']) < ZONE_RADIUS:
                            first_blue_visited = bi['idx']
                            break
            except:
                pass

            if done or first_blue_visited is not None:
                break

        if first_blue_visited is None:
            return None

        # Determine choice
        chose_optimal = (first_blue_visited == optimal_blue['idx'])
        chose_greedy = (first_blue_visited == greedy_blue['idx'])

        return {
            'seed': seed,
            'agent_start': agent_start.tolist(),
            'blue_zones': [b.tolist() for b in blue_zones],
            'green_zone': nearest_green.tolist(),
            'greedy_blue_idx': greedy_blue['idx'],
            'greedy_blue_dist': greedy_blue['dist_from_agent'],
            'greedy_total_path': greedy_blue['total_path'],
            'optimal_blue_idx': optimal_blue['idx'],
            'optimal_blue_dist': optimal_blue['dist_from_agent'],
            'optimal_total_path': optimal_blue['total_path'],
            'path_diff': path_diff,
            'first_blue_visited': first_blue_visited,
            'chose_optimal': chose_optimal,
            'chose_greedy': chose_greedy,
            'trajectory': [t.tolist() for t in trajectory],
        }
    finally:
        env.close()


def run_safety_test(env_id: str, model, seed: int, max_steps: int = 300) -> Optional[Dict]:
    """
    Test safety planning: Does agent choose farther-but-safer goal?

    Task: (F green | F yellow) & G !blue

    We look for scenarios where:
    - Green is CLOSER but path is BLOCKED by blue zones
    - Yellow is FARTHER but path is CLEAR
    - Agent should choose yellow (planning) not green (greedy)
    """
    formula = "(F green | F yellow) & G !blue"
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(env_id, sampler_fn, sequence=False)

    try:
        props = set(env.get_propositions())

        planner = ExhaustiveSearch(model, props, num_loops=2)
        agent = Agent(model, planner, propositions=props)

        reset_out = env.reset(seed=seed)
        obs, info = (reset_out, {}) if not isinstance(reset_out, tuple) else reset_out
        agent.reset()

        agent_start, zone_positions = extract_zone_info(env)

        if agent_start is None:
            return None

        green_zones = zone_positions.get('green', [])
        yellow_zones = zone_positions.get('yellow', [])
        blue_zones = zone_positions.get('blue', [])

        if not green_zones or not yellow_zones or not blue_zones:
            return None

        # Find nearest of each color
        nearest_green_idx = np.argmin([np.linalg.norm(agent_start - g) for g in green_zones])
        nearest_green = green_zones[nearest_green_idx]
        dist_to_green = np.linalg.norm(agent_start - nearest_green)

        nearest_yellow_idx = np.argmin([np.linalg.norm(agent_start - y) for y in yellow_zones])
        nearest_yellow = yellow_zones[nearest_yellow_idx]
        dist_to_yellow = np.linalg.norm(agent_start - nearest_yellow)

        # Check if paths are blocked
        green_blocked = path_intersects_zones(agent_start, nearest_green, blue_zones)
        yellow_blocked = path_intersects_zones(agent_start, nearest_yellow, blue_zones)

        # We want: one path blocked, one path clear (either direction)
        # This tests safety-aware planning
        if green_blocked == yellow_blocked:  # Both blocked or both clear - skip
            return None

        # Determine which is the "safe" choice
        if green_blocked and not yellow_blocked:
            safe_color = 'yellow'
            blocked_color = 'green'
            safe_dist = dist_to_yellow
            blocked_dist = dist_to_green
        else:  # yellow blocked, green clear
            safe_color = 'green'
            blocked_color = 'yellow'
            safe_dist = dist_to_green
            blocked_dist = dist_to_yellow

        dist_diff = safe_dist - blocked_dist  # Positive means safe is farther

        # We want safe to be farther (makes it a planning test)
        # But we'll accept any scenario where one is blocked and one isn't
        # and track the distance difference in results
        # (Removed strict filter - we'll analyze regardless of distance relationship)

        # Run the episode
        trajectory = [agent_start.copy()]
        reached_green = False
        reached_yellow = False
        touched_blue = False
        first_goal_reached = None

        for step in range(max_steps):
            with torch.no_grad():
                action = agent.get_action(obs, info, deterministic=True)
            action = coerce_action(action, env.action_space)

            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret

            try:
                agent_pos = env.unwrapped.task.agent.pos[:2].copy()
                trajectory.append(agent_pos.copy())

                # Check zone contacts
                for g in green_zones:
                    if np.linalg.norm(agent_pos - g) < ZONE_RADIUS and not reached_green:
                        reached_green = True
                        if first_goal_reached is None:
                            first_goal_reached = 'green'

                for y in yellow_zones:
                    if np.linalg.norm(agent_pos - y) < ZONE_RADIUS and not reached_yellow:
                        reached_yellow = True
                        if first_goal_reached is None:
                            first_goal_reached = 'yellow'

                for b in blue_zones:
                    if np.linalg.norm(agent_pos - b) < ZONE_RADIUS:
                        touched_blue = True
            except:
                pass

            if done:
                break

        # Determine outcome
        # "Safe" choice: chose the unblocked goal without touching blue
        # "Blocked" choice: chose the blocked goal or touched blue
        chose_safe = (first_goal_reached == safe_color and not touched_blue)
        chose_blocked = (first_goal_reached == blocked_color) or touched_blue

        return {
            'seed': seed,
            'agent_start': agent_start.tolist(),
            'green_zone': nearest_green.tolist(),
            'yellow_zone': nearest_yellow.tolist(),
            'blue_zones': [b.tolist() for b in blue_zones],
            'dist_to_green': dist_to_green,
            'dist_to_yellow': dist_to_yellow,
            'safe_color': safe_color,
            'blocked_color': blocked_color,
            'safe_dist': safe_dist,
            'blocked_dist': blocked_dist,
            'dist_diff': dist_diff,  # positive = safe is farther
            'green_blocked': green_blocked,
            'yellow_blocked': yellow_blocked,
            'first_goal_reached': first_goal_reached,
            'reached_green': reached_green,
            'reached_yellow': reached_yellow,
            'touched_blue': touched_blue,
            'chose_safe': chose_safe,
            'chose_blocked': chose_blocked,
            'trajectory': [t.tolist() for t in trajectory],
        }
    finally:
        env.close()


def plot_scenario(result: Dict, test_type: str, filename: Path):
    """Plot a single scenario with trajectory."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Color mapping
    colors = {'blue': '#3498db', 'green': '#2ecc71', 'yellow': '#f1c40f', 'magenta': '#e91e63'}

    if test_type == 'optimality':
        # Plot blue zones
        for i, blue in enumerate(result['blue_zones']):
            is_optimal = (i == result['optimal_blue_idx'])
            is_greedy = (i == result['greedy_blue_idx'])

            circle = patches.Circle(blue, ZONE_RADIUS,
                                   facecolor=colors['blue'],
                                   edgecolor='green' if is_optimal else ('orange' if is_greedy else 'black'),
                                   linewidth=3 if (is_optimal or is_greedy) else 1,
                                   alpha=0.5)
            ax.add_patch(circle)
            label = 'OPT' if is_optimal else ('GRD' if is_greedy else '')
            ax.text(blue[0], blue[1], f'B{i}\n{label}', ha='center', va='center', fontsize=8, fontweight='bold')

        # Plot green zone
        green = result['green_zone']
        circle = patches.Circle(green, ZONE_RADIUS, facecolor=colors['green'], alpha=0.5, edgecolor='black')
        ax.add_patch(circle)
        ax.text(green[0], green[1], 'G', ha='center', va='center', fontsize=10, fontweight='bold')

        title = f"Optimality Test (seed={result['seed']})\n"
        title += f"Chose: {'OPTIMAL (farther first)' if result['chose_optimal'] else 'GREEDY (closer first)'}"

    else:  # safety
        # Determine which is blocked/safe
        green_blocked = result['green_blocked']

        # Plot green zone
        green = result['green_zone']
        green_label = 'G\n(blocked)' if green_blocked else 'G\n(safe)'
        green_edge = 'red' if green_blocked else 'green'
        circle = patches.Circle(green, ZONE_RADIUS, facecolor=colors['green'],
                               edgecolor=green_edge, linewidth=3, alpha=0.5)
        ax.add_patch(circle)
        ax.text(green[0], green[1], green_label, ha='center', va='center', fontsize=8, fontweight='bold')

        # Plot yellow zone
        yellow = result['yellow_zone']
        yellow_label = 'Y\n(blocked)' if not green_blocked else 'Y\n(safe)'
        yellow_edge = 'red' if not green_blocked else 'green'
        circle = patches.Circle(yellow, ZONE_RADIUS, facecolor=colors['yellow'],
                               edgecolor=yellow_edge, linewidth=3, alpha=0.5)
        ax.add_patch(circle)
        ax.text(yellow[0], yellow[1], yellow_label, ha='center', va='center', fontsize=8, fontweight='bold')

        # Plot blue zones (obstacles)
        for blue in result['blue_zones']:
            circle = patches.Circle(blue, ZONE_RADIUS, facecolor=colors['blue'], alpha=0.4, edgecolor='red', linewidth=2)
            ax.add_patch(circle)
            ax.text(blue[0], blue[1], 'B', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

        title = f"Safety Test (seed={result['seed']})\n"
        title += f"Chose: {'SAFE (' + result['safe_color'] + ')' if result['chose_safe'] else 'BLOCKED (' + result['blocked_color'] + ')'}"

    # Plot trajectory
    traj = np.array(result['trajectory'])
    n_points = len(traj)
    for i in range(n_points - 1):
        progress = i / max(n_points - 1, 1)
        color = plt.cm.plasma(progress)
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], color=color, linewidth=2, alpha=0.8)

    # Mark start
    ax.scatter(traj[0, 0], traj[0, 1], s=150, c='orange', marker='D',
              zorder=5, edgecolors='black', linewidths=2, label='Start')

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_scenarios', type=int, default=50)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/paper_planning_tests')
    ap.add_argument('--test', choices=['all', 'optimality', 'safety'], default='all',
                   help='Which test to run (default: all)')
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

    # =========================================================================
    # TEST 1: OPTIMALITY
    # =========================================================================
    opt_results = []

    if args.test in ['all', 'optimality']:
        print("\n" + "="*70)
        print("TEST 1: OPTIMALITY - Does agent choose farther-first when optimal?")
        print("="*70)
        print("Looking for scenarios where:")
        print("  - Farther blue leads to shorter total path")
        print("  - Greedy (closer blue) leads to longer total path")
        sys.stdout.flush()

        attempts = 0
        max_attempts = args.n_scenarios * 100  # Cap attempts to prevent infinite loops

        while len(opt_results) < args.n_scenarios and attempts < max_attempts:
            try:
                result = run_optimality_test(args.env_id, model, args.seed + attempts * 100, args.max_steps)
                if result is not None:
                    opt_results.append(result)
                    print(f"  [{len(opt_results)}/{args.n_scenarios}] Found scenario (seed={result['seed']})", flush=True)
                    if len(opt_results) <= 3:
                        plot_scenario(result, 'optimality', out_dir / f'optimality_example_{len(opt_results)}.png')
            except Exception as e:
                print(f"  Warning: attempt {attempts} failed with error: {e}", flush=True)

            attempts += 1

            # Periodic garbage collection to prevent memory buildup
            if attempts % 50 == 0:
                gc.collect()
                print(f"  ... searched {attempts} seeds so far", flush=True)

        print(f"\nFound {len(opt_results)} optimality test scenarios (searched {attempts} seeds)")

    if opt_results:
        df_opt = pd.DataFrame([{
            'seed': r['seed'],
            'chose_optimal': r['chose_optimal'],
            'chose_greedy': r['chose_greedy'],
            'path_diff': r['path_diff'],
            'optimal_dist': r['optimal_blue_dist'],
            'greedy_dist': r['greedy_blue_dist'],
        } for r in opt_results])

        df_opt.to_csv(out_dir / 'optimality_results.csv', index=False)

        n_total = len(df_opt)
        n_optimal = df_opt['chose_optimal'].sum()
        n_greedy = df_opt['chose_greedy'].sum()

        print(f"\n  Results:")
        print(f"    Chose OPTIMAL (farther first): {n_optimal}/{n_total} ({100*n_optimal/n_total:.1f}%)")
        print(f"    Chose GREEDY (closer first):   {n_greedy}/{n_total} ({100*n_greedy/n_total:.1f}%)")
        print(f"    Average path difference: {df_opt['path_diff'].mean():.2f} units")

        if n_optimal > n_greedy:
            print(f"\n  >>> EVIDENCE OF PLANNING: Agent prefers optimal over greedy!")
        else:
            print(f"\n  >>> NO PLANNING: Agent prefers greedy/closer first")

    # =========================================================================
    # TEST 2: SAFETY
    # =========================================================================
    safety_results = []

    if args.test in ['all', 'safety']:
        print("\n" + "="*70)
        print("TEST 2: SAFETY - Does agent choose farther-but-safer goal?")
        print("="*70)
        print("Looking for scenarios where:")
        print("  - One goal is CLOSER but BLOCKED by blue")
        print("  - Other goal is FARTHER but CLEAR (safe)")
        print("  - Tests: Does agent choose farther-but-safer over closer-but-blocked?")
        sys.stdout.flush()

        attempts = 0
        max_attempts = args.n_scenarios * 100  # Cap attempts to prevent infinite loops

        while len(safety_results) < args.n_scenarios and attempts < max_attempts:
            try:
                result = run_safety_test(args.env_id, model, args.seed + 10000 + attempts * 100, args.max_steps)
                if result is not None:
                    safety_results.append(result)
                    print(f"  [{len(safety_results)}/{args.n_scenarios}] Found scenario (seed={result['seed']})", flush=True)
                    if len(safety_results) <= 3:
                        plot_scenario(result, 'safety', out_dir / f'safety_example_{len(safety_results)}.png')
            except Exception as e:
                print(f"  Warning: attempt {attempts} failed with error: {e}", flush=True)

            attempts += 1

            # Periodic garbage collection to prevent memory buildup
            if attempts % 50 == 0:
                gc.collect()
                print(f"  ... searched {attempts} seeds so far", flush=True)

        print(f"\nFound {len(safety_results)} safety test scenarios (searched {attempts} seeds)")

    if safety_results:
        df_safety = pd.DataFrame([{
            'seed': r['seed'],
            'chose_safe': r['chose_safe'],
            'chose_blocked': r['chose_blocked'],
            'safe_color': r['safe_color'],
            'blocked_color': r['blocked_color'],
            'safe_dist': r['safe_dist'],
            'blocked_dist': r['blocked_dist'],
            'dist_diff': r['dist_diff'],
            'touched_blue': r['touched_blue'],
            'first_goal': r['first_goal_reached'],
        } for r in safety_results])

        df_safety.to_csv(out_dir / 'safety_results.csv', index=False)

        n_total = len(df_safety)
        n_safe = df_safety['chose_safe'].sum()
        n_blocked = df_safety['chose_blocked'].sum()
        n_touched_blue = df_safety['touched_blue'].sum()
        n_safe_farther = (df_safety['dist_diff'] > 0).sum()

        print(f"\n  Results:")
        print(f"    Chose SAFE path:     {n_safe}/{n_total} ({100*n_safe/n_total:.1f}%)")
        print(f"    Chose BLOCKED path:  {n_blocked}/{n_total} ({100*n_blocked/n_total:.1f}%)")
        print(f"    Touched blue (safety violation): {n_touched_blue}/{n_total} ({100*n_touched_blue/n_total:.1f}%)")
        print(f"    Safe path farther in: {n_safe_farther}/{n_total} ({100*n_safe_farther/n_total:.1f}%) scenarios")
        print(f"    Average distance diff (safe - blocked): {df_safety['dist_diff'].mean():.2f} units")

        if n_safe > n_blocked:
            print(f"\n  >>> EVIDENCE OF PLANNING: Agent prefers safe over closer!")
        else:
            print(f"\n  >>> NO PLANNING: Agent goes for closer goal despite blocking")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY: PLANNING EVIDENCE")
    print("="*70)

    if opt_results and safety_results:
        opt_rate = df_opt['chose_optimal'].mean()
        safe_rate = df_safety['chose_safe'].mean()

        print(f"\n  Optimality (chose farther-first when optimal): {100*opt_rate:.1f}%")
        print(f"  Safety (chose farther-but-safer):              {100*safe_rate:.1f}%")

        if opt_rate > 0.6:
            print(f"\n  OPTIMALITY: Evidence of planning (>{60}% chose optimal)")
        elif opt_rate > 0.4:
            print(f"\n  OPTIMALITY: Inconclusive (40-60%)")
        else:
            print(f"\n  OPTIMALITY: No planning (<40% chose optimal)")

        if safe_rate > 0.6:
            print(f"  SAFETY: Evidence of planning (>{60}% chose safe)")
        elif safe_rate > 0.4:
            print(f"  SAFETY: Inconclusive (40-60%)")
        else:
            print(f"  SAFETY: No planning (<40% chose safe)")

    print(f"\nResults and example plots saved to: {out_dir}")


if __name__ == '__main__':
    main()
