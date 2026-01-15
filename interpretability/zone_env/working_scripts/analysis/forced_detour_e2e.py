#!/usr/bin/env python3
"""
Forced Detour Experiment: Testing for Transition Function Evidence

This experiment creates scenarios where:
1. The DIRECT path to the goal is blocked by avoid zones
2. A DETOUR path exists that requires initially moving away from or sideways to the goal
3. A distance-based heuristic would fail; a transition function would succeed

We also test different LTL formula types:
- Simple reach: F blue
- Reach-avoid: !yellow U blue
- Sequencing: F (blue & F green) - reach blue, then green

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/forced_detour_e2e.py
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
    """Coerce action to match the action space format."""
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
    """Extract agent position and zone positions from environment."""
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
    except Exception as e:
        pass

    return agent_pos, zone_positions


def compute_blocking_score(agent_pos: np.ndarray, goal_pos: np.ndarray,
                           avoid_positions: List[np.ndarray]) -> Tuple[float, int]:
    """
    Compute how severely the direct path is blocked.

    Returns:
        - blocking_score: sum of how much each avoid zone blocks the path
        - num_blocking: number of avoid zones that block the path
    """
    if not avoid_positions:
        return 0.0, 0

    vec_to_goal = goal_pos - agent_pos
    dist_to_goal = np.linalg.norm(vec_to_goal)

    if dist_to_goal < 0.1:
        return 0.0, 0

    unit_vec = vec_to_goal / dist_to_goal

    blocking_score = 0.0
    num_blocking = 0

    for avoid_pos in avoid_positions:
        vec_to_avoid = avoid_pos - agent_pos
        proj_length = np.dot(vec_to_avoid, unit_vec)

        # Check if avoid is between agent and goal
        if proj_length < ZONE_RADIUS or proj_length > dist_to_goal - ZONE_RADIUS:
            continue

        # Perpendicular distance from avoid to the line
        proj_point = agent_pos + proj_length * unit_vec
        perp_dist = np.linalg.norm(avoid_pos - proj_point)

        # Score inversely proportional to perpendicular distance
        # Higher score = more blocking
        if perp_dist < ZONE_RADIUS * 3:  # Within 3 radii of path
            blocking_score += max(0, ZONE_RADIUS * 3 - perp_dist)
            if perp_dist < ZONE_RADIUS * 1.5:  # Significantly blocking
                num_blocking += 1

    return blocking_score, num_blocking


def check_detour_exists(agent_pos: np.ndarray, goal_pos: np.ndarray,
                        avoid_positions: List[np.ndarray],
                        num_angles: int = 16) -> bool:
    """
    Check if there's likely a detour path around the obstacles.

    Simple heuristic: check if there are clear angles to move initially
    that don't immediately hit avoid zones.
    """
    clear_angles = 0

    for i in range(num_angles):
        angle = 2 * np.pi * i / num_angles
        direction = np.array([np.cos(angle), np.sin(angle)])
        test_pos = agent_pos + direction * ZONE_RADIUS * 2

        # Check if this direction is clear of avoid zones
        is_clear = True
        for avoid_pos in avoid_positions:
            if np.linalg.norm(test_pos - avoid_pos) < ZONE_RADIUS * 1.5:
                is_clear = False
                break

        if is_clear:
            clear_angles += 1

    # Need at least some clear directions
    return clear_angles >= 4


def find_forced_detour_scenario(
    env,
    reach_color: str,
    avoid_color: str,
    seed: int,
    min_blocking_zones: int = 2,
    min_blocking_score: float = 0.5,
) -> Optional[Dict]:
    """
    Find scenarios where direct path is blocked but detour exists.
    """
    reset_out = env.reset(seed=seed)
    agent_pos, zone_positions = extract_zone_info(env)

    if agent_pos is None:
        return None

    reach_zones = zone_positions.get(reach_color, [])
    avoid_zones = zone_positions.get(avoid_color, [])

    if len(reach_zones) < 1 or len(avoid_zones) < 2:
        return None

    # Find the closest reach zone
    reach_dists = [np.linalg.norm(rz - agent_pos) for rz in reach_zones]
    closest_idx = np.argmin(reach_dists)
    goal_pos = reach_zones[closest_idx]
    goal_dist = reach_dists[closest_idx]

    # Check blocking score
    blocking_score, num_blocking = compute_blocking_score(agent_pos, goal_pos, avoid_zones)

    if num_blocking < min_blocking_zones or blocking_score < min_blocking_score:
        return None

    # Check that a detour likely exists
    if not check_detour_exists(agent_pos, goal_pos, avoid_zones):
        return None

    # Check agent isn't starting too close to any zone
    for zpos in reach_zones + avoid_zones:
        if np.linalg.norm(agent_pos - zpos) < ZONE_RADIUS * 1.5:
            return None

    return {
        'seed': seed,
        'reach_color': reach_color,
        'avoid_color': avoid_color,
        'agent_pos': agent_pos.tolist(),
        'goal_pos': goal_pos.tolist(),
        'goal_dist': float(goal_dist),
        'blocking_score': float(blocking_score),
        'num_blocking': num_blocking,
        'avoid_zones': [az.tolist() for az in avoid_zones],
        'all_zone_positions': {k: [v.tolist() for v in vs]
                               for k, vs in zone_positions.items()},
    }


def run_scenario(
    scenario: Dict,
    model,
    env_id: str,
    formula: str,
    max_steps: int = 240,
    deterministic: bool = True,
) -> Dict:
    """Run a scenario with a given formula."""
    seed = scenario['seed']
    reach_color = scenario['reach_color']
    avoid_color = scenario['avoid_color']

    # Create environment with the formula
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(env_id, sampler_fn, sequence=False)
    props = set(env.get_propositions())

    # Set up agent
    planner = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, planner, propositions=props)

    # Reset
    reset_out = env.reset(seed=seed)
    obs, info = (reset_out, {}) if not isinstance(reset_out, tuple) else reset_out
    agent.reset()

    goal_pos = np.array(scenario['goal_pos'])
    avoid_zones = [np.array(az) for az in scenario['avoid_zones']]

    # Run trajectory
    trajectory = []
    reached_goal = False
    touched_avoid = False
    goal_step = -1
    avoid_step = -1
    min_dist_to_goal = float('inf')
    max_dist_from_start = 0

    agent_start = np.array(scenario['agent_pos'])

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

        dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
        dist_from_start = np.linalg.norm(agent_pos - agent_start)
        in_goal = dist_to_goal < ZONE_RADIUS
        in_avoid = any(np.linalg.norm(agent_pos - az) < ZONE_RADIUS for az in avoid_zones)

        min_dist_to_goal = min(min_dist_to_goal, dist_to_goal)
        max_dist_from_start = max(max_dist_from_start, dist_from_start)

        trajectory.append({
            'step': step,
            'pos_x': float(agent_pos[0]),
            'pos_y': float(agent_pos[1]),
            'dist_to_goal': float(dist_to_goal),
            'dist_from_start': float(dist_from_start),
            'in_goal': in_goal,
            'in_avoid': in_avoid,
        })

        if in_goal and not reached_goal:
            reached_goal = True
            goal_step = step

        if in_avoid and not touched_avoid:
            touched_avoid = True
            avoid_step = step

        if done:
            break

    env.close()

    # Determine outcome
    if reached_goal and not touched_avoid:
        outcome = 'safe_success'
    elif reached_goal and touched_avoid:
        outcome = 'risky_success'
    elif touched_avoid:
        outcome = 'fail'
    else:
        outcome = 'neither'

    # Check if agent took a detour (moved away from goal initially)
    if len(trajectory) > 10:
        early_dists = [t['dist_to_goal'] for t in trajectory[:10]]
        took_detour = max(early_dists) > scenario['goal_dist'] * 1.1  # Moved 10% further away
    else:
        took_detour = False

    return {
        'seed': seed,
        'formula': formula,
        'reach_color': reach_color,
        'avoid_color': avoid_color,
        'outcome': outcome,
        'reached_goal': reached_goal,
        'touched_avoid': touched_avoid,
        'goal_step': goal_step,
        'avoid_step': avoid_step,
        'min_dist_to_goal': float(min_dist_to_goal),
        'max_dist_from_start': float(max_dist_from_start),
        'took_detour': took_detour,
        'blocking_score': scenario['blocking_score'],
        'num_blocking': scenario['num_blocking'],
        'trajectory': trajectory,
        'all_zone_positions': scenario['all_zone_positions'],
        'agent_start': scenario['agent_pos'],
        'goal_pos': scenario['goal_pos'],
        'avoid_zones': scenario['avoid_zones'],
    }


def plot_scenario(result: Dict, out_path: Path):
    """Plot scenario using visualize/zones.py style."""
    fig, ax = plt.subplots(figsize=(10, 10))
    setup_axis(ax)

    traj = result['trajectory']
    path = [(t['pos_x'], t['pos_y']) for t in traj]

    # Convert zone positions
    zone_pos = result['all_zone_positions']
    zone_positions_dict = {}
    for color, positions in zone_pos.items():
        for i, pos in enumerate(positions):
            zone_positions_dict[f'{color}_zone{i}'] = tuple(pos)

    draw_zones(ax, zone_positions_dict)

    agent_start = result['agent_start']
    draw_diamond(ax, agent_start, color='orange')

    # Draw trajectory
    draw_path(ax, path, color='darkgreen', linewidth=3)

    # Red squares at intervals
    for i in range(20, len(path), 20):
        ax.plot(path[i][0], path[i][1], 's', color='red', markersize=6, zorder=5)
    ax.plot(path[-1][0], path[-1][1], 's', color='red', markersize=8, zorder=5)

    # Label goal and blocking avoid zones
    goal_pos = result['goal_pos']
    ax.annotate('GOAL', xy=goal_pos, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.8))

    for avoid_pos in result['avoid_zones']:
        ax.annotate('AVOID', xy=avoid_pos, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))

    # Draw direct path line (dashed) to show blocking
    ax.plot([agent_start[0], goal_pos[0]], [agent_start[1], goal_pos[1]],
            'r--', linewidth=2, alpha=0.4, zorder=1)

    outcome = result['outcome']
    formula = result['formula']
    detour = "YES" if result['took_detour'] else "NO"
    title = f"Seed {result['seed']}: {outcome}\n"
    title += f"Formula: {formula}\n"
    title += f"Took detour: {detour}, Blocking zones: {result['num_blocking']}"
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_target', type=int, default=30)
    ap.add_argument('--max_attempts', type=int, default=2000)
    ap.add_argument('--max_steps', type=int, default=240)
    ap.add_argument('--min_blocking_zones', type=int, default=2)
    ap.add_argument('--min_blocking_score', type=float, default=0.5)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/forced_detour')
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

    # Color pairs
    color_pairs = [
        ('blue', 'green'),
        ('green', 'blue'),
        ('yellow', 'magenta'),
        ('magenta', 'yellow'),
    ]

    # Find forced detour scenarios
    print(f"\nSearching for {args.n_target} forced detour scenarios...")
    print(f"  Min blocking zones: {args.min_blocking_zones}")
    print(f"  Min blocking score: {args.min_blocking_score}")

    scenarios = []

    envs_by_color = {}
    for reach_color, avoid_color in color_pairs:
        formula = f"F {reach_color}"
        sampler_fn = FixedSampler.partial(formula)
        envs_by_color[(reach_color, avoid_color)] = make_env(args.env_id, sampler_fn, sequence=False)

    for seed_offset in range(args.max_attempts):
        if len(scenarios) >= args.n_target:
            break

        seed = args.seed + seed_offset * 100
        reach_color, avoid_color = color_pairs[seed_offset % len(color_pairs)]
        env = envs_by_color[(reach_color, avoid_color)]

        scenario = find_forced_detour_scenario(
            env, reach_color, avoid_color, seed,
            min_blocking_zones=args.min_blocking_zones,
            min_blocking_score=args.min_blocking_score,
        )

        if scenario is not None:
            scenarios.append(scenario)
            if len(scenarios) % 5 == 0:
                print(f"  Found {len(scenarios)}/{args.n_target}")

    for env in envs_by_color.values():
        env.close()

    print(f"\nFound {len(scenarios)} forced detour scenarios")

    if len(scenarios) == 0:
        print("No valid scenarios found!")
        return

    # Save scenarios
    with open(out_dir / 'scenarios.json', 'w') as f:
        json.dump(scenarios, f, indent=2)

    # Define formulas to test
    formulas_to_test = []
    for scenario in scenarios:
        reach = scenario['reach_color']
        avoid = scenario['avoid_color']
        formulas_to_test.append({
            'scenario': scenario,
            'formulas': [
                ('simple_reach', f'F {reach}'),
                ('reach_avoid', f'!{avoid} U {reach}'),
            ]
        })

    # Run experiments
    print(f"\nRunning {len(scenarios)} scenarios with multiple formulas...")
    all_results = []

    for i, item in enumerate(formulas_to_test):
        scenario = item['scenario']
        for formula_name, formula in item['formulas']:
            result = run_scenario(
                scenario, model, args.env_id, formula,
                max_steps=args.max_steps,
            )
            result['formula_name'] = formula_name
            all_results.append(result)

            # Save plot
            plot_path = plots_dir / f"scenario_{scenario['seed']}_{formula_name}_{result['outcome']}.png"
            try:
                plot_scenario(result, plot_path)
            except Exception as e:
                print(f"  Plot failed: {e}")

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(scenarios)} scenarios completed")

    # Save results
    with open(out_dir / 'results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Create summary
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'seed': r['seed'],
            'formula_name': r['formula_name'],
            'formula': r['formula'],
            'outcome': r['outcome'],
            'reached_goal': r['reached_goal'],
            'touched_avoid': r['touched_avoid'],
            'took_detour': r['took_detour'],
            'blocking_score': r['blocking_score'],
            'num_blocking': r['num_blocking'],
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / 'summary.csv', index=False)

    # Print statistics
    print("\n" + "="*70)
    print("FORCED DETOUR EXPERIMENT RESULTS")
    print("="*70)

    for formula_name in df['formula_name'].unique():
        subset = df[df['formula_name'] == formula_name]
        total = len(subset)

        print(f"\n{'='*50}")
        print(f"Formula: {formula_name}")
        print(f"{'='*50}")

        outcomes = subset['outcome'].value_counts()
        for outcome in ['safe_success', 'risky_success', 'fail', 'neither']:
            count = outcomes.get(outcome, 0)
            pct = 100 * count / total
            print(f"  {outcome:15s}: {count:3d} ({pct:5.1f}%)")

        detours = subset['took_detour'].sum()
        print(f"\n  Took detour: {detours}/{total} ({100*detours/total:.1f}%)")

        # Success rate by blocking severity
        high_blocking = subset[subset['num_blocking'] >= 3]
        if len(high_blocking) > 0:
            high_success = (high_blocking['outcome'].isin(['safe_success', 'risky_success'])).sum()
            print(f"  Success with 3+ blocking zones: {high_success}/{len(high_blocking)} "
                  f"({100*high_success/len(high_blocking):.1f}%)")

    print(f"\n{'='*50}")
    print("DETOUR ANALYSIS")
    print(f"{'='*50}")

    # Compare detour rates between formulas
    for formula_name in df['formula_name'].unique():
        subset = df[df['formula_name'] == formula_name]
        successful = subset[subset['outcome'].isin(['safe_success', 'risky_success'])]
        if len(successful) > 0:
            detour_in_success = successful['took_detour'].sum()
            print(f"  {formula_name}: {detour_in_success}/{len(successful)} successful runs took detour "
                  f"({100*detour_in_success/len(successful):.1f}%)")

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
