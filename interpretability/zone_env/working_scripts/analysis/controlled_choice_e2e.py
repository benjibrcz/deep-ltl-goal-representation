#!/usr/bin/env python3
"""
Controlled zone choice experiments with rigorous scenario criteria.

Creates scenarios where:
1. Two REACH zones are at EQUAL distance from agent start
2. One REACH zone has an AVOID zone blocking the direct path
3. The other REACH zone has a clear path

This provides a clean test of planning: distance-based heuristics give 50/50,
but a planning agent should consistently choose the unblocked zone.

Strategy:
- Run many random environment resets
- Filter for scenarios meeting our criteria
- Analyze agent choice behavior

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/controlled_choice_e2e.py \
        --n_target 50 --max_attempts 500 --out_dir interpretability/zone_env/results/controlled_choice
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
import matplotlib.patches as patches

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
from visualize.zones import draw_zones, draw_diamond, draw_path, setup_axis, _color_map

try:
    from gymnasium import spaces as gspaces
except Exception:
    from gym import spaces as gspaces

torch.set_grad_enabled(False)

# Color palette
CMAP_RGB = {
    "blue": "#4C72B0",
    "green": "#55A868",
    "yellow": "#E1C027",
    "magenta": "#BB78A5",
}

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


def extract_zone_info(env) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
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


def check_path_blocked(agent_pos: np.ndarray, reach_pos: np.ndarray,
                       avoid_positions: List[np.ndarray],
                       zone_radius: float = ZONE_RADIUS,
                       blocking_threshold: float = 0.6) -> bool:
    """
    Check if any avoid zone blocks the direct path from agent to reach zone.

    A zone "blocks" if it's close to the line segment between agent and reach,
    and is between them (not behind agent or beyond reach).
    """
    if not avoid_positions:
        return False

    vec_to_reach = reach_pos - agent_pos
    dist_to_reach = np.linalg.norm(vec_to_reach)

    if dist_to_reach < 0.1:
        return False

    unit_vec = vec_to_reach / dist_to_reach

    for avoid_pos in avoid_positions:
        vec_to_avoid = avoid_pos - agent_pos

        # Project avoid onto the line to reach
        proj_length = np.dot(vec_to_avoid, unit_vec)

        # Check if avoid is between agent and reach (with margin)
        if proj_length < zone_radius or proj_length > dist_to_reach - zone_radius:
            continue

        # Perpendicular distance from avoid to the line
        proj_point = agent_pos + proj_length * unit_vec
        perp_dist = np.linalg.norm(avoid_pos - proj_point)

        # Blocks if perpendicular distance is less than threshold
        if perp_dist < blocking_threshold:
            return True

    return False


def check_zone_overlap(pos1: np.ndarray, pos2: np.ndarray,
                       min_separation: float = ZONE_RADIUS * 2) -> bool:
    """Check if two zones overlap or are too close."""
    return np.linalg.norm(pos1 - pos2) < min_separation


def find_controlled_scenario(
    env,
    reach_color: str,
    avoid_color: str,
    seed: int,
    distance_tolerance: float = 0.3,
    blocking_threshold: float = 0.6,
    min_reach_separation: float = 1.0,  # Min distance between the two reach zones
    min_agent_zone_dist: float = 0.6,   # Agent shouldn't start too close to any zone
) -> Optional[Dict]:
    """
    Check if a given seed produces a valid controlled scenario.

    Returns scenario dict if valid, None otherwise.

    Valid scenario requires:
    1. At least 2 reach zones
    2. At least 1 avoid zone
    3. Two reach zones within distance_tolerance of each other from agent
    4. One reach zone blocked by avoid, one not blocked
    5. No zone overlaps between reach and avoid zones
    6. Reach zones are sufficiently separated from each other
    7. Agent doesn't start too close to any zone
    """
    reset_out = env.reset(seed=seed)
    agent_pos, zone_positions = extract_zone_info(env)

    if agent_pos is None:
        return None

    reach_zones = zone_positions.get(reach_color, [])
    avoid_zones = zone_positions.get(avoid_color, [])

    # Check agent isn't starting too close to any zone
    all_zones = reach_zones + avoid_zones
    for zpos in all_zones:
        if np.linalg.norm(agent_pos - zpos) < min_agent_zone_dist:
            return None

    if len(reach_zones) < 2 or len(avoid_zones) < 1:
        return None

    # Check for overlaps between reach and avoid zones
    for rpos in reach_zones:
        for apos in avoid_zones:
            if check_zone_overlap(rpos, apos):
                return None  # Reject scenarios with overlapping zones

    # Calculate distances to all reach zones
    reach_info = []
    for i, rpos in enumerate(reach_zones):
        dist = np.linalg.norm(rpos - agent_pos)
        blocked = check_path_blocked(agent_pos, rpos, avoid_zones,
                                     blocking_threshold=blocking_threshold)
        reach_info.append({
            'id': i,
            'pos': rpos,
            'distance': dist,
            'blocked': blocked,
        })

    # Find pairs of reach zones with similar distance
    # where one is blocked and one is not
    valid_pairs = []
    for i in range(len(reach_info)):
        for j in range(i+1, len(reach_info)):
            ri, rj = reach_info[i], reach_info[j]

            # Check distance similarity
            dist_diff = abs(ri['distance'] - rj['distance'])
            if dist_diff > distance_tolerance:
                continue

            # Check one blocked, one not
            if ri['blocked'] == rj['blocked']:
                continue

            # Check reach zones are sufficiently separated
            reach_separation = np.linalg.norm(ri['pos'] - rj['pos'])
            if reach_separation < min_reach_separation:
                continue

            # Valid pair found!
            blocked_zone = ri if ri['blocked'] else rj
            safe_zone = rj if ri['blocked'] else ri

            valid_pairs.append({
                'blocked': blocked_zone,
                'safe': safe_zone,
                'distance_diff': dist_diff,
                'avg_distance': (ri['distance'] + rj['distance']) / 2,
            })

    if not valid_pairs:
        return None

    # Pick the best pair (smallest distance difference)
    best_pair = min(valid_pairs, key=lambda p: p['distance_diff'])

    return {
        'seed': seed,
        'reach_color': reach_color,
        'avoid_color': avoid_color,
        'agent_pos': agent_pos.tolist(),
        'blocked_zone': {
            'id': best_pair['blocked']['id'],
            'pos': best_pair['blocked']['pos'].tolist(),
            'distance': best_pair['blocked']['distance'],
        },
        'safe_zone': {
            'id': best_pair['safe']['id'],
            'pos': best_pair['safe']['pos'].tolist(),
            'distance': best_pair['safe']['distance'],
        },
        'distance_diff': best_pair['distance_diff'],
        'avg_distance': best_pair['avg_distance'],
        'all_reach_zones': [(r['id'], r['pos'].tolist(), r['distance'], r['blocked'])
                           for r in reach_info],
        'avoid_zones': [az.tolist() for az in avoid_zones],
        'all_zone_positions': {k: [v.tolist() for v in vs]
                               for k, vs in zone_positions.items()},
    }


def run_controlled_scenario(
    scenario: Dict,
    model,
    env_id: str,
    max_steps: int = 240,  # Increased from 200 to give agent more time
    deterministic: bool = True,
) -> Dict:
    """Run a controlled scenario and track which zone the agent chooses."""
    reach_color = scenario['reach_color']
    avoid_color = scenario['avoid_color']
    seed = scenario['seed']

    # Create environment
    formula = f"F {reach_color}"
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(env_id, sampler_fn, sequence=False)
    props = set(env.get_propositions())

    # Set up agent
    planner = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, planner, propositions=props)

    # Reset with the same seed that created the scenario
    reset_out = env.reset(seed=seed)
    obs, info = (reset_out, {}) if not isinstance(reset_out, tuple) else reset_out
    agent.reset()

    # Get zone positions (should match scenario)
    agent_pos_init, zone_positions = extract_zone_info(env)

    blocked_zone_pos = np.array(scenario['blocked_zone']['pos'])
    safe_zone_pos = np.array(scenario['safe_zone']['pos'])
    avoid_zones = [np.array(az) for az in scenario['avoid_zones']]

    # Run trajectory
    trajectory = []
    reached_blocked = False
    reached_safe = False
    touched_avoid = False
    blocked_step = -1
    safe_step = -1
    avoid_step = -1

    for step in range(max_steps):
        # Get action
        with torch.no_grad():
            action = agent.get_action(obs, info, deterministic=deterministic)
        action = coerce_action(action, env.action_space)

        # Get agent position
        try:
            agent_pos = env.unwrapped.task.agent.pos[:2].copy()
        except:
            agent_pos = np.array([np.nan, np.nan])

        # Step environment
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret

        # Check zone entries
        dist_to_blocked = np.linalg.norm(agent_pos - blocked_zone_pos)
        dist_to_safe = np.linalg.norm(agent_pos - safe_zone_pos)
        in_blocked = dist_to_blocked < ZONE_RADIUS
        in_safe = dist_to_safe < ZONE_RADIUS

        # Check avoid zones
        in_avoid = any(np.linalg.norm(agent_pos - az) < ZONE_RADIUS for az in avoid_zones)

        trajectory.append({
            'step': step,
            'pos_x': float(agent_pos[0]),
            'pos_y': float(agent_pos[1]),
            'dist_to_blocked': float(dist_to_blocked),
            'dist_to_safe': float(dist_to_safe),
            'in_blocked': in_blocked,
            'in_safe': in_safe,
            'in_avoid': in_avoid,
        })

        if in_blocked and not reached_blocked:
            reached_blocked = True
            blocked_step = step

        if in_safe and not reached_safe:
            reached_safe = True
            safe_step = step

        if in_avoid and not touched_avoid:
            touched_avoid = True
            avoid_step = step

        if done:
            break

    env.close()

    # Determine outcome and choice
    if reached_safe and not reached_blocked:
        choice = 'safe'
        outcome = 'safe_success' if not touched_avoid else 'risky_success'
    elif reached_blocked and not reached_safe:
        choice = 'blocked'
        outcome = 'risky_success' if touched_avoid else 'unsafe_success'
    elif reached_safe and reached_blocked:
        # Both reached - which was first?
        if safe_step < blocked_step:
            choice = 'safe'
            outcome = 'safe_success' if not touched_avoid else 'risky_success'
        else:
            choice = 'blocked'
            outcome = 'risky_success' if touched_avoid else 'unsafe_success'
    else:
        choice = 'neither'
        outcome = 'fail' if touched_avoid else 'neither'

    return {
        'seed': seed,
        'reach_color': reach_color,
        'avoid_color': avoid_color,
        'blocked_zone_pos': scenario['blocked_zone']['pos'],
        'safe_zone_pos': scenario['safe_zone']['pos'],
        'distance_diff': scenario['distance_diff'],
        'avg_distance': scenario['avg_distance'],
        'choice': choice,
        'outcome': outcome,
        'reached_blocked': reached_blocked,
        'reached_safe': reached_safe,
        'touched_avoid': touched_avoid,
        'blocked_step': blocked_step,
        'safe_step': safe_step,
        'avoid_step': avoid_step,
        'trajectory': trajectory,
        'all_zone_positions': scenario['all_zone_positions'],
        'agent_start': scenario['agent_pos'],
    }


def plot_controlled_scenario(result: Dict, scenario: Dict, out_path: Path):
    """Plot controlled scenario using visualize/zones.py style."""
    fig, ax = plt.subplots(figsize=(10, 10))
    setup_axis(ax)

    # Extract trajectory data
    traj = result['trajectory']
    path = [(t['pos_x'], t['pos_y']) for t in traj]

    reach_color = result['reach_color']
    avoid_color = result['avoid_color']

    # Convert zone positions to format expected by draw_zones
    zone_pos = result['all_zone_positions']
    zone_positions_dict = {}
    for color, positions in zone_pos.items():
        for i, pos in enumerate(positions):
            zone_positions_dict[f'{color}_zone{i}'] = tuple(pos)

    # Draw all zones
    draw_zones(ax, zone_positions_dict)

    # Draw agent start as orange diamond
    agent_start = result['agent_start']
    draw_diamond(ax, agent_start, color='orange')

    # Draw trajectory as green path
    draw_path(ax, path, color='darkgreen', linewidth=3)

    # Add red squares at key points (every 20 steps + end)
    for i in range(20, len(path), 20):
        ax.plot(path[i][0], path[i][1], 's', color='red', markersize=6, zorder=5)
    ax.plot(path[-1][0], path[-1][1], 's', color='red', markersize=8, zorder=5)

    # Add labels for SAFE, BLOCKED, and AVOID zones
    blocked_pos = np.array(result['blocked_zone_pos'])
    safe_pos = np.array(result['safe_zone_pos'])

    # Label the key zones with small text
    ax.annotate('SAFE', xy=safe_pos, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.7))
    ax.annotate('BLOCKED', xy=blocked_pos, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.7))

    # Label avoid zones
    avoid_zones = scenario.get('avoid_zones', [])
    for avoid_pos in avoid_zones:
        ax.annotate('AVOID', xy=avoid_pos, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))

    # Add title
    choice = result['choice']
    outcome = result['outcome']
    dist_diff = result['distance_diff']
    title = f"Seed {result['seed']}: Choice={choice.upper()}, Outcome={outcome}"
    ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_target', type=int, default=50,
                    help='Target number of valid controlled scenarios')
    ap.add_argument('--max_attempts', type=int, default=1000,
                    help='Max seeds to try to find valid scenarios')
    ap.add_argument('--max_steps', type=int, default=240)
    ap.add_argument('--distance_tolerance', type=float, default=0.3,
                    help='Max allowed distance difference between reach zones')
    ap.add_argument('--blocking_threshold', type=float, default=0.6,
                    help='Max perpendicular distance for blocking')
    ap.add_argument('--deterministic', action='store_true', default=True)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/controlled_choice')
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

    # Color pairs to try
    color_pairs = [
        ('blue', 'green'),
        ('green', 'blue'),
        ('yellow', 'magenta'),
        ('magenta', 'yellow'),
        ('blue', 'yellow'),
        ('green', 'magenta'),
    ]

    # Find valid controlled scenarios
    print(f"\nSearching for {args.n_target} controlled scenarios...")
    print(f"  Distance tolerance: {args.distance_tolerance}")
    print(f"  Blocking threshold: {args.blocking_threshold}")

    scenarios = []
    attempts = 0

    # Create one environment per color pair (reuse for efficiency)
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

        scenario = find_controlled_scenario(
            env, reach_color, avoid_color, seed,
            distance_tolerance=args.distance_tolerance,
            blocking_threshold=args.blocking_threshold,
        )

        attempts += 1

        if scenario is not None:
            scenarios.append(scenario)
            if len(scenarios) % 5 == 0:
                print(f"  Found {len(scenarios)}/{args.n_target} valid scenarios "
                      f"(tried {attempts} seeds)")

    # Close all environments
    for env in envs_by_color.values():
        env.close()

    print(f"\nFound {len(scenarios)} valid scenarios from {attempts} attempts")
    print(f"Success rate: {100*len(scenarios)/attempts:.1f}%")

    if len(scenarios) == 0:
        print("No valid scenarios found! Try adjusting tolerances.")
        return

    # Save scenarios
    with open(out_dir / 'scenarios.json', 'w') as f:
        json.dump(scenarios, f, indent=2)

    # Run scenarios
    print(f"\nRunning {len(scenarios)} controlled scenarios...")
    results = []

    for i, scenario in enumerate(scenarios):
        result = run_controlled_scenario(
            scenario, model, args.env_id,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
        )
        results.append(result)

        # Save plot
        plot_path = plots_dir / f"scenario_{scenario['seed']}_{result['choice']}.png"
        try:
            plot_controlled_scenario(result, scenario, plot_path)
        except Exception as e:
            print(f"  Failed to plot {scenario['seed']}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(scenarios)} completed")

    # Save results
    with open(out_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Create summary
    summary_rows = []
    for r in results:
        summary_rows.append({
            'seed': r['seed'],
            'reach_color': r['reach_color'],
            'avoid_color': r['avoid_color'],
            'choice': r['choice'],
            'outcome': r['outcome'],
            'distance_diff': r['distance_diff'],
            'avg_distance': r['avg_distance'],
            'reached_blocked': r['reached_blocked'],
            'reached_safe': r['reached_safe'],
            'touched_avoid': r['touched_avoid'],
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / 'summary.csv', index=False)

    # Print detailed statistics
    print("\n" + "="*70)
    print("CONTROLLED CHOICE EXPERIMENT RESULTS")
    print("="*70)

    total = len(df)

    print(f"\n{'='*50}")
    print("SCENARIO CHARACTERISTICS")
    print(f"{'='*50}")
    print(f"Total controlled scenarios: {total}")
    print(f"Mean distance difference: {df['distance_diff'].mean():.3f} "
          f"(max: {df['distance_diff'].max():.3f})")
    print(f"Mean avg distance to zones: {df['avg_distance'].mean():.2f}")

    print(f"\n{'='*50}")
    print("AGENT CHOICE (THE KEY RESULT)")
    print(f"{'='*50}")

    choice_counts = df['choice'].value_counts()
    for choice in ['safe', 'blocked', 'neither']:
        count = choice_counts.get(choice, 0)
        pct = 100 * count / total
        print(f"  Chose {choice:10s}: {count:3d} ({pct:5.1f}%)")

    # Key metric: of scenarios where agent reached a zone, did it choose safe?
    reached_any = df[df['choice'].isin(['safe', 'blocked'])]
    if len(reached_any) > 0:
        chose_safe = (reached_any['choice'] == 'safe').sum()
        print(f"\n  >>> Of {len(reached_any)} where agent reached a zone:")
        print(f"  >>> Chose SAFE zone: {chose_safe}/{len(reached_any)} "
              f"({100*chose_safe/len(reached_any):.1f}%)")
        print(f"  >>> Chose BLOCKED zone: {len(reached_any)-chose_safe}/{len(reached_any)} "
              f"({100*(len(reached_any)-chose_safe)/len(reached_any):.1f}%)")

    print(f"\n{'='*50}")
    print("OUTCOME DISTRIBUTION")
    print(f"{'='*50}")

    outcome_counts = df['outcome'].value_counts()
    for outcome in ['safe_success', 'risky_success', 'unsafe_success', 'fail', 'neither']:
        count = outcome_counts.get(outcome, 0)
        pct = 100 * count / total
        if count > 0:
            print(f"  {outcome:15s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n{'='*50}")
    print("AVOID ZONE CONTACT")
    print(f"{'='*50}")

    touched = df['touched_avoid'].sum()
    print(f"  Touched avoid zone: {touched}/{total} ({100*touched/total:.1f}%)")

    # Breakdown by choice
    for choice in ['safe', 'blocked']:
        subset = df[df['choice'] == choice]
        if len(subset) > 0:
            touched_subset = subset['touched_avoid'].sum()
            print(f"    When chose {choice}: {touched_subset}/{len(subset)} "
                  f"({100*touched_subset/len(subset):.1f}%) touched avoid")

    print(f"\n{'='*50}")
    print("INTERPRETATION")
    print(f"{'='*50}")

    if len(reached_any) > 0:
        safe_pct = 100 * chose_safe / len(reached_any)
        if safe_pct > 80:
            print("  STRONG evidence of planning: Agent consistently chooses safe path")
        elif safe_pct > 60:
            print("  MODERATE evidence of planning: Agent prefers safe path")
        elif safe_pct > 40:
            print("  WEAK/NO evidence: Near 50/50 suggests distance-based heuristic")
        else:
            print("  NEGATIVE evidence: Agent prefers blocked path (unexpected)")

    print(f"\nResults saved to: {out_dir}")
    print("  - scenarios.json: scenario definitions")
    print("  - results.pkl: full results with trajectories")
    print("  - summary.csv: summary statistics")
    print("  - plots/: trajectory visualizations")


if __name__ == '__main__':
    main()
