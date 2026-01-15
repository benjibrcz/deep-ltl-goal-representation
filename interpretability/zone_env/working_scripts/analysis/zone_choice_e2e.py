#!/usr/bin/env python3
"""
Zone choice end-to-end experiments.

Tests if agent makes "smart" choices between two reach zones when one is safer:
- Two zones of the same REACH color
- One AVOID zone positioned to block path to one reach zone
- Agent should choose the unblocked (safer) reach zone

This tests whether the agent has planning capability to anticipate obstacles.

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/zone_choice_e2e.py \
        --n_scenarios 50 --max_steps 200 --out_dir interpretability/zone_env/results/zone_choice_e2e
"""
import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

try:
    from gymnasium import spaces as gspaces
except Exception:
    from gym import spaces as gspaces

torch.set_grad_enabled(False)


# Color palette matching zone_env style
CMAP_RGB = {
    "blue": "#4C72B0",
    "green": "#55A868",
    "yellow": "#E1C027",
    "magenta": "#BB78A5",
}


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


def create_choice_scenario(
    reach_color: str,
    avoid_color: str,
    scenario_id: int,
    rng: random.Random,
) -> Dict:
    """
    Create a choice scenario:
    - Agent starts at center-ish position
    - SAFE reach zone in one direction (unobstructed)
    - BLOCKED reach zone in another direction
    - AVOID zone between agent and blocked reach zone
    """
    # Define positions (in normalized space -2 to 2)
    # Agent will start near origin

    # Pick two opposite-ish directions for the two reach zones
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    rng.shuffle(angles)
    safe_angle = angles[0]
    blocked_angle = angles[1]

    # Reach zones at distance ~1.5-2.0 from origin
    safe_dist = rng.uniform(1.5, 2.0)
    blocked_dist = rng.uniform(1.5, 2.0)

    safe_reach_pos = (
        safe_dist * np.cos(safe_angle),
        safe_dist * np.sin(safe_angle),
    )
    blocked_reach_pos = (
        blocked_dist * np.cos(blocked_angle),
        blocked_dist * np.sin(blocked_angle),
    )

    # Avoid zone is between agent (origin) and blocked reach zone
    # Place it at ~60-80% of the way to blocked zone
    avoid_fraction = rng.uniform(0.5, 0.7)
    avoid_pos = (
        blocked_reach_pos[0] * avoid_fraction + rng.uniform(-0.2, 0.2),
        blocked_reach_pos[1] * avoid_fraction + rng.uniform(-0.2, 0.2),
    )

    # Calculate if avoid truly blocks blocked zone
    # (distance from avoid to line between agent and blocked zone)
    blocked_blocked = True  # By construction

    return {
        'scenario_id': scenario_id,
        'reach_color': reach_color,
        'avoid_color': avoid_color,
        'safe_reach_pos': safe_reach_pos,
        'blocked_reach_pos': blocked_reach_pos,
        'avoid_pos': avoid_pos,
        'safe_angle_deg': np.degrees(safe_angle),
        'blocked_angle_deg': np.degrees(blocked_angle),
    }


def run_scenario(
    scenario: Dict,
    model,
    env_id: str,
    max_steps: int = 200,
    deterministic: bool = True,
) -> Dict:
    """
    Run a scenario using the standard environment.

    Since we can't control exact zone placement, we use the environment's
    random placement and analyze which zones the agent actually visits.
    """
    reach_color = scenario['reach_color']
    avoid_color = scenario['avoid_color']

    # Create environment
    formula = f"F {reach_color}"
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(env_id, sampler_fn, sequence=False)
    props = set(env.get_propositions())

    # Set up agent
    planner = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, planner, propositions=props)

    # Reset with scenario-specific seed
    seed = scenario['scenario_id'] + 2000
    reset_out = env.reset(seed=seed)
    obs, info = (reset_out, {}) if not isinstance(reset_out, tuple) else reset_out
    agent.reset()

    # Extract zone positions
    zone_positions = {}
    reach_zones = []
    avoid_zones = []
    try:
        task = env.unwrapped.task
        for geom_name, geom in task._geoms.items():
            if hasattr(geom, 'color_name') and hasattr(geom, 'num'):
                color = geom.color_name
                if color not in zone_positions:
                    zone_positions[color] = []
                for i in range(geom.num):
                    try:
                        body_name = f'{color}_zone{i}'
                        pos = task.data.body(body_name).xpos[:2].copy()
                        zone_positions[color].append(pos.tolist())
                        if color == reach_color:
                            reach_zones.append({'id': f'{color}_{i}', 'pos': pos.tolist()})
                        elif color == avoid_color:
                            avoid_zones.append({'id': f'{color}_{i}', 'pos': pos.tolist()})
                    except:
                        pass
    except:
        pass

    # Get initial agent position
    try:
        initial_pos = env.unwrapped.task.agent.pos[:2].copy()
    except:
        initial_pos = np.array([0.0, 0.0])

    # Compute distances to reach zones from initial position
    reach_distances = []
    for rz in reach_zones:
        d = np.linalg.norm(np.array(rz['pos']) - initial_pos)
        reach_distances.append({'zone': rz['id'], 'distance': d, 'pos': rz['pos']})
    reach_distances.sort(key=lambda x: x['distance'])

    # Check if avoid zone blocks any reach zone
    blocking_info = []
    for rz in reach_zones:
        reach_pos = np.array(rz['pos'])
        # Check each avoid zone
        for az in avoid_zones:
            avoid_pos = np.array(az['pos'])
            # Is avoid zone "between" agent and reach zone?
            # Check if avoid is closer than reach AND in similar direction
            d_reach = np.linalg.norm(reach_pos - initial_pos)
            d_avoid = np.linalg.norm(avoid_pos - initial_pos)

            # Vector analysis
            vec_to_reach = reach_pos - initial_pos
            vec_to_avoid = avoid_pos - initial_pos

            # Project avoid onto reach direction
            if d_reach > 0:
                proj = np.dot(vec_to_avoid, vec_to_reach) / np.dot(vec_to_reach, vec_to_reach)
                perp_dist = np.linalg.norm(vec_to_avoid - proj * vec_to_reach)

                # Avoid "blocks" if it's in the path (proj between 0 and 1) and close to line
                blocks = 0.2 < proj < 0.9 and perp_dist < 0.6
                if blocks:
                    blocking_info.append({
                        'reach_zone': rz['id'],
                        'avoid_zone': az['id'],
                        'proj': proj,
                        'perp_dist': perp_dist,
                    })

    # Run trajectory
    trajectory = []
    reached_zones = []
    avoided_zones = []

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

        # Check propositions
        current_props = info.get('propositions', set())

        trajectory.append({
            'step': step,
            'pos_x': float(agent_pos[0]),
            'pos_y': float(agent_pos[1]),
            'in_reach': reach_color in current_props,
            'in_avoid': avoid_color in current_props,
        })

        # Track which reach zone was reached first
        if reach_color in current_props and not reached_zones:
            # Find which reach zone
            for rz in reach_zones:
                d = np.linalg.norm(np.array(rz['pos']) - agent_pos)
                if d < 0.5:  # Zone radius
                    reached_zones.append({'zone': rz['id'], 'step': step})
                    break

        if avoid_color in current_props and not avoided_zones:
            for az in avoid_zones:
                d = np.linalg.norm(np.array(az['pos']) - agent_pos)
                if d < 0.5:
                    avoided_zones.append({'zone': az['id'], 'step': step})
                    break

        if done:
            break

    env.close()

    # Determine outcome
    touched_reach = len(reached_zones) > 0
    touched_avoid = len(avoided_zones) > 0

    if touched_reach and not touched_avoid:
        outcome = 'safe_success'
    elif touched_reach and touched_avoid:
        reach_step = reached_zones[0]['step'] if reached_zones else 999
        avoid_step = avoided_zones[0]['step'] if avoided_zones else 999
        outcome = 'safe_success' if reach_step < avoid_step else 'risky_success'
    elif touched_avoid and not touched_reach:
        outcome = 'fail'
    else:
        outcome = 'neither'

    # Did agent choose the closer/safer zone?
    chose_closer = False
    chose_blocked = False
    if reached_zones and len(reach_distances) >= 2:
        closest_zone = reach_distances[0]['zone']
        chosen_zone = reached_zones[0]['zone']
        chose_closer = chosen_zone == closest_zone

        # Check if chosen zone was blocked
        blocked_zones = [b['reach_zone'] for b in blocking_info]
        chose_blocked = chosen_zone in blocked_zones

    return {
        'scenario_id': scenario['scenario_id'],
        'reach_color': reach_color,
        'avoid_color': avoid_color,
        'outcome': outcome,
        'touched_reach': touched_reach,
        'touched_avoid': touched_avoid,
        'reach_zones': reach_zones,
        'avoid_zones': avoid_zones,
        'blocking_info': blocking_info,
        'n_blocked_zones': len(set(b['reach_zone'] for b in blocking_info)),
        'chose_closer': chose_closer,
        'chose_blocked': chose_blocked,
        'trajectory': trajectory,
        'zone_positions': zone_positions,
        'initial_pos': initial_pos.tolist(),
        'reach_distances': reach_distances,
    }


def plot_scenario(result: Dict, out_path: Path, zone_radius: float = 0.4):
    """Plot scenario with zone choice information."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Extract data
    traj = result['trajectory']
    xs = [t['pos_x'] for t in traj]
    ys = [t['pos_y'] for t in traj]

    # Plot all zones
    zone_pos = result.get('zone_positions', {})
    reach_color = result['reach_color']
    avoid_color = result['avoid_color']

    for color, positions in zone_pos.items():
        for pos in positions:
            facecolor = CMAP_RGB.get(color, 'gray')
            alpha = 0.4

            # Highlight reach and avoid zones
            if color == reach_color:
                edgecolor = 'green'
                linewidth = 3
                label = 'REACH'
            elif color == avoid_color:
                edgecolor = 'red'
                linewidth = 3
                label = 'AVOID'
            else:
                edgecolor = 'black'
                linewidth = 1
                label = ''

            circle = patches.Circle(
                pos, zone_radius,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
            )
            ax.add_patch(circle)

            if label:
                ax.annotate(label, xy=pos, ha='center', va='center',
                           fontsize=10, fontweight='bold', color=edgecolor)

    # Mark blocking zones
    for block in result.get('blocking_info', []):
        # Draw line from agent start to blocked reach zone through avoid
        pass  # Could add visual indicator

    # Plot trajectory
    ax.plot(xs, ys, 'k-', linewidth=1.5, alpha=0.7, label='Trajectory')
    ax.plot(xs[0], ys[0], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(xs[-1], ys[-1], 'rs', markersize=12, label='End', zorder=5)

    # Mark zone entries
    for t in traj:
        if t['in_reach']:
            ax.plot(t['pos_x'], t['pos_y'], 'g^', markersize=8, alpha=0.8)
        if t['in_avoid']:
            ax.plot(t['pos_x'], t['pos_y'], 'rv', markersize=8, alpha=0.8)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    outcome = result['outcome']
    n_blocked = result.get('n_blocked_zones', 0)
    chose_blocked = result.get('chose_blocked', False)
    title = f"Scenario {result['scenario_id']}: reach={reach_color}, avoid={avoid_color}\n"
    title += f"Outcome: {outcome}, Blocked zones: {n_blocked}, Chose blocked: {chose_blocked}"
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_scenarios', type=int, default=50)
    ap.add_argument('--max_steps', type=int, default=200)
    ap.add_argument('--deterministic', action='store_true', default=True)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/zone_choice_e2e')
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

    # Generate scenarios - use different color pairs
    print(f"\nGenerating {args.n_scenarios} scenarios...")
    scenarios = []
    rng = random.Random(args.seed)
    color_pairs = [
        ('blue', 'green'),
        ('green', 'blue'),
        ('yellow', 'magenta'),
        ('magenta', 'yellow'),
    ]

    for i in range(args.n_scenarios):
        reach, avoid = color_pairs[i % len(color_pairs)]
        scenario = create_choice_scenario(reach, avoid, i, rng)
        scenarios.append(scenario)

    # Run scenarios
    print(f"\nRunning {len(scenarios)} scenarios...")
    results = []
    for i, scenario in enumerate(scenarios):
        result = run_scenario(scenario, model, args.env_id, args.max_steps, args.deterministic)
        results.append(result)

        # Save plot for some scenarios
        if i < 20 or result['outcome'] != 'safe_success':
            plot_path = plots_dir / f"scenario_{i}_{result['outcome']}.png"
            try:
                plot_scenario(result, plot_path)
            except Exception as e:
                print(f"  Failed to plot {i}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(scenarios)} completed")

    # Save results
    with open(out_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Create summary
    summary_rows = []
    for r in results:
        summary_rows.append({
            'scenario_id': r['scenario_id'],
            'reach_color': r['reach_color'],
            'avoid_color': r['avoid_color'],
            'outcome': r['outcome'],
            'touched_reach': r['touched_reach'],
            'touched_avoid': r['touched_avoid'],
            'n_blocked_zones': r['n_blocked_zones'],
            'chose_closer': r['chose_closer'],
            'chose_blocked': r['chose_blocked'],
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / 'summary.csv', index=False)

    # Print statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    outcome_counts = df['outcome'].value_counts()
    total = len(df)

    print(f"\nOutcome distribution (n={total}):")
    for outcome in ['safe_success', 'risky_success', 'fail', 'neither']:
        count = outcome_counts.get(outcome, 0)
        pct = 100 * count / total
        print(f"  {outcome:15s}: {count:3d} ({pct:5.1f}%)")

    # Blocking analysis
    has_blocking = df[df['n_blocked_zones'] > 0]
    no_blocking = df[df['n_blocked_zones'] == 0]

    print(f"\n--- BLOCKING ANALYSIS ---")
    print(f"Scenarios with blocking: {len(has_blocking)}")
    print(f"Scenarios without blocking: {len(no_blocking)}")

    if len(has_blocking) > 0:
        safe_with_block = (has_blocking['outcome'] == 'safe_success').sum()
        print(f"\nWhen avoid blocks a reach zone (n={len(has_blocking)}):")
        print(f"  Safe success: {100*safe_with_block/len(has_blocking):.1f}%")

        chose_blocked = has_blocking['chose_blocked'].sum()
        print(f"  Chose blocked zone: {100*chose_blocked/len(has_blocking):.1f}%")

    if len(no_blocking) > 0:
        safe_no_block = (no_blocking['outcome'] == 'safe_success').sum()
        print(f"\nWhen no blocking (n={len(no_blocking)}):")
        print(f"  Safe success: {100*safe_no_block/len(no_blocking):.1f}%")

    # Closer zone analysis
    chose_closer_total = df['chose_closer'].sum()
    print(f"\n--- DISTANCE PREFERENCE ---")
    print(f"Chose closer zone: {chose_closer_total}/{total} ({100*chose_closer_total/total:.1f}%)")

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
