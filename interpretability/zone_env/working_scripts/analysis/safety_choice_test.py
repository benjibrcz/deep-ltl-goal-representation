#!/usr/bin/env python3
"""
Safety Choice Test: Does the agent choose safe goals over blocked ones?

From DeepLTL paper Figure 1(c):
- Task: (F green ∨ F yellow) ∧ G ¬blue - reach green OR yellow, while ALWAYS avoiding blue
- One goal is blocked by blue zones (dangerous path)
- Other goal is clear (safe path)
- Optimal: Choose the safe/unblocked goal

This tests whether the agent anticipates obstacles when making goal choices.

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/safety_choice_test.py
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


def is_path_blocked(start: np.ndarray, goal: np.ndarray, obstacles: List[np.ndarray],
                    radius: float = ZONE_RADIUS) -> bool:
    """Check if direct path from start to goal crosses any obstacle."""
    if not obstacles:
        return False

    direction = goal - start
    path_length = np.linalg.norm(direction)
    if path_length < 0.01:
        return False

    direction = direction / path_length

    # Check multiple points along path
    for t in np.linspace(0, 1, 20):
        point = start + t * (goal - start)
        for obs in obstacles:
            if np.linalg.norm(point - obs) < radius * 1.5:  # Some margin
                return True
    return False


def run_safety_choice_scenario(
    env_id: str,
    model,
    goal1_color: str,
    goal2_color: str,
    avoid_color: str,
    seed: int,
    max_steps: int = 300,
) -> Optional[Dict]:
    """Run a scenario testing if agent chooses safe goal over blocked one."""

    # (F goal1 | F goal2) & G !avoid - reach either goal while always avoiding
    formula = f"(F {goal1_color} | F {goal2_color}) & G !{avoid_color}"
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

    goal1_zones = zone_positions.get(goal1_color, [])
    goal2_zones = zone_positions.get(goal2_color, [])
    avoid_zones = zone_positions.get(avoid_color, [])

    if not goal1_zones or not goal2_zones or not avoid_zones:
        env.close()
        return None

    # Find nearest goal of each color
    dist_to_goal1 = min(np.linalg.norm(agent_start - g) for g in goal1_zones)
    dist_to_goal2 = min(np.linalg.norm(agent_start - g) for g in goal2_zones)
    nearest_goal1 = goal1_zones[np.argmin([np.linalg.norm(agent_start - g) for g in goal1_zones])]
    nearest_goal2 = goal2_zones[np.argmin([np.linalg.norm(agent_start - g) for g in goal2_zones])]

    # Check if paths are blocked
    goal1_blocked = is_path_blocked(agent_start, nearest_goal1, avoid_zones)
    goal2_blocked = is_path_blocked(agent_start, nearest_goal2, avoid_zones)

    # We want scenarios where exactly one goal is blocked
    if goal1_blocked == goal2_blocked:
        env.close()
        return None

    safe_color = goal2_color if goal1_blocked else goal1_color
    blocked_color = goal1_color if goal1_blocked else goal2_color
    safe_dist = dist_to_goal2 if goal1_blocked else dist_to_goal1
    blocked_dist = dist_to_goal1 if goal1_blocked else dist_to_goal2

    # Run the rollout
    trajectory = []
    reached_goal1 = False
    reached_goal2 = False
    touched_avoid = False

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
        for g in goal1_zones:
            if np.linalg.norm(agent_pos - g) < ZONE_RADIUS:
                reached_goal1 = True
        for g in goal2_zones:
            if np.linalg.norm(agent_pos - g) < ZONE_RADIUS:
                reached_goal2 = True
        for a in avoid_zones:
            if np.linalg.norm(agent_pos - a) < ZONE_RADIUS:
                touched_avoid = True

        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret

        if done:
            break

    env.close()

    # Determine outcome
    if goal1_blocked:
        chose_safe = reached_goal2 and not reached_goal1
        chose_blocked = reached_goal1
    else:
        chose_safe = reached_goal1 and not reached_goal2
        chose_blocked = reached_goal2

    if chose_safe and not touched_avoid:
        outcome = 'safe_success'
    elif chose_blocked and not touched_avoid:
        outcome = 'blocked_success_clean'  # Got through blocked path safely (lucky?)
    elif chose_blocked and touched_avoid:
        outcome = 'blocked_failed'  # Tried blocked path and hit avoid
    elif touched_avoid:
        outcome = 'failed_safety'
    else:
        outcome = 'neither'

    return {
        'seed': seed,
        'formula': formula,
        'goal1_color': goal1_color,
        'goal2_color': goal2_color,
        'avoid_color': avoid_color,
        'agent_start': agent_start.tolist(),
        'goal1_zones': [g.tolist() for g in goal1_zones],
        'goal2_zones': [g.tolist() for g in goal2_zones],
        'avoid_zones': [a.tolist() for a in avoid_zones],
        'goal1_blocked': goal1_blocked,
        'goal2_blocked': goal2_blocked,
        'safe_color': safe_color,
        'blocked_color': blocked_color,
        'safe_dist': safe_dist,
        'blocked_dist': blocked_dist,
        'reached_goal1': reached_goal1,
        'reached_goal2': reached_goal2,
        'touched_avoid': touched_avoid,
        'chose_safe': chose_safe,
        'chose_blocked': chose_blocked,
        'outcome': outcome,
        'trajectory': trajectory,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_scenarios', type=int, default=100)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/safety_choice')
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

    # Color combinations
    color_combos = [
        ('green', 'yellow', 'blue'),
        ('blue', 'magenta', 'yellow'),
        ('yellow', 'green', 'magenta'),
        ('magenta', 'blue', 'green'),
    ]

    print(f"\nSearching for safety choice scenarios (one goal blocked, one clear)...")
    results = []
    attempts = 0

    while len(results) < args.n_scenarios and attempts < args.n_scenarios * 30:
        seed = args.seed + attempts * 100
        goal1, goal2, avoid = color_combos[attempts % len(color_combos)]

        result = run_safety_choice_scenario(
            args.env_id, model, goal1, goal2, avoid, seed, args.max_steps
        )

        if result is not None:
            results.append(result)
            if len(results) % 10 == 0:
                print(f"  Found {len(results)} valid scenarios (tried {attempts})")

        attempts += 1

    print(f"\nCollected {len(results)} valid safety choice scenarios")

    if not results:
        print("No valid scenarios found!")
        return

    # Save results
    with open(out_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Analyze results
    df = pd.DataFrame([{
        'seed': r['seed'],
        'safe_color': r['safe_color'],
        'blocked_color': r['blocked_color'],
        'safe_dist': r['safe_dist'],
        'blocked_dist': r['blocked_dist'],
        'chose_safe': r['chose_safe'],
        'chose_blocked': r['chose_blocked'],
        'touched_avoid': r['touched_avoid'],
        'outcome': r['outcome'],
    } for r in results])

    df.to_csv(out_dir / 'summary.csv', index=False)

    # Print statistics
    print("\n" + "="*70)
    print("SAFETY CHOICE TEST RESULTS")
    print("="*70)

    total = len(df)
    safe_count = df['chose_safe'].sum()
    blocked_count = df['chose_blocked'].sum()
    neither_count = total - safe_count - blocked_count

    print(f"\n  Total scenarios: {total}")
    print(f"  (All scenarios have exactly one blocked path)")
    print(f"\n  Agent chose:")
    print(f"    SAFE goal:    {safe_count} ({100*safe_count/total:.1f}%)")
    print(f"    BLOCKED goal: {blocked_count} ({100*blocked_count/total:.1f}%)")
    print(f"    Neither:      {neither_count} ({100*neither_count/total:.1f}%)")

    # Check if distance was a factor
    safe_closer = (df['safe_dist'] < df['blocked_dist']).sum()
    blocked_closer = (df['safe_dist'] > df['blocked_dist']).sum()

    print(f"\n  Distance analysis:")
    print(f"    Safe goal was closer:    {safe_closer} scenarios")
    print(f"    Blocked goal was closer: {blocked_closer} scenarios")

    # When blocked goal is closer, what does agent do?
    blocked_closer_df = df[df['safe_dist'] > df['blocked_dist']]
    if len(blocked_closer_df) > 0:
        chose_safe_when_blocked_closer = blocked_closer_df['chose_safe'].sum()
        total_blocked_closer = len(blocked_closer_df)
        print(f"\n  When BLOCKED goal is closer ({total_blocked_closer} scenarios):")
        print(f"    Agent chose SAFE anyway: {chose_safe_when_blocked_closer} ({100*chose_safe_when_blocked_closer/total_blocked_closer:.1f}%)")

    # Outcome breakdown
    print(f"\n  Outcomes:")
    for outcome in df['outcome'].unique():
        count = (df['outcome'] == outcome).sum()
        print(f"    {outcome}: {count} ({100*count/total:.1f}%)")

    print(f"\n{'='*50}")
    print("INTERPRETATION")
    print(f"{'='*50}")

    if safe_count > blocked_count * 1.5:
        print(f"\n  >>> AGENT SHOWS SAFETY AWARENESS")
        print(f"  >>> Chose safe path {100*safe_count/total:.1f}% of the time")
        print(f"  >>> This suggests planning/anticipation of obstacles")
    elif blocked_count > safe_count:
        print(f"\n  >>> AGENT IS NOT SAFETY-AWARE")
        print(f"  >>> Chose blocked path {100*blocked_count/total:.1f}% of the time")
        print(f"  >>> No evidence of obstacle anticipation")
    else:
        print(f"\n  >>> INCONCLUSIVE")
        print(f"  >>> Agent split between safe and blocked")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Choice distribution
    ax = axes[0]
    choices = ['Safe', 'Blocked', 'Neither']
    counts = [safe_count, blocked_count, neither_count]
    colors = ['green', 'red', 'gray']
    ax.bar(choices, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Safety Choice Test: Agent Goal Selection')
    ax.grid(True, alpha=0.3, axis='y')
    for i, count in enumerate(counts):
        ax.text(i, count + 1, f'{100*count/total:.1f}%', ha='center', fontsize=10)

    # Plot 2: Outcome distribution
    ax = axes[1]
    outcomes = df['outcome'].value_counts()
    outcome_colors = {
        'safe_success': 'green',
        'blocked_success_clean': 'yellow',
        'blocked_failed': 'red',
        'failed_safety': 'darkred',
        'neither': 'gray',
    }
    bars = ax.bar(range(len(outcomes)), outcomes.values, alpha=0.7,
                  color=[outcome_colors.get(o, 'blue') for o in outcomes.index])
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels(outcomes.index, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Detailed Outcomes')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / 'safety_results.png', dpi=150)
    plt.close()

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
