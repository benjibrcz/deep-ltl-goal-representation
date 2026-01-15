#!/usr/bin/env python3
"""
Value Anticipation Experiment

Tests whether the agent's value function anticipates obstacles BEFORE contact,
or only reacts AFTER contact. This reveals whether the agent has a predictive
world model.

Key question: Does value drop when APPROACHING an obstacle, or only when IN it?

Anticipation would suggest:
- Agent predicts future states
- Has some form of world model
- Plans ahead

No anticipation suggests:
- Purely reactive
- No prediction of future states

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/value_anticipation.py
"""
import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
import preprocessing

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


def run_value_tracking_rollout(
    env_id: str,
    model,
    reach_color: str,
    avoid_color: str,
    seed: int,
    max_steps: int = 240,
) -> Dict:
    """Run a rollout tracking value estimates and distances."""

    formula = f"!{avoid_color} U {reach_color}"
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

    reach_zones = zone_positions.get(reach_color, [])
    avoid_zones = zone_positions.get(avoid_color, [])

    if not reach_zones or not avoid_zones:
        env.close()
        return None

    # Track data
    steps_data = []

    for step in range(max_steps):
        # Get action from agent (this sets up obs['goal'])
        with torch.no_grad():
            action = agent.get_action(obs, info, deterministic=True)

            # Now obs has 'goal' set by the agent, we can get value
            try:
                preprocessed = preprocessing.preprocess_obss([obs], props)
                _, value_tensor = model(preprocessed)
                value = value_tensor.item()
            except Exception as e:
                if step == 0:
                    print(f"  Value extraction error: {e}")
                value = None

        action = coerce_action(action, env.action_space)

        # Get current position
        try:
            agent_pos = env.unwrapped.task.agent.pos[:2].copy()
            agent_vel = env.unwrapped.task.agent.vel[:2].copy()
        except:
            agent_pos = np.array([np.nan, np.nan])
            agent_vel = np.array([np.nan, np.nan])

        # Calculate distances
        dist_to_nearest_reach = min(np.linalg.norm(agent_pos - rz) for rz in reach_zones)
        dist_to_nearest_avoid = min(np.linalg.norm(agent_pos - az) for az in avoid_zones)

        in_reach = dist_to_nearest_reach < ZONE_RADIUS
        in_avoid = dist_to_nearest_avoid < ZONE_RADIUS

        # Step environment
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret

        steps_data.append({
            'step': step,
            'pos_x': float(agent_pos[0]),
            'pos_y': float(agent_pos[1]),
            'vel_x': float(agent_vel[0]),
            'vel_y': float(agent_vel[1]),
            'speed': float(np.linalg.norm(agent_vel)),
            'dist_to_reach': float(dist_to_nearest_reach),
            'dist_to_avoid': float(dist_to_nearest_avoid),
            'in_reach': in_reach,
            'in_avoid': in_avoid,
            'value': value,
            'reward': float(rew),
        })

        if done:
            break

    env.close()

    # Determine outcome
    reached_goal = any(s['in_reach'] for s in steps_data)
    touched_avoid = any(s['in_avoid'] for s in steps_data)

    if reached_goal and not touched_avoid:
        outcome = 'safe_success'
    elif reached_goal and touched_avoid:
        outcome = 'risky_success'
    elif touched_avoid:
        outcome = 'fail'
    else:
        outcome = 'neither'

    return {
        'seed': seed,
        'reach_color': reach_color,
        'avoid_color': avoid_color,
        'outcome': outcome,
        'steps': steps_data,
        'zone_positions': {k: [v.tolist() for v in vs] for k, vs in zone_positions.items()},
    }


def analyze_value_anticipation(results: List[Dict]) -> pd.DataFrame:
    """Analyze whether value drops before or after obstacle contact."""

    analysis_rows = []

    for result in results:
        if result is None:
            continue

        steps = result['steps']

        # Find steps approaching avoid zones
        for i, step in enumerate(steps):
            if step['value'] is None:
                continue

            dist = step['dist_to_avoid']
            in_avoid = step['in_avoid']

            # Categorize by distance to avoid
            if dist < ZONE_RADIUS:
                dist_category = 'in_zone'
            elif dist < ZONE_RADIUS * 2:
                dist_category = 'very_close'
            elif dist < ZONE_RADIUS * 4:
                dist_category = 'close'
            elif dist < ZONE_RADIUS * 8:
                dist_category = 'medium'
            else:
                dist_category = 'far'

            # Check if approaching (getting closer)
            if i > 0 and steps[i-1]['dist_to_avoid'] is not None:
                approaching = dist < steps[i-1]['dist_to_avoid']
            else:
                approaching = None

            analysis_rows.append({
                'seed': result['seed'],
                'step': step['step'],
                'dist_to_avoid': dist,
                'dist_category': dist_category,
                'in_avoid': in_avoid,
                'value': step['value'],
                'approaching': approaching,
                'outcome': result['outcome'],
            })

    return pd.DataFrame(analysis_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_rollouts', type=int, default=50)
    ap.add_argument('--max_steps', type=int, default=240)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/value_anticipation')
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

    # Color pairs
    color_pairs = [
        ('blue', 'yellow'),
        ('green', 'magenta'),
        ('yellow', 'blue'),
        ('magenta', 'green'),
    ]

    # Run rollouts
    print(f"\nRunning {args.n_rollouts} rollouts with value tracking...")
    results = []

    for i in range(args.n_rollouts):
        seed = args.seed + i * 100
        reach_color, avoid_color = color_pairs[i % len(color_pairs)]

        result = run_value_tracking_rollout(
            args.env_id, model, reach_color, avoid_color, seed, args.max_steps
        )

        if result is not None:
            results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_rollouts} completed")

    print(f"\nCollected {len(results)} valid rollouts")

    # Check if values were collected
    total_values = sum(sum(1 for s in r['steps'] if s['value'] is not None) for r in results)
    total_steps = sum(len(r['steps']) for r in results)
    print(f"Value data: {total_values}/{total_steps} steps have value estimates")

    if total_values == 0:
        print("WARNING: No value estimates collected! Check model output format.")
        # Try to get a sample value for debugging
        if results:
            sample_step = results[0]['steps'][0] if results[0]['steps'] else None
            print(f"Sample step data: {sample_step}")

    # Save raw results
    with open(out_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Analyze value anticipation
    df = analyze_value_anticipation(results)
    print(f"Analysis DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    df.to_csv(out_dir / 'value_by_distance.csv', index=False)

    # Print statistics
    print("\n" + "="*70)
    print("VALUE ANTICIPATION ANALYSIS")
    print("="*70)

    print(f"\n{'='*50}")
    print("VALUE BY DISTANCE TO AVOID ZONE")
    print(f"{'='*50}")

    if df.empty or 'dist_category' not in df.columns:
        print("  No value data available for analysis!")
        print(f"\nResults saved to: {out_dir}")
        return

    # Group by distance category
    dist_order = ['far', 'medium', 'close', 'very_close', 'in_zone']

    for cat in dist_order:
        subset = df[df['dist_category'] == cat]
        if len(subset) > 0:
            mean_val = subset['value'].mean()
            std_val = subset['value'].std()
            print(f"  {cat:12s}: mean={mean_val:.4f} (std={std_val:.4f}, n={len(subset)})")

    print(f"\n{'='*50}")
    print("VALUE WHEN APPROACHING VS NOT")
    print(f"{'='*50}")

    approaching = df[df['approaching'] == True]
    not_approaching = df[df['approaching'] == False]

    if len(approaching) > 0 and len(not_approaching) > 0:
        print(f"  Approaching obstacle : mean={approaching['value'].mean():.4f}")
        print(f"  Moving away         : mean={not_approaching['value'].mean():.4f}")

    print(f"\n{'='*50}")
    print("VALUE TRAJECTORY AROUND AVOID CONTACT")
    print(f"{'='*50}")

    # Find rollouts where agent touched avoid zone
    touched_rollouts = [r for r in results if r['outcome'] in ['risky_success', 'fail']]

    if touched_rollouts:
        # Analyze value before/during/after contact
        before_contact = []
        during_contact = []
        after_contact = []

        for result in touched_rollouts:
            steps = result['steps']
            contact_started = False
            contact_ended = False

            for i, step in enumerate(steps):
                if step['value'] is None:
                    continue

                if step['in_avoid']:
                    during_contact.append(step['value'])
                    contact_started = True
                elif not contact_started:
                    before_contact.append(step['value'])
                elif contact_started:
                    after_contact.append(step['value'])
                    contact_ended = True

        if before_contact:
            print(f"  Before contact: mean={np.mean(before_contact):.4f} (n={len(before_contact)})")
        if during_contact:
            print(f"  During contact: mean={np.mean(during_contact):.4f} (n={len(during_contact)})")
        if after_contact:
            print(f"  After contact : mean={np.mean(after_contact):.4f} (n={len(after_contact)})")

    # Create visualization
    print(f"\n{'='*50}")
    print("INTERPRETATION")
    print(f"{'='*50}")

    # Check for anticipation
    far_val = df[df['dist_category'] == 'far']['value'].mean()
    close_val = df[df['dist_category'] == 'close']['value'].mean()
    very_close_val = df[df['dist_category'] == 'very_close']['value'].mean()
    in_zone_val = df[df['dist_category'] == 'in_zone']['value'].mean()

    if not np.isnan(far_val) and not np.isnan(close_val):
        drop_before_contact = far_val - very_close_val if not np.isnan(very_close_val) else 0
        drop_at_contact = very_close_val - in_zone_val if not np.isnan(in_zone_val) and not np.isnan(very_close_val) else 0

        print(f"\n  Value drop approaching (far→very_close): {drop_before_contact:.4f}")
        print(f"  Value drop at contact (very_close→in): {drop_at_contact:.4f}")

        if drop_before_contact > 0.01:
            print(f"\n  >>> ANTICIPATION DETECTED: Value drops BEFORE contact")
            print(f"  >>> This suggests some predictive world model")
        else:
            print(f"\n  >>> NO ANTICIPATION: Value only drops AT/AFTER contact")
            print(f"  >>> Agent is purely reactive, no prediction")

    # Plot value vs distance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Value vs distance
    ax = axes[0]
    valid = df[df['value'].notna() & (df['dist_to_avoid'] < 3)]
    ax.scatter(valid['dist_to_avoid'], valid['value'], alpha=0.3, s=10)
    ax.axvline(x=ZONE_RADIUS, color='red', linestyle='--', label=f'Zone radius ({ZONE_RADIUS})')
    ax.set_xlabel('Distance to nearest avoid zone')
    ax.set_ylabel('Value estimate')
    ax.set_title('Value vs Distance to Avoid Zone')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean value by distance category
    ax = axes[1]
    cat_means = df.groupby('dist_category')['value'].mean().reindex(dist_order)
    cat_stds = df.groupby('dist_category')['value'].std().reindex(dist_order)
    x = range(len(dist_order))
    ax.bar(x, cat_means.values, yerr=cat_stds.values, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(dist_order, rotation=45)
    ax.set_xlabel('Distance category')
    ax.set_ylabel('Mean value')
    ax.set_title('Mean Value by Distance to Avoid Zone')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / 'value_anticipation.png', dpi=150)
    plt.close()

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
