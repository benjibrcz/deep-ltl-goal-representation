#!/usr/bin/env python3
"""
Position Probing Experiment

Tests whether the agent's hidden state encodes:
1. Current position (x, y) - basic state representation
2. Next position (x', y') - one-step transition function
3. Velocity (vx, vy) - dynamics awareness
4. Distance to goal - goal representation
5. Distance to nearest obstacle - obstacle awareness

If we can decode NEXT position better than baseline, the agent has learned
some form of transition function / world model.

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/position_probing.py
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

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


def collect_hidden_states(
    env_id: str,
    model,
    reach_color: str,
    avoid_color: str,
    seed: int,
    max_steps: int = 240,
) -> List[Dict]:
    """Collect hidden states along with position data during rollout."""

    formula = f"!{avoid_color} U {reach_color}"
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(env_id, sampler_fn, sequence=False)
    props = set(env.get_propositions())

    planner = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, planner, propositions=props)

    reset_out = env.reset(seed=seed)
    obs, info = (reset_out, {}) if not isinstance(reset_out, tuple) else reset_out
    agent.reset()

    _, zone_positions = extract_zone_info(env)

    reach_zones = zone_positions.get(reach_color, [])
    avoid_zones = zone_positions.get(avoid_color, [])

    if not reach_zones:
        env.close()
        return []

    data_points = []

    for step in range(max_steps):
        # Get current state
        try:
            agent_pos = env.unwrapped.task.agent.pos[:2].copy()
            agent_vel = env.unwrapped.task.agent.vel[:2].copy()
        except:
            break

        # Get action first (this sets obs['goal'] via sequence search)
        with torch.no_grad():
            action = agent.get_action(obs, info, deterministic=True)

            # Now extract hidden state using proper model interface
            try:
                preprocessed = preprocessing.preprocess_obss([obs], props)
                embedding = model.compute_embedding(preprocessed)
                hidden_np = embedding.squeeze().cpu().numpy()
            except Exception as e:
                if step == 0:
                    print(f"  Hidden extraction error: {e}")
                # Fallback: use raw observation features
                if isinstance(obs, dict) and 'features' in obs:
                    hidden_np = np.array(obs['features'], dtype=np.float32)
                else:
                    hidden_np = np.zeros(64, dtype=np.float32)  # Placeholder

        action = coerce_action(action, env.action_space)
        action_np = np.array(action).flatten()

        # Calculate distances
        dist_to_goal = min(np.linalg.norm(agent_pos - rz) for rz in reach_zones)
        dist_to_avoid = min(np.linalg.norm(agent_pos - az) for az in avoid_zones) if avoid_zones else 10.0

        # Store current state data
        current_data = {
            'pos': agent_pos.copy(),
            'vel': agent_vel.copy(),
            'hidden': hidden_np.copy(),
            'action': action_np.copy(),
            'dist_to_goal': dist_to_goal,
            'dist_to_avoid': dist_to_avoid,
        }

        # Step environment
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret

        # Get next position
        try:
            next_pos = env.unwrapped.task.agent.pos[:2].copy()
            current_data['next_pos'] = next_pos.copy()
            data_points.append(current_data)
        except:
            pass

        if done:
            break

    env.close()
    return data_points


def train_linear_probe(X_train, y_train, X_test, y_test, alpha=1.0):
    """Train a ridge regression probe and return metrics."""
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, y_train)

    y_pred_train = probe.predict(X_train)
    y_pred_test = probe.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)

    return {
        'probe': probe,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mse_test': mse_test,
        'rmse_test': rmse_test,
        'y_pred_test': y_pred_test,
        'y_test': y_test,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_rollouts', type=int, default=100)
    ap.add_argument('--max_steps', type=int, default=200)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/position_probing')
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

    # Collect data
    print(f"\nCollecting hidden states from {args.n_rollouts} rollouts...")
    all_data = []

    for i in range(args.n_rollouts):
        seed = args.seed + i * 100
        reach_color, avoid_color = color_pairs[i % len(color_pairs)]

        data = collect_hidden_states(
            args.env_id, model, reach_color, avoid_color, seed, args.max_steps
        )
        all_data.extend(data)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{args.n_rollouts} rollouts, {len(all_data)} data points")

    print(f"\nCollected {len(all_data)} data points total")

    if len(all_data) < 100:
        print("Not enough data points!")
        return

    # Prepare data arrays
    hiddens = np.array([d['hidden'] for d in all_data])
    positions = np.array([d['pos'] for d in all_data])
    next_positions = np.array([d['next_pos'] for d in all_data])
    velocities = np.array([d['vel'] for d in all_data])
    actions = np.array([d['action'] for d in all_data])
    dist_to_goal = np.array([d['dist_to_goal'] for d in all_data])
    dist_to_avoid = np.array([d['dist_to_avoid'] for d in all_data])

    print(f"\nHidden state shape: {hiddens.shape}")
    print(f"Position shape: {positions.shape}")

    # Split data
    indices = np.arange(len(all_data))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed)

    H_train, H_test = hiddens[train_idx], hiddens[test_idx]
    pos_train, pos_test = positions[train_idx], positions[test_idx]
    next_pos_train, next_pos_test = next_positions[train_idx], next_positions[test_idx]
    vel_train, vel_test = velocities[train_idx], velocities[test_idx]
    act_train, act_test = actions[train_idx], actions[test_idx]
    dist_goal_train, dist_goal_test = dist_to_goal[train_idx], dist_to_goal[test_idx]
    dist_avoid_train, dist_avoid_test = dist_to_avoid[train_idx], dist_to_avoid[test_idx]

    # Train probes
    print("\n" + "="*70)
    print("TRAINING LINEAR PROBES")
    print("="*70)

    results = {}

    # 1. Current position from hidden state
    print("\n1. Decoding CURRENT POSITION from hidden state...")
    results['pos_from_hidden'] = train_linear_probe(H_train, pos_train, H_test, pos_test)
    print(f"   R² = {results['pos_from_hidden']['r2_test']:.4f}, RMSE = {results['pos_from_hidden']['rmse_test']:.4f}")

    # 2. Next position from hidden state (THE KEY TEST)
    print("\n2. Decoding NEXT POSITION from hidden state...")
    results['next_pos_from_hidden'] = train_linear_probe(H_train, next_pos_train, H_test, next_pos_test)
    print(f"   R² = {results['next_pos_from_hidden']['r2_test']:.4f}, RMSE = {results['next_pos_from_hidden']['rmse_test']:.4f}")

    # 3. Next position from hidden state + action (should be better if action matters)
    print("\n3. Decoding NEXT POSITION from hidden + action...")
    H_act_train = np.concatenate([H_train, act_train], axis=1)
    H_act_test = np.concatenate([H_test, act_test], axis=1)
    results['next_pos_from_hidden_action'] = train_linear_probe(H_act_train, next_pos_train, H_act_test, next_pos_test)
    print(f"   R² = {results['next_pos_from_hidden_action']['r2_test']:.4f}, RMSE = {results['next_pos_from_hidden_action']['rmse_test']:.4f}")

    # 4. Baseline: Next position from current position (trivial baseline)
    print("\n4. Baseline: NEXT POSITION from current position...")
    results['next_pos_from_pos'] = train_linear_probe(pos_train, next_pos_train, pos_test, next_pos_test)
    print(f"   R² = {results['next_pos_from_pos']['r2_test']:.4f}, RMSE = {results['next_pos_from_pos']['rmse_test']:.4f}")

    # 5. Baseline: Next position from current position + action
    print("\n5. Baseline: NEXT POSITION from position + action...")
    pos_act_train = np.concatenate([pos_train, act_train], axis=1)
    pos_act_test = np.concatenate([pos_test, act_test], axis=1)
    results['next_pos_from_pos_action'] = train_linear_probe(pos_act_train, next_pos_train, pos_act_test, next_pos_test)
    print(f"   R² = {results['next_pos_from_pos_action']['r2_test']:.4f}, RMSE = {results['next_pos_from_pos_action']['rmse_test']:.4f}")

    # 6. Velocity from hidden state
    print("\n6. Decoding VELOCITY from hidden state...")
    results['vel_from_hidden'] = train_linear_probe(H_train, vel_train, H_test, vel_test)
    print(f"   R² = {results['vel_from_hidden']['r2_test']:.4f}, RMSE = {results['vel_from_hidden']['rmse_test']:.4f}")

    # 7. Distance to goal from hidden state
    print("\n7. Decoding DISTANCE TO GOAL from hidden state...")
    results['dist_goal_from_hidden'] = train_linear_probe(H_train, dist_goal_train.reshape(-1, 1), H_test, dist_goal_test.reshape(-1, 1))
    print(f"   R² = {results['dist_goal_from_hidden']['r2_test']:.4f}, RMSE = {results['dist_goal_from_hidden']['rmse_test']:.4f}")

    # 8. Distance to avoid from hidden state
    print("\n8. Decoding DISTANCE TO AVOID from hidden state...")
    results['dist_avoid_from_hidden'] = train_linear_probe(H_train, dist_avoid_train.reshape(-1, 1), H_test, dist_avoid_test.reshape(-1, 1))
    print(f"   R² = {results['dist_avoid_from_hidden']['r2_test']:.4f}, RMSE = {results['dist_avoid_from_hidden']['rmse_test']:.4f}")

    # Save results
    summary = {
        'n_datapoints': len(all_data),
        'hidden_dim': hiddens.shape[1],
        'probes': {k: {'r2_test': v['r2_test'], 'rmse_test': v['rmse_test']} for k, v in results.items()},
    }

    with open(out_dir / 'summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2)

    # Print interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    pos_r2 = results['pos_from_hidden']['r2_test']
    next_pos_r2 = results['next_pos_from_hidden']['r2_test']
    next_pos_action_r2 = results['next_pos_from_hidden_action']['r2_test']
    baseline_r2 = results['next_pos_from_pos']['r2_test']
    baseline_action_r2 = results['next_pos_from_pos_action']['r2_test']

    print(f"\n  Current position decoding:     R² = {pos_r2:.4f}")
    print(f"  Next position (hidden only):   R² = {next_pos_r2:.4f}")
    print(f"  Next position (hidden+action): R² = {next_pos_action_r2:.4f}")
    print(f"  Baseline (pos only):           R² = {baseline_r2:.4f}")
    print(f"  Baseline (pos+action):         R² = {baseline_action_r2:.4f}")

    if pos_r2 > 0.9:
        print(f"\n  ✓ Hidden state encodes CURRENT POSITION well")
    else:
        print(f"\n  ✗ Hidden state does NOT encode current position well")

    if next_pos_r2 > baseline_r2:
        print(f"  ✓ Hidden state contains MORE than just position for next-step prediction")
        print(f"    (hidden: {next_pos_r2:.4f} > baseline: {baseline_r2:.4f})")
    else:
        print(f"  ✗ Hidden state doesn't help beyond knowing current position")

    if next_pos_action_r2 > next_pos_r2 + 0.01:
        print(f"  ✓ Adding ACTION improves prediction")
        print(f"    This suggests action-conditioned transition function potential")
    else:
        print(f"  ✗ Adding action doesn't help much - prediction is action-agnostic")

    vel_r2 = results['vel_from_hidden']['r2_test']
    if vel_r2 > 0.5:
        print(f"  ✓ Hidden state encodes VELOCITY (R² = {vel_r2:.4f})")
    else:
        print(f"  ✗ Hidden state does NOT encode velocity well (R² = {vel_r2:.4f})")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Current position decoding
    ax = axes[0, 0]
    y_pred = results['pos_from_hidden']['y_pred_test']
    y_true = results['pos_from_hidden']['y_test']
    ax.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.3, s=10, label='X')
    ax.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.3, s=10, label='Y')
    ax.plot([-3, 3], [-3, 3], 'k--', alpha=0.5)
    ax.set_xlabel('True position')
    ax.set_ylabel('Predicted position')
    ax.set_title(f'Current Position Decoding (R²={pos_r2:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Next position from hidden
    ax = axes[0, 1]
    y_pred = results['next_pos_from_hidden']['y_pred_test']
    y_true = results['next_pos_from_hidden']['y_test']
    ax.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.3, s=10, label='X')
    ax.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.3, s=10, label='Y')
    ax.plot([-3, 3], [-3, 3], 'k--', alpha=0.5)
    ax.set_xlabel('True next position')
    ax.set_ylabel('Predicted next position')
    ax.set_title(f'Next Position from Hidden (R²={next_pos_r2:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Comparison bar chart
    ax = axes[1, 0]
    probes = ['pos\n(hidden)', 'next_pos\n(hidden)', 'next_pos\n(h+action)', 'next_pos\n(pos)', 'next_pos\n(pos+act)']
    r2_values = [pos_r2, next_pos_r2, next_pos_action_r2, baseline_r2, baseline_action_r2]
    colors = ['steelblue', 'darkorange', 'darkorange', 'gray', 'gray']
    ax.bar(probes, r2_values, color=colors, alpha=0.7)
    ax.set_ylabel('R² score')
    ax.set_title('Probe Comparison')
    ax.axhline(y=baseline_r2, color='red', linestyle='--', label='Position baseline')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Other probes
    ax = axes[1, 1]
    other_probes = ['velocity', 'dist_goal', 'dist_avoid']
    other_r2 = [
        results['vel_from_hidden']['r2_test'],
        results['dist_goal_from_hidden']['r2_test'],
        results['dist_avoid_from_hidden']['r2_test'],
    ]
    ax.bar(other_probes, other_r2, color='seagreen', alpha=0.7)
    ax.set_ylabel('R² score')
    ax.set_title('Other Representations in Hidden State')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / 'probing_results.png', dpi=150)
    plt.close()

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
