#!/usr/bin/env python3
"""
Non-linear Position Probing Experiment

Addresses limitations of linear probing:
1. Lidar uses exp(-dist) encoding, so linear probes fail even if info is present
2. Different network layers may encode differently

This experiment uses:
- MLP probes (non-linear) in addition to linear
- Layer-wise probing: env_net, ltl_net, combined embedding
- Transformed targets: raw distance vs exp(-dist) encoding

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/nonlinear_probing.py
"""
import argparse
import json
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


class MLPProbe(nn.Module):
    """2-layer MLP probe for non-linear decoding."""
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_probe(X_train, y_train, X_test, y_test, hidden_dim=64, epochs=100, lr=1e-3):
    """Train an MLP probe and return metrics."""
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Create model
    probe = MLPProbe(input_dim, output_dim, hidden_dim)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    probe.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = probe(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        y_pred_train = probe(X_train_t).numpy()
        y_pred_test = probe(X_test_t).numpy()

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
    }


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
    }


def collect_layerwise_representations(
    env_id: str,
    model,
    reach_color: str,
    avoid_color: str,
    seed: int,
    props: set,
    max_steps: int = 240,
) -> List[Dict]:
    """Collect representations from different network layers."""

    formula = f"!{avoid_color} U {reach_color}"
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(env_id, sampler_fn, sequence=False)

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
        try:
            agent_pos = env.unwrapped.task.agent.pos[:2].copy()
            agent_vel = env.unwrapped.task.agent.vel[:2].copy()
        except:
            break

        # Get action first (sets up obs['goal'] via sequence search)
        with torch.no_grad():
            action = agent.get_action(obs, info, deterministic=True)

        # Now extract representations from different layers (obs now has 'goal')
        with torch.no_grad():
            try:
                preprocessed = preprocessing.preprocess_obss([obs], props)

                # Layer 1: env_net output (processes observation features)
                env_features = preprocessed.features
                env_out = model.env_net(env_features)

                # Layer 2: ltl_net output (processes LTL sequence)
                ltl_out = model.ltl_net(preprocessed.seq)

                # Layer 3: combined embedding (input to actor/critic)
                embedding = model.compute_embedding(preprocessed)

                # Also get raw observation features
                raw_features = env_features.squeeze().cpu().numpy()
                env_out_np = env_out.squeeze().cpu().numpy()
                ltl_out_np = ltl_out.squeeze().cpu().numpy()
                embedding_np = embedding.squeeze().cpu().numpy()

            except Exception as e:
                if step == 0:
                    print(f"  Extraction error: {e}")
                continue

        action = coerce_action(action, env.action_space)

        # Calculate distances
        dist_to_goal = min(np.linalg.norm(agent_pos - rz) for rz in reach_zones)
        dist_to_avoid = min(np.linalg.norm(agent_pos - az) for az in avoid_zones) if avoid_zones else 10.0

        # Store data
        data_points.append({
            'pos': agent_pos.copy(),
            'vel': agent_vel.copy(),
            'raw_features': raw_features.copy(),
            'env_out': env_out_np.copy(),
            'ltl_out': ltl_out_np.copy(),
            'embedding': embedding_np.copy(),
            'dist_to_goal': dist_to_goal,
            'dist_to_avoid': dist_to_avoid,
        })

        # Step
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret

        if done:
            break

    env.close()
    return data_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_rollouts', type=int, default=50)
    ap.add_argument('--max_steps', type=int, default=200)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/nonlinear_probing')
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
    props = set(dummy_env.get_propositions())
    dummy_env.close()

    # Color pairs
    color_pairs = [
        ('blue', 'yellow'),
        ('green', 'magenta'),
        ('yellow', 'blue'),
        ('magenta', 'green'),
    ]

    # Collect data
    print(f"\nCollecting layer-wise representations from {args.n_rollouts} rollouts...")
    all_data = []

    for i in range(args.n_rollouts):
        seed = args.seed + i * 100
        reach_color, avoid_color = color_pairs[i % len(color_pairs)]

        data = collect_layerwise_representations(
            args.env_id, model, reach_color, avoid_color, seed, props, args.max_steps
        )
        all_data.extend(data)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_rollouts} rollouts, {len(all_data)} data points")

    print(f"\nCollected {len(all_data)} data points total")

    if len(all_data) < 100:
        print("Not enough data points!")
        return

    # Prepare data arrays
    raw_features = np.array([d['raw_features'] for d in all_data])
    env_out = np.array([d['env_out'] for d in all_data])
    ltl_out = np.array([d['ltl_out'] for d in all_data])
    embeddings = np.array([d['embedding'] for d in all_data])
    positions = np.array([d['pos'] for d in all_data])
    velocities = np.array([d['vel'] for d in all_data])
    dist_to_goal = np.array([d['dist_to_goal'] for d in all_data])
    dist_to_avoid = np.array([d['dist_to_avoid'] for d in all_data])

    print(f"\nLayer dimensions:")
    print(f"  Raw features: {raw_features.shape}")
    print(f"  env_net output: {env_out.shape}")
    print(f"  ltl_net output: {ltl_out.shape}")
    print(f"  Combined embedding: {embeddings.shape}")

    # Split data
    indices = np.arange(len(all_data))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed)

    # Prepare targets
    # Raw targets
    pos_train, pos_test = positions[train_idx], positions[test_idx]
    vel_train, vel_test = velocities[train_idx], velocities[test_idx]
    dist_goal_train, dist_goal_test = dist_to_goal[train_idx], dist_to_goal[test_idx]
    dist_avoid_train, dist_avoid_test = dist_to_avoid[train_idx], dist_to_avoid[test_idx]

    # Transformed targets (lidar-style encoding)
    exp_dist_goal_train = np.exp(-dist_goal_train)
    exp_dist_goal_test = np.exp(-dist_goal_test)
    exp_dist_avoid_train = np.exp(-dist_avoid_train)
    exp_dist_avoid_test = np.exp(-dist_avoid_test)

    # Layers to probe
    layers = {
        'raw_features': (raw_features[train_idx], raw_features[test_idx]),
        'env_net': (env_out[train_idx], env_out[test_idx]),
        'ltl_net': (ltl_out[train_idx], ltl_out[test_idx]),
        'embedding': (embeddings[train_idx], embeddings[test_idx]),
    }

    # Targets to probe
    targets = {
        'position': (pos_train, pos_test),
        'velocity': (vel_train, vel_test),
        'dist_to_goal': (dist_goal_train.reshape(-1, 1), dist_goal_test.reshape(-1, 1)),
        'dist_to_avoid': (dist_avoid_train.reshape(-1, 1), dist_avoid_test.reshape(-1, 1)),
        'exp_dist_goal': (exp_dist_goal_train.reshape(-1, 1), exp_dist_goal_test.reshape(-1, 1)),
        'exp_dist_avoid': (exp_dist_avoid_train.reshape(-1, 1), exp_dist_avoid_test.reshape(-1, 1)),
    }

    # Run all probing experiments
    print("\n" + "="*80)
    print("PROBING EXPERIMENTS: LINEAR vs MLP, LAYER-WISE")
    print("="*80)

    results = {}

    for layer_name, (X_train, X_test) in layers.items():
        print(f"\n{'='*60}")
        print(f"LAYER: {layer_name} (dim={X_train.shape[1]})")
        print(f"{'='*60}")

        results[layer_name] = {}

        for target_name, (y_train, y_test) in targets.items():
            print(f"\n  Target: {target_name}")

            # Linear probe
            linear_result = train_linear_probe(X_train, y_train, X_test, y_test)
            print(f"    Linear:  R² = {linear_result['r2_test']:.4f}")

            # MLP probe
            mlp_result = train_mlp_probe(X_train, y_train, X_test, y_test, epochs=100)
            print(f"    MLP:     R² = {mlp_result['r2_test']:.4f}")

            results[layer_name][target_name] = {
                'linear_r2': linear_result['r2_test'],
                'mlp_r2': mlp_result['r2_test'],
            }

    # Save results
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY: R² SCORES BY LAYER AND TARGET")
    print("="*80)

    # Header
    print(f"\n{'Target':<20} | {'raw_features':^24} | {'env_net':^24} | {'ltl_net':^24} | {'embedding':^24}")
    print(f"{'':20} | {'Linear':^11} {'MLP':^11} | {'Linear':^11} {'MLP':^11} | {'Linear':^11} {'MLP':^11} | {'Linear':^11} {'MLP':^11}")
    print("-" * 130)

    for target_name in targets.keys():
        row = f"{target_name:<20}"
        for layer_name in layers.keys():
            lin = results[layer_name][target_name]['linear_r2']
            mlp = results[layer_name][target_name]['mlp_r2']
            row += f" | {lin:^11.3f} {mlp:^11.3f}"
        print(row)

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Compare linear vs MLP for distance
    emb_dist_linear = results['embedding']['dist_to_goal']['linear_r2']
    emb_dist_mlp = results['embedding']['dist_to_goal']['mlp_r2']
    emb_exp_dist_linear = results['embedding']['exp_dist_goal']['linear_r2']

    print(f"\n1. Distance to goal from embedding:")
    print(f"   Linear probe (raw dist):     R² = {emb_dist_linear:.3f}")
    print(f"   MLP probe (raw dist):        R² = {emb_dist_mlp:.3f}")
    print(f"   Linear probe (exp(-dist)):   R² = {emb_exp_dist_linear:.3f}")

    if emb_dist_mlp > emb_dist_linear + 0.1:
        print(f"   >>> MLP significantly better! Non-linear encoding confirmed.")
    if emb_exp_dist_linear > emb_dist_linear + 0.1:
        print(f"   >>> Exp-transformed target helps! Lidar-style encoding confirmed.")

    # Compare layers
    print(f"\n2. Best layer for position decoding (MLP):")
    for layer_name in layers.keys():
        r2 = results[layer_name]['position']['mlp_r2']
        print(f"   {layer_name}: R² = {r2:.3f}")

    # env_net should be best for spatial info
    env_pos = results['env_net']['position']['mlp_r2']
    emb_pos = results['embedding']['position']['mlp_r2']
    if env_pos > emb_pos:
        print(f"   >>> env_net encodes position better than full embedding!")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Linear vs MLP for distance (embedding layer)
    ax = axes[0, 0]
    targets_dist = ['dist_to_goal', 'exp_dist_goal', 'dist_to_avoid', 'exp_dist_avoid']
    x = np.arange(len(targets_dist))
    width = 0.35
    linear_vals = [results['embedding'][t]['linear_r2'] for t in targets_dist]
    mlp_vals = [results['embedding'][t]['mlp_r2'] for t in targets_dist]
    ax.bar(x - width/2, linear_vals, width, label='Linear', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, mlp_vals, width, label='MLP', color='darkorange', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['dist_goal', 'exp(-dist_goal)', 'dist_avoid', 'exp(-dist_avoid)'], rotation=15)
    ax.set_ylabel('R² Score')
    ax.set_title('Embedding Layer: Linear vs MLP Probes')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Layer comparison for position (MLP)
    ax = axes[0, 1]
    layer_names = list(layers.keys())
    pos_r2 = [results[l]['position']['mlp_r2'] for l in layer_names]
    colors = ['lightcoral', 'lightgreen', 'lightskyblue', 'plum']
    ax.bar(layer_names, pos_r2, color=colors, alpha=0.7)
    ax.set_ylabel('R² Score (MLP)')
    ax.set_title('Position Decoding by Layer')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Layer comparison for distance (MLP)
    ax = axes[1, 0]
    dist_r2 = [results[l]['dist_to_goal']['mlp_r2'] for l in layer_names]
    ax.bar(layer_names, dist_r2, color=colors, alpha=0.7)
    ax.set_ylabel('R² Score (MLP)')
    ax.set_title('Distance to Goal Decoding by Layer')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: All targets for env_net (best spatial layer)
    ax = axes[1, 1]
    all_targets = list(targets.keys())
    env_linear = [results['env_net'][t]['linear_r2'] for t in all_targets]
    env_mlp = [results['env_net'][t]['mlp_r2'] for t in all_targets]
    x = np.arange(len(all_targets))
    ax.bar(x - width/2, env_linear, width, label='Linear', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, env_mlp, width, label='MLP', color='darkorange', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(all_targets, rotation=30, ha='right')
    ax.set_ylabel('R² Score')
    ax.set_title('env_net Layer: All Targets')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / 'nonlinear_probing_results.png', dpi=150)
    plt.close()

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
