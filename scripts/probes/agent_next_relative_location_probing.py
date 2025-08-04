#!/usr/bin/env python3
"""
Agent Relative Location Probing for Zone Environment

This script analyzes the internal representations in the network by probing for the agent's relative position to each zone (agent_pos[:2] - zone_center for each zone) at each time step, instead of the absolute agent position.

1. Relative location prediction analysis (linear and non-linear)
2. Spatial relationship encoding
3. Zone proximity and direction encoding
4. Action-outcome prediction
5. Temporal dynamics analysis
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from tqdm import trange, tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import pandas as pd
import gc

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from ltl import FixedSampler
from envs import make_env
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from sequence.search import ExhaustiveSearch
from model.agent import Agent

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 1
NUM_ROLLOUTS = 30
MAX_STEPS = 200
PREDICTION_HORIZON = 0  # Predict t+k steps ahead (0 = current state, 1 = next state, etc.)

class MLPProbe(nn.Module):
    """Two-layer MLP probe for non-linear analysis"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

def train_mlp_probe(X_train, y_train, X_test, y_test, input_dim, output_dim, epochs=100, lr=0.001):
    """Train a non-linear MLP probe"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    model = MLPProbe(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            test_outputs = model(X_test_tensor)
            train_loss = criterion(train_outputs, y_train_tensor).item()
            test_loss = criterion(test_outputs, y_test_tensor).item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        y_pred = test_outputs.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        r2 = np.mean(r2)
    return model, r2, y_pred, train_losses, test_losses

def get_zone_centers():
    """Get zone centers from the environment"""
    sampler_fn = FixedSampler.partial("GF blue")
    temp_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    temp_env.reset(seed=1)
    zone_centers = {}
    if hasattr(temp_env, 'zone_positions'):
        zone_positions = temp_env.zone_positions
        for zone_name, position in zone_positions.items():
            color = zone_name.split('_')[0]
            if color not in zone_centers:
                zone_centers[color] = []
            zone_centers[color].append(position[:2])
    temp_env.close()
    return zone_centers

def collect_relative_location_data(layer_name, formula):
    """Collect data for relative location probing for a specific layer"""
    print(f"Collecting data for {layer_name} with formula {formula}")
    print(f"Prediction horizon: t+{PREDICTION_HORIZON}")
    all_relative_locations = []
    all_activations = []
    all_actions = []
    all_zone_distances = []
    all_zone_directions = []
    sampler_fn = FixedSampler.partial(formula)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, 0)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    print(f"Observation shape: {build_env.observation_space.shape}")
    print(f"Num LTLNet params: {sum(p.numel() for p in model.ltl_net.parameters())}")
    print(f"Prediction horizon: t+{PREDICTION_HORIZON}")
    zone_centers = get_zone_centers()
    zone_colors = list(zone_centers.keys())
    for rollout_idx in trange(NUM_ROLLOUTS, desc=f"Rollouts for {layer_name}"):
        rollout_activations = []
        rollout_relative_locations = []
        rollout_actions = []
        rollout_zone_distances = []
        rollout_zone_directions = []
        rollout_target_relative_locations = []
        def hook_fn(mod, inp, out):
            if layer_name == "env_net":
                arr = out.detach().squeeze().cpu().numpy() if hasattr(out, 'detach') else out.squeeze().cpu().numpy()
            elif layer_name == "policy_encoder":
                arr = out.detach().squeeze().cpu().numpy() if hasattr(out, 'detach') else out.squeeze().cpu().numpy()
            elif layer_name == "ltl_rnn":
                arr = out[1].detach().squeeze(0).squeeze(0).cpu().numpy() if isinstance(out, tuple) else out.detach().squeeze().cpu().numpy()
            else:
                arr = out.detach().squeeze().cpu().numpy() if hasattr(out, 'detach') else out.squeeze().cpu().numpy()
            rollout_activations.append(arr)
        rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        ret = rollout_env.reset(seed=rollout_idx + 1)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        props = set(rollout_env.get_propositions())
        search = ExhaustiveSearch(model, props, num_loops=2)
        agent = Agent(model, search=search, propositions=props, verbose=False)
        handle = None
        if layer_name == "env_net":
            if hasattr(model, 'env_net') and model.env_net is not None:
                handle = model.env_net.register_forward_hook(hook_fn)
        elif layer_name == "policy_encoder":
            if hasattr(model, 'actor') and hasattr(model.actor, 'enc'):
                handle = model.actor.enc.register_forward_hook(hook_fn)
        elif layer_name == "ltl_rnn":
            if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
                handle = model.ltl_net.rnn.register_forward_hook(hook_fn)
        for step in range(MAX_STEPS):
            current_pos = rollout_env.agent_pos[:2].copy()
            rel_locs = []
            for color in zone_colors:
                centers = zone_centers[color]
                if centers:
                    distances = [np.linalg.norm(current_pos - center) for center in centers]
                    min_idx = np.argmin(distances)
                    closest_center = centers[min_idx]
                    rel_loc = current_pos - closest_center
                    rel_locs.extend(rel_loc)
                else:
                    rel_locs.extend([0.0, 0.0])
            rollout_relative_locations.append(rel_locs)
            action = agent.get_action(obs, info, deterministic=True).flatten()
            rollout_actions.append(action)
            zone_dists = []
            zone_dirs = []
            for color in zone_colors:
                centers = zone_centers[color]
                if centers:
                    distances = [np.linalg.norm(current_pos - center) for center in centers]
                    min_dist = min(distances)
                    closest_center = centers[np.argmin(distances)]
                    direction = (closest_center - current_pos) / (min_dist + 1e-8)
                    zone_dists.append(min_dist)
                    zone_dirs.extend(direction)
                else:
                    zone_dists.append(0.0)
                    zone_dirs.extend([0.0, 0.0])
            rollout_zone_distances.append(zone_dists)
            rollout_zone_directions.append(zone_dirs)
            ret = rollout_env.step(action)
            if len(ret) == 5:
                next_obs, rew, term, trunc, next_info = ret
                done = term or trunc
            else:
                next_obs, rew, done, next_info = ret
            obs, info = next_obs, next_info
            if done:
                break
        min_len = min(len(rollout_activations), len(rollout_relative_locations), len(rollout_actions), len(rollout_zone_distances), len(rollout_zone_directions))
        rollout_activations = rollout_activations[:min_len]
        rollout_relative_locations = rollout_relative_locations[:min_len]
        rollout_actions = rollout_actions[:min_len]
        rollout_zone_distances = rollout_zone_distances[:min_len]
        rollout_zone_directions = rollout_zone_directions[:min_len]
        all_activations.extend(rollout_activations)
        all_relative_locations.extend(rollout_relative_locations)
        all_actions.extend(rollout_actions)
        all_zone_distances.extend(rollout_zone_distances)
        all_zone_directions.extend(rollout_zone_directions)
        if handle is not None:
            handle.remove()
        rollout_env.close()
        for i in range(min_len):
            rollout_target_relative_locations.append(rollout_relative_locations[min(i + PREDICTION_HORIZON, len(rollout_relative_locations) - 1)])
    del model, agent
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return (np.array(all_activations), np.array(all_relative_locations), np.array(all_actions), np.array(all_zone_distances), np.array(all_zone_directions), zone_colors, np.array(rollout_target_relative_locations))

def train_time_step_mlp_probes(activations, relative_locations, layer_name, num_time_steps=10):
    print(f"\n=== Time Step MLP Analysis for {layer_name} ===")
    X = activations
    y = relative_locations
    data_per_step = len(X) // num_time_steps
    step_results = {}
    print(f"Training {num_time_steps} separate MLPs (one per time step)")
    print(f"Data points per time step: {data_per_step}")
    for step in range(num_time_steps):
        start_idx = step * data_per_step
        end_idx = (step + 1) * data_per_step
        if end_idx > len(X):
            break
        X_step = X[start_idx:end_idx]
        y_step = y[start_idx:end_idx]
        split_idx = int(0.8 * len(X_step))
        X_train, X_test = X_step[:split_idx], X_step[split_idx:]
        y_train, y_test = y_step[:split_idx], y_step[split_idx:]
        if len(X_train) < 10 or len(X_test) < 5:
            continue
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        mlp_model, r2_mlp, y_pred_mlp, train_losses, test_losses = train_mlp_probe(
            X_train, y_train, X_test, y_test, input_dim, output_dim, epochs=30, lr=0.001
        )
        mse_mlp = mean_squared_error(y_test, y_pred_mlp)
        step_results[step] = {
            'r2': r2_mlp,
            'mse': mse_mlp,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'predictions': y_pred_mlp,
            'ground_truth': y_test,
            'data_points': len(X_step)
        }
        print(f"Time step {step}: R² = {r2_mlp:.3f}, MSE = {mse_mlp:.6f}, Data = {len(X_step)}")
    return step_results

def analyze_relative_location_prediction(activations, relative_locations, layer_name):
    print(f"\n=== Relative Location Prediction Analysis for {layer_name} ===")
    X = activations
    y = relative_locations
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print("Training Ridge probe...")
    reg = Ridge()
    reg.fit(X_train, y_train)
    y_pred_ridge = reg.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    print(f"Ridge probe - MSE: {mse_ridge:.6f}, R²: {r2_ridge:.3f}")
    feature_importance = np.abs(reg.coef_).mean(axis=0)
    print(f"Ridge feature importance: {feature_importance.mean():.6f}")
    print("Training non-linear MLP probe (global)...")
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    mlp_model, r2_mlp, y_pred_mlp, train_losses, test_losses = train_mlp_probe(
        X_train, y_train, X_test, y_test, input_dim, output_dim, epochs=50, lr=0.001
    )
    mse_mlp = mean_squared_error(y_test, y_pred_mlp)
    print(f"Global MLP probe - MSE: {mse_mlp:.6f}, R²: {r2_mlp:.3f}")
    time_step_results = train_time_step_mlp_probes(activations, relative_locations, layer_name)
    improvement_global = r2_mlp - r2_ridge
    print(f"Global MLP improvement over Ridge: {improvement_global:.3f}")
    if time_step_results:
        avg_time_step_r2 = np.mean([r['r2'] for r in time_step_results.values()])
        best_time_step_r2 = max([r['r2'] for r in time_step_results.values()])
        print(f"Average time-step MLP R²: {avg_time_step_r2:.3f}")
        print(f"Best time-step MLP R²: {best_time_step_r2:.3f}")
        print(f"Time-step vs Global improvement: {avg_time_step_r2 - r2_mlp:.3f}")
    return {
        'ridge': {
            'mse': mse_ridge,
            'r2': r2_ridge,
            'feature_importance': feature_importance,
            'predictions': y_pred_ridge,
            'ground_truth': y_test
        },
        'mlp': {
            'mse': mse_mlp,
            'r2': r2_mlp,
            'predictions': y_pred_mlp,
            'train_losses': train_losses,
            'test_losses': test_losses
        },
        'time_step_mlp': time_step_results,
        'improvement': improvement_global
    }

def analyze_zone_relationships(activations, zone_distances, zone_directions, layer_name):
    """Analyze how the layer encodes zone relationships"""
    print(f"\n=== Zone Relationship Analysis for {layer_name} ===")
    zone_colors = ['red', 'magenta', 'yellow', 'orange', 'blue', 'green', 'aqua']
    # Analyze distance encoding
    distance_results = {}
    for i, color in enumerate(zone_colors):
        if i < zone_distances.shape[1]:
            distances = zone_distances[:, i]
            reg = LinearRegression()
            reg.fit(activations, distances)
            r2 = reg.score(activations, distances)
            distance_results[color] = r2
            print(f"{color} distance R²: {r2:.3f}")
    # Analyze direction encoding
    direction_results = {}
    for i, color in enumerate(zone_colors):
        if i * 2 + 1 < zone_directions.shape[1]:
            directions = zone_directions[:, i*2:i*2+2]  # x, y components
            reg = LinearRegression()
            reg.fit(activations, directions)
            r2 = reg.score(activations, directions)
            direction_results[color] = r2
            print(f"{color} direction R²: {r2:.3f}")
    return {
        'distance_encoding': distance_results,
        'direction_encoding': direction_results
    }

def analyze_action_prediction(activations, actions, layer_name):
    """Analyze how well the layer predicts actions"""
    print(f"\n=== Action Prediction Analysis for {layer_name} ===")
    split_idx = int(0.8 * len(activations))
    X_train, X_test = activations[:split_idx], activations[split_idx:]
    y_train, y_test = actions[:split_idx], actions[split_idx:]
    action_dims = y_train.shape[1]
    action_results = {}
    for dim in range(action_dims):
        reg = LinearRegression()
        reg.fit(X_train, y_train[:, dim])
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test[:, dim], y_pred)
        r2 = reg.score(X_test, y_test[:, dim])
        action_results[f'dim_{dim}'] = {'mse': mse, 'r2': r2}
        print(f"Action dim {dim} - MSE: {mse:.6f}, R²: {r2:.3f}")
    return action_results

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs('world_model_analysis_plots', exist_ok=True)
    FORMULA = "GF blue & GF green"
    layers_to_analyze = ["env_net", "policy_encoder"]
    all_results = {}
    print(f"Analyzing relative location for {len(layers_to_analyze)} layers")
    print(f"Formula: {FORMULA}")
    print(f"Number of rollouts: {NUM_ROLLOUTS}")
    for layer_name in layers_to_analyze:
        print(f"\n{'='*60}")
        print(f"ANALYZING LAYER: {layer_name}")
        print(f"{'='*60}")
        try:
            data = collect_relative_location_data(layer_name, FORMULA)
            if data is not None and all(d is not None for d in data):
                activations, relative_locations, actions, zone_distances, zone_directions, zone_colors, rollout_target_relative_locations = data
                print(f"Collected {len(activations)} data points")
                print(f"Activation shape: {activations.shape}")
                print(f"Relative location shape: {relative_locations.shape}")
                rel_loc_results = analyze_relative_location_prediction(
                    activations, relative_locations, layer_name)
                zone_results = analyze_zone_relationships(
                    activations, zone_distances, zone_directions, layer_name)
                action_results = analyze_action_prediction(
                    activations, actions, layer_name)
                all_results[layer_name] = {
                    'relative_location': rel_loc_results,
                    'zone_relationships': zone_results,
                    'action_prediction': action_results
                }
                # Save raw data
                np.save(f'world_model_analysis_plots/{layer_name}_relative_location_activations.npy', activations)
                np.save(f'world_model_analysis_plots/{layer_name}_relative_locations.npy', relative_locations)
                np.save(f'world_model_analysis_plots/{layer_name}_actions.npy', actions)
                np.save(f'world_model_analysis_plots/{layer_name}_zone_distances.npy', zone_distances)
                np.save(f'world_model_analysis_plots/{layer_name}_rollout_target_relative_locations.npy', rollout_target_relative_locations)
                print(f"✓ Completed analysis for {layer_name}")
            else:
                print(f"✗ No data collected for {layer_name}")
        except Exception as e:
            print(f"✗ Error analyzing {layer_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"\n{'='*60}")
    print("RELATIVE LOCATION PROBING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Layer':<15} {'Ridge R²':<12} {'Global MLP R²':<15} {'Time-Step MLP R²':<18} {'Action R²':<12} {'Zone R²':<12}")
    print("-" * 90)
    for layer_name, results in all_results.items():
        ridge_r2 = results['relative_location']['ridge']['r2'] if 'relative_location' in results else 0
        global_mlp_r2 = results['relative_location']['mlp']['r2'] if 'relative_location' in results else 0
        time_step_r2 = 0
        if 'relative_location' in results and 'time_step_mlp' in results['relative_location']:
            time_step_results = results['relative_location']['time_step_mlp']
            if time_step_results:
                time_step_r2 = np.mean([r['r2'] for r in time_step_results.values()])
        action_r2 = np.mean([r['r2'] for r in results['action_prediction'].values()]) if 'action_prediction' in results else 0
        zone_r2 = np.mean(list(results['zone_relationships']['distance_encoding'].values())) if 'zone_relationships' in results else 0
        print(f"{layer_name:<15} {ridge_r2:<12.3f} {global_mlp_r2:<15.3f} {time_step_r2:<18.3f} {action_r2:<12.3f} {zone_r2:<12.3f}")
        if 'relative_location' in results:
            if 'improvement' in results['relative_location']:
                improvement = results['relative_location']['improvement']
                print(f"  → Global MLP improvement: {improvement:+.3f}")
            if 'time_step_mlp' in results['relative_location'] and time_step_results:
                time_step_improvement = time_step_r2 - global_mlp_r2
                best_time_step = max([r['r2'] for r in time_step_results.values()])
                print(f"  → Time-step vs Global: {time_step_improvement:+.3f} (best: {best_time_step:.3f})")
    summary_df = pd.DataFrame(all_results).T
    summary_df.to_csv('world_model_analysis_plots/relative_location_probing_summary.csv')
    print(f"\nSummary saved to: world_model_analysis_plots/relative_location_probing_summary.csv")

if __name__ == '__main__':
    main() 