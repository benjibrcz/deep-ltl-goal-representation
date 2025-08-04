#!/usr/bin/env python3
"""
World Model Analysis for Zone Environment

This script analyzes the internal world model representations in the network:
1. Next-state prediction analysis (linear and non-linear)
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
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Initialize model
    model = MLPProbe(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Evaluation
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
    
    # Calculate R² score
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        y_pred = test_outputs.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        
        # R² calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        r2 = np.mean(r2)  # Average across output dimensions
    
    return model, r2, y_pred, train_losses, test_losses

def get_zone_centers():
    """Get zone centers from the environment"""
    # For PointLtl2-v0, we need to get zone positions from the environment
    # The zones are dynamically placed, so we'll use a sample environment to get them
    sampler_fn = FixedSampler.partial("GF blue")
    temp_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    temp_env.reset(seed=1)
    
    # Get zone positions from the environment
    zone_centers = {}
    if hasattr(temp_env, 'zone_positions'):
        zone_positions = temp_env.zone_positions
        # zone_positions is a dict with keys like "blue_0", "green_1", etc.
        for zone_name, position in zone_positions.items():
            # Extract color from zone name (e.g., "blue_0" -> "blue")
            color = zone_name.split('_')[0]
            if color not in zone_centers:
                zone_centers[color] = []
            zone_centers[color].append(position[:2])  # Take only x, y coordinates
    
    temp_env.close()
    return zone_centers

def collect_world_model_data(layer_name, formula):
    """Collect world model data for a specific layer"""
    print(f"Collecting data for {layer_name} with formula {formula}")
    print(f"Prediction horizon: t+{PREDICTION_HORIZON}")
    
    all_current_states = []
    all_target_states = []  # States to predict (t+k)
    all_actions = []
    all_zone_distances = []
    all_zone_directions = []
    
    # Setup environment and model
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
    
    # Get zone centers for distance calculations
    zone_centers = get_zone_centers()
    
    # Collect data across multiple rollouts
    all_activations = []
    for rollout_idx in trange(NUM_ROLLOUTS, desc=f"Rollouts for {layer_name}"):
        rollout_activations = []  # Local to this rollout
        rollout_current_states = []
        rollout_actions = []
        rollout_zone_distances = []
        rollout_zone_directions = []
        
        def hook_fn(mod, inp, out):
            if layer_name == "env_net":
                if hasattr(out, 'detach'):
                    arr = out.detach().squeeze().cpu().numpy()
                else:
                    arr = out.squeeze().cpu().numpy()
            elif layer_name == "policy_encoder":
                if hasattr(out, 'detach'):
                    arr = out.detach().squeeze().cpu().numpy()
                else:
                    arr = out.squeeze().cpu().numpy()
            elif layer_name == "ltl_rnn":
                if isinstance(out, tuple):
                    h_n = out[1]  # Final hidden state
                    arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy()
                else:
                    arr = out.detach().squeeze().cpu().numpy()
            else:
                if hasattr(out, 'detach'):
                    arr = out.detach().squeeze().cpu().numpy()
                else:
                    arr = out.squeeze().cpu().numpy()
            rollout_activations.append(arr)

        # Setup environment and model for this rollout
        rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        ret = rollout_env.reset(seed=rollout_idx + 1)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        props = set(rollout_env.get_propositions())
        search = ExhaustiveSearch(model, props, num_loops=2)
        agent = Agent(model, search=search, propositions=props, verbose=False)
        # Register hook (now after hook_fn is defined)
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
            # Record current state directly from the environment
            current_pos = rollout_env.agent_pos[:2].copy()
            rollout_current_states.append(current_pos)
            # Get action and record activation
            action = agent.get_action(obs, info, deterministic=True).flatten()
            rollout_actions.append(action)
            # Calculate zone distances and directions
            zone_dists = []
            zone_dirs = []
            for color, centers in zone_centers.items():
                if centers:  # If there are centers for this color
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
            # Take step to get next state
            ret = rollout_env.step(action)
            if len(ret) == 5:
                next_obs, rew, term, trunc, next_info = ret
                done = term or trunc
            else:
                next_obs, rew, done, next_info = ret
            obs, info = next_obs, next_info
            if done:
                break
        # Truncate all lists to the same length (minimum)
        min_len = min(len(rollout_activations), len(rollout_current_states), len(rollout_actions), len(rollout_zone_distances), len(rollout_zone_directions))
        rollout_activations = rollout_activations[:min_len]
        rollout_current_states = rollout_current_states[:min_len]
        rollout_actions = rollout_actions[:min_len]
        rollout_zone_distances = rollout_zone_distances[:min_len]
        rollout_zone_directions = rollout_zone_directions[:min_len]
        # Create target states based on prediction horizon
        rollout_target_states = []
        for i in range(len(rollout_current_states)):
            target_idx = min(i + PREDICTION_HORIZON, len(rollout_current_states) - 1)
            rollout_target_states.append(rollout_current_states[target_idx])
        rollout_target_states = rollout_target_states[:min_len]
        # Append to global lists
        all_activations.extend(rollout_activations)
        all_current_states.extend(rollout_current_states)
        all_target_states.extend(rollout_target_states)
        all_actions.extend(rollout_actions)
        all_zone_distances.extend(rollout_zone_distances)
        all_zone_directions.extend(rollout_zone_directions)
        # Remove hook
        if handle is not None:
            handle.remove()
        rollout_env.close()
    # Clean up memory
    del model, agent
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return (np.array(all_activations), np.array(all_current_states), 
            np.array(all_target_states), np.array(all_actions),
            np.array(all_zone_distances), np.array(all_zone_directions))

def train_time_step_mlp_probes(activations, current_states, target_states, layer_name, num_time_steps=10):
    """Train separate MLP probes for different time steps"""
    print(f"\n=== Time Step MLP Analysis for {layer_name} ===")
    
    # Prepare data
    X = activations
    y = target_states  # Always predict absolute target state
    
    # Split data by time steps (assuming data is ordered by time steps)
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
        
        # Split into train/test for this time step
        split_idx = int(0.8 * len(X_step))
        X_train, X_test = X_step[:split_idx], X_step[split_idx:]
        y_train, y_test = y_step[:split_idx], y_step[split_idx:]
        
        if len(X_train) < 10 or len(X_test) < 5:  # Need minimum data
            continue
            
        # Train MLP for this time step
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

def analyze_next_state_prediction(activations, current_states, target_states, layer_name):
    """Analyze how well the layer predicts target states (linear and non-linear)"""
    horizon_text = "current" if PREDICTION_HORIZON == 0 else f"t+{PREDICTION_HORIZON}"
    print(f"\n=== {horizon_text.title()} State Prediction Analysis for {layer_name} ===")
    
    # Prepare data
    X = activations
    y = target_states  # Always predict absolute target state
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 1. Ridge probe (like the successful previous probes)
    print("Training Ridge probe...")
    reg = Ridge()
    reg.fit(X_train, y_train)
    
    # Evaluate Ridge probe
    y_pred_ridge = reg.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    
    print(f"Ridge probe - MSE: {mse_ridge:.6f}, R²: {r2_ridge:.3f}")
    
    # Analyze feature importance
    feature_importance = np.abs(reg.coef_).mean(axis=0)
    print(f"Ridge feature importance: {feature_importance.mean():.6f}")
    
    # 2. Non-linear MLP probe (global)
    print("Training non-linear MLP probe (global)...")
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    mlp_model, r2_mlp, y_pred_mlp, train_losses, test_losses = train_mlp_probe(
        X_train, y_train, X_test, y_test, input_dim, output_dim, epochs=50, lr=0.001
    )
    
    mse_mlp = mean_squared_error(y_test, y_pred_mlp)
    print(f"Global MLP probe - MSE: {mse_mlp:.6f}, R²: {r2_mlp:.3f}")
    
    # 3. Time-step-specific MLP probes
    time_step_results = train_time_step_mlp_probes(activations, current_states, target_states, layer_name)
    
    # Compare results
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
    
    # Split data
    split_idx = int(0.8 * len(activations))
    X_train, X_test = activations[:split_idx], activations[split_idx:]
    y_train, y_test = actions[:split_idx], actions[split_idx:]
    
    # Train model for each action dimension
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

def visualize_world_model_analysis(activations, current_states, next_states, 
                                  zone_distances, layer_name, results):
    """Create visualizations for world model analysis"""
    print(f"\n=== Creating visualizations for {layer_name} ===")
    
    # PCA for activation space
    pca = PCA(n_components=2)
    activations_pca = pca.fit_transform(activations)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'World Model Analysis: {layer_name}', fontsize=16)
    
    # 1. Activation space PCA
    scatter1 = axes[0,0].scatter(activations_pca[:, 0], activations_pca[:, 1], 
                                c=zone_distances[:, 0], cmap='viridis', alpha=0.6)  # Use first zone (red)
    axes[0,0].set_xlabel('PC1')
    axes[0,0].set_ylabel('PC2')
    axes[0,0].set_title('Activation Space (colored by distance to red zone)')
    plt.colorbar(scatter1, ax=axes[0,0])
    
    # 2. State prediction accuracy (Ridge vs MLP)
    if 'next_state' in results:
        # Ridge predictions
        axes[0,1].scatter(results['next_state']['ridge']['ground_truth'][:, 0], 
                         results['next_state']['ridge']['predictions'][:, 0], 
                         alpha=0.6, label=f'Ridge (R²={results["next_state"]["ridge"]["r2"]:.3f})')
        
        # MLP predictions
        axes[0,1].scatter(results['next_state']['ridge']['ground_truth'][:, 0], 
                         results['next_state']['mlp']['predictions'][:, 0], 
                         alpha=0.6, label=f'MLP (R²={results["next_state"]["mlp"]["r2"]:.3f})')
        
        axes[0,1].plot([-0.1, 0.1], [-0.1, 0.1], 'r--', alpha=0.8)
        axes[0,1].set_xlabel('Ground Truth Δx')
        axes[0,1].set_ylabel('Predicted Δx')
        axes[0,1].set_title('State Prediction Comparison')
        axes[0,1].legend()
    
    # 3. Zone distance encoding
    if 'zone_relationships' in results:
        colors = list(results['zone_relationships']['distance_encoding'].keys())
        r2_values = list(results['zone_relationships']['distance_encoding'].values())
        bars = axes[0,2].bar(colors, r2_values)
        axes[0,2].set_ylabel('R²')
        axes[0,2].set_title('Zone Distance Encoding')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # Color bars by zone color
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    # 4. Trajectory visualization
    axes[1,0].scatter(current_states[:, 0], current_states[:, 1], 
                     c=zone_distances[:, 0], cmap='viridis', alpha=0.6, s=20)  # Use first zone (red)
    axes[1,0].set_xlabel('X Position')
    axes[1,0].set_ylabel('Y Position')
    axes[1,0].set_title('Agent Positions (colored by distance to red zone)')
    axes[1,0].set_xlim(-2, 2)
    axes[1,0].set_ylim(-2, 2)
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Action prediction
    if 'action_prediction' in results:
        action_dims = list(results['action_prediction'].keys())
        action_r2 = [results['action_prediction'][dim]['r2'] for dim in action_dims]
        axes[1,1].bar(action_dims, action_r2)
        axes[1,1].set_ylabel('R²')
        axes[1,1].set_title('Action Prediction Accuracy')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    # 6. Time-step MLP performance
    if 'next_state' in results and 'time_step_mlp' in results['next_state']:
        time_step_results = results['next_state']['time_step_mlp']
        if time_step_results:
            steps = list(time_step_results.keys())
            r2_values = [time_step_results[step]['r2'] for step in steps]
            axes[1,2].plot(steps, r2_values, 'o-', linewidth=2, markersize=6)
            axes[1,2].set_xlabel('Time Step')
            axes[1,2].set_ylabel('R² Score')
            axes[1,2].set_title('Time-Step MLP Performance')
            axes[1,2].grid(True, alpha=0.3)
            
            # Add global MLP performance as reference line
            if 'mlp' in results['next_state']:
                global_r2 = results['next_state']['mlp']['r2']
                axes[1,2].axhline(y=global_r2, color='red', linestyle='--', 
                                label=f'Global MLP (R²={global_r2:.3f})')
                axes[1,2].legend()
    else:
        # Fallback to feature importance if no time-step results
        if 'next_state' in results:
            importance = results['next_state']['ridge']['feature_importance']
            top_features = np.argsort(importance)[-20:]  # Top 20 features
            axes[1,2].bar(range(len(top_features)), importance[top_features])
            axes[1,2].set_xlabel('Feature Index')
            axes[1,2].set_ylabel('Importance')
            axes[1,2].set_title('Top 20 Feature Importance (Ridge)')
    
    plt.tight_layout()
    plt.savefig(f'world_model_analysis_plots/{layer_name}_world_model_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Create output directory
    os.makedirs('world_model_analysis_plots', exist_ok=True)
    
    # Test formula
    FORMULA = "GF blue & GF green"
    
    # Layers to analyze
    layers_to_analyze = ["env_net", "policy_encoder"]
    
    # Store results for each layer
    all_results = {}
    
    print(f"Analyzing world model for {len(layers_to_analyze)} layers")
    print(f"Formula: {FORMULA}")
    print(f"Number of rollouts: {NUM_ROLLOUTS}")
    
    for layer_name in layers_to_analyze:
        print(f"\n{'='*60}")
        print(f"ANALYZING LAYER: {layer_name}")
        print(f"{'='*60}")
        
        try:
            # Collect data for this layer
            data = collect_world_model_data(layer_name, FORMULA)
            
            if data is not None and all(d is not None for d in data):
                activations, current_states, target_states, actions, zone_distances, zone_directions = data
                
                print(f"Collected {len(activations)} data points")
                print(f"Activation shape: {activations.shape}")
                print(f"State shape: {current_states.shape}")
                
                # Analyze target state prediction
                next_state_results = analyze_next_state_prediction(
                    activations, current_states, target_states, layer_name)
                
                # Analyze zone relationships
                zone_results = analyze_zone_relationships(
                    activations, zone_distances, zone_directions, layer_name)
                
                # Analyze action prediction
                action_results = analyze_action_prediction(
                    activations, actions, layer_name)
                
                # Store results
                all_results[layer_name] = {
                    'next_state': next_state_results,
                    'zone_relationships': zone_results,
                    'action_prediction': action_results
                }
                
                # Create visualizations
                visualize_world_model_analysis(
                    activations, current_states, target_states, zone_distances, 
                    layer_name, all_results[layer_name])
                
                # Save raw data
                np.save(f'world_model_analysis_plots/{layer_name}_activations.npy', activations)
                np.save(f'world_model_analysis_plots/{layer_name}_states.npy', current_states)
                np.save(f'world_model_analysis_plots/{layer_name}_target_states.npy', target_states)
                np.save(f'world_model_analysis_plots/{layer_name}_actions.npy', actions)
                np.save(f'world_model_analysis_plots/{layer_name}_zone_distances.npy', zone_distances)
                
                print(f"✓ Completed analysis for {layer_name}")
            else:
                print(f"✗ No data collected for {layer_name}")
                
        except Exception as e:
            print(f"✗ Error analyzing {layer_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print summary
    print(f"\n{'='*60}")
    print("WORLD MODEL ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Layer':<15} {'Ridge R²':<12} {'Global MLP R²':<15} {'Time-Step MLP R²':<18} {'Action R²':<12} {'Zone R²':<12}")
    print("-" * 90)
    
    for layer_name, results in all_results.items():
        ridge_r2 = results['next_state']['ridge']['r2'] if 'next_state' in results else 0
        global_mlp_r2 = results['next_state']['mlp']['r2'] if 'next_state' in results else 0
        
        # Calculate time-step MLP performance
        time_step_r2 = 0
        if 'next_state' in results and 'time_step_mlp' in results['next_state']:
            time_step_results = results['next_state']['time_step_mlp']
            if time_step_results:
                time_step_r2 = np.mean([r['r2'] for r in time_step_results.values()])
        
        action_r2 = np.mean([r['r2'] for r in results['action_prediction'].values()]) if 'action_prediction' in results else 0
        zone_r2 = np.mean(list(results['zone_relationships']['distance_encoding'].values())) if 'zone_relationships' in results else 0
        
        print(f"{layer_name:<15} {ridge_r2:<12.3f} {global_mlp_r2:<15.3f} {time_step_r2:<18.3f} {action_r2:<12.3f} {zone_r2:<12.3f}")
        
        # Print improvements
        if 'next_state' in results:
            if 'improvement' in results['next_state']:
                improvement = results['next_state']['improvement']
                print(f"  → Global MLP improvement: {improvement:+.3f}")
            
            if 'time_step_mlp' in results['next_state'] and time_step_results:
                time_step_improvement = time_step_r2 - global_mlp_r2
                best_time_step = max([r['r2'] for r in time_step_results.values()])
                print(f"  → Time-step vs Global: {time_step_improvement:+.3f} (best: {best_time_step:.3f})")
    
    # Save summary
    summary_df = pd.DataFrame(all_results).T
    summary_df.to_csv('world_model_analysis_plots/world_model_analysis_summary.csv')
    print(f"\nSummary saved to: world_model_analysis_plots/world_model_analysis_summary.csv")

if __name__ == '__main__':
    main() 