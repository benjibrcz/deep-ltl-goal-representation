#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0
FORMULA = "GF blue & GF green"
N_WORLDS = 100
MAX_STEPS = 50

# Color mapping
COLOR_MAP = {
    "blue": 0, "green": 1, "yellow": 2, "pink": 3, "magenta": 4, "orange": 5, "red": 6
}
COLOR_RGB = {
    0: "#4C72B0",  # blue
    1: "#55A868",  # green  
    2: "#E1C027",  # yellow
    3: "#BB78A5",  # pink
    4: "#C44E52",  # magenta
    5: "#FF7F0E",  # orange
    6: "#D62728",  # red
}
COLOR_NAMES = ["blue", "green", "yellow", "pink", "magenta", "orange", "red"]

class ZoneProbe:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.layer = dict(model.named_modules())[layer_name]
        self.features = []
        self.positions = []
        self.colors = []
        self.world_info = []
        
    def hook_fn(self, module, input, output):
        """Hook function to capture layer activations"""
        if hasattr(output, 'detach'):
            feat = output.detach().cpu().squeeze().numpy()
        else:
            feat = output.squeeze().cpu().numpy()
        self.features.append(feat)
    
    def collect_data(self, env, sampler_fn):
        """Collect zone data and activations from multiple worlds"""
        print(f"Collecting data from layer: {self.layer_name}")
        
        # Register hook
        handle = self.layer.register_forward_hook(self.hook_fn)
        
        # Create agent for consistent behavior
        props = set(env.get_propositions())
        search = ExhaustiveSearch(self.model, props, num_loops=2)
        agent = Agent(self.model, search=search, propositions=props, verbose=False)
        
        for world_idx in trange(N_WORLDS, desc="Collecting worlds"):
            # Reset environment with new world
            ret = env.reset(seed=SEED + world_idx)
            obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
            agent.reset()
            
            # Extract zone information first
            self.extract_zone_info(env, world_idx)
            
            # Get agent action to trigger forward pass
            try:
                action = agent.get_action(obs, info, deterministic=True)
            except:
                # Fallback if agent fails
                action = env.action_space.sample()
        
        handle.remove()
        
        # Convert to arrays
        self.features = np.array(self.features)
        self.positions = np.array(self.positions)
        self.colors = np.array(self.colors)
        
        print(f"Collected {len(self.features)} samples")
        print(f"Feature shape: {self.features.shape}")
        print(f"Position shape: {self.positions.shape}")
        print(f"Color shape: {self.colors.shape}")
        
        # Verify shapes match
        if len(self.features) != len(self.positions):
            print(f"WARNING: Feature count ({len(self.features)}) doesn't match position count ({len(self.positions)})")
            # Truncate to match
            min_len = min(len(self.features), len(self.positions))
            self.features = self.features[:min_len]
            self.positions = self.positions[:min_len]
            self.colors = self.colors[:min_len]
            print(f"Truncated to {min_len} samples")
    
    def extract_zone_info(self, env, world_idx):
        """Extract zone positions and colors from environment"""
        try:
            # Try to get zone positions from environment
            if hasattr(env, 'zone_positions'):
                zone_pos = env.zone_positions
                if isinstance(zone_pos, dict):
                    positions = []
                    colors = []
                    for key, pos in sorted(zone_pos.items()):
                        positions.extend(pos[:2])  # Take x, y coordinates
                        # Extract color from key
                        color_name = None
                        for color in COLOR_MAP:
                            if color in key.lower():
                                color_name = color
                                break
                        if color_name:
                            colors.append(COLOR_MAP[color_name])
                        else:
                            colors.append(0)  # Default to blue
                    
                    self.positions.append(positions)
                    self.colors.append(colors)
                    self.world_info.append({
                        'world_idx': world_idx,
                        'num_zones': len(colors),
                        'positions': positions,
                        'colors': colors
                    })
                else:
                    # Fallback: use layout information
                    self.extract_from_layout(env, world_idx)
            else:
                # Fallback: use layout information
                self.extract_from_layout(env, world_idx)
                
        except Exception as e:
            print(f"Error extracting zone info for world {world_idx}: {e}")
            # Add dummy data to maintain alignment
            self.positions.append([0, 0, 0, 0])  # 2 zones at origin
            self.colors.append([0, 1])  # blue, green
            self.world_info.append({
                'world_idx': world_idx,
                'num_zones': 2,
                'positions': [0, 0, 0, 0],
                'colors': [0, 1]
            })
    
    def extract_from_layout(self, env, world_idx):
        """Extract zone info from environment layout"""
        try:
            layout = env.task.world_info.layout
            positions = []
            colors = []
            
            for key, val in sorted(layout.items()):
                if 'zone' in key.lower():
                    # Extract position
                    if isinstance(val, dict) and 'pos' in val:
                        pos = val['pos']
                        positions.extend(pos[:2])
                    elif isinstance(val, np.ndarray):
                        positions.extend(val[:2].tolist())
                    else:
                        positions.extend([0, 0])
                    
                    # Extract color
                    color_name = None
                    for color in COLOR_MAP:
                        if color in key.lower():
                            color_name = color
                            break
                    if color_name:
                        colors.append(COLOR_MAP[color_name])
                    else:
                        colors.append(0)  # Default to blue
            
            if not positions:  # No zones found
                positions = [0, 0, 0, 0]
                colors = [0, 1]
            
            self.positions.append(positions)
            self.colors.append(colors)
            self.world_info.append({
                'world_idx': world_idx,
                'num_zones': len(colors),
                'positions': positions,
                'colors': colors
            })
            
        except Exception as e:
            print(f"Error extracting from layout for world {world_idx}: {e}")
            self.positions.append([0, 0, 0, 0])
            self.colors.append([0, 1])
            self.world_info.append({
                'world_idx': world_idx,
                'num_zones': 2,
                'positions': [0, 0, 0, 0],
                'colors': [0, 1]
            })
    
    def train_probes(self):
        """Train position and color prediction probes"""
        print("Training position and color probes...")
        
        # Train position probe (Ridge regression)
        self.position_probe = Ridge(alpha=1.0)
        self.position_probe.fit(self.features, self.positions)
        
        # Calculate position prediction accuracy
        pos_pred = self.position_probe.predict(self.features)
        pos_mse = mean_squared_error(self.positions, pos_pred)
        print(f"Position MSE: {pos_mse:.4f}")
        
        # Train color probes (Logistic regression for each zone)
        self.color_probes = []
        color_accuracies = []
        
        for zone_idx in range(self.colors.shape[1]):
            zone_colors = self.colors[:, zone_idx]
            unique_colors = np.unique(zone_colors)
            
            if len(unique_colors) > 1:
                # Train classifier
                clf = LogisticRegression(max_iter=1000, random_state=SEED)
                clf.fit(self.features, zone_colors)
                self.color_probes.append(clf)
                
                # Calculate accuracy
                pred_colors = clf.predict(self.features)
                acc = accuracy_score(zone_colors, pred_colors)
                color_accuracies.append(acc)
                print(f"Zone {zone_idx} color accuracy: {acc:.3f}")
            else:
                # Only one color present
                self.color_probes.append(None)
                color_accuracies.append(1.0)
                print(f"Zone {zone_idx} color accuracy: 1.000 (single class)")
        
        return {
            'position_mse': pos_mse,
            'color_accuracies': color_accuracies,
            'avg_color_accuracy': np.mean(color_accuracies)
        }
    
    def predict_zones(self, features):
        """Predict zone positions and colors from features"""
        # Predict positions
        pos_pred = self.position_probe.predict(features)
        
        # Predict colors
        color_pred = []
        for zone_idx, probe in enumerate(self.color_probes):
            if probe is not None:
                zone_colors = probe.predict(features)
                color_pred.append(zone_colors)
            else:
                # Use most common color for this zone
                most_common = np.bincount(self.colors[:, zone_idx]).argmax()
                color_pred.append([most_common] * len(features))
        
        color_pred = np.array(color_pred).T
        
        return pos_pred, color_pred

def find_network_layers(model):
    """Find all hookable layers in the model"""
    layers = []
    
    # Environment network layers
    if hasattr(model, 'env_net'):
        for name, module in model.env_net.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                layers.append(f"env_net.{name}")
    
    # LTL network layers
    if hasattr(model, 'ltl_net'):
        for name, module in model.ltl_net.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                layers.append(f"ltl_net.{name}")
    
    # Policy/Actor layers
    if hasattr(model, 'actor'):
        for name, module in model.actor.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                layers.append(f"actor.{name}")
    
    return layers

def create_zone_visualization(probe, world_idx=0, save_path=None):
    """Create visualization of true vs predicted zones"""
    # Get true and predicted data for one world
    true_pos = probe.positions[world_idx]
    true_col = probe.colors[world_idx]
    
    features = probe.features[world_idx:world_idx+1]
    pred_pos, pred_col = probe.predict_zones(features)
    pred_pos = pred_pos[0]
    pred_col = pred_col[0]
    
    # Reshape positions to (num_zones, 2)
    num_zones = len(true_col)
    true_pos_reshaped = np.array(true_pos).reshape(num_zones, 2)
    pred_pos_reshaped = np.array(pred_pos).reshape(num_zones, 2)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot true zones
    ax1.set_title("True Zones", fontsize=14, fontweight='bold')
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    for i, (pos, col) in enumerate(zip(true_pos_reshaped, true_col)):
        color = COLOR_RGB.get(col, "#000000")
        circle = patches.Circle(pos, 0.3, facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], f"Zone {i}\n{COLOR_NAMES[col]}", 
                ha='center', va='center', fontweight='bold', color='white')
    
    # Plot predicted zones
    ax2.set_title("Predicted Zones", fontsize=14, fontweight='bold')
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    for i, (pos, col) in enumerate(zip(pred_pos_reshaped, pred_col)):
        color = COLOR_RGB.get(col, "#000000")
        circle = patches.Circle(pos, 0.3, facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(pos[0], pos[1], f"Zone {i}\n{COLOR_NAMES[col]}", 
                ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def create_comparison_plot(probes, save_path=None):
    """Create comparison plot of all layers"""
    layers = list(probes.keys())
    position_mses = [probes[layer]['position_mse'] for layer in layers]
    color_accuracies = [probes[layer]['avg_color_accuracy'] for layer in layers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Position MSE comparison
    bars1 = ax1.bar(range(len(layers)), position_mses, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Network Layer')
    ax1.set_ylabel('Position MSE')
    ax1.set_title('Zone Position Prediction Error')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([layer.split('.')[-1] for layer in layers], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, mse in zip(bars1, position_mses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{mse:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Color accuracy comparison
    bars2 = ax2.bar(range(len(layers)), color_accuracies, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Network Layer')
    ax2.set_ylabel('Average Color Accuracy')
    ax2.set_title('Zone Color Prediction Accuracy')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([layer.split('.')[-1] for layer in layers], rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars2, color_accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    plt.show()

def probe_during_rollout(probe, model, env, layer_name, world_idx=0, max_steps=500, snapshots=[0, 50, 100, 200, 300, 400], save_path=None):
    """Probe zone representations dynamically during a rollout with multiple snapshots."""
    print(f"\nProbing {layer_name} during rollout in world {world_idx}...")
    print(f"Rollout length: {max_steps} steps, snapshots at: {snapshots}")
    
    # Reset environment to a specific world
    ret = env.reset(seed=SEED + world_idx)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    
    # Prepare agent
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    agent.reset()
    
    # Prepare to collect activations
    layer = dict(model.named_modules())[layer_name]
    rollout_features = []
    agent_positions = []
    
    def hook_fn(module, input, output):
        if hasattr(output, 'detach'):
            feat = output.detach().cpu().squeeze().numpy()
        else:
            feat = output.squeeze().cpu().numpy()
        rollout_features.append(feat)
    
    handle = layer.register_forward_hook(hook_fn)
    
    # Rollout
    for t in range(max_steps):
        agent_positions.append(np.array(env.agent_pos[:2]))
        try:
            action = agent.get_action(obs, info, deterministic=True)
            if hasattr(action, 'cpu'):
                action = action.cpu().numpy()
            if hasattr(action, 'flatten'):
                action = action.flatten()
            action = np.asarray(action)
        except:
            action = env.action_space.sample()
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
        if done:
            break
    
    handle.remove()
    
    rollout_features = np.array(rollout_features)
    agent_positions = np.array(agent_positions)
    actual_steps = len(rollout_features)
    
    print(f"Actual rollout length: {actual_steps} steps")
    
    # Predict zones at each step
    pos_preds, col_preds = probe.predict_zones(rollout_features)
    num_zones = len(probe.colors[world_idx])
    true_pos = np.array(probe.positions[world_idx]).reshape(num_zones, 2)
    true_col = probe.colors[world_idx]
    
    # Filter snapshots to valid steps
    valid_snapshots = [s for s in snapshots if s < actual_steps]
    if actual_steps - 1 not in valid_snapshots:
        valid_snapshots.append(actual_steps - 1)  # Always include final step
    
    print(f"Valid snapshots: {valid_snapshots}")
    
    # Create multi-panel visualization
    n_snapshots = len(valid_snapshots)
    fig, axes = plt.subplots(2, n_snapshots, figsize=(4*n_snapshots, 8))
    if n_snapshots == 1:
        axes = axes.reshape(2, 1)
    
    # Plot true zones and trajectory (top row)
    for i, step in enumerate(valid_snapshots):
        ax = axes[0, i]
        ax.set_title(f"True Zones + Trajectory (t={step})", fontsize=12, fontweight='bold')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Plot true zones
        for j, (pos, col) in enumerate(zip(true_pos, true_col)):
            color = COLOR_RGB.get(col, "#000000")
            circle = patches.Circle(pos, 0.3, facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], f"Zone {j}\n{COLOR_NAMES[col]}", 
                   ha='center', va='center', fontweight='bold', color='white', fontsize=8)
        
        # Plot trajectory up to this step
        trajectory = agent_positions[:step+1]
        ax.plot(trajectory[:,0], trajectory[:,1], '-o', color='k', markersize=3, alpha=0.7)
        if len(trajectory) > 0:
            ax.plot(trajectory[0,0], trajectory[0,1], 'ko', markersize=6, label='start')
            ax.plot(trajectory[-1,0], trajectory[-1,1], 'ks', markersize=6, label='current')
        ax.legend(fontsize=8)
    
    # Plot predicted zones (bottom row)
    for i, step in enumerate(valid_snapshots):
        ax = axes[1, i]
        ax.set_title(f"Predicted Zones (t={step})", fontsize=12, fontweight='bold')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Plot predicted zones at this step
        pos_step = pos_preds[step].reshape(num_zones, 2)
        col_step = col_preds[step]
        for j, (pos, col) in enumerate(zip(pos_step, col_step)):
            color = COLOR_RGB.get(col, "#000000")
            circle = patches.Circle(pos, 0.3, facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], f"Zone {j}\n{COLOR_NAMES[col]}", 
                   ha='center', va='center', fontweight='bold', color='white', fontsize=8)
        
        # Plot agent position at this step
        if step < len(agent_positions):
            ax.plot(agent_positions[step,0], agent_positions[step,1], 'ko', markersize=8, label='agent')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved rollout probe visualization to {save_path}")
    plt.show()

def main():
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("=== Improved Zone Probing Analysis ===")
    print(f"Environment: {ENV}")
    print(f"Experiment: {EXP}")
    print(f"Formula: {FORMULA}")
    print(f"Number of worlds: {N_WORLDS}")
    print()
    
    # Load model
    print("Loading model...")
    sampler_fn = FixedSampler.partial(FORMULA)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()
    
    # Find hookable layers
    print("Finding hookable layers...")
    all_layers = find_network_layers(model)
    print(f"Found {len(all_layers)} hookable layers:")
    for layer in all_layers:
        print(f"  - {layer}")
    print()
    
    # Select layers to probe (focus on key layers)
    key_layers = [
        "env_net.mlp.0",  # First environment layer
        "env_net.mlp.1",  # Second environment layer  
        "env_net.mlp.2",  # Third environment layer
        "env_net.mlp.3",  # Fourth environment layer
        "ltl_net.rnn",    # LTL RNN
        "actor.enc.0",    # First actor layer
        "actor.enc.2",    # Third actor layer
    ]
    
    # Filter to only include layers that exist
    layers_to_probe = [layer for layer in key_layers if layer in all_layers]
    print(f"Probing {len(layers_to_probe)} key layers:")
    for layer in layers_to_probe:
        print(f"  - {layer}")
    print()
    
    # Create environment for data collection
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    
    # Probe each layer
    probe_results = {}
    
    for layer_name in layers_to_probe:
        print(f"\n{'='*50}")
        print(f"Probing layer: {layer_name}")
        print(f"{'='*50}")
        
        # Create probe
        probe = ZoneProbe(model, layer_name)
        
        # Collect data
        probe.collect_data(env, sampler_fn)
        
        # Train probes
        results = probe.train_probes()
        probe_results[layer_name] = results
        
        # Store probe for later use
        probe_results[layer_name]['probe'] = probe
        
        print(f"Results for {layer_name}:")
        print(f"  Position MSE: {results['position_mse']:.4f}")
        print(f"  Average Color Accuracy: {results['avg_color_accuracy']:.3f}")
        print()
    
    env.close()
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Comparison plot
    create_comparison_plot(probe_results, 'improved_zone_probe_comparison.png')
    
    # 2. Individual visualizations for best layers
    best_position_layer = min(probe_results.keys(), 
                            key=lambda x: probe_results[x]['position_mse'])
    best_color_layer = max(probe_results.keys(), 
                          key=lambda x: probe_results[x]['avg_color_accuracy'])
    
    print(f"\nBest position prediction: {best_position_layer}")
    print(f"Best color prediction: {best_color_layer}")
    
    create_zone_visualization(probe_results[best_position_layer]['probe'], 
                             save_path='best_position_prediction.png')
    create_zone_visualization(probe_results[best_color_layer]['probe'], 
                             save_path='best_color_prediction.png')
    
    # 3. Create detailed results table
    print("\nDetailed Results:")
    print("-" * 80)
    print(f"{'Layer':<20} {'Position MSE':<15} {'Color Acc':<12} {'Best Zone':<12}")
    print("-" * 80)
    
    for layer_name in layers_to_probe:
        results = probe_results[layer_name]
        best_zone_acc = max(results['color_accuracies'])
        print(f"{layer_name:<20} {results['position_mse']:<15.4f} "
              f"{results['avg_color_accuracy']:<12.3f} {best_zone_acc:<12.3f}")
    
    # 3. Probe during rollout for best position layer
    print("\nProbing zone representations during agent rollout...")
    probe_during_rollout(
        probe_results[best_position_layer]['probe'],
        model,
        env,
        best_position_layer,
        world_idx=0,
        max_steps=200,
        save_path='zone_probe_rollout_long.png'
    )
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- improved_zone_probe_comparison.png")
    print("- best_position_prediction.png") 
    print("- best_color_prediction.png")
    print("- zone_probe_rollout_long.png")

if __name__ == '__main__':
    main() 