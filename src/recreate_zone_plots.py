#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.patches as patches

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
MAX_STEPS = 1000
N_WORLDS = 10

def extract_color_positions(env):
    """Extract the true positions of colors from the environment."""
    color_positions = {}
    if hasattr(env, 'zone_positions'):
        for zone_name, position in env.zone_positions.items():
            color = zone_name.split('_')[0]  # Extract color from zone name
            if color not in color_positions:
                color_positions[color] = []
            color_positions[color].append(position)
    return color_positions

def train_position_probe(model, env, sampler_fn, layer_name):
    """Train a probe to predict zone positions from model activations."""
    print(f"Training position probe for layer: {layer_name}")
    
    # Get the layer to hook into
    layer = dict(model.named_modules())[layer_name]
    
    # Collect data
    features_list = []
    positions_list = []
    colors_list = []
    
    def hook_fn(module, input, output):
        if hasattr(output, 'detach'):
            feat = output.detach().cpu().squeeze().numpy()
        else:
            feat = output.squeeze().cpu().numpy()
        hook_fn.current_feature = feat
    
    handle = layer.register_forward_hook(hook_fn)
    
    # Create agent
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    
    for world_idx in trange(N_WORLDS, desc="Collecting data"):
        # Reset environment
        ret = env.reset(seed=SEED + world_idx)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        agent.reset()
        
        # Extract zone information
        color_positions = extract_color_positions(env)
        
        # Get agent action to trigger forward pass
        try:
            action = agent.get_action(obs, info, deterministic=True)
        except:
            action = env.action_space.sample()
        
        # Store data - one feature per world, multiple positions per world
        current_feature = hook_fn.current_feature
        for color, positions in color_positions.items():
            for pos in positions:
                features_list.append(current_feature)
                positions_list.append(pos)
                colors_list.append(color)
    
    handle.remove()
    
    # Convert to arrays
    features = np.array(features_list)
    positions = np.array(positions_list)
    colors = np.array(colors_list)
    
    print(f"Collected {len(features)} samples")
    print(f"Features shape: {features.shape}")
    print(f"Positions shape: {positions.shape}")
    print(f"Colors: {np.unique(colors)}")
    
    # Train position probe
    position_probe = Ridge(alpha=1.0)
    position_probe.fit(features, positions)
    
    # Calculate accuracy
    pos_pred = position_probe.predict(features)
    pos_mse = mean_squared_error(positions, pos_pred)
    print(f"Position MSE: {pos_mse:.4f}")
    
    return position_probe, features, positions, colors

def visualize_predictions(model, env, sampler_fn, layer_name, position_probe, world_idx=0):
    """Visualize true vs predicted zone positions."""
    print(f"\nVisualizing predictions for world {world_idx}...")
    
    # Reset environment to specific world
    ret = env.reset(seed=SEED + world_idx)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    
    # Get true zone positions
    color_positions = extract_color_positions(env)
    
    # Get agent position
    agent_pos = env.agent_pos[:2]
    
    # Get model activation for this world
    layer = dict(model.named_modules())[layer_name]
    
    def hook_fn(module, input, output):
        if hasattr(output, 'detach'):
            feat = output.detach().cpu().squeeze().numpy()
        else:
            feat = output.squeeze().cpu().numpy()
        hook_fn.feature = feat
    
    hook_fn.feature = None
    handle = layer.register_forward_hook(hook_fn)
    
    # Create agent and get action to trigger forward pass
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    agent.reset()
    
    try:
        action = agent.get_action(obs, info, deterministic=True)
    except:
        action = env.action_space.sample()
    
    handle.remove()
    
    if hook_fn.feature is None:
        print("Failed to get model activation")
        return
    
    # Predict zone positions for this world
    # We need to know how many positions to predict (total number of zones in this world)
    n_zones = sum(len(v) for v in color_positions.values())
    predicted_positions = position_probe.predict(np.tile(hook_fn.feature, (n_zones, 1)))
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot agent position
    ax.scatter([agent_pos[0]], [agent_pos[1]], c='black', s=100, marker='o', label='Agent', zorder=5)
    
    # Color mapping
    color_map = {
        'blue': '#1f77b4',
        'green': '#2ca02c', 
        'yellow': '#ff7f0e',
        'magenta': '#d62728',
        'red': '#e377c2',
        'orange': '#ff7f0e',
        'aqua': '#17becf'
    }
    
    # Plot true and predicted zones
    idx = 0
    for color, positions_list in color_positions.items():
        for i, pos in enumerate(positions_list):
            # True zone: filled circle
            ax.scatter([pos[0]], [pos[1]], c=color_map.get(color, 'gray'), 
                      s=200, marker='o', alpha=0.7, label=f'True {color}' if i == 0 else "")
            # Predicted zone: dashed circle
            pred_pos = predicted_positions[idx]
            ax.scatter([pred_pos[0]], [pred_pos[1]], c=color_map.get(color, 'gray'),
                      s=200, marker='o', facecolors='none', edgecolors=color_map.get(color, 'gray'),
                      linewidth=2, label=f'Predicted {color}' if i == 0 else "")
            circle = patches.Circle((pred_pos[0], pred_pos[1]), 0.18, color=color_map.get(color, 'gray'), fill=False, linestyle='dashed', linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            idx += 1
    
    # Set up plot
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'True vs Predicted Zone Positions (World {world_idx})\nLayer: {layer_name}')
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'zone_predictions_world_{world_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print(f"Agent position: {agent_pos}")
    print(f"True zone positions: {color_positions}")
    print(f"Predicted positions shape: {predicted_positions.shape}")
    print(f"First few predicted positions: {predicted_positions[:6]}")

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Load model
    sampler_fn = FixedSampler.partial(FORMULA)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, 0)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()
    
    # Create environment for data collection
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    
    # Train position probe on environment network
    layer_name = "env_net.mlp.3"  # Environment network layer
    position_probe, features, positions, colors = train_position_probe(model, env, sampler_fn, layer_name)
    
    # Visualize predictions for a few worlds
    for world_idx in range(3):
        visualize_predictions(model, env, sampler_fn, layer_name, position_probe, world_idx)
    
    env.close()

if __name__ == '__main__':
    main() 