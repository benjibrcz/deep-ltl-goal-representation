#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

def extract_color_positions(env):
    """Extract the true positions of colors from the environment."""
    color_positions = {}
    if hasattr(env, 'zone_positions'):
        for zone_name, position in env.zone_positions.items():
            color = zone_name.split('_')[0]  # Extract color from zone name
            color_positions[color] = position
    return color_positions

def visualize_color_location_predictions():
    """Visualize true vs predicted color locations to understand the perfect correlations."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("=== Visualizing Color Location Predictions ===")
    
    # Setup model and environment
    sampler_fn = FixedSampler.partial(FORMULA)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()
    
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    
    # Hook into environment network
    env_features = []
    
    def env_hook(mod, inp, out):
        arr = out.detach().squeeze().cpu().numpy()
        env_features.append(arr)
    
    # Register hook
    handle = None
    if hasattr(model.env_net, 'mlp'):
        handle = model.env_net.mlp.register_forward_hook(env_hook)
    
    # Collect data
    true_blue_positions = []
    true_green_positions = []
    agent_positions = []
    
    rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    ret = rollout_env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    for step in trange(MAX_STEPS, desc="Collecting data"):
        # Get true color positions
        color_positions = extract_color_positions(rollout_env)
        if 'blue' in color_positions:
            true_blue_positions.append(color_positions['blue'])
        if 'green' in color_positions:
            true_green_positions.append(color_positions['green'])
        
        # Get agent position
        agent_positions.append(rollout_env.agent_pos[:2])
        
        # Get action (triggers forward pass and hooks)
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        ret = rollout_env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
        
        if done:
            break
    
    # Remove hook
    if handle:
        handle.remove()
    rollout_env.close()
    
    # Convert to arrays
    true_blue_positions = np.array(true_blue_positions)
    true_green_positions = np.array(true_green_positions)
    agent_positions = np.array(agent_positions)
    env_features = np.array(env_features)
    
    print(f"Collected {len(true_blue_positions)} samples")
    print(f"Environment features shape: {env_features.shape}")
    
    # Train probes
    min_len = min(len(env_features), len(true_blue_positions), len(true_green_positions))
    X = env_features[:min_len]
    blue_pos = true_blue_positions[:min_len]
    green_pos = true_green_positions[:min_len]
    
    # Train blue position probe
    blue_probe = LinearRegression()
    blue_probe.fit(X, blue_pos)
    blue_pred = blue_probe.predict(X)
    
    # Train green position probe
    green_probe = LinearRegression()
    green_probe.fit(X, green_pos)
    green_pred = green_probe.predict(X)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: True vs Predicted Blue X
    axes[0, 0].scatter(blue_pos[:, 0], blue_pred[:, 0], alpha=0.6)
    axes[0, 0].plot([blue_pos[:, 0].min(), blue_pos[:, 0].max()], 
                    [blue_pos[:, 0].min(), blue_pos[:, 0].max()], 'r--')
    axes[0, 0].set_xlabel('True Blue X')
    axes[0, 0].set_ylabel('Predicted Blue X')
    axes[0, 0].set_title('Blue X: True vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: True vs Predicted Blue Y
    axes[0, 1].scatter(blue_pos[:, 1], blue_pred[:, 1], alpha=0.6)
    axes[0, 1].plot([blue_pos[:, 1].min(), blue_pos[:, 1].max()], 
                    [blue_pos[:, 1].min(), blue_pos[:, 1].max()], 'r--')
    axes[0, 1].set_xlabel('True Blue Y')
    axes[0, 1].set_ylabel('Predicted Blue Y')
    axes[0, 1].set_title('Blue Y: True vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Blue positions over time
    axes[0, 2].plot(blue_pos[:, 0], label='True X', alpha=0.7)
    axes[0, 2].plot(blue_pos[:, 1], label='True Y', alpha=0.7)
    axes[0, 2].plot(blue_pred[:, 0], label='Pred X', alpha=0.7, linestyle='--')
    axes[0, 2].plot(blue_pred[:, 1], label='Pred Y', alpha=0.7, linestyle='--')
    axes[0, 2].set_xlabel('Time Step')
    axes[0, 2].set_ylabel('Position')
    axes[0, 2].set_title('Blue Position Over Time')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: True vs Predicted Green X
    axes[1, 0].scatter(green_pos[:, 0], green_pred[:, 0], alpha=0.6, color='green')
    axes[1, 0].plot([green_pos[:, 0].min(), green_pos[:, 0].max()], 
                    [green_pos[:, 0].min(), green_pos[:, 0].max()], 'r--')
    axes[1, 0].set_xlabel('True Green X')
    axes[1, 0].set_ylabel('Predicted Green X')
    axes[1, 0].set_title('Green X: True vs Predicted')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: True vs Predicted Green Y
    axes[1, 1].scatter(green_pos[:, 1], green_pred[:, 1], alpha=0.6, color='green')
    axes[1, 1].plot([green_pos[:, 1].min(), green_pos[:, 1].max()], 
                    [green_pos[:, 1].min(), green_pos[:, 1].max()], 'r--')
    axes[1, 1].set_xlabel('True Green Y')
    axes[1, 1].set_ylabel('Predicted Green Y')
    axes[1, 1].set_title('Green Y: True vs Predicted')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Green positions over time
    axes[1, 2].plot(green_pos[:, 0], label='True X', alpha=0.7, color='green')
    axes[1, 2].plot(green_pos[:, 1], label='True Y', alpha=0.7, color='green')
    axes[1, 2].plot(green_pred[:, 0], label='Pred X', alpha=0.7, color='green', linestyle='--')
    axes[1, 2].plot(green_pred[:, 1], label='Pred Y', alpha=0.7, color='green', linestyle='--')
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Position')
    axes[1, 2].set_title('Green Position Over Time')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('color_location_predictions.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to color_location_predictions.png")
    plt.close()
    
    # Print detailed statistics
    print("\n=== Detailed Statistics ===")
    print(f"Blue X - MSE: {np.mean((blue_pos[:, 0] - blue_pred[:, 0])**2):.6f}")
    print(f"Blue Y - MSE: {np.mean((blue_pos[:, 1] - blue_pred[:, 1])**2):.6f}")
    print(f"Green X - MSE: {np.mean((green_pos[:, 0] - green_pred[:, 0])**2):.6f}")
    print(f"Green Y - MSE: {np.mean((green_pos[:, 1] - green_pred[:, 1])**2):.6f}")
    
    print(f"\nBlue X correlation: {np.corrcoef(blue_pos[:, 0], blue_pred[:, 0])[0, 1]:.6f}")
    print(f"Blue Y correlation: {np.corrcoef(blue_pos[:, 1], blue_pred[:, 1])[0, 1]:.6f}")
    print(f"Green X correlation: {np.corrcoef(green_pos[:, 0], green_pred[:, 0])[0, 1]:.6f}")
    print(f"Green Y correlation: {np.corrcoef(green_pos[:, 1], green_pred[:, 1])[0, 1]:.6f}")
    
    # Check if positions are constant
    print(f"\nBlue position variance: {np.var(blue_pos, axis=0)}")
    print(f"Green position variance: {np.var(green_pos, axis=0)}")
    
    # Check if predictions are constant
    print(f"Blue prediction variance: {np.var(blue_pred, axis=0)}")
    print(f"Green prediction variance: {np.var(green_pred, axis=0)}")

if __name__ == '__main__':
    visualize_color_location_predictions() 