#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

def probe_color_locations():
    """Probe the model's internal representations for color location encoding."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("=== Probing Color Location Representations ===")
    
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
    
    # Hook into different network layers
    ltl_features = []
    env_features = []
    policy_features = []
    
    def ltl_hook(mod, inp, out):
        if hasattr(out, '__iter__') and len(out) > 1:
            h_n = out[1]  # Hidden state
        else:
            h_n = out
        arr = h_n.detach().squeeze().cpu().numpy()
        ltl_features.append(arr)
    
    def env_hook(mod, inp, out):
        arr = out.detach().squeeze().cpu().numpy()
        env_features.append(arr)
    
    def policy_hook(mod, inp, out):
        arr = out.detach().squeeze().cpu().numpy()
        policy_features.append(arr)
    
    # Register hooks
    handles = []
    if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
        handles.append(model.ltl_net.rnn.register_forward_hook(ltl_hook))
    if hasattr(model.env_net, 'mlp'):
        handles.append(model.env_net.mlp.register_forward_hook(env_hook))
    if hasattr(model, 'policy_net') and hasattr(model.policy_net, 'mlp'):
        handles.append(model.policy_net.mlp.register_forward_hook(policy_hook))
    
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
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    rollout_env.close()
    
    # Convert to arrays
    true_blue_positions = np.array(true_blue_positions)
    true_green_positions = np.array(true_green_positions)
    agent_positions = np.array(agent_positions)
    
    print(f"Collected {len(true_blue_positions)} samples")
    print(f"Blue positions shape: {true_blue_positions.shape}")
    print(f"Green positions shape: {true_green_positions.shape}")
    print(f"Agent positions shape: {agent_positions.shape}")
    
    # Analyze each network
    networks = [
        ("LTL Network", ltl_features),
        ("Environment Network", env_features),
        ("Policy Network", policy_features)
    ]
    
    results = {}
    
    for network_name, features in networks:
        if not features:
            print(f"\n{network_name}: No features collected")
            continue
            
        print(f"\n=== {network_name} Analysis ===")
        X = np.array(features)
        print(f"Feature shape: {X.shape}")
        
        # Ensure we have the same number of samples
        min_len = min(len(X), len(true_blue_positions), len(true_green_positions))
        X = X[:min_len]
        blue_pos = true_blue_positions[:min_len]
        green_pos = true_green_positions[:min_len]
        agent_pos = agent_positions[:min_len]
        
        # Probe for blue position prediction
        blue_probe = LinearRegression()
        blue_probe.fit(X, blue_pos)
        blue_pred = blue_probe.predict(X)
        blue_mse = mean_squared_error(blue_pos, blue_pred)
        blue_corr = np.corrcoef(blue_pos.flatten(), blue_pred.flatten())[0, 1]
        
        # Probe for green position prediction
        green_probe = LinearRegression()
        green_probe.fit(X, green_pos)
        green_pred = green_probe.predict(X)
        green_mse = mean_squared_error(green_pos, green_pred)
        green_corr = np.corrcoef(green_pos.flatten(), green_pred.flatten())[0, 1]
        
        # Probe for agent position prediction
        agent_probe = LinearRegression()
        agent_probe.fit(X, agent_pos)
        agent_pred = agent_probe.predict(X)
        agent_mse = mean_squared_error(agent_pos, agent_pred)
        agent_corr = np.corrcoef(agent_pos.flatten(), agent_pred.flatten())[0, 1]
        
        print(f"Blue position - MSE: {blue_mse:.4f}, Correlation: {blue_corr:.4f}")
        print(f"Green position - MSE: {green_mse:.4f}, Correlation: {green_corr:.4f}")
        print(f"Agent position - MSE: {agent_mse:.4f}, Correlation: {agent_corr:.4f}")
        
        results[network_name] = {
            'blue_mse': blue_mse, 'blue_corr': blue_corr,
            'green_mse': green_mse, 'green_corr': green_corr,
            'agent_mse': agent_mse, 'agent_corr': agent_corr,
            'blue_probe': blue_probe, 'green_probe': green_probe, 'agent_probe': agent_probe
        }
    
    return results

if __name__ == '__main__':
    results = probe_color_locations()
    print("\n=== Summary ===")
    for network, metrics in results.items():
        print(f"{network}:")
        print(f"  Blue position correlation: {metrics['blue_corr']:.4f}")
        print(f"  Green position correlation: {metrics['green_corr']:.4f}")
        print(f"  Agent position correlation: {metrics['agent_corr']:.4f}") 