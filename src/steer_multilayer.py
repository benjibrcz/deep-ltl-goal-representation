#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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

def train_multilayer_probes(model, env, sampler_fn):
    """Train probes for both LTL and environment networks"""
    print("Training multi-layer probes...")
    
    # Hook into both networks
    ltl_feats = []
    env_feats = []
    
    def ltl_hook_fn(mod, inp, out):
        h_n = out[1]
        arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy()
        ltl_feats.append(arr)
    
    def env_hook_fn(mod, inp, out):
        arr = out.detach().squeeze(0).cpu().numpy()
        env_feats.append(arr)
    
    # Register hooks
    ltl_handle = model.ltl_net.rnn.register_forward_hook(ltl_hook_fn)
    env_handle = model.env_net.register_forward_hook(env_hook_fn)
    
    # Create agent and collect data
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    
    rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    ret = rollout_env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    labels = []
    for step in range(200):  # Shorter rollout for probe training
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        # Get current goal
        seq = getattr(agent, "sequence", None)
        if seq and len(seq) > 0:
            goal_set = seq[0][0]
            if len(goal_set) == 1:
                assignment = next(iter(goal_set))
                true_props = {p for p, v in assignment.assignment if v}
                if len(true_props) == 1:
                    prop = next(iter(true_props))
                    if prop in ['blue', 'green']:
                        labels.append(1 if prop == 'blue' else 0)
                    else:
                        labels.append(-1)
                else:
                    labels.append(-1)
            else:
                labels.append(-1)
        else:
            labels.append(-1)
            
        ret = rollout_env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
        
        if done:
            break
    
    ltl_handle.remove()
    env_handle.remove()
    rollout_env.close()
    
    # Train LTL probe
    X_ltl = np.array(ltl_feats)
    y = np.array(labels)
    valid_idxs = (y != -1)
    if len(X_ltl) > len(y):
        X_ltl = X_ltl[:len(y)]
    X_ltl, y_ltl = X_ltl[valid_idxs], y[valid_idxs]
    
    if len(np.unique(y_ltl)) <= 1:
        print("Warning: Only one class found for LTL probe training")
        ltl_probe = None
    else:
        ltl_probe = LogisticRegression(max_iter=1000)
        ltl_probe.fit(X_ltl, y_ltl)
        print(f"LTL probe accuracy: {ltl_probe.score(X_ltl, y_ltl):.2%}")
    
    # Train environment probe
    X_env = np.array(env_feats)
    if len(X_env) > len(y):
        X_env = X_env[:len(y)]
    X_env, y_env = X_env[valid_idxs], y[valid_idxs]
    
    if len(np.unique(y_env)) <= 1:
        print("Warning: Only one class found for environment probe training")
        env_probe = None
    else:
        env_probe = LogisticRegression(max_iter=1000)
        env_probe.fit(X_env, y_env)
        print(f"Environment probe accuracy: {env_probe.score(X_env, y_env):.2%}")
    
    return ltl_probe, env_probe

def steer_multilayer(model, env, sampler_fn, ltl_probe, env_probe, steering_strength=1.0):
    """Steer both LTL and environment networks simultaneously"""
    print(f"Running multi-layer steering with strength {steering_strength}...")
    
    # Hook into both networks for steering
    ltl_steered = []
    env_steered = []
    
    def ltl_steer_hook_fn(mod, inp, out):
        h_n = out[1]
        # Apply steering to LTL hidden state
        if ltl_probe is not None:
            steer_vector = ltl_probe.coef_[0] * steering_strength
            h_n_steered = h_n + torch.tensor(steer_vector, dtype=h_n.dtype, device=h_n.device).unsqueeze(0).unsqueeze(0)
            out = (out[0], h_n_steered)
        ltl_steered.append(1)
    
    def env_steer_hook_fn(mod, inp, out):
        # Apply steering to environment features
        if env_probe is not None:
            steer_vector = env_probe.coef_[0] * steering_strength
            out_steered = out + torch.tensor(steer_vector, dtype=out.dtype, device=out.device).unsqueeze(0)
            out = out_steered
        env_steered.append(1)
    
    # Register steering hooks
    ltl_handle = model.ltl_net.rnn.register_forward_hook(ltl_steer_hook_fn)
    env_handle = model.env_net.register_forward_hook(env_steer_hook_fn)
    
    # Create agent and run rollout
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    
    rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    ret = rollout_env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    # Track goals and steering effects
    goals = []
    steering_interventions = 0
    
    for step in trange(MAX_STEPS, desc="Multi-layer steering rollout"):
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        # Count steering interventions
        if len(ltl_steered) > step or len(env_steered) > step:
            steering_interventions += 1
        
        # Get current goal
        seq = getattr(agent, "sequence", None)
        if seq and len(seq) > 0:
            goal_set = seq[0][0]
            if len(goal_set) == 1:
                assignment = next(iter(goal_set))
                true_props = {p for p, v in assignment.assignment if v}
                if len(true_props) == 1:
                    prop = next(iter(true_props))
                    goals.append(prop)
                else:
                    goals.append('other')
            else:
                goals.append('other')
        else:
            goals.append('none')
            
        ret = rollout_env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
        
        if done:
            break
    
    ltl_handle.remove()
    env_handle.remove()
    rollout_env.close()
    
    # Analyze results
    blue_count = sum(1 for g in goals if g == 'blue')
    green_count = sum(1 for g in goals if g == 'green')
    total_goals = blue_count + green_count
    
    if total_goals > 0:
        blue_ratio = blue_count / total_goals
        print(f"Multi-layer steering results:")
        print(f"  Blue goals: {blue_count}")
        print(f"  Green goals: {green_count}")
        print(f"  Blue ratio: {blue_ratio:.2%}")
        print(f"  Steering interventions: {steering_interventions}")
    else:
        print("No goals detected during steering")
        blue_ratio = 0.0
    
    return {
        'blue_count': blue_count,
        'green_count': green_count,
        'blue_ratio': blue_ratio,
        'steering_interventions': steering_interventions,
        'goals': goals
    }

def main():
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("=== Multi-Layer Steering Analysis ===")
    print(f"Environment: {ENV}")
    print(f"Experiment: {EXP}")
    print(f"Formula: {FORMULA}")
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
    
    # Create environment
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    
    # Train probes
    ltl_probe, env_probe = train_multilayer_probes(model, env, sampler_fn)
    
    if ltl_probe is None and env_probe is None:
        print("Error: No probes could be trained")
        return
    
    # Test different steering strengths
    steering_strengths = [0.5, 1.0, 2.0, 5.0]
    results = {}
    
    print(f"\n=== Testing Multi-Layer Steering ===")
    for strength in steering_strengths:
        print(f"\nTesting steering strength: {strength}")
        result = steer_multilayer(model, env, sampler_fn, ltl_probe, env_probe, strength)
        results[strength] = result
    
    # Baseline (no steering)
    print(f"\n=== Baseline (No Steering) ===")
    baseline_result = steer_multilayer(model, env, sampler_fn, None, None, 0.0)
    results[0.0] = baseline_result
    
    # Summary
    print(f"\n=== Multi-Layer Steering Summary ===")
    print("Steering Strength | Blue Ratio | Interventions")
    print("-" * 45)
    for strength in sorted(results.keys()):
        result = results[strength]
        print(f"{strength:15.1f} | {result['blue_ratio']:10.2%} | {result['steering_interventions']:13d}")
    
    # Visualization
    strengths = sorted(results.keys())
    blue_ratios = [results[s]['blue_ratio'] for s in strengths]
    interventions = [results[s]['steering_interventions'] for s in strengths]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Blue ratio plot
    ax1.plot(strengths, blue_ratios, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    ax1.set_xlabel('Steering Strength')
    ax1.set_ylabel('Blue Goal Ratio')
    ax1.set_title('Multi-Layer Steering: Blue Goal Ratio')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Interventions plot
    ax2.plot(strengths, interventions, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Steering Strength')
    ax2.set_ylabel('Steering Interventions')
    ax2.set_title('Multi-Layer Steering: Interventions')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multilayer_steering_results.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to multilayer_steering_results.png")
    plt.show()
    
    env.close()
    print("\n=== Multi-Layer Steering Complete ===")

if __name__ == '__main__':
    main() 