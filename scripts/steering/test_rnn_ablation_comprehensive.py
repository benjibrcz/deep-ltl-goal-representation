#!/usr/bin/env python3
"""
Comprehensive LTL RNN layer ablation test

This script tests different ablation ratios to find the threshold where 
RNN feature ablation affects goal-directed behavior.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import random
import numpy as np
import torch
from tqdm import trange

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from envs.flatworld    import FlatWorld
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

ENV       = "PointLtl2-v0"
EXP       = "big_test"
SEED      = 1
MAX_STEPS = 1000
FORMULA   = "GF blue & GF green"

def test_comprehensive_rnn_ablation():
    """Test different ablation ratios on LTL RNN features"""
    print("=== Comprehensive LTL RNN Layer Ablation Test ===\n")
    
    # Set up
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
    print("Model loaded successfully")

    # Set up env and agent for data collection
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    search = ExhaustiveSearch(model, set(props), num_loops=2)
    agent = Agent(model, search=search, propositions=set(props), verbose=False)

    # Collect activations and train probe
    print("\n--- Collecting LTL RNN activations ---")
    activations, labels = collect_rnn_activations_and_labels(model, env, agent, max_steps=500)
    
    # Filter and train probe
    valid_idxs = [i for i, label in enumerate(labels) if label is not None]
    X = np.array([activations[i] for i in valid_idxs])
    y_str = [labels[i] for i in valid_idxs]
    unique_goals = sorted(set(y_str))
    goal_to_idx = {goal: idx for idx, goal in enumerate(unique_goals)}
    y = np.array([goal_to_idx[label] for label in y_str])
    
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"LTL RNN probe accuracy: {acc:.3f}")
    
    # Get feature importance
    abs_weights = np.abs(clf.coef_[0])
    feature_count = len(abs_weights)
    print(f"Total RNN features: {feature_count}")
    
    # Test baseline behavior
    print("\n--- Testing Baseline Behavior ---")
    baseline_results = run_rollouts(model, num_rollouts=3)
    print(f"Baseline goal completion rate: {baseline_results['completion_rate']:.3f}")
    print(f"Baseline average steps: {baseline_results['avg_steps']:.1f}")
    print(f"Baseline goals visited: {baseline_results['avg_goals']:.1f}")
    
    # Test different ablation ratios
    ablation_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\n--- Testing Different Ablation Ratios ---")
    print(f"{'Ratio':<8} {'Features':<10} {'Completion':<12} {'Steps':<8} {'Goals':<8} {'Change':<10}")
    print("-" * 60)
    
    results = []
    
    for ratio in ablation_ratios:
        num_features = int(ratio * feature_count)
        topk_indices = np.argsort(abs_weights)[-num_features:]
        
        # Create ablation hook
        def ablation_hook(module, inp, out):
            if isinstance(out, tuple):
                packed_seq, h_n = out
                h_n_ablated = h_n.clone()
                h_n_ablated[..., topk_indices] = 0.0
                return (packed_seq, h_n_ablated)
            else:
                out_ablated = out.clone()
                out_ablated[..., topk_indices] = 0.0
                return out_ablated
        
        # Register hook and test
        if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
            handle = model.ltl_net.rnn.register_forward_hook(ablation_hook)
            
            ablated_results = run_rollouts(model, num_rollouts=3)
            
            # Calculate changes
            completion_change = ablated_results['completion_rate'] - baseline_results['completion_rate']
            steps_change = ablated_results['avg_steps'] - baseline_results['avg_steps']
            goals_change = ablated_results['avg_goals'] - baseline_results['avg_goals']
            
            print(f"{ratio:<8.1f} {num_features:<10} {ablated_results['completion_rate']:<12.3f} "
                  f"{ablated_results['avg_steps']:<8.1f} {ablated_results['avg_goals']:<8.1f} "
                  f"{completion_change:+.3f}")
            
            results.append({
                'ratio': ratio,
                'num_features': num_features,
                'completion_rate': ablated_results['completion_rate'],
                'avg_steps': ablated_results['avg_steps'],
                'avg_goals': ablated_results['avg_goals'],
                'completion_change': completion_change,
                'steps_change': steps_change,
                'goals_change': goals_change
            })
            
            handle.remove()
    
    # Summary
    print(f"\n--- Summary ---")
    print(f"Baseline completion rate: {baseline_results['completion_rate']:.3f}")
    
    # Find threshold where behavior changes significantly
    significant_changes = [r for r in results if abs(r['completion_change']) > 0.1]
    if significant_changes:
        first_change = min(significant_changes, key=lambda x: x['ratio'])
        print(f"First significant change at {first_change['ratio']:.1f} ablation ratio "
              f"({first_change['num_features']} features)")
    else:
        print("No significant changes in completion rate observed")
    
    # Check for any changes in efficiency (steps)
    efficiency_changes = [r for r in results if abs(r['steps_change']) > 50]
    if efficiency_changes:
        print(f"Efficiency changes observed at ratios: {[r['ratio'] for r in efficiency_changes]}")

def collect_rnn_activations_and_labels(model, env, agent, max_steps=1000):
    """Collect activations and labels from LTL RNN layer"""
    activations = []
    labels = []
    
    def hook_fn(mod, inp, out):
        if isinstance(out, tuple):
            h_n = out[1]  # Final hidden state
            arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy()
        else:
            arr = out.detach().squeeze().cpu().numpy()
        activations.append(arr)
    
    # Register hook on LTL RNN
    if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
        handle = model.ltl_net.rnn.register_forward_hook(hook_fn)
    else:
        print("LTL RNN layer not found!")
        return [], []
    
    # Run rollout
    ret = env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    for step in trange(max_steps, desc="Collecting RNN activations"):
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        # Get current goal from agent's sequence
        seq = getattr(agent, "sequence", None)
        if seq and len(seq) > 0:
            goal_set = seq[0][0]
            if len(goal_set) == 1:
                try:
                    assignment = next(iter(goal_set))
                    true_props = {p for p, v in assignment.assignment if v}
                    if len(true_props) == 1:
                        prop = next(iter(true_props))
                        labels.append(prop)
                    else:
                        labels.append(None)
                except (StopIteration, AttributeError):
                    labels.append(None)
            else:
                labels.append(None)
        else:
            labels.append(None)
            
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
        
        if done:
            break
    
    # Remove hook
    handle.remove()
    
    return activations, labels

def run_rollouts(model, num_rollouts=3):
    """Run multiple rollouts and collect metrics"""
    sampler_fn = FixedSampler.partial(FORMULA)
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    search = ExhaustiveSearch(model, set(props), num_loops=2)
    agent = Agent(model, search=search, propositions=set(props), verbose=False)
    
    completion_count = 0
    total_steps = 0
    total_goals = 0
    
    for rollout in range(num_rollouts):
        ret = env.reset(seed=SEED + rollout)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        agent.reset()
        
        visited_goals = set()
        steps = 0
        
        for step in range(MAX_STEPS):
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
            # Track current goal
            seq = getattr(agent, "sequence", None)
            if seq and len(seq) > 0:
                goal_set = seq[0][0]
                if len(goal_set) == 1:
                    try:
                        assignment = next(iter(goal_set))
                        true_props = {p for p, v in assignment.assignment if v}
                        if len(true_props) == 1:
                            prop = next(iter(true_props))
                            visited_goals.add(prop)
                    except (StopIteration, AttributeError):
                        pass
            
            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret
            
            steps += 1
            
            if done:
                break
        
        # Check if all goals were visited (completion)
        if len(visited_goals) >= 2:  # For "GF blue & GF green", need at least 2 goals
            completion_count += 1
        
        total_steps += steps
        total_goals += len(visited_goals)
    
    env.close()
    
    return {
        'completion_rate': completion_count / num_rollouts,
        'avg_steps': total_steps / num_rollouts,
        'avg_goals': total_goals / num_rollouts
    }

if __name__ == "__main__":
    test_comprehensive_rnn_ablation() 