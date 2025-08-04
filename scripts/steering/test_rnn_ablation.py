#!/usr/bin/env python3
"""
Test script for LTL RNN layer ablation

This script implements ablation tests specifically on the LTL RNN layer to see if 
removing RNN features causally affects goal-directed behavior.
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

def test_rnn_ablation():
    """Test if ablating LTL RNN features affects goal-directed behavior"""
    print("=== Testing LTL RNN Layer Ablation ===\n")
    
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
    print(f"LTL RNN layer exists: {hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None}")

    # Set up env and agent for data collection
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    search = ExhaustiveSearch(model, set(props), num_loops=2)
    agent = Agent(model, search=search, propositions=set(props), verbose=False)

    # Collect activations and labels from LTL RNN layer
    print("\n--- Collecting LTL RNN activations ---")
    activations, labels = collect_rnn_activations_and_labels(model, env, agent, max_steps=500)
    print(f"Collected {len(activations)} activations and {len(labels)} labels from LTL RNN.")
    print(f"First 10 labels: {labels[:10]}")

    # Filter out None labels
    valid_idxs = [i for i, label in enumerate(labels) if label is not None]
    X = np.array([activations[i] for i in valid_idxs])
    y_str = [labels[i] for i in valid_idxs]
    print(f"Filtered to {len(y_str)} valid samples for probe training.")

    # Encode labels as integers
    unique_goals = sorted(set(y_str))
    goal_to_idx = {goal: idx for idx, goal in enumerate(unique_goals)}
    y = np.array([goal_to_idx[label] for label in y_str])
    print(f"Goal to index mapping: {goal_to_idx}")

    # Train logistic regression probe
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"LTL RNN probe accuracy: {acc:.3f}")
    print(f"Probe weights shape: {clf.coef_.shape}")
    
    # Get top features for ablation
    abs_weights = np.abs(clf.coef_[0])
    topk = 10
    topk_indices = np.argsort(abs_weights)[-topk:]
    print(f"LTL RNN top {topk} feature indices: {topk_indices}")
    print(f"Top feature weights: {abs_weights[topk_indices]}")

    # Test baseline behavior
    print("\n--- Testing Baseline Behavior ---")
    baseline_results = run_rollouts(model, num_rollouts=3)
    print(f"Baseline goal completion rate: {baseline_results['completion_rate']:.3f}")
    print(f"Baseline average steps: {baseline_results['avg_steps']:.1f}")
    print(f"Baseline goals visited: {baseline_results['avg_goals']:.1f}")
    
    # Test with RNN ablation
    print("\n--- Testing LTL RNN Ablation ---")
    
    # Create ablation hook for RNN
    def rnn_ablation_hook(module, inp, out):
        # RNN output is (packed_sequence, h_n) where h_n is the final hidden state
        if isinstance(out, tuple):
            packed_seq, h_n = out
            # Clone the hidden state and zero out top features
            h_n_ablated = h_n.clone()
            h_n_ablated[..., topk_indices] = 0.0
            return (packed_seq, h_n_ablated)
        else:
            # If not tuple, just zero out the output
            out_ablated = out.clone()
            out_ablated[..., topk_indices] = 0.0
            return out_ablated
    
    # Register ablation hook
    if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
        handle = model.ltl_net.rnn.register_forward_hook(rnn_ablation_hook)
        print("Registered RNN ablation hook")
        
        # Test ablated behavior
        ablated_results = run_rollouts(model, num_rollouts=3)
        print(f"Ablated goal completion rate: {ablated_results['completion_rate']:.3f}")
        print(f"Ablated average steps: {ablated_results['avg_steps']:.1f}")
        print(f"Ablated goals visited: {ablated_results['avg_goals']:.1f}")
        
        # Calculate changes
        completion_change = ablated_results['completion_rate'] - baseline_results['completion_rate']
        steps_change = ablated_results['avg_steps'] - baseline_results['avg_steps']
        goals_change = ablated_results['avg_goals'] - baseline_results['avg_goals']
        
        print(f"\n--- Ablation Effects ---")
        print(f"Goal completion change: {completion_change:+.3f}")
        print(f"Steps change: {steps_change:+.1f}")
        print(f"Goals visited change: {goals_change:+.1f}")
        
        # Remove hook
        handle.remove()
        print("Removed RNN ablation hook")
    else:
        print("LTL RNN layer not found!")

def collect_rnn_activations_and_labels(model, env, agent, max_steps=1000):
    """Collect activations and labels from LTL RNN layer"""
    activations = []
    labels = []
    
    def hook_fn(mod, inp, out):
        # RNN output is (packed_sequence, h_n) where h_n is the final hidden state
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
    test_rnn_ablation() 