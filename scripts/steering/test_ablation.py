#!/usr/bin/env python3
"""
Test script for goal representation ablation

This script implements a simple ablation test to see if removing goal-related
features causally affects goal-directed behavior.
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

def test_ablation():
    """Test if ablating features affects goal-directed behavior"""
    print("=== Testing Goal Representation Ablation ===\n")
    
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

    # Collect activations and labels from policy_mlp_0 (first policy layer)
    activations, labels = collect_layer_activations_and_labels(model, env, agent, 'policy_mlp_0', max_steps=500)
    print(f"Collected {len(activations)} activations and {len(labels)} labels from policy_mlp_0.")
    print(f"First 10 labels: {labels[:10]}")
    env.close()
    
    # Collect activations and labels from policy_mlp_1 (second policy layer)
    activations, labels = collect_layer_activations_and_labels(model, env, agent, 'policy_mlp_1', max_steps=500)
    print(f"Collected {len(activations)} activations and {len(labels)} labels from policy_mlp_1.")
    print(f"First 10 labels: {labels[:10]}")

    # Test baseline behavior
    print("\n--- Testing Baseline Behavior ---")
    baseline_results = run_rollouts(model, num_rollouts=3)
    print(f"Baseline goal completion rate: {baseline_results['completion_rate']:.3f}")
    print(f"Baseline average steps: {baseline_results['avg_steps']:.1f}")
    print(f"Baseline goals visited: {baseline_results['avg_goals']:.1f}")
    
    # Test with different ablation strategies
    ablation_tests = [
        ("No ablation (control)", 0.0),
        ("Random 25% ablation", 0.25),
        ("Random 50% ablation", 0.50),
        ("Random 75% ablation", 0.75),
    ]
    
    for test_name, ablation_ratio in ablation_tests:
        print(f"\n--- Testing {test_name} ---")
        
        # For now, we'll simulate ablation effects
        # In a real implementation, you'd hook into the model layers
        simulated_results = simulate_ablation_effect(baseline_results, ablation_ratio)
        
        print(f"Simulated goal completion rate: {simulated_results['completion_rate']:.3f}")
        print(f"Simulated average steps: {simulated_results['avg_steps']:.1f}")
        print(f"Simulated goals visited: {simulated_results['avg_goals']:.1f}")
        
        # Calculate changes
        completion_change = simulated_results['completion_rate'] - baseline_results['completion_rate']
        steps_change = simulated_results['avg_steps'] - baseline_results['avg_steps']
        
        print(f"Goal completion change: {completion_change:+.3f}")
        print(f"Steps change: {steps_change:+.1f}")

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
    print(f"Probe accuracy: {acc:.3f}")
    print(f"Probe weights shape: {clf.coef_.shape}")
    print(f"Probe weights (first 10): {clf.coef_[0][:10]}")

    # --- Step 1: Train probe and get top 10 features for env_net_mlp_0 ---
    activations_e0, labels_e0 = collect_layer_activations_and_labels(model, env, agent, 'env_net_mlp_0', max_steps=500)
    valid_idxs_e0 = [i for i, label in enumerate(labels_e0) if label is not None]
    Xe0 = np.array([activations_e0[i] for i in valid_idxs_e0])
    y_str_e0 = [labels_e0[i] for i in valid_idxs_e0]
    unique_goals_e0 = sorted(set(y_str_e0))
    goal_to_idx_e0 = {goal: idx for idx, goal in enumerate(unique_goals_e0)}
    ye0 = np.array([goal_to_idx_e0[label] for label in y_str_e0])
    clf_e0 = LogisticRegression(max_iter=1000)
    clf_e0.fit(Xe0, ye0)
    abs_weights_e0 = np.abs(clf_e0.coef_[0])
    topk_e0 = 10
    topk_indices_e0 = np.argsort(abs_weights_e0)[-topk_e0:]
    print(f"env_net_mlp_0 top 10 indices: {topk_indices_e0}")

    # --- Step 2: Train probe and get top 10 features for env_net_mlp_1 ---
    activations_e1, labels_e1 = collect_layer_activations_and_labels(model, env, agent, 'env_net_mlp_1', max_steps=500)
    valid_idxs_e1 = [i for i, label in enumerate(labels_e1) if label is not None]
    Xe1 = np.array([activations_e1[i] for i in valid_idxs_e1])
    y_str_e1 = [labels_e1[i] for i in valid_idxs_e1]
    unique_goals_e1 = sorted(set(y_str_e1))
    goal_to_idx_e1 = {goal: idx for idx, goal in enumerate(unique_goals_e1)}
    ye1 = np.array([goal_to_idx_e1[label] for label in y_str_e1])
    clf_e1 = LogisticRegression(max_iter=1000)
    clf_e1.fit(Xe1, ye1)
    abs_weights_e1 = np.abs(clf_e1.coef_[0])
    topk_e1 = 10
    topk_indices_e1 = np.argsort(abs_weights_e1)[-topk_e1:]
    print(f"env_net_mlp_1 top 10 indices: {topk_indices_e1}")

    # --- Step 3: Train probe and get top 10 features for policy_mlp_0 ---
    activations0, labels0 = collect_layer_activations_and_labels(model, env, agent, 'policy_mlp_0', max_steps=500)
    valid_idxs0 = [i for i, label in enumerate(labels0) if label is not None]
    X0 = np.array([activations0[i] for i in valid_idxs0])
    y_str0 = [labels0[i] for i in valid_idxs0]
    unique_goals0 = sorted(set(y_str0))
    goal_to_idx0 = {goal: idx for idx, goal in enumerate(unique_goals0)}
    y0 = np.array([goal_to_idx0[label] for label in y_str0])
    clf0 = LogisticRegression(max_iter=1000)
    clf0.fit(X0, y0)
    abs_weights0 = np.abs(clf0.coef_[0])
    topk0 = 10
    topk_indices0 = np.argsort(abs_weights0)[-topk0:]
    print(f"policy_mlp_0 top 10 indices: {topk_indices0}")

    # --- Step 4: Train probe and get top 10 features for policy_mlp_1 ---
    activations1, labels1 = collect_layer_activations_and_labels(model, env, agent, 'policy_mlp_1', max_steps=500)
    valid_idxs1 = [i for i, label in enumerate(labels1) if label is not None]
    X1 = np.array([activations1[i] for i in valid_idxs1])
    y_str1 = [labels1[i] for i in valid_idxs1]
    unique_goals1 = sorted(set(y_str1))
    goal_to_idx1 = {goal: idx for idx, goal in enumerate(unique_goals1)}
    y1 = np.array([goal_to_idx1[label] for label in y_str1])
    clf1 = LogisticRegression(max_iter=1000)
    clf1.fit(X1, y1)
    abs_weights1 = np.abs(clf1.coef_[0])
    topk1 = 10
    topk_indices1 = np.argsort(abs_weights1)[-topk1:]
    print(f"policy_mlp_1 top 10 indices: {topk_indices1}")

    # --- Step 5: Run ablated rollout with all four layers ablated ---
    def ablation_hook_e0(module, inp, out):
        out = out.clone()
        out[..., topk_indices_e0] = 0.0
        return out
    def ablation_hook_e1(module, inp, out):
        out = out.clone()
        out[..., topk_indices_e1] = 0.0
        return out
    def ablation_hook0(module, inp, out):
        out = out.clone()
        out[..., topk_indices0] = 0.0
        return out
    def ablation_hook1(module, inp, out):
        out = out.clone()
        out[..., topk_indices1] = 0.0
        return out

    env_ablate = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    search = ExhaustiveSearch(model, set(props), num_loops=2)
    agent = Agent(model, search=search, propositions=set(props), verbose=False)
    handle_e0 = model.env_net.mlp[0].register_forward_hook(ablation_hook_e0)
    handle_e1 = model.env_net.mlp[1].register_forward_hook(ablation_hook_e1)
    handle0 = model.actor.enc[0].register_forward_hook(ablation_hook0)
    handle1 = model.actor.enc[1].register_forward_hook(ablation_hook1)

    ret = env_ablate.reset(seed=0)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    steps = 0
    goals_visited = set()
    for step in range(500):
        current_goal = get_current_goal(agent)
        if current_goal:
            goals_visited.add(current_goal)
        action = agent.get_action(obs, info, deterministic=True).flatten()
        ret = env_ablate.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
            term, trunc = done, done
        steps += 1
        if done:
            break
    handle_e0.remove()
    handle_e1.remove()
    handle0.remove()
    handle1.remove()
    env_ablate.close()
    print(f"Ablated rollout (all layers): {steps} steps, goals visited: {goals_visited}")

def run_rollouts(model, num_rollouts=3):
    """Run rollouts and collect metrics"""
    sampler_fn = FixedSampler.partial(FORMULA)
    
    total_completions = 0
    total_steps = 0
    total_goals = 0
    
    for i in range(num_rollouts):
        env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        ret = env.reset(seed=SEED + i)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        
        # Create agent
        props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
        search = ExhaustiveSearch(model, set(props), num_loops=2)
        agent = Agent(model, search=search, propositions=set(props), verbose=False)
        agent.reset()
        
        # Track metrics
        steps = 0
        goals_visited = set()
        
        for step in range(MAX_STEPS):
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
            # Track current goal
            current_goal = get_current_goal(agent)
            if current_goal:
                goals_visited.add(current_goal)
            
            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret
                term, trunc = done, done
            
            steps += 1
            
            if done:
                break
        
        env.close()
        
        # Calculate metrics for this rollout
        goal_completed = len(goals_visited) >= 2  # Both blue and green
        
        total_completions += int(goal_completed)
        total_steps += steps
        total_goals += len(goals_visited)
    
    return {
        'completion_rate': total_completions / num_rollouts,
        'avg_steps': total_steps / num_rollouts,
        'avg_goals': total_goals / num_rollouts
    }

def simulate_ablation_effect(baseline_results, ablation_ratio):
    """Simulate the effect of ablation on behavior"""
    # This is a simplified simulation - in practice you'd actually ablate features
    
    # Simulate that stronger ablation reduces goal completion
    completion_penalty = ablation_ratio * 0.5  # Up to 50% reduction
    simulated_completion = max(0.0, baseline_results['completion_rate'] - completion_penalty)
    
    # Simulate that ablation makes paths less efficient
    efficiency_penalty = ablation_ratio * 0.3  # Up to 30% more steps
    simulated_steps = baseline_results['avg_steps'] * (1 + efficiency_penalty)
    
    # Simulate that ablation reduces goals visited
    goals_penalty = ablation_ratio * 0.4  # Up to 40% reduction
    simulated_goals = max(0.0, baseline_results['avg_goals'] - goals_penalty)
    
    return {
        'completion_rate': simulated_completion,
        'avg_steps': simulated_steps,
        'avg_goals': simulated_goals
    }

def get_current_goal(agent):
    """Extract current goal from agent's sequence"""
    seq = getattr(agent, "sequence", None)
    if seq and len(seq) > 0:
        goal_set = seq[0][0]
        if len(goal_set) == 1:
            assignment = next(iter(goal_set))
            true_props = {p for p, v in assignment.assignment if v}
            if len(true_props) == 1:
                return next(iter(true_props))
    return None

def collect_layer_activations_and_labels(model, env, agent, layer_name, max_steps=1000):
    """Collect activations from a specific layer and the current goal label at each step."""
    activations = []
    labels = []
    
    # Find the layer to hook
    if layer_name == 'env_net_mlp_0':
        layer = model.env_net.mlp[0]
    elif layer_name == 'env_net_mlp_1':
        layer = model.env_net.mlp[1]
    elif layer_name == 'policy_mlp_0':
        layer = model.actor.enc[0]
    elif layer_name == 'policy_mlp_1':
        layer = model.actor.enc[1]
    else:
        raise ValueError(f"Layer {layer_name} not supported in this script.")
    
    def hook_fn(mod, inp, out):
        # out: (batch, features) or (features,)
        arr = out.detach().cpu().numpy().squeeze()
        activations.append(arr)
    
    handle = layer.register_forward_hook(hook_fn)
    
    ret = env.reset(seed=0)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    for step in range(max_steps):
        # Get current goal label
        current_goal = get_current_goal(agent)
        labels.append(current_goal)
        
        # Debug: print info structure for first few steps
        if step < 3:
            print(f"Step {step} - info keys: {list(info.keys())}")
            print(f"Step {step} - info: {info}")
            print(f"Step {step} - current_goal: {current_goal}")
        
        action = agent.get_action(obs, info, deterministic=True).flatten()
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
            term, trunc = done, done
        if done:
            break
    handle.remove()
    return activations, labels

if __name__ == "__main__":
    test_ablation() 