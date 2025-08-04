#!/usr/bin/env python3
"""
Multi-Layer Representation Surgery Test

This script tests representation surgery at multiple layers to find where
causal control of goal selection actually happens.
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

def test_multi_layer_surgery():
    """Test representation surgery at multiple layers"""
    print("=== Multi-Layer Representation Surgery Test ===\n")
    
    # Set up
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Load model
    build_env = make_env(ENV, FixedSampler.partial("GF yellow & GF blue"), sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, 0)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()
    print("Model loaded successfully")

    # Define layers to test
    layers_to_test = [
        ("ltl_net.set_network", "LTL Set Network"),
        ("actor.mlp[0]", "Actor MLP 0"),
        ("actor.mlp[1]", "Actor MLP 1"), 
        ("env_net.mlp[0]", "Environment MLP 0"),
        ("env_net.mlp[1]", "Environment MLP 1")
    ]
    
    # Collect yellow activations from yellow & blue formula
    print("\n--- Collecting Yellow Activations ---")
    yellow_activations = {}
    
    for layer_name, layer_desc in layers_to_test:
        print(f"Collecting from {layer_desc}...")
        activations = collect_layer_activations(model, "GF yellow & GF blue", target_goal="yellow", 
                                              layer_name=layer_name, max_steps=300)
        yellow_activations[layer_name] = activations
        print(f"  Collected {len(activations)} activations")
    
    # Baseline behavior with green & blue formula
    print("\n--- Baseline Green & Blue Behavior ---")
    baseline_results = run_goal_pursuit_test(model, "GF green & GF blue", num_rollouts=3)
    print(f"Baseline completion rate: {baseline_results['completion_rate']:.3f}")
    print(f"Baseline goals visited: {baseline_results['avg_goals']:.1f}")
    
    # Test surgery at each layer
    print("\n--- Testing Surgery at Each Layer ---")
    surgery_results = {}
    
    for layer_name, layer_desc in layers_to_test:
        print(f"\n--- Testing {layer_desc} ---")
        
        if len(yellow_activations[layer_name]) > 0:
            results = run_goal_pursuit_with_layer_surgery(model, "GF green & GF blue", 
                                                        yellow_activations[layer_name], 
                                                        layer_name=layer_name, num_rollouts=3)
            
            surgery_results[layer_name] = results
            
            # Check if yellow appears in goal sequences
            yellow_count = sum(1 for seq in results['goal_sequences'] if 'yellow' in seq)
            print(f"  Completion rate: {results['completion_rate']:.3f}")
            print(f"  Goals visited: {results['avg_goals']:.1f}")
            print(f"  Rollouts with yellow: {yellow_count}/{len(results['goal_sequences'])}")
            
            # Compare with baseline
            completion_change = results['completion_rate'] - baseline_results['completion_rate']
            goals_change = results['avg_goals'] - baseline_results['avg_goals']
            print(f"  Completion change: {completion_change:+.3f}")
            print(f"  Goals change: {goals_change:+.1f}")
        else:
            print(f"  No activations collected for {layer_desc}")
    
    # Summary
    print("\n--- Surgery Summary ---")
    for layer_name, layer_desc in layers_to_test:
        if layer_name in surgery_results:
            results = surgery_results[layer_name]
            yellow_count = sum(1 for seq in results['goal_sequences'] if 'yellow' in seq)
            print(f"{layer_desc:<20}: {yellow_count}/{len(results['goal_sequences'])} rollouts with yellow")

def collect_layer_activations(model, formula, target_goal, layer_name, max_steps=1000):
    """Collect activations from a specific layer when pursuing target goal"""
    activations = []
    
    def hook_fn(mod, inp, out):
        if isinstance(out, torch.Tensor):
            arr = out.detach().squeeze().cpu().numpy()
        elif isinstance(out, tuple):
            # Handle tuple outputs (like RNN)
            if len(out) == 2:  # (packed_seq, h_n)
                arr = out[1].detach().squeeze(0).squeeze(0).cpu().numpy()
            else:
                arr = out[0].detach().squeeze().cpu().numpy()
        else:
            return
        activations.append(arr)
    
    # Register hook on specified layer
    layer = get_layer_by_name(model, layer_name)
    if layer is None:
        print(f"Layer {layer_name} not found!")
        return []
    
    handle = layer.register_forward_hook(hook_fn)
    
    # Set up environment and agent
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    search = ExhaustiveSearch(model, set(props), num_loops=2)
    agent = Agent(model, search=search, propositions=set(props), verbose=False)
    
    # Run rollout and collect activations only when pursuing target goal
    ret = env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    target_activations = []
    
    for step in trange(max_steps, desc=f"Collecting {target_goal} activations from {layer_name}"):
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        # Check if currently pursuing target goal
        seq = getattr(agent, "sequence", None)
        if seq and len(seq) > 0:
            goal_set = seq[0][0]
            if len(goal_set) == 1:
                try:
                    assignment = next(iter(goal_set))
                    true_props = {p for p, v in assignment.assignment if v}
                    if len(true_props) == 1:
                        current_goal = next(iter(true_props))
                        if current_goal == target_goal and len(activations) > 0:
                            # Store the activation when pursuing target goal
                            target_activations.append(activations[-1].copy())
                except (StopIteration, AttributeError):
                    pass
        
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
    env.close()
    
    return target_activations

def get_layer_by_name(model, layer_name):
    """Get a layer by its name"""
    if layer_name == "ltl_net.set_network":
        return model.ltl_net.set_network
    elif layer_name == "actor.mlp[0]":
        return model.actor.enc[0]  # First layer of actor encoder
    elif layer_name == "actor.mlp[1]":
        return model.actor.enc[2]  # Second layer of actor encoder (after activation)
    elif layer_name == "env_net.mlp[0]":
        return model.env_net.mlp[0]  # First layer of env net
    elif layer_name == "env_net.mlp[1]":
        return model.env_net.mlp[2]  # Second layer of env net (after activation)
    else:
        return None

def run_goal_pursuit_test(model, formula, num_rollouts=5):
    """Run goal pursuit test and collect metrics"""
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    search = ExhaustiveSearch(model, set(props), num_loops=2)
    agent = Agent(model, search=search, propositions=set(props), verbose=False)
    
    completion_count = 0
    total_goals = 0
    goal_sequences = []
    
    for rollout in range(num_rollouts):
        ret = env.reset(seed=SEED + rollout)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        agent.reset()
        
        visited_goals = set()
        achieved_goals = []
        last_goal = None
        
        for step in range(MAX_STEPS):
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
            # Track current goal and achievements
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
                            
                            # Track goal achievements (when goal changes)
                            if prop != last_goal:
                                achieved_goals.append(prop)
                                last_goal = prop
                    except (StopIteration, AttributeError):
                        pass
            
            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret
            
            if done:
                break
        
        # Check if all goals were visited (completion)
        if len(visited_goals) >= 2:  # Need at least 2 goals for completion
            completion_count += 1
        
        total_goals += len(visited_goals)
        goal_sequences.append(achieved_goals)
    
    env.close()
    
    return {
        'completion_rate': completion_count / num_rollouts,
        'avg_goals': total_goals / num_rollouts,
        'goal_sequences': goal_sequences
    }

def run_goal_pursuit_with_layer_surgery(model, formula, injected_activations, layer_name, num_rollouts=5):
    """Run goal pursuit with injected activations at a specific layer"""
    if len(injected_activations) == 0:
        return run_goal_pursuit_test(model, formula, num_rollouts)
    
    # Create surgery hook
    activation_idx = 0
    
    def surgery_hook(module, inp, out):
        nonlocal activation_idx
        if isinstance(out, torch.Tensor):
            if activation_idx < len(injected_activations):
                out_injected = out.clone()
                inject = torch.tensor(injected_activations[activation_idx], dtype=out.dtype, device=out.device)
                # Only inject if the last dimension matches
                if out_injected.shape[-1] == inject.shape[-1]:
                    if out_injected.ndim == 2 and out_injected.shape[1] == inject.shape[0]:
                        out_injected[0, :] = inject
                    elif out_injected.ndim == 3 and out_injected.shape[2] == inject.shape[0]:
                        out_injected[0, 0, :] = inject
                    elif out_injected.ndim == 1 and out_injected.shape[0] == inject.shape[0]:
                        out_injected[:] = inject
                    else:
                        print(f"[WARN] Unhandled tensor shape {out_injected.shape} for injection, skipping.")
                        return out
                    activation_idx += 1
                    return out_injected
                else:
                    print(f"[WARN] Shape mismatch: out {out_injected.shape}, inject {inject.shape}. Skipping injection.")
                    return out
        elif isinstance(out, tuple) and len(out) == 2:
            # Handle RNN outputs
            packed_seq, h_n = out
            if activation_idx < len(injected_activations):
                h_n_injected = h_n.clone()
                inject = torch.tensor(injected_activations[activation_idx], dtype=h_n.dtype, device=h_n.device)
                if h_n_injected.shape[-1] == inject.shape[-1]:
                    if h_n_injected.ndim == 3 and h_n_injected.shape[2] == inject.shape[0]:
                        h_n_injected[0, 0, :] = inject
                    elif h_n_injected.ndim == 2 and h_n_injected.shape[1] == inject.shape[0]:
                        h_n_injected[0, :] = inject
                    elif h_n_injected.ndim == 1 and h_n_injected.shape[0] == inject.shape[0]:
                        h_n_injected[:] = inject
                    else:
                        print(f"[WARN] Unhandled RNN tensor shape {h_n_injected.shape} for injection, skipping.")
                        return out
                    activation_idx += 1
                    return (packed_seq, h_n_injected)
                else:
                    print(f"[WARN] RNN shape mismatch: h_n {h_n_injected.shape}, inject {inject.shape}. Skipping injection.")
                    return out
        return out
    
    # Register surgery hook on specified layer
    layer = get_layer_by_name(model, layer_name)
    if layer is None:
        print(f"Layer {layer_name} not found!")
        return run_goal_pursuit_test(model, formula, num_rollouts)
    
    handle = layer.register_forward_hook(surgery_hook)
    
    # Run test with surgery
    results = run_goal_pursuit_test(model, formula, num_rollouts)
    
    # Remove hook
    handle.remove()
    
    return results

if __name__ == "__main__":
    test_multi_layer_surgery() 