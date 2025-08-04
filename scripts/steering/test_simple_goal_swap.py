#!/usr/bin/env python3
"""
Simple Goal Swap Test

This script tests if we can make the agent pursue yellow instead of green
by using a formula that only includes yellow and green (no blue).
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

def test_simple_goal_swap():
    """Test if we can make the agent pursue yellow instead of green"""
    print("=== Simple Goal Swap Test ===\n")
    
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

    # Test 1: Baseline with yellow & blue formula
    print("\n--- Test 1: Baseline Yellow & Blue Pursuit ---")
    yellow_blue_results = run_goal_pursuit_test(model, "GF yellow & GF blue", num_rollouts=5)
    print(f"Yellow & Blue completion rate: {yellow_blue_results['completion_rate']:.3f}")
    print(f"Yellow & Blue goals visited: {yellow_blue_results['avg_goals']:.1f}")
    print(f"Yellow & Blue goal sequence: {yellow_blue_results['goal_sequences'][:2]}...")
    
    # Test 2: Baseline with green & blue formula
    print("\n--- Test 2: Baseline Green & Blue Pursuit ---")
    green_blue_results = run_goal_pursuit_test(model, "GF green & GF blue", num_rollouts=5)
    print(f"Green & Blue completion rate: {green_blue_results['completion_rate']:.3f}")
    print(f"Green & Blue goals visited: {green_blue_results['avg_goals']:.1f}")
    print(f"Green & Blue goal sequence: {green_blue_results['goal_sequences'][:2]}...")
    
    # Test 3: Try yellow & green formula (no blue)
    print("\n--- Test 3: Yellow & Green Pursuit (No Blue) ---")
    yellow_green_results = run_goal_pursuit_test(model, "GF yellow & GF green", num_rollouts=5)
    print(f"Yellow & Green completion rate: {yellow_green_results['completion_rate']:.3f}")
    print(f"Yellow & Green goals visited: {yellow_green_results['avg_goals']:.1f}")
    print(f"Yellow & Green goal sequence: {yellow_green_results['goal_sequences'][:2]}...")
    
    # Test 4: Try to inject yellow activations during green & blue pursuit
    print("\n--- Test 4: Injecting Yellow Activations During Green & Blue Pursuit ---")
    
    # Collect yellow activations from yellow & blue formula
    yellow_activations = collect_goal_activations(model, "GF yellow & GF blue", target_goal="yellow", max_steps=500)
    print(f"Collected {len(yellow_activations)} yellow activations")
    
    if len(yellow_activations) > 0:
        # Inject yellow activations during green & blue pursuit
        surgery_results = run_goal_pursuit_with_surgery(model, "GF green & GF blue", yellow_activations, num_rollouts=5)
        print(f"Surgery completion rate: {surgery_results['completion_rate']:.3f}")
        print(f"Surgery goals visited: {surgery_results['avg_goals']:.1f}")
        print(f"Surgery goal sequence: {surgery_results['goal_sequences'][:2]}...")
        
        # Check if yellow appears in goal sequences
        yellow_count = sum(1 for seq in surgery_results['goal_sequences'] if 'yellow' in seq)
        print(f"Rollouts with yellow pursuit: {yellow_count}/{len(surgery_results['goal_sequences'])}")
        
        # Compare with baseline
        print(f"\n--- Surgery Effects ---")
        completion_change = surgery_results['completion_rate'] - green_blue_results['completion_rate']
        goals_change = surgery_results['avg_goals'] - green_blue_results['avg_goals']
        print(f"Completion rate change: {completion_change:+.3f}")
        print(f"Goals visited change: {goals_change:+.1f}")
    else:
        print("No yellow activations collected")

def collect_goal_activations(model, formula, target_goal, max_steps=1000):
    """Collect activations when agent is pursuing a specific goal"""
    activations = []
    
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
        return []
    
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
    
    for step in trange(max_steps, desc=f"Collecting {target_goal} activations"):
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

def run_goal_pursuit_with_surgery(model, formula, injected_activations, num_rollouts=5):
    """Run goal pursuit with injected activations"""
    if len(injected_activations) == 0:
        return run_goal_pursuit_test(model, formula, num_rollouts)
    
    # Create surgery hook
    activation_idx = 0
    
    def surgery_hook(module, inp, out):
        nonlocal activation_idx
        if isinstance(out, tuple):
            packed_seq, h_n = out
            # Inject the collected activation
            if activation_idx < len(injected_activations):
                h_n_injected = h_n.clone()
                h_n_injected[0, 0, :] = torch.tensor(injected_activations[activation_idx], dtype=h_n.dtype, device=h_n.device)
                activation_idx += 1
                return (packed_seq, h_n_injected)
        return out
    
    # Register surgery hook
    if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
        handle = model.ltl_net.rnn.register_forward_hook(surgery_hook)
    else:
        print("LTL RNN layer not found!")
        return run_goal_pursuit_test(model, formula, num_rollouts)
    
    # Run test with surgery
    results = run_goal_pursuit_test(model, formula, num_rollouts)
    
    # Remove hook
    handle.remove()
    
    return results

if __name__ == "__main__":
    test_simple_goal_swap() 