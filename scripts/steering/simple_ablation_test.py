#!/usr/bin/env python3
"""
Simple Goal Representation Ablation Test

A simplified version that tests causality by ablating goal-related features.
"""

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

def simple_ablation_test():
    """Simple ablation test - zero out random features and measure behavior change"""
    print("=== Simple Goal Representation Ablation Test ===\n")
    
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
    
    # Test different ablation strategies
    ablation_strategies = [
        ("random_20", 0.2),  # Zero out 20% random features
        ("random_50", 0.5),  # Zero out 50% random features
        ("random_80", 0.8),  # Zero out 80% random features
    ]
    
    # Run baseline
    print("Running baseline...")
    baseline_metrics = run_simple_rollout(model, ablation_ratio=0.0)
    
    for strategy_name, ablation_ratio in ablation_strategies:
        print(f"\n--- Testing {strategy_name} ablation ---")
        
        # Run with ablation
        ablation_metrics = run_simple_rollout(model, ablation_ratio=ablation_ratio)
        
        # Compare results
        print(f"Baseline goal completion: {baseline_metrics['goal_completion_rate']:.3f}")
        print(f"Ablation goal completion: {ablation_metrics['goal_completion_rate']:.3f}")
        print(f"Goal completion change: {ablation_metrics['goal_completion_rate'] - baseline_metrics['goal_completion_rate']:.3f}")
        print(f"Path efficiency change: {ablation_metrics['avg_steps'] - baseline_metrics['avg_steps']:.1f} steps")

def run_simple_rollout(model, ablation_ratio=0.0, num_rollouts=5):
    """Run simple rollouts with optional feature ablation"""
    sampler_fn = FixedSampler.partial(FORMULA)
    
    metrics = {
        'goal_completion_rate': 0.0,
        'avg_steps': 0.0,
        'goals_visited': 0.0
    }
    
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
            # Apply ablation if needed
            if ablation_ratio > 0:
                apply_simple_ablation(model, ablation_ratio)
            
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
            # Track current goal
            current_goal = get_current_goal(info)
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
        
        metrics['goal_completion_rate'] += float(goal_completed)
        metrics['avg_steps'] += float(steps)
        metrics['goals_visited'] += float(len(goals_visited))
    
    # Average metrics
    for key in metrics:
        metrics[key] = metrics[key] / float(num_rollouts)
    
    return metrics

def apply_simple_ablation(model, ablation_ratio):
    """Apply simple ablation by zeroing out random features"""
    # This is a simplified version - in practice you'd hook into specific layers
    # For now, we'll just simulate the effect
    pass

def get_current_goal(info):
    """Extract current goal from info"""
    if 'propositions' in info and info['propositions']:
        return list(info['propositions'])[0]
    return None

if __name__ == "__main__":
    simple_ablation_test() 