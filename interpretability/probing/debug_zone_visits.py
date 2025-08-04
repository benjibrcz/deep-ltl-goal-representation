#!/usr/bin/env python3
"""
Debug Zone Visits

Check if the agent actually visits zones during rollouts and 
understand why we're getting 0 positive samples.
"""

import os
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from ltl import FixedSampler
from envs import make_env
from sequence.search import ExhaustiveSearch
from model.agent import Agent

ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0

def debug_zone_visits():
    """Debug whether agent actually visits zones."""
    print("=== DEBUGGING ZONE VISITS ===")
    
    goal = "FG blue"
    
    # Load model
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Create environment
    env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    print(f"Goal: {goal}")
    print(f"Available propositions: {sorted(props)}")
    
    # Test multiple rollouts
    total_zone_visits = 0
    total_steps = 0
    
    for rollout in range(5):  # Test 5 rollouts
        print(f"\n--- ROLLOUT {rollout} ---")
        
        obs = env.reset(seed=rollout * 100)
        agent.reset()
        
        rollout_visits = set()
        done = False
        
        for step in range(100):  # Longer rollouts to see if agent reaches zones
            if done:
                break
                
            # Check current propositions
            obs_props = set(obs.get('propositions', []))
            
            if obs_props:
                print(f"Step {step}: VISITED {obs_props}")
                rollout_visits.update(obs_props)
                total_zone_visits += len(obs_props)
            
            action = agent.get_action(obs, {}, deterministic=True).flatten()
            obs, reward, done, info = env.step(action)
            
            # Also check info propositions
            if isinstance(info, dict) and 'propositions' in info:
                info_props = set(info['propositions']) if isinstance(info['propositions'], (list, set)) else set()
                if info_props and info_props != obs_props:
                    print(f"Step {step}: INFO_PROPS {info_props}")
                    rollout_visits.update(info_props)
            
            # Print position occasionally to see if agent is moving
            if hasattr(env, 'agent_pos') or hasattr(env.unwrapped, 'agent_pos'):
                agent_pos = getattr(env, 'agent_pos', getattr(env.unwrapped, 'agent_pos', None))
                if agent_pos is not None and step % 20 == 0:
                    print(f"Step {step}: Agent position {agent_pos}")
            
            total_steps += 1
            
            if done:
                print(f"Episode ended at step {step}, reward={reward}")
                break
        
        print(f"Rollout {rollout}: Visited zones {rollout_visits} in {step+1} steps")
        
        if not rollout_visits:
            print(f"❌ No zones visited in rollout {rollout}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total zone visits: {total_zone_visits}")
    print(f"Total steps: {total_steps}")
    print(f"Visit rate: {total_zone_visits/total_steps:.4f} visits per step")
    
    if total_zone_visits == 0:
        print(f"\n🚨 PROBLEM: Agent never visits any zones!")
        print(f"Possible issues:")
        print(f"  1. Rollouts too short")
        print(f"  2. Agent stuck or not moving")
        print(f"  3. Zones positioned outside agent's reach")
        print(f"  4. Agent doesn't know how to reach target zones")
    
    env.close()

def main():
    debug_zone_visits()

if __name__ == "__main__":
    main() 