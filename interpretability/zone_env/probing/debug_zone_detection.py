#!/usr/bin/env python3
"""
Debug Zone Detection

Figure out what's actually in the PointLtl2-v0 observation space
and how to properly detect zone colors and zone visits.
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

def debug_observation_space():
    """Debug what's in the observation space."""
    print("=== DEBUGGING OBSERVATION SPACE ===")
    
    # Load model
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Create environment
    env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    
    print(f"Environment: {env}")
    print(f"Environment type: {type(env)}")
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Observation space type: {type(env.observation_space)}")
    
    if hasattr(env.observation_space, 'spaces'):
        print(f"\nObservation space keys: {list(env.observation_space.spaces.keys())}")
        
        for key, space in env.observation_space.spaces.items():
            print(f"  {key}: {space} (shape: {getattr(space, 'shape', 'N/A')})")
    
    # Try to get actual observation
    obs = env.reset(seed=0)
    print(f"\nActual observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
    
    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if key.endswith('_lidar') and value.size <= 20:
                    print(f"    Values: {value}")
            else:
                print(f"  {key}: {type(value)} = {value}")
    
    # Check for propositions
    print(f"\nPropositions method: {hasattr(env, 'get_propositions')}")
    if hasattr(env, 'get_propositions'):
        props = env.get_propositions()
        print(f"Available propositions: {props}")
    
    # Check environment attributes
    print(f"\nEnvironment attributes:")
    if hasattr(env, 'colors'):
        print(f"  colors: {env.colors}")
    if hasattr(env, 'zone_colors'):
        print(f"  zone_colors: {env.zone_colors}")
    
    # Unwrap to see inner environment
    unwrapped = env
    wrapper_chain = []
    while hasattr(unwrapped, 'env'):
        wrapper_chain.append(type(unwrapped).__name__)
        unwrapped = unwrapped.env
    wrapper_chain.append(type(unwrapped).__name__)
    
    print(f"\nWrapper chain: {' -> '.join(wrapper_chain)}")
    print(f"Unwrapped environment: {type(unwrapped)}")
    
    # Check unwrapped environment for zone info
    if hasattr(unwrapped, 'task'):
        print(f"Task: {type(unwrapped.task)}")
        task = unwrapped.task
        if hasattr(task, 'geoms'):
            print(f"Task geoms: {[type(g).__name__ for g in task.geoms]}")
            for g in task.geoms:
                if hasattr(g, 'color_name'):
                    print(f"  Geom {type(g).__name__}: color_name='{g.color_name}'")
                elif hasattr(g, 'name'):
                    print(f"  Geom {type(g).__name__}: name='{g.name}'")
    
    # Test a few steps to see propositions
    print(f"\n=== TESTING PROPOSITIONS DURING ROLLOUT ===")
    props = set(env.get_propositions()) if hasattr(env, 'get_propositions') else set()
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    for step in range(10):
        current_props = obs.get('propositions', set()) if isinstance(obs, dict) else set()
        info_props = set()
        
        print(f"Step {step}: obs_props={current_props}")
        
        action = agent.get_action(obs, {}, deterministic=True).flatten()
        obs, reward, done, info = env.step(action)
        
        if isinstance(info, dict) and 'propositions' in info:
            info_props = info['propositions']
            print(f"         info_props={info_props}")
            
        if done:
            break
    
    env.close()

def main():
    debug_observation_space()

if __name__ == "__main__":
    main() 