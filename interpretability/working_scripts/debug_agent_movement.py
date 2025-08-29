#!/usr/bin/env python3
"""
Debug script to check agent movement
"""

import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from envs import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search import ExhaustiveSearch
from model.agent import Agent

# Quick test
ENV, EXP, SEED = "PointLtl2-v0", "big_test", 0

# Build model
dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
cfg = model_configs[ENV]
store = ModelStore(ENV, EXP, SEED)
store.load_vocab()
status = store.load_training_status(map_location="cpu")
model = build_model(dummy, status, cfg).eval()
dummy.close()

# Test one rollout
env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
props = set(env.get_propositions())
planner = ExhaustiveSearch(model, props, num_loops=2)
agent = Agent(model, planner, propositions=props, verbose=False)

obs = env.reset(seed=SEED)
agent.reset()

print("Initial position:", env.agent_pos[:2])
print("Initial observation keys:", obs.keys())

positions = []
speeds = []

for step in range(50):
    action = agent.get_action(obs, {}, deterministic=True)
    
    # Ensure action is in correct format
    if isinstance(action, np.ndarray):
        action = action.flatten()
    elif isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy().flatten()
    
    obs, _, done, _ = env.step(action)
    
    pos = env.agent_pos[:2]
    positions.append(pos.copy())
    
    if step > 0:
        delta = pos - positions[-2]
        speed = np.linalg.norm(delta) / 0.02
        speeds.append(speed)
        print(f"Step {step}: pos={pos}, delta={delta}, speed={speed:.6f}")
    
    if done:
        break

env.close()

print(f"\nPosition range: {np.min(positions, axis=0)} to {np.max(positions, axis=0)}")
print(f"Speed range: {np.min(speeds):.6f} to {np.max(speeds):.6f}")
print(f"Average speed: {np.mean(speeds):.6f}")
print(f"Non-zero speeds: {np.sum(np.array(speeds) > 0.001)}/{len(speeds)}") 