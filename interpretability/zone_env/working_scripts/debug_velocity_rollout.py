#!/usr/bin/env python3
"""
Debug velocity components during a long rollout.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config import model_configs
from model.agent import Agent
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from utils.model_store import ModelStore

def main():
    print("🔍 Debug Velocity Components During Long Rollout")
    print("=" * 60)
    
    # Load model and agent
    ENV = "PointLtl2-v0"
    EXP = "big_test"
    SEED = 0
    
    model_store = ModelStore(ENV, EXP, SEED)
    model_store.load_vocab()
    status = model_store.load_training_status(map_location="cpu")
    
    # Create environment for model building
    from envs import make_env
    from ltl import FixedSampler
    
    dummy_env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
    
    # Build model
    from model.model import build_model
    from config import model_configs
    
    cfg = model_configs[ENV]
    model = build_model(dummy_env, status, cfg).eval()
    dummy_env.close()
    
    # Create environment
    env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
    
    # Create agent
    from sequence.search import ExhaustiveSearch
    
    props = set(env.get_propositions())
    planner = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, planner, propositions=props, verbose=False)
    
    # Reset environment
    obs = env.reset(seed=SEED)
    agent.reset()
    
    print(f"Observation keys: {obs.keys()}")
    print(f"Features shape: {obs['features'].shape}")
    print(f"Action space: {env.action_space}")
    print(f"Agent position: {env.agent_pos}")
    print()
    
    # Run long rollout
    max_steps = 1000
    velocity_history = []
    angular_velocity_history = []
    action_history = []
    position_history = []
    
    print("🔄 Running long rollout...")
    print("Step | Action | Position | Velocity [vx, vy, vz] | Speed | Direction | Angular [wx, wy, wz]")
    print("-" * 100)
    
    for step in range(max_steps):
        # Get action
        action = agent.get_action(obs, {})
        
        # Record current state
        features = obs['features']
        vx, vy, vz = features[35], features[36], features[37]
        wx, wy, wz = features[38], features[39], features[40]  # Angular velocity
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        direction = [np.sign(vx), np.sign(vy), np.sign(vz)]
        angular_direction = [np.sign(wx), np.sign(wy), np.sign(wz)]
        
        # Store history
        velocity_history.append([vx, vy, vz])
        angular_velocity_history.append([wx, wy, wz])
        action_history.append(action)
        position_history.append(env.agent_pos.copy())
        
        # Print every 10 steps or when velocity changes significantly
        if step % 10 == 0 or abs(vx) > 0.1 or abs(vy) > 0.1:
            print(f"{step:4d} | {action.flatten()} | {env.agent_pos} | [{vx:6.3f}, {vy:6.3f}, {vz:6.3f}] | {speed:5.3f} | {direction} | [{wx:6.3f}, {wy:6.3f}, {wz:6.3f}]")
        
        # Step environment
        obs, reward, done, info = env.step(action.flatten())
        
        if done:
            print(f"\nEpisode ended at step {step}")
            break
    
    # Analyze velocity patterns
    velocity_history = np.array(velocity_history)
    angular_velocity_history = np.array(angular_velocity_history)
    action_history = np.array(action_history)
    position_history = np.array(position_history)
    
    print("\n📊 Velocity Analysis")
    print("=" * 40)
    print(f"Total steps: {len(velocity_history)}")
    print(f"vx range: [{velocity_history[:, 0].min():.3f}, {velocity_history[:, 0].max():.3f}]")
    print(f"vy range: [{velocity_history[:, 1].min():.3f}, {velocity_history[:, 1].max():.3f}]")
    print(f"vz range: [{velocity_history[:, 2].min():.3f}, {velocity_history[:, 2].max():.3f}]")
    print(f"wx range: [{angular_velocity_history[:, 0].min():.3f}, {angular_velocity_history[:, 0].max():.3f}]")
    print(f"wy range: [{angular_velocity_history[:, 1].min():.3f}, {angular_velocity_history[:, 1].max():.3f}]")
    print(f"wz range: [{angular_velocity_history[:, 2].min():.3f}, {angular_velocity_history[:, 2].max():.3f}]")
    
    # Check for non-zero components
    vx_nonzero = np.count_nonzero(velocity_history[:, 0])
    vy_nonzero = np.count_nonzero(velocity_history[:, 1])
    vz_nonzero = np.count_nonzero(velocity_history[:, 2])
    
    print(f"\nNon-zero velocity components:")
    print(f"vx non-zero: {vx_nonzero}/{len(velocity_history)} ({100*vx_nonzero/len(velocity_history):.1f}%)")
    print(f"vy non-zero: {vy_nonzero}/{len(velocity_history)} ({100*vy_nonzero/len(velocity_history):.1f}%)")
    print(f"vz non-zero: {vz_nonzero}/{len(velocity_history)} ({100*vz_nonzero/len(velocity_history):.1f}%)")
    
    # Check for non-zero angular velocity components
    wx_nonzero = np.count_nonzero(angular_velocity_history[:, 0])
    wy_nonzero = np.count_nonzero(angular_velocity_history[:, 1])
    wz_nonzero = np.count_nonzero(angular_velocity_history[:, 2])
    
    print(f"\nNon-zero angular velocity components:")
    print(f"wx non-zero: {wx_nonzero}/{len(angular_velocity_history)} ({100*wx_nonzero/len(angular_velocity_history):.1f}%)")
    print(f"wy non-zero: {wy_nonzero}/{len(angular_velocity_history)} ({100*wy_nonzero/len(angular_velocity_history):.1f}%)")
    print(f"wz non-zero: {wz_nonzero}/{len(angular_velocity_history)} ({100*wz_nonzero/len(angular_velocity_history):.1f}%)")
    
    # Check for sign changes (direction changes)
    vx_sign_changes = np.sum(np.diff(np.sign(velocity_history[:, 0])) != 0)
    vy_sign_changes = np.sum(np.diff(np.sign(velocity_history[:, 1])) != 0)
    vz_sign_changes = np.sum(np.diff(np.sign(velocity_history[:, 2])) != 0)
    
    print(f"\nDirection changes:")
    print(f"vx sign changes: {vx_sign_changes}")
    print(f"vy sign changes: {vy_sign_changes}")
    print(f"vz sign changes: {vz_sign_changes}")
    
    # Check for angular velocity sign changes
    wx_sign_changes = np.sum(np.diff(np.sign(angular_velocity_history[:, 0])) != 0)
    wy_sign_changes = np.sum(np.diff(np.sign(angular_velocity_history[:, 1])) != 0)
    wz_sign_changes = np.sum(np.diff(np.sign(angular_velocity_history[:, 2])) != 0)
    
    print(f"\nAngular velocity direction changes:")
    print(f"wx sign changes: {wx_sign_changes}")
    print(f"wy sign changes: {wy_sign_changes}")
    print(f"wz sign changes: {wz_sign_changes}")
    
    # Position analysis
    position_changes = np.diff(position_history, axis=0)
    total_movement = np.sum(np.linalg.norm(position_changes, axis=1))
    
    print(f"\nMovement analysis:")
    print(f"Total movement: {total_movement:.3f}")
    print(f"Average step size: {total_movement/len(position_changes):.3f}")
    print(f"Final position: {position_history[-1]}")
    print(f"Position range: [{position_history.min():.3f}, {position_history.max():.3f}]")
    
    # Find steps with non-zero horizontal velocity
    horizontal_movement = np.where(np.abs(velocity_history[:, :2]).any(axis=1))[0]
    if len(horizontal_movement) > 0:
        print(f"\nSteps with horizontal movement: {horizontal_movement[:10]}...")
        for step in horizontal_movement[:5]:
            print(f"  Step {step}: vx={velocity_history[step, 0]:.3f}, vy={velocity_history[step, 1]:.3f}")
    else:
        print("\nNo horizontal movement detected!")
    
    print("\n✅ Velocity debug complete!")

if __name__ == "__main__":
    main() 