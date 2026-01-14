#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0
FORMULA = "GF blue & GF green"

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Create environment
    sampler_fn = FixedSampler.partial(FORMULA)
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    
    # Reset environment
    ret = env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    
    print("=== OBSERVATION ANALYSIS ===")
    print(f"Observation type: {type(obs)}")
    
    if isinstance(obs, dict):
        print(f"Observation keys: {list(obs.keys())}")
        
        # Examine features in detail
        if 'features' in obs:
            features = obs['features']
            print(f"\nFeatures shape: {features.shape}")
            print(f"Features dtype: {features.dtype}")
            print(f"Features range: [{features.min():.3f}, {features.max():.3f}]")
            
            # Break down the features vector
            print("\n=== FEATURES BREAKDOWN ===")
            
            # First 3 values are likely agent sensors (accelerometer, velocimeter, gyro)
            print(f"Agent sensors (0-2): {features[0:3]}")
            
            # Next 16 values are likely lidar readings for zones
            print(f"Zone lidar readings (3-18): {features[3:19]}")
            
            # Next 16 values are likely lidar readings for walls
            print(f"Wall lidar readings (19-34): {features[19:35]}")
            
            # Next 4 values are likely wall sensor
            print(f"Wall sensor (35-38): {features[35:39]}")
            
            # Remaining values
            print(f"Remaining values (39-79): {features[39:80]}")
            
            # Check which zones are being detected
            zone_lidar = features[3:19]
            print(f"\nZone lidar max values: {zone_lidar.max():.3f}")
            print(f"Zone lidar non-zero bins: {np.sum(zone_lidar > 0.01)}")
            print(f"Zone lidar values: {zone_lidar}")
            
            # Check wall lidar
            wall_lidar = features[19:35]
            print(f"\nWall lidar max values: {wall_lidar.max():.3f}")
            print(f"Wall lidar non-zero bins: {np.sum(wall_lidar > 0.01)}")
            print(f"Wall lidar values: {wall_lidar}")
    
    print("\n=== ZONE POSITIONS ===")
    if hasattr(env, 'zone_positions'):
        zone_positions = env.zone_positions
        print(f"Zone positions: {zone_positions}")
        
        # Get agent position
        agent_pos = env.agent_pos[:2]
        print(f"Agent position: {agent_pos}")
        
        # Calculate distances to zones
        print("\n=== DISTANCES TO ZONES ===")
        for zone_name, zone_pos in zone_positions.items():
            distance = np.linalg.norm(agent_pos - zone_pos)
            print(f"{zone_name}: distance = {distance:.3f}")
    
    # Check if we can access the task directly
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'task'):
        task = env.unwrapped.task
        print(f"\nTask type: {type(task)}")
        
        # Look for obstacles that might be zones
        if hasattr(task, '_obstacles'):
            print(f"Number of obstacles: {len(task._obstacles)}")
            for i, obstacle in enumerate(task._obstacles):
                print(f"  Obstacle {i}: {obstacle.name}, type: {type(obstacle)}")
                if hasattr(obstacle, 'pos'):
                    print(f"    Position: {obstacle.pos}")
                if hasattr(obstacle, 'color_name'):
                    print(f"    Color: {obstacle.color_name}")
    
    env.close()

if __name__ == '__main__':
    main() 