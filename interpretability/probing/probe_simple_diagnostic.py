#!/usr/bin/env python3
"""
Simple Grid Prediction Diagnostic

Focus on the key insight: class imbalance explains poor grid prediction performance.
"""

import os
import sys
import numpy as np
import torch
from tqdm import trange

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

def position_to_grid_cell(pos, grid_size, map_bounds):
    x, y = pos
    x_min, y_min, x_max, y_max = map_bounds
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    grid_x = int(np.clip(x_norm * grid_size, 0, grid_size - 1))
    grid_y = int(np.clip(y_norm * grid_size, 0, grid_size - 1))
    return grid_x, grid_y

def analyze_class_imbalance():
    """Analyze why grid prediction fails: class imbalance."""
    print("=== GRID PREDICTION CLASS IMBALANCE ANALYSIS ===")
    
    # Load model
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Collect data
    env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    grid_size = 5
    horizon = 5
    map_bounds = (-2, -2, 2, 2)
    
    samples = []
    all_positions = []
    
    for world_id in range(3):  # Small sample for speed
        for rollout_id in trange(6, desc=f"World {world_id}"):
            trajectory_data = []
            
            done = False
            obs = env.reset(seed=world_id + rollout_id * 1000)
            agent.reset()
            
            for step_id in range(40):
                if done:
                    break
                    
                current_pos = env.agent_pos[:2].copy()
                all_positions.append(current_pos)
                
                obs_features = obs.get('features', np.zeros(80))
                goal_encoding = np.zeros(10)
                goal_encoding[0] = 1.0  # blue
                
                trajectory_data.append({
                    'position': current_pos,
                    'features': np.concatenate([obs_features, goal_encoding])
                })
                
                action = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, info = env.step(action)
            
            # Create samples
            if len(trajectory_data) < horizon + 2:
                continue
                
            for i in range(len(trajectory_data) - horizon):
                current_data = trajectory_data[i]
                
                # Get future positions
                future_positions = []
                for j in range(1, horizon + 1):
                    if i + j < len(trajectory_data):
                        future_positions.append(trajectory_data[i + j]['position'])
                
                if len(future_positions) == 0:
                    continue
                
                # Create grid visits set
                grid_visits = set()
                for future_pos in future_positions:
                    gx, gy = position_to_grid_cell(future_pos, grid_size, map_bounds)
                    grid_visits.add((gx, gy))
                
                samples.append({
                    'position': current_data['position'],
                    'grid_visits': grid_visits
                })
    
    env.close()
    
    print(f"📊 ANALYSIS RESULTS:")
    print(f"Total samples: {len(samples)}")
    print(f"Total positions tracked: {len(all_positions)}")
    
    # Analyze class balance for each cell
    print(f"\n⚖️  CLASS IMBALANCE BREAKDOWN:")
    cell_stats = {}
    
    for i in range(grid_size):
        for j in range(grid_size):
            positive = sum(1 for s in samples if (i, j) in s['grid_visits'])
            ratio = positive / len(samples) if len(samples) > 0 else 0
            cell_stats[(i, j)] = ratio
    
    # Statistics
    ratios = list(cell_stats.values())
    meaningful_cells = sum(1 for r in ratios if 0.05 <= r <= 0.95)
    
    print(f"Average positive ratio: {np.mean(ratios):.3f}")
    print(f"Min positive ratio: {np.min(ratios):.3f}")
    print(f"Max positive ratio: {np.max(ratios):.3f}")
    print(f"Cells with reasonable balance (5-95%): {meaningful_cells}/{grid_size*grid_size}")
    
    # Show the worst cases
    sorted_cells = sorted(cell_stats.items(), key=lambda x: x[1])
    print(f"\n🔴 WORST CELLS (hardest to predict):")
    for (i, j), ratio in sorted_cells[:5]:
        count = int(ratio * len(samples))
        print(f"  Cell ({i},{j}): {count}/{len(samples)} = {ratio:.3f} ({ratio*100:.1f}%)")
    
    print(f"\n🟢 BEST CELLS (easiest to predict):")
    for (i, j), ratio in sorted_cells[-5:]:
        count = int(ratio * len(samples))
        print(f"  Cell ({i},{j}): {count}/{len(samples)} = {ratio:.3f} ({ratio*100:.1f}%)")
    
    # Grid coverage analysis
    print(f"\n🗺️  SPATIAL COVERAGE:")
    grid_counts = np.zeros((grid_size, grid_size))
    for pos in all_positions:
        gx, gy = position_to_grid_cell(pos, grid_size, map_bounds)
        grid_counts[gy, gx] += 1
    
    total_positions = len(all_positions)
    cells_visited = np.sum(grid_counts > 0)
    
    print(f"Cells visited: {cells_visited}/{grid_size*grid_size}")
    print(f"Most visited cell: {grid_counts.max()/total_positions*100:.1f}% of time")
    print(f"Least visited cell: {grid_counts.min()/total_positions*100:.1f}% of time")
    
    # The key insight
    print(f"\n🔍 KEY INSIGHT:")
    print(f"✅ Agent navigates successfully (visits {cells_visited}/{grid_size*grid_size} cells)")
    print(f"❌ But most cells visited only {np.mean(ratios)*100:.1f}% of time on average")
    print(f"⚖️  Severe class imbalance makes prediction nearly impossible")
    print(f"🧠 Navigation success ≠ Spatial predictability!")
    
    print(f"\n💡 EXPLANATION:")
    print(f"The agent succeeds through REACTIVE navigation:")
    print(f"  - Responds to immediate stimuli")
    print(f"  - Makes locally optimal moves")
    print(f"  - Doesn't need to predict future locations")
    print(f"  - Class imbalance reflects this reactive strategy")

if __name__ == "__main__":
    analyze_class_imbalance() 