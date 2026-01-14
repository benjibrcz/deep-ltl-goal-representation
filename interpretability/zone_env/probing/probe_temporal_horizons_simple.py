#!/usr/bin/env python3
"""
Simplified Temporal Horizons × Generalization Splits Probe

Tests "will this grid cell be visited in the next N steps?" across different values of N
and three generalization splits (temporal, spatial, environmental).
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from ltl import FixedSampler
from envs import make_env
from sequence.search import ExhaustiveSearch
from model.agent import Agent

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0

def position_to_grid_cell(pos, grid_size, map_bounds):
    """Convert continuous position to grid cell indices."""
    x, y = pos
    x_min, y_min, x_max, y_max = map_bounds
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    grid_x = int(np.clip(x_norm * grid_size, 0, grid_size - 1))
    grid_y = int(np.clip(y_norm * grid_size, 0, grid_size - 1))
    return grid_x, grid_y

def collect_data_with_metadata(model, world_ids, goal="FG blue", grid_size=5, 
                              n_rollouts_per_world=6, max_steps=50, 
                              horizon_steps=[1, 3, 5, 10],
                              map_bounds=(-2, -2, 2, 2)):
    """Collect spatial prediction data with metadata for generalization splits."""
    print(f"Collecting data for {len(horizon_steps)} temporal horizons...")
    print(f"Goal: {goal}, Grid: {grid_size}x{grid_size}")
    
    env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    # Collect all data samples with metadata
    all_samples = []
    
    for world_id in world_ids:
        print(f"Processing world {world_id}...")
        for rollout_id in trange(n_rollouts_per_world, desc=f"Rollouts"):
            trajectory_data = []
            
            done = False
            obs = env.reset(seed=world_id + rollout_id * 1000)
            agent.reset()
            
            # Collect complete trajectory
            for step_id in range(max_steps):
                if done:
                    break
                    
                current_pos = env.agent_pos[:2].copy()
                obs_features = obs.get('features', np.zeros(80))
                goal_encoding = np.zeros(10)
                if 'blue' in goal.lower():
                    goal_encoding[0] = 1.0
                elif 'red' in goal.lower():
                    goal_encoding[1] = 1.0
                elif 'green' in goal.lower():
                    goal_encoding[2] = 1.0
                
                network_representation = np.concatenate([obs_features, goal_encoding])
                
                trajectory_data.append({
                    'features': network_representation,
                    'position': current_pos,
                    'step_id': step_id,
                    'world_id': world_id,
                    'rollout_id': rollout_id
                })
                
                action = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, info = env.step(action)
            
            # Create samples for each horizon
            max_horizon = max(horizon_steps)
            if len(trajectory_data) < max_horizon + 5:
                continue
                
            for N in horizon_steps:
                valid_samples = len(trajectory_data) - N
                if valid_samples <= 0:
                    continue
                    
                for i in range(valid_samples):
                    current_data = trajectory_data[i]
                    
                    # Get future positions for next N steps
                    future_positions = []
                    for future_step in range(1, N + 1):
                        if i + future_step < len(trajectory_data):
                            future_pos = trajectory_data[i + future_step]['position']
                            future_positions.append(future_pos)
                    
                    if len(future_positions) == 0:
                        continue
                    
                    # Create binary labels for each grid cell
                    grid_visits = set()
                    for future_pos in future_positions:
                        grid_x, grid_y = position_to_grid_cell(future_pos, grid_size, map_bounds)
                        grid_visits.add((grid_x, grid_y))
                    
                    # Create sample with all metadata
                    sample = {
                        'features': current_data['features'],
                        'horizon': N,
                        'world_id': world_id,
                        'rollout_id': rollout_id,
                        'step_id': current_data['step_id'],
                        'grid_visits': grid_visits
                    }
                    all_samples.append(sample)
    
    env.close()
    print(f"Collected {len(all_samples)} total samples")
    return all_samples, grid_size

def create_splits(samples):
    """Create temporal, spatial, environmental splits."""
    splits = {}
    
    # Extract metadata
    step_ids = [s['step_id'] for s in samples]
    rollout_keys = [(s['world_id'], s['rollout_id']) for s in samples]
    world_ids = [s['world_id'] for s in samples]
    
    # 1. TEMPORAL SPLIT: Early vs late steps
    median_step = np.median(step_ids)
    temporal_train = [i for i, step in enumerate(step_ids) if step <= median_step]
    temporal_test = [i for i, step in enumerate(step_ids) if step > median_step]
    splits['temporal'] = (temporal_train, temporal_test)
    
    # 2. SPATIAL SPLIT: Some rollouts vs others
    unique_rollouts = list(set(rollout_keys))
    random.shuffle(unique_rollouts)
    n_train = int(len(unique_rollouts) * 0.7)
    train_rollouts = set(unique_rollouts[:n_train])
    
    spatial_train = [i for i, key in enumerate(rollout_keys) if key in train_rollouts]
    spatial_test = [i for i, key in enumerate(rollout_keys) if key not in train_rollouts]
    splits['spatial'] = (spatial_train, spatial_test)
    
    # 3. ENVIRONMENTAL SPLIT: Some worlds vs others
    unique_worlds = list(set(world_ids))
    random.shuffle(unique_worlds)
    n_train_worlds = max(1, int(len(unique_worlds) * 0.7))
    train_worlds = set(unique_worlds[:n_train_worlds])
    
    env_train = [i for i, w in enumerate(world_ids) if w in train_worlds]
    env_test = [i for i, w in enumerate(world_ids) if w not in train_worlds]
    splits['environmental'] = (env_train, env_test)
    
    return splits

def evaluate_horizon_split(samples, horizon, split_name, train_indices, test_indices, grid_size):
    """Evaluate one horizon-split combination."""
    print(f"  Evaluating {split_name} split, horizon {horizon}...")
    
    # Filter samples for this horizon
    horizon_samples = [s for s in samples if s['horizon'] == horizon]
    if len(horizon_samples) == 0:
        return None
    
    # Filter by train/test split
    train_samples = [horizon_samples[i] for i in train_indices if i < len(horizon_samples)]
    test_samples = [horizon_samples[i] for i in test_indices if i < len(horizon_samples)]
    
    if len(train_samples) == 0 or len(test_samples) == 0:
        return None
    
    # Extract features
    X_train = np.array([s['features'] for s in train_samples])
    X_test = np.array([s['features'] for s in test_samples])
    
    # Apply PCA
    if X_train.shape[1] > 50:
        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    
    # Test prediction for each grid cell
    accuracies = []
    meaningful_cells = 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Create labels for this cell
            y_train = [1 if (i, j) in s['grid_visits'] else 0 for s in train_samples]
            y_test = [1 if (i, j) in s['grid_visits'] else 0 for s in test_samples]
            
            # Check if we have both classes and sufficient visits
            if len(set(y_train)) < 2 or np.mean(y_train + y_test) < 0.05:
                accuracies.append(0.5)  # Random performance
                continue
            
            meaningful_cells += 1
            
            try:
                # Train balanced classifier
                model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = balanced_accuracy_score(y_test, y_pred)
                accuracies.append(acc)
            except:
                accuracies.append(0.5)
    
    result = {
        'horizon': horizon,
        'split': split_name,
        'n_train': len(train_samples),
        'n_test': len(test_samples),
        'meaningful_cells': meaningful_cells,
        'balanced_accuracy': np.mean(accuracies)
    }
    
    print(f"    Train: {len(train_samples)}, Test: {len(test_samples)}")
    print(f"    Meaningful cells: {meaningful_cells}/{grid_size*grid_size}")
    print(f"    Balanced accuracy: {result['balanced_accuracy']:.3f}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Temporal Horizons × Generalization Splits')
    parser.add_argument('--goal', default="FG blue", help='LTL goal to test')
    parser.add_argument('--grid-size', type=int, default=5, help='Grid size')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 3, 5, 10],
                       help='Prediction horizons in steps')
    parser.add_argument('--n-worlds', type=int, default=4, help='Number of worlds')
    parser.add_argument('--n-rollouts', type=int, default=6, help='Rollouts per world')
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps per rollout')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=== TEMPORAL HORIZONS × GENERALIZATION SPLITS PROBE ===")
    print(f"🎯 Testing: Will grid cells be visited in the next N steps?")
    print(f"🧠 Across: Temporal, Spatial, Environmental generalization")
    print(f"Goal: {args.goal}")
    print(f"Horizons: {args.horizons} steps")
    
    # Load model
    print("\nLoading model...")
    store = ModelStore(ENV, EXP, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial(args.goal), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Collect data
    world_ids = list(range(args.n_worlds))
    samples, grid_size = collect_data_with_metadata(
        model, world_ids, args.goal, args.grid_size, 
        args.n_rollouts, args.max_steps, args.horizons
    )
    
    if len(samples) == 0:
        print("❌ No samples collected!")
        return
    
    # Create generalization splits
    print("\nCreating generalization splits...")
    splits = create_splits(samples)
    
    # Evaluate all combinations
    print("\nEvaluating all horizon × split combinations...")
    results = []
    
    for horizon in args.horizons:
        print(f"\n--- HORIZON {horizon} STEPS ---")
        for split_name, (train_indices, test_indices) in splits.items():
            result = evaluate_horizon_split(
                samples, horizon, split_name, train_indices, test_indices, grid_size
            )
            if result:
                results.append(result)
    
    # Analysis
    print(f"\n=== COMPREHENSIVE ANALYSIS ===")
    if results:
        print(f"📊 PERFORMANCE SUMMARY:")
        
        # Group by split
        for split_name in ['temporal', 'spatial', 'environmental']:
            split_results = [r for r in results if r['split'] == split_name]
            if split_results:
                best = max(split_results, key=lambda x: x['balanced_accuracy'])
                avg = np.mean([r['balanced_accuracy'] for r in split_results])
                print(f"  {split_name.upper()} split:")
                print(f"    Best: {best['horizon']} steps (Acc = {best['balanced_accuracy']:.3f})")
                print(f"    Average: {avg:.3f}")
        
        # Temporal patterns
        print(f"\n🕐 TEMPORAL PATTERNS:")
        for split_name in ['temporal', 'spatial', 'environmental']:
            split_results = [r for r in results if r['split'] == split_name]
            if split_results:
                short_term = [r for r in split_results if r['horizon'] <= 5]
                long_term = [r for r in split_results if r['horizon'] >= 10]
                
                if short_term and long_term:
                    short_avg = np.mean([r['balanced_accuracy'] for r in short_term])
                    long_avg = np.mean([r['balanced_accuracy'] for r in long_term])
                    
                    print(f"  {split_name.upper()}: Short-term {short_avg:.3f} vs Long-term {long_avg:.3f}")
                    if short_avg > long_avg + 0.02:
                        print(f"    → ✅ SHORT-TERM advantage")
                    elif long_avg > short_avg + 0.02:
                        print(f"    → 🤔 LONG-TERM advantage")
                    else:
                        print(f"    → ❌ No temporal preference")
        
        # Overall best
        best_overall = max(results, key=lambda x: x['balanced_accuracy'])
        max_acc = best_overall['balanced_accuracy']
        
        print(f"\n🏆 BEST OVERALL:")
        print(f"Split: {best_overall['split']}, Horizon: {best_overall['horizon']} steps")
        print(f"Balanced Accuracy: {max_acc:.3f}")
        
        # Interpretation
        if max_acc > 0.6:
            print(f"\n🚀 SIGNIFICANT spatial prediction capability!")
        elif max_acc > 0.55:
            print(f"\n🤔 WEAK spatial prediction capability")
        else:
            print(f"\n❌ NO meaningful spatial prediction capability")
            print(f"   All combinations near chance level (0.5)")

if __name__ == "__main__":
    main() 