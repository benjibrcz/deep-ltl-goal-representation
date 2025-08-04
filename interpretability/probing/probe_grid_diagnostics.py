#!/usr/bin/env python3
"""
Grid Prediction Diagnostics

Diagnose why grid cell prediction is performing so poorly despite good agent navigation.
Let's verify our methodology and identify the real issues.
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
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

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

def collect_diagnostic_data(model, world_ids, goal="FG blue", grid_size=5, 
                           n_rollouts_per_world=8, max_steps=60, 
                           horizon=5, map_bounds=(-2, -2, 2, 2)):
    """Collect data with detailed diagnostics."""
    print(f"🔍 DIAGNOSTIC DATA COLLECTION")
    print(f"Goal: {goal}, Grid: {grid_size}x{grid_size}, Horizon: {horizon} steps")
    
    env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    samples = []
    all_positions = []
    all_future_positions = []
    
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
                all_positions.append(current_pos)
                
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
            
            # Create samples for this horizon
            if len(trajectory_data) < horizon + 5:
                continue
                
            valid_samples = len(trajectory_data) - horizon
            for i in range(valid_samples):
                current_data = trajectory_data[i]
                
                # Get future positions for next horizon steps
                future_positions = []
                for future_step in range(1, horizon + 1):
                    if i + future_step < len(trajectory_data):
                        future_pos = trajectory_data[i + future_step]['position']
                        future_positions.append(future_pos)
                        all_future_positions.append(future_pos)
                
                if len(future_positions) == 0:
                    continue
                
                # Create binary labels for each grid cell
                grid_visits = set()
                for future_pos in future_positions:
                    grid_x, grid_y = position_to_grid_cell(future_pos, grid_size, map_bounds)
                    grid_visits.add((grid_x, grid_y))
                
                sample = {
                    'features': current_data['features'],
                    'position': current_data['position'],
                    'future_positions': future_positions,
                    'grid_visits': grid_visits,
                    'world_id': world_id,
                    'rollout_id': rollout_id,
                    'step_id': current_data['step_id']
                }
                samples.append(sample)
    
    env.close()
    print(f"Collected {len(samples)} samples from {len(all_positions)} total positions")
    
    return samples, np.array(all_positions), np.array(all_future_positions), grid_size

def diagnose_grid_coverage(samples, all_positions, all_future_positions, grid_size, map_bounds):
    """Analyze grid coverage and visit patterns."""
    print(f"\n🗺️  GRID COVERAGE ANALYSIS")
    
    # Create visit frequency matrix
    current_grid_counts = np.zeros((grid_size, grid_size))
    future_grid_counts = np.zeros((grid_size, grid_size))
    
    # Count current positions
    for pos in all_positions:
        grid_x, grid_y = position_to_grid_cell(pos, grid_size, map_bounds)
        current_grid_counts[grid_y, grid_x] += 1
    
    # Count future positions
    for pos in all_future_positions:
        grid_x, grid_y = position_to_grid_cell(pos, grid_size, map_bounds)
        future_grid_counts[grid_y, grid_x] += 1
    
    # Normalize to percentages
    current_grid_freq = current_grid_counts / len(all_positions) * 100
    future_grid_freq = future_grid_counts / len(all_future_positions) * 100
    
    print(f"Current position coverage:")
    print(f"  Cells visited: {np.sum(current_grid_counts > 0)}/{grid_size*grid_size}")
    print(f"  Most visited cell: {current_grid_freq.max():.1f}% of time")
    print(f"  Least visited cell: {current_grid_freq.min():.1f}% of time")
    
    print(f"Future position coverage:")
    print(f"  Cells visited: {np.sum(future_grid_counts > 0)}/{grid_size*grid_size}")
    print(f"  Most visited cell: {future_grid_freq.max():.1f}% of time")
    print(f"  Least visited cell: {future_grid_freq.min():.1f}% of time")
    
    # Visualize grid coverage
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Current positions heatmap
    sns.heatmap(current_grid_freq, annot=True, fmt='.1f', cmap='Blues', 
                ax=axes[0], cbar_kws={'label': 'Visit %'})
    axes[0].set_title('Current Position Distribution (%)')
    axes[0].set_xlabel('Grid X')
    axes[0].set_ylabel('Grid Y')
    
    # Future positions heatmap
    sns.heatmap(future_grid_freq, annot=True, fmt='.1f', cmap='Reds', 
                ax=axes[1], cbar_kws={'label': 'Visit %'})
    axes[1].set_title('Future Position Distribution (%)')
    axes[1].set_xlabel('Grid X')
    axes[1].set_ylabel('Grid Y')
    
    plt.tight_layout()
    plt.savefig('interpretability/probing/corrected_results/grid_coverage_diagnostic.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return current_grid_freq, future_grid_freq

def diagnose_class_balance(samples, grid_size):
    """Analyze class balance for each grid cell prediction."""
    print(f"\n⚖️  CLASS BALANCE ANALYSIS")
    
    # Count positive/negative samples for each cell
    cell_stats = {}
    
    for i in range(grid_size):
        for j in range(grid_size):
            positive_count = sum(1 for s in samples if (i, j) in s['grid_visits'])
            negative_count = len(samples) - positive_count
            positive_ratio = positive_count / len(samples)
            
            cell_stats[(i, j)] = {
                'positive': positive_count,
                'negative': negative_count,
                'positive_ratio': positive_ratio,
                'total': len(samples)
            }
    
    # Print statistics
    ratios = [stats['positive_ratio'] for stats in cell_stats.values()]
    meaningful_cells = sum(1 for ratio in ratios if 0.05 <= ratio <= 0.95)
    
    print(f"Class balance statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Cells with 5-95% positive ratio: {meaningful_cells}/{grid_size*grid_size}")
    print(f"  Average positive ratio: {np.mean(ratios):.3f}")
    print(f"  Min positive ratio: {np.min(ratios):.3f}")
    print(f"  Max positive ratio: {np.max(ratios):.3f}")
    
    # Show per-cell breakdown
    print(f"\nPer-cell breakdown (grid_x, grid_y: positive/total = ratio):")
    for i in range(grid_size):
        for j in range(grid_size):
            stats = cell_stats[(i, j)]
            print(f"  ({i},{j}): {stats['positive']}/{stats['total']} = {stats['positive_ratio']:.3f}")
    
    return cell_stats

def test_individual_probes(samples, grid_size, test_cells=[(2, 2), (1, 3), (3, 1)]):
    """Test individual probes for specific cells with detailed diagnostics."""
    print(f"\n🔬 INDIVIDUAL PROBE TESTING")
    
    # Extract features
    X = np.array([s['features'] for s in samples])
    print(f"Features shape: {X.shape}")
    
    # Apply PCA
    if X.shape[1] > 50:
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X)
        print(f"PCA: {X.shape[1]} → {X_pca.shape[1]} dimensions")
    else:
        X_pca = X
    
    # Simple train/test split
    n_train = int(0.7 * len(samples))
    indices = list(range(len(samples)))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X_pca[train_indices]
    X_test = X_pca[test_indices]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Test specific cells
    for cell_x, cell_y in test_cells:
        print(f"\n--- Testing cell ({cell_x}, {cell_y}) ---")
        
        # Create labels
        y_all = [1 if (cell_x, cell_y) in s['grid_visits'] else 0 for s in samples]
        y_train = [y_all[i] for i in train_indices]
        y_test = [y_all[i] for i in test_indices]
        
        positive_train = sum(y_train)
        positive_test = sum(y_test)
        
        print(f"Train: {positive_train}/{len(y_train)} positive ({positive_train/len(y_train)*100:.1f}%)")
        print(f"Test: {positive_test}/{len(y_test)} positive ({positive_test/len(y_test)*100:.1f}%)")
        
        # Check if we can train
        if len(set(y_train)) < 2:
            print("❌ Cannot train - only one class in training data")
            continue
        
        if positive_train < 5 or positive_test < 2:
            print("⚠️  Very few positive samples - results may be unreliable")
        
        # Train probe
        try:
            # Try both balanced and unbalanced
            for balance_name, balance_param in [("Unbalanced", None), ("Balanced", 'balanced')]:
                model = LogisticRegression(max_iter=1000, random_state=42, class_weight=balance_param)
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Metrics
                                 accuracy = accuracy_score(y_test, y_pred)
                 balanced_acc = balanced_accuracy_score(y_test, y_pred)
                 f1 = f1_score(y_test, y_pred, zero_division='0.0')
                
                print(f"  {balance_name} classifier:")
                print(f"    Accuracy: {accuracy:.3f}")
                print(f"    Balanced Accuracy: {balanced_acc:.3f}")
                print(f"    F1 Score: {f1:.3f}")
                
                # Feature importance (top 5)
                if hasattr(model, 'coef_'):
                    feature_importance = np.abs(model.coef_[0])
                    top_features = np.argsort(feature_importance)[-5:]
                    print(f"    Top 5 feature indices: {top_features}")
                    print(f"    Top 5 feature weights: {feature_importance[top_features]}")
        
        except Exception as e:
            print(f"❌ Error training probe: {str(e)}")

def test_simpler_predictions(samples):
    """Test simpler predictions that should be easier."""
    print(f"\n🎯 SIMPLER PREDICTION TESTS")
    
    # Extract features
    X = np.array([s['features'] for s in samples])
    if X.shape[1] > 50:
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = X
    
    # Simple train/test split
    n_train = int(0.7 * len(samples))
    indices = list(range(len(samples)))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X_pca[train_indices]
    X_test = X_pca[test_indices]
    
    # Test 1: Will agent move to ANY new grid cell?
    print(f"\n--- Test 1: Will agent visit ANY new grid cell? ---")
    y_move = []
    for s in samples:
        current_grid = position_to_grid_cell(s['position'], 5, (-2, -2, 2, 2))
        will_move = any(pos for pos in s['future_positions'] 
                       if tuple(position_to_grid_cell(pos, 5, (-2, -2, 2, 2))) != tuple(current_grid))
        y_move.append(1 if will_move else 0)
    
    y_train_move = [y_move[i] for i in train_indices]
    y_test_move = [y_move[i] for i in test_indices]
    
    if len(set(y_train_move)) >= 2:
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train_move)
        y_pred = model.predict(X_test)
        
        acc = balanced_accuracy_score(y_test_move, y_pred)
        print(f"Will move to new cell - Balanced Accuracy: {acc:.3f}")
        print(f"Positive samples: {sum(y_train_move)}/{len(y_train_move)} train, {sum(y_test_move)}/{len(y_test_move)} test")
    
    # Test 2: Distance moved
    print(f"\n--- Test 2: Will agent move far (>0.5 units)? ---")
    y_far = []
    for s in samples:
        max_distance = 0
        for future_pos in s['future_positions']:
            dist = np.linalg.norm(np.array(future_pos) - np.array(s['position']))
            max_distance = max(max_distance, dist)
        y_far.append(1 if max_distance > 0.5 else 0)
    
    y_train_far = [y_far[i] for i in train_indices]
    y_test_far = [y_far[i] for i in test_indices]
    
    if len(set(y_train_far)) >= 2:
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train_far)
        y_pred = model.predict(X_test)
        
        acc = balanced_accuracy_score(y_test_far, y_pred)
        print(f"Will move far - Balanced Accuracy: {acc:.3f}")
        print(f"Positive samples: {sum(y_train_far)}/{len(y_train_far)} train, {sum(y_test_far)}/{len(y_test_far)} test")

def main():
    parser = argparse.ArgumentParser(description='Grid Prediction Diagnostics')
    parser.add_argument('--goal', default="FG blue", help='LTL goal to test')
    parser.add_argument('--grid-size', type=int, default=5, help='Grid size')
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--n-worlds', type=int, default=4, help='Number of worlds')
    parser.add_argument('--n-rollouts', type=int, default=8, help='Rollouts per world')
    parser.add_argument('--max-steps', type=int, default=60, help='Max steps per rollout')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=== GRID PREDICTION DIAGNOSTICS ===")
    print(f"🔍 Investigating why grid prediction performs so poorly")
    print(f"Goal: {args.goal}, Grid: {args.grid_size}x{args.grid_size}, Horizon: {args.horizon}")
    
    # Load model
    print("\nLoading model...")
    store = ModelStore(ENV, EXP, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial(args.goal), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Collect data with diagnostics
    world_ids = list(range(args.n_worlds))
    samples, all_positions, all_future_positions, grid_size = collect_diagnostic_data(
        model, world_ids, args.goal, args.grid_size, 
        args.n_rollouts, args.max_steps, args.horizon
    )
    
    if len(samples) == 0:
        print("❌ No samples collected!")
        return
    
    # Run diagnostics
    current_freq, future_freq = diagnose_grid_coverage(
        samples, all_positions, all_future_positions, grid_size, (-2, -2, 2, 2)
    )
    
    cell_stats = diagnose_class_balance(samples, grid_size)
    
    test_individual_probes(samples, grid_size)
    
    test_simpler_predictions(samples)
    
    print(f"\n=== DIAGNOSTIC CONCLUSIONS ===")
    print(f"✅ We ARE training separate probes for each grid cell")
    print(f"✅ Each probe is a proper binary classifier")
    print(f"⚠️  Check the grid coverage and class balance results above")
    print(f"⚠️  The issue might be severe class imbalance or insufficient data")
    print(f"📊 Review the heatmaps saved to corrected_results/")

if __name__ == "__main__":
    main() 