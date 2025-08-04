#!/usr/bin/env python3
"""
Temporal Horizons Spatial Grid Probing with Generalization Splits

Tests "will this grid cell be visited in the next N steps?" across different values of N
and across three generalization splits (temporal, spatial, environmental) to understand 
if the network has short-term vs long-term spatial planning capabilities and how this
varies across different types of generalization.

HYPOTHESIS: 
- Short-term (N=1-3): May show some predictive capability, especially temporal split
- Medium-term (N=5-10): Our previous findings 
- Long-term (N=15-30): Likely worse performance due to planning limitations
- Temporal split may show better performance than spatial/environmental

Usage: python interpretability/probing/probe_temporal_horizons.py
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from datetime import datetime

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
    
    # Normalize to [0, 1]
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    
    # Convert to grid indices
    grid_x = int(np.clip(x_norm * grid_size, 0, grid_size - 1))
    grid_y = int(np.clip(y_norm * grid_size, 0, grid_size - 1))
    
    return grid_x, grid_y

def collect_temporal_horizon_data(model, world_ids, goal="FG blue", grid_size=5, 
                                 n_rollouts_per_world=8, max_steps=60, 
                                 horizon_steps=[1, 3, 5, 10, 15, 20],
                                 map_bounds=(-2, -2, 2, 2)):
    """Collect data for temporal horizon analysis with metadata for splits."""
    print(f"Collecting temporal horizon data for goal: {goal}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Max trajectory length: {max_steps} steps") 
    print(f"Testing horizons: {horizon_steps} steps")
    
    max_horizon = max(horizon_steps)
    
    # Store data for each horizon with metadata for splits
    horizon_data = {}
    for N in horizon_steps:
        horizon_data[N] = {
            'features': [],
            'grid_labels': {f"cell_{i}_{j}": [] for i in range(grid_size) for j in range(grid_size)},
            'metadata': []
        }
    
    env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    total_trajectories = 0
    total_samples = {N: 0 for N in horizon_steps}
    
    for world_id in world_ids:
        print(f"Processing world {world_id}...")
        for rollout_id in trange(n_rollouts_per_world, desc=f"Rollouts for world {world_id}"):
            trajectory_data = []
            
            done = False
            obs = env.reset(seed=world_id + rollout_id * 1000)
            agent.reset()
            
            # Collect complete trajectory
            for step_id in range(max_steps):
                if done:
                    break
                    
                current_pos = env.agent_pos[:2].copy()
                
                # Get network representation
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
                    'network_representation': network_representation,
                    'position': current_pos,
                    'step_id': step_id,
                    'world_id': world_id,
                    'rollout_id': rollout_id
                })
                
                action = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, info = env.step(action)
            
            # Skip short trajectories
            if len(trajectory_data) < max_horizon + 5:
                continue
                
            total_trajectories += 1
            
            # Create labels for different temporal horizons
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
                    
                    # Store features (same for all horizons)
                    horizon_data[N]['features'].append(current_data['network_representation'])
                    
                    # Create binary labels for each grid cell
                    grid_visits = set()
                    for future_pos in future_positions:
                        grid_x, grid_y = position_to_grid_cell(future_pos, grid_size, map_bounds)
                        grid_visits.add((grid_x, grid_y))
                    
                    # Binary label for each cell: 1 if visited in next N steps, 0 if not
                    for i_cell in range(grid_size):
                        for j_cell in range(grid_size):
                            will_visit = 1 if (i_cell, j_cell) in grid_visits else 0
                            horizon_data[N]['grid_labels'][f"cell_{i_cell}_{j_cell}"].append(will_visit)
                    
                    horizon_data[N]['metadata'].append({
                        'world_id': world_id,
                        'rollout_id': rollout_id,
                        'step_id': current_data['step_id'],
                        'current_pos': current_data['position'],
                        'horizon': N
                    })
                    
                    total_samples[N] += 1
    
    env.close()
    
    print(f"Processed {total_trajectories} complete trajectories")
    for N in horizon_steps:
        print(f"  Horizon {N} steps: {total_samples[N]} samples")
    
    # Convert to numpy arrays
    for N in horizon_steps:
        horizon_data[N]['features'] = np.array(horizon_data[N]['features'])
        for cell_key in horizon_data[N]['grid_labels']:
            horizon_data[N]['grid_labels'][cell_key] = np.array(horizon_data[N]['grid_labels'][cell_key])
    
    return horizon_data, total_trajectories

def create_generalization_splits(features, labels, metadata, split_ratio=0.7):
    """Create temporal, spatial, and environmental generalization splits."""
    n_samples = len(features)
    splits = {}
    
    # 1. TEMPORAL SPLIT: Early vs late steps within same rollouts
    step_ids = np.array([meta['step_id'] for meta in metadata])
    median_step = np.median(step_ids)
    
    early_mask = step_ids <= median_step
    late_mask = step_ids > median_step
    
    # Use early steps for training, late steps for testing
    splits['temporal'] = {
        'train_indices': np.where(early_mask)[0],
        'test_indices': np.where(late_mask)[0],
        'description': f'Train: early steps (≤{median_step:.1f}), Test: late steps (>{median_step:.1f})'
    }
    
    # 2. SPATIAL SPLIT: Some rollouts vs other rollouts within same worlds
    unique_rollouts = list(set((meta['world_id'], meta['rollout_id']) for meta in metadata))
    n_train_rollouts = int(len(unique_rollouts) * split_ratio)
    
    random.shuffle(unique_rollouts)
    train_rollouts = set(unique_rollouts[:n_train_rollouts])
    test_rollouts = set(unique_rollouts[n_train_rollouts:])
    
    train_mask = np.array([(meta['world_id'], meta['rollout_id']) in train_rollouts for meta in metadata])
    test_mask = np.array([(meta['world_id'], meta['rollout_id']) in test_rollouts for meta in metadata])
    
    splits['spatial'] = {
        'train_indices': np.where(train_mask)[0],
        'test_indices': np.where(test_mask)[0],
        'description': f'Train: {len(train_rollouts)} rollouts, Test: {len(test_rollouts)} rollouts'
    }
    
    # 3. ENVIRONMENTAL SPLIT: Some worlds vs other worlds
    unique_worlds = list(set(meta['world_id'] for meta in metadata))
    n_train_worlds = max(1, int(len(unique_worlds) * split_ratio))
    
    random.shuffle(unique_worlds)
    train_worlds = set(unique_worlds[:n_train_worlds])
    test_worlds = set(unique_worlds[n_train_worlds:])
    
    world_train_mask = np.array([meta['world_id'] in train_worlds for meta in metadata])
    world_test_mask = np.array([meta['world_id'] in test_worlds for meta in metadata])
    
    splits['environmental'] = {
        'train_indices': np.where(world_train_mask)[0],
        'test_indices': np.where(world_test_mask)[0],
        'description': f'Train: worlds {sorted(train_worlds)}, Test: worlds {sorted(test_worlds)}'
    }
    
    return splits

def evaluate_temporal_horizon_with_splits(data, N, grid_size, min_visit_freq=0.05):
    """Evaluate spatial prediction for a specific temporal horizon N across all splits."""
    features = data['features']
    grid_labels = data['grid_labels']
    metadata = data['metadata']
    
    if len(features) == 0:
        return None
    
    print(f"\n--- Evaluating N={N} step prediction ---")
    print(f"Features shape: {features.shape}")
    
    # Apply PCA if needed
    if features.shape[1] > 50:
        pca = PCA(n_components=0.95)
        features_pca = pca.fit_transform(features)
        print(f"PCA: {features.shape[1]} → {features_pca.shape[1]} dimensions")
    else:
        features_pca = features
    
    # Create generalization splits
    # We'll use the first cell's labels as a dummy for split creation (all cells have same metadata)
    dummy_labels = grid_labels[list(grid_labels.keys())[0]]
    splits = create_generalization_splits(features_pca, dummy_labels, metadata)
    
    results_by_split = {}
    
    for split_name, split_info in splits.items():
        print(f"\n  === {split_name.upper()} SPLIT ===")
        print(f"  {split_info['description']}")
        
        train_indices = split_info['train_indices']
        test_indices = split_info['test_indices']
        
        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"  ❌ Insufficient data for {split_name} split")
            continue
        
        X_train = features_pca[train_indices]
        X_test = features_pca[test_indices]
        
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        meaningful_cells = 0
        balanced_accuracies = []
        f1_scores = []
        visit_frequencies = []
        cell_results = {}
        
        for cell_key in grid_labels:
            y_all = grid_labels[cell_key]
            y_train = y_all[train_indices]
            y_test = y_all[test_indices]
            
            # Calculate visit frequency
            visit_freq = np.mean(y_all) if len(y_all) > 0 else 0
            visit_frequencies.append(visit_freq)
            
            # Check if we have both classes and sufficient visit frequency
            unique_train = np.unique(y_train)
            
            if len(unique_train) < 2 or len(y_train) == 0 or len(y_test) == 0 or visit_freq < min_visit_freq:
                # Insufficient data
                balanced_acc = 0.5  # Random performance
                f1 = 0.0
                status = "insufficient_data"
            else:
                meaningful_cells += 1
                # Train balanced classifier
                try:
                    classes = np.unique(y_train)
                    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                    class_weight_dict = dict(zip(classes, class_weights))
                    
                    model = LogisticRegression(max_iter=1000, random_state=42, 
                                             class_weight=class_weight_dict)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    
                    balanced_acc = balanced_accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, zero_division='0')
                    
                    status = "success"
                except Exception as e:
                    balanced_acc = 0.5
                    f1 = 0.0
                    status = f"error: {str(e)[:30]}"
            
            balanced_accuracies.append(balanced_acc)
            f1_scores.append(f1)
            
            cell_results[cell_key] = {
                'balanced_accuracy': balanced_acc,
                'f1_score': f1,
                'visit_frequency': visit_freq,
                'status': status
            }
        
        # Overall metrics for this split
        overall_balanced_acc = np.mean(balanced_accuracies)
        overall_f1 = np.mean(f1_scores)
        avg_visit_freq = np.mean(visit_frequencies)
        
        results_by_split[split_name] = {
            'meaningful_cells': meaningful_cells,
            'total_cells': grid_size * grid_size,
            'overall_balanced_accuracy': overall_balanced_acc,
            'overall_f1_score': overall_f1,
            'avg_visit_frequency': avg_visit_freq,
            'cell_results': cell_results,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        print(f"  Meaningful cells: {meaningful_cells}/{grid_size*grid_size}")
        print(f"  Overall balanced accuracy: {overall_balanced_acc:.3f}")
        print(f"  Overall F1 score: {overall_f1:.3f}")
    
    summary = {
        'horizon': N,
        'n_samples': len(features),
        'results_by_split': results_by_split
    }
    
    return summary

def visualize_temporal_horizon_splits_results(horizon_results, output_dir):
    """Visualize how spatial prediction performance varies with temporal horizon across splits."""
    # Extract data for visualization
    horizons = []
    split_names = ['temporal', 'spatial', 'environmental']
    split_data = {split: {'balanced_accs': [], 'f1_scores': [], 'meaningful_cells': []} 
                  for split in split_names}
    
    for result in horizon_results:
        if result is not None:
            horizons.append(result['horizon'])
            for split_name in split_names:
                if split_name in result['results_by_split']:
                    split_result = result['results_by_split'][split_name]
                    split_data[split_name]['balanced_accs'].append(split_result['overall_balanced_accuracy'])
                    split_data[split_name]['f1_scores'].append(split_result['overall_f1_score'])
                    split_data[split_name]['meaningful_cells'].append(split_result['meaningful_cells'])
                else:
                    # Missing data
                    split_data[split_name]['balanced_accs'].append(0.5)
                    split_data[split_name]['f1_scores'].append(0.0)
                    split_data[split_name]['meaningful_cells'].append(0)
    
    if len(horizons) == 0:
        print("No data to visualize")
        return
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'temporal': 'blue', 'spatial': 'green', 'environmental': 'red'}
    markers = {'temporal': 'o', 'spatial': 's', 'environmental': '^'}
    
    # Plot 1: Balanced Accuracy vs Horizon (all splits)
    for split_name in split_names:
        if len(split_data[split_name]['balanced_accs']) > 0:
            axes[0,0].plot(horizons, split_data[split_name]['balanced_accs'], 
                          color=colors[split_name], marker=markers[split_name], 
                          linewidth=2, markersize=8, label=f'{split_name.capitalize()} split')
    
    axes[0,0].axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Random baseline')
    axes[0,0].set_xlabel('Prediction Horizon (steps)')
    axes[0,0].set_ylabel('Balanced Accuracy')
    axes[0,0].set_title('Spatial Prediction vs Temporal Horizon (All Splits)', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    axes[0,0].set_ylim(0.45, 0.65)
    
    # Plot 2: F1 Score vs Horizon (all splits)
    for split_name in split_names:
        if len(split_data[split_name]['f1_scores']) > 0:
            axes[0,1].plot(horizons, split_data[split_name]['f1_scores'], 
                          color=colors[split_name], marker=markers[split_name], 
                          linewidth=2, markersize=8, label=f'{split_name.capitalize()} split')
    
    axes[0,1].set_xlabel('Prediction Horizon (steps)')
    axes[0,1].set_ylabel('F1 Score')
    axes[0,1].set_title('F1 Score vs Temporal Horizon (All Splits)', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Plot 3: Meaningful Cells vs Horizon (all splits)
    for split_name in split_names:
        if len(split_data[split_name]['meaningful_cells']) > 0:
            axes[1,0].plot(horizons, split_data[split_name]['meaningful_cells'], 
                          color=colors[split_name], marker=markers[split_name], 
                          linewidth=2, markersize=8, label=f'{split_name.capitalize()} split')
    
    axes[1,0].set_xlabel('Prediction Horizon (steps)')
    axes[1,0].set_ylabel('Meaningful Cells')
    axes[1,0].set_title('Meaningful Cells vs Temporal Horizon (All Splits)', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # Plot 4: Heatmap of balanced accuracy by horizon and split
    split_labels = [s.capitalize() for s in split_names]
    heatmap_data = np.array([split_data[split]['balanced_accs'] for split in split_names])
    
    im = axes[1,1].imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0.45, vmax=0.65)
    axes[1,1].set_xticks(range(len(horizons)))
    axes[1,1].set_xticklabels(horizons)
    axes[1,1].set_yticks(range(len(split_labels)))
    axes[1,1].set_yticklabels(split_labels)
    axes[1,1].set_xlabel('Prediction Horizon (steps)')
    axes[1,1].set_ylabel('Generalization Split')
    axes[1,1].set_title('Balanced Accuracy Heatmap', fontweight='bold')
    
    # Add text annotations to heatmap
    for i in range(len(split_labels)):
        for j in range(len(horizons)):
            text = axes[1,1].text(j, i, f'{heatmap_data[i, j]:.3f}', 
                                ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1,1])
    cbar.set_label('Balanced Accuracy')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/temporal_horizons_splits_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Temporal horizon splits visualization saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Temporal Horizons Spatial Grid Probing with Generalization Splits')
    parser.add_argument('--goal', default="FG blue", help='LTL goal to test')
    parser.add_argument('--grid-size', type=int, default=5, help='Grid size')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 3, 5, 10, 15, 20],
                       help='Prediction horizons to test (in steps)')
    parser.add_argument('--min-visit-freq', type=float, default=0.05, 
                       help='Minimum visit frequency for meaningful cells')
    parser.add_argument('--n-worlds', type=int, default=6, help='Number of worlds')
    parser.add_argument('--n-rollouts', type=int, default=8, help='Rollouts per world')
    parser.add_argument('--max-steps', type=int, default=60, help='Max steps per rollout')
    parser.add_argument('--output-dir', type=str, 
                       default='interpretability/probing/corrected_results',
                       help='Output directory for results')
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
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Temporal horizons: {args.horizons} steps")
    print(f"This will reveal how planning capabilities vary across generalization types!")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    horizon_data, n_trajectories = collect_temporal_horizon_data(
        model, world_ids, args.goal, args.grid_size, 
        args.n_rollouts, args.max_steps, args.horizons
    )
    
    if n_trajectories == 0:
        print("❌ No trajectories collected!")
        return
    
    # Evaluate each temporal horizon across all splits
    print(f"\nEvaluating spatial prediction across temporal horizons and generalization splits...")
    horizon_results = []
    
    for N in args.horizons:
        if N in horizon_data and len(horizon_data[N]['features']) > 0:
            result = evaluate_temporal_horizon_with_splits(horizon_data[N], N, args.grid_size, args.min_visit_freq)
            horizon_results.append(result)
        else:
            print(f"\n--- No data for N={N} step prediction ---")
            horizon_results.append(None)
    
    # Save results
    print("\nSaving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create summary dataframe
    summary_data = []
    for result in horizon_results:
        if result is not None:
            for split_name, split_result in result['results_by_split'].items():
                summary_data.append({
                    'horizon': result['horizon'],
                    'split_type': split_name,
                    'n_samples': result['n_samples'],
                    'n_train': split_result['n_train'],
                    'n_test': split_result['n_test'],
                    'meaningful_cells': split_result['meaningful_cells'],
                    'total_cells': split_result['total_cells'],
                    'balanced_accuracy': split_result['overall_balanced_accuracy'],
                    'f1_score': split_result['overall_f1_score'],
                    'avg_visit_frequency': split_result['avg_visit_frequency']
                })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = f"{args.output_dir}/temporal_horizons_splits_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Summary results saved to: {csv_path}")
    
    # Visualize
    print("\nCreating temporal horizon × splits visualization...")
    visualize_temporal_horizon_splits_results(horizon_results, args.output_dir)
    
    # Comprehensive analysis and interpretation
    print(f"\n=== TEMPORAL HORIZON × GENERALIZATION ANALYSIS ===")
    if summary_data:
        df = pd.DataFrame(summary_data)
        
                                   print(f"📊 PERFORMANCE SUMMARY BY SPLIT:")
         for split_name in ['temporal', 'spatial', 'environmental']:
             split_df = df[df['split_type'] == split_name]
             if len(split_df) > 0:
                 max_acc_for_split = 0.0
                 best_horizon = 0
                 for _, row in split_df.iterrows():
                     if row['balanced_accuracy'] > max_acc_for_split:
                         max_acc_for_split = row['balanced_accuracy']
                         best_horizon = row['horizon']
                 avg_performance = split_df['balanced_accuracy'].mean()
                 print(f"  {split_name.upper()} split:")
                 print(f"    Best: {best_horizon} steps (Bal Acc = {max_acc_for_split:.3f})")
                 print(f"    Average: {avg_performance:.3f}")
         
         # Check for temporal patterns across splits
         print(f"\n🕐 TEMPORAL PLANNING ANALYSIS BY SPLIT:")
         for split_name in ['temporal', 'spatial', 'environmental']:
             split_df = df[df['split_type'] == split_name]
             if len(split_df) > 0:
                 short_term = split_df[split_df['horizon'] <= 5]['balanced_accuracy'].mean()
                 long_term = split_df[split_df['horizon'] >= 10]['balanced_accuracy'].mean()
                 
                 print(f"  {split_name.upper()} split:")
                 print(f"    Short-term (≤5 steps): {short_term:.3f}")
                 print(f"    Long-term (≥10 steps): {long_term:.3f}")
                 
                 if short_term > long_term + 0.02:
                     print(f"    → ✅ SHORT-TERM advantage in {split_name} generalization")
                 elif long_term > short_term + 0.02:
                     print(f"    → 🤔 LONG-TERM advantage in {split_name} generalization")
                 else:
                     print(f"    → ❌ No temporal preference in {split_name} generalization")
         
         # Best performing combination
         max_acc_overall = 0.0
         best_split = ""
         best_horizon_overall = 0
         for _, row in df.iterrows():
             if row['balanced_accuracy'] > max_acc_overall:
                 max_acc_overall = row['balanced_accuracy']
                 best_split = row['split_type']
                 best_horizon_overall = row['horizon']
         
         print(f"\n🏆 BEST COMBINATION:")
         print(f"Split: {best_split}, Horizon: {best_horizon_overall} steps")
         print(f"Balanced Accuracy: {max_acc_overall:.3f}")
         
         # Overall interpretation
         max_acc = max_acc_overall
        if max_acc > 0.6:
            print(f"\n🚀 SIGNIFICANT spatial prediction capability detected!")
            print(f"   Found in {best_combo['split_type']} split at {best_combo['horizon']} steps")
        elif max_acc > 0.55:
            print(f"\n🤔 WEAK spatial prediction capability")
            print(f"   Some evidence in {best_combo['split_type']} split")
        else:
            print(f"\n❌ NO meaningful spatial prediction capability")
            print(f"   All combinations perform near chance level")
            print(f"   Consistent across temporal, spatial, and environmental generalization")

if __name__ == "__main__":
    main() 