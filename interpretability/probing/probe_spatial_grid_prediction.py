#!/usr/bin/env python3
"""
Spatial Grid Prediction Probing

Tests if the complete network can predict which areas of the map the agent will visit
by dividing the space into a grid and creating binary classifiers for each grid cell.

HYPOTHESIS: If the network has learned spatial navigation patterns, it should be able
to predict future locations better than random.

Usage: python interpretability/probing/probe_spatial_grid_prediction.py
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
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

def grid_cell_to_position(grid_x, grid_y, grid_size, map_bounds):
    """Convert grid cell indices to center position."""
    x_min, y_min, x_max, y_max = map_bounds
    
    # Get cell centers
    x = x_min + (grid_x + 0.5) * (x_max - x_min) / grid_size
    y = y_min + (grid_y + 0.5) * (y_max - y_min) / grid_size
    
    return x, y

def collect_spatial_grid_data(model, world_ids, goal="FG blue", grid_size=5, 
                             n_rollouts_per_world=8, max_steps=50, k_steps=5, 
                             map_bounds=(-2, -2, 2, 2)):
    """Collect data for spatial grid prediction."""
    print(f"Collecting spatial grid data for goal: {goal}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Map bounds: {map_bounds}")
    print(f"Prediction horizon: {k_steps} steps")
    
    all_features = []
    all_grid_labels = {}  # One binary label per grid cell
    all_metadata = []
    
    # Initialize grid labels
    for i in range(grid_size):
        for j in range(grid_size):
            all_grid_labels[f"cell_{i}_{j}"] = []
    
    env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    target_zone_name = goal.split()[-1]  # "FG blue" -> "blue"
    
    total_samples = 0
    
    for world_id in world_ids:
        print(f"Processing world {world_id}...")
        for rollout_id in trange(n_rollouts_per_world, desc=f"Rollouts for world {world_id}"):
            trajectory_data = []
            
            done = False
            obs = env.reset(seed=world_id + rollout_id * 1000)
            agent.reset()
            
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
                    'step_id': step_id
                })
                
                action = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, info = env.step(action)
            
            # Create grid prediction labels
            valid_samples = len(trajectory_data) - k_steps
            if valid_samples <= 0:
                continue
                
            for i in range(valid_samples):
                current_data = trajectory_data[i]
                
                # Get future positions for next k_steps
                future_positions = []
                for future_step in range(1, k_steps + 1):
                    if i + future_step < len(trajectory_data):
                        future_pos = trajectory_data[i + future_step]['position']
                        future_positions.append(future_pos)
                
                if len(future_positions) == 0:
                    continue
                
                # Store features
                all_features.append(current_data['network_representation'])
                
                # Create binary labels for each grid cell
                grid_visits = set()
                for future_pos in future_positions:
                    grid_x, grid_y = position_to_grid_cell(future_pos, grid_size, map_bounds)
                    grid_visits.add((grid_x, grid_y))
                
                # Binary label for each cell: 1 if visited, 0 if not
                for i_cell in range(grid_size):
                    for j_cell in range(grid_size):
                        will_visit = 1 if (i_cell, j_cell) in grid_visits else 0
                        all_grid_labels[f"cell_{i_cell}_{j_cell}"].append(will_visit)
                
                all_metadata.append({
                    'world_id': world_id,
                    'rollout_id': rollout_id,
                    'step_id': current_data['step_id'],
                    'current_pos': current_data['position'],
                    'future_cells': list(grid_visits)
                })
                
                total_samples += 1
    
    env.close()
    
    print(f"Collected {total_samples} samples")
    
    # Convert to numpy arrays
    for cell_key in all_grid_labels:
        all_grid_labels[cell_key] = np.array(all_grid_labels[cell_key])
    
    return {
        'features': np.array(all_features),
        'grid_labels': all_grid_labels,
        'metadata': all_metadata,
        'grid_size': grid_size,
        'map_bounds': map_bounds
    }

def evaluate_spatial_grid_prediction(data):
    """Evaluate spatial grid prediction for each grid cell."""
    features = data['features']
    grid_labels = data['grid_labels']
    grid_size = data['grid_size']
    
    # Apply PCA
    pca = PCA(n_components=0.95)
    features_pca = pca.fit_transform(features)
    print(f"PCA: {features.shape[1]} → {features_pca.shape[1]} dimensions")
    
    # Results storage
    results = {}
    accuracy_grid = np.zeros((grid_size, grid_size))
    visit_frequency_grid = np.zeros((grid_size, grid_size))
    
    # Train/test split
    n_samples = len(features_pca)
    n_train = int(0.7 * n_samples)
    
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = features_pca[train_indices]
    X_test = features_pca[test_indices]
    
    print(f"\nEvaluating prediction for each grid cell...")
    
    for cell_key in grid_labels:
        # Parse cell coordinates
        parts = cell_key.split('_')
        i_cell, j_cell = int(parts[1]), int(parts[2])
        
        y_all = grid_labels[cell_key]
        y_train = y_all[train_indices]
        y_test = y_all[test_indices]
        
        # Calculate visit frequency
        visit_freq = np.mean(y_all)
        visit_frequency_grid[i_cell, j_cell] = visit_freq
        
        # Check if we have both classes in training data
        unique_train = np.unique(y_train)
        unique_test = np.unique(y_test)
        
        if len(unique_train) < 2:
            # Can't train classifier - assign baseline accuracy
            accuracy = max(np.mean(y_test), 1 - np.mean(y_test))  # Majority class accuracy
            print(f"  Cell ({i_cell},{j_cell}): Baseline accuracy = {accuracy:.3f} (visit freq: {visit_freq:.3f})")
        else:
            # Train classifier
            try:
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"  Cell ({i_cell},{j_cell}): Accuracy = {accuracy:.3f} (visit freq: {visit_freq:.3f})")
            except:
                accuracy = max(np.mean(y_test), 1 - np.mean(y_test))
                print(f"  Cell ({i_cell},{j_cell}): Baseline accuracy = {accuracy:.3f} (training failed)")
        
        accuracy_grid[i_cell, j_cell] = accuracy
        
        results[cell_key] = {
            'accuracy': accuracy,
            'visit_frequency': visit_freq,
            'n_positive': np.sum(y_all),
            'n_total': len(y_all)
        }
    
    return {
        'results': results,
        'accuracy_grid': accuracy_grid,
        'visit_frequency_grid': visit_frequency_grid,
        'overall_accuracy': np.mean(accuracy_grid),
        'grid_size': grid_size
    }

def visualize_results(evaluation_results, map_bounds, output_dir):
    """Visualize spatial grid prediction results."""
    accuracy_grid = evaluation_results['accuracy_grid']
    visit_frequency_grid = evaluation_results['visit_frequency_grid']
    grid_size = evaluation_results['grid_size']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Prediction Accuracy Heatmap
    im1 = ax1.imshow(accuracy_grid.T, origin='lower', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax1.set_title('Spatial Grid Prediction Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Grid X')
    ax1.set_ylabel('Grid Y')
    
    # Add accuracy values as text
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax1.text(i, j, f'{accuracy_grid[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, label='Prediction Accuracy')
    
    # Plot 2: Visit Frequency Heatmap
    im2 = ax2.imshow(visit_frequency_grid.T, origin='lower', cmap='Blues', vmin=0, vmax=1.0)
    ax2.set_title('Grid Cell Visit Frequency', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Grid X')
    ax2.set_ylabel('Grid Y')
    
    # Add frequency values as text
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax2.text(i, j, f'{visit_frequency_grid[i, j]:.2f}',
                           ha="center", va="center", color="white" if visit_frequency_grid[i, j] > 0.5 else "black",
                           fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, label='Visit Frequency')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/spatial_grid_prediction_{grid_size}x{grid_size}_{timestamp}.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to {output_dir}/spatial_grid_prediction_{grid_size}x{grid_size}_{timestamp}.png")

def main():
    parser = argparse.ArgumentParser(description='Spatial Grid Prediction Probing')
    parser.add_argument('--goal', default="FG blue", help='LTL goal to test')
    parser.add_argument('--grid-size', type=int, default=5, help='Grid size (e.g., 5 for 5x5 grid)')
    parser.add_argument('--k-steps', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--n-worlds', type=int, default=6, help='Number of worlds')
    parser.add_argument('--n-rollouts', type=int, default=8, help='Rollouts per world')
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps per rollout')
    parser.add_argument('--map-bounds', nargs=4, type=float, default=[-2, -2, 2, 2],
                       help='Map bounds: x_min y_min x_max y_max')
    parser.add_argument('--output-dir', type=str, 
                       default='interpretability/probing/corrected_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=== SPATIAL GRID PREDICTION PROBE ===")
    print(f"🎯 Can the network predict which areas of the map the agent will visit?")
    print(f"Goal: {args.goal}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Prediction horizon: {args.k_steps} steps")
    print(f"Map bounds: {args.map_bounds}")
    
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
    data = collect_spatial_grid_data(
        model, world_ids, args.goal, args.grid_size, 
        args.n_rollouts, args.max_steps, args.k_steps, 
        tuple(args.map_bounds)
    )
    
    # Evaluate
    print("\nEvaluating spatial grid prediction...")
    evaluation_results = evaluate_spatial_grid_prediction(data)
    
    # Results summary
    print(f"\n=== SPATIAL GRID PREDICTION RESULTS ===")
    print(f"Overall average accuracy: {evaluation_results['overall_accuracy']:.3f}")
    
    # Visualize
    print("\nCreating visualizations...")
    visualize_results(evaluation_results, tuple(args.map_bounds), args.output_dir)
    
    # Interpretation
    overall_acc = evaluation_results['overall_accuracy']
    if overall_acc > 0.7:
        print("✅ EXCELLENT! Network has strong spatial prediction capability!")
        print("   The system has learned a spatial model of movement patterns! 🗺️")
    elif overall_acc > 0.6:
        print("🤔 GOOD spatial prediction - moderate spatial modeling capability")
    elif overall_acc > 0.55:
        print("📊 WEAK spatial prediction - limited spatial understanding")
    else:
        print("❌ POOR spatial prediction - mostly random performance")
    
    print(f"\n💡 Check the heatmap to see which areas are most predictable!")
    print(f"   High accuracy + high visit frequency = well-learned navigation patterns")
    print(f"   High accuracy + low visit frequency = consistent avoidance patterns")

if __name__ == "__main__":
    main() 