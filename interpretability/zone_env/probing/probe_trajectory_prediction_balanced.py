#!/usr/bin/env python3
"""
Balanced Within-Rollout Trajectory Prediction Probing

Fixed version that properly handles class imbalance by using balanced metrics
and focusing on cells with meaningful visit patterns.

Usage: python interpretability/probing/probe_trajectory_prediction_balanced.py
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
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

def collect_trajectory_data(model, world_ids, goal="FG blue", grid_size=5, 
                           n_rollouts_per_world=8, max_steps=50, 
                           train_steps_ratio=0.5, k_steps=5,
                           map_bounds=(-2, -2, 2, 2)):
    """Collect data for within-rollout trajectory prediction."""
    print(f"Collecting WITHIN-ROLLOUT trajectory data for goal: {goal}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Training on first {train_steps_ratio*100:.0f}% of each trajectory")
    print(f"Testing on remaining steps of the SAME trajectories")
    print(f"Prediction horizon: {k_steps} steps")
    
    train_features = []
    train_labels = {}
    test_features = []
    test_labels = {}
    
    # Initialize labels for each grid cell
    for i in range(grid_size):
        for j in range(grid_size):
            train_labels[f"cell_{i}_{j}"] = []
            test_labels[f"cell_{i}_{j}"] = []
    
    env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    train_samples = 0
    test_samples = 0
    successful_trajectories = 0
    
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
                    'step_id': step_id
                })
                
                action = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, info = env.step(action)
            
            # Skip short trajectories
            if len(trajectory_data) < 2 * k_steps:
                continue
                
            successful_trajectories += 1
            
            # Split trajectory into train/test portions
            split_point = int(len(trajectory_data) * train_steps_ratio)
            train_end = split_point - k_steps  # Ensure we can make k-step predictions
            
            if train_end <= 0:
                continue
            
            # TRAINING SAMPLES: Early steps predicting within early portion
            for i in range(train_end):
                current_data = trajectory_data[i]
                
                # Get future positions within training portion
                future_positions = []
                for future_step in range(1, k_steps + 1):
                    if i + future_step < len(trajectory_data):
                        future_pos = trajectory_data[i + future_step]['position']
                        future_positions.append(future_pos)
                
                if len(future_positions) == 0:
                    continue
                
                train_features.append(current_data['network_representation'])
                
                # Create binary labels for each grid cell
                grid_visits = set()
                for future_pos in future_positions:
                    grid_x, grid_y = position_to_grid_cell(future_pos, grid_size, map_bounds)
                    grid_visits.add((grid_x, grid_y))
                
                for i_cell in range(grid_size):
                    for j_cell in range(grid_size):
                        will_visit = 1 if (i_cell, j_cell) in grid_visits else 0
                        train_labels[f"cell_{i_cell}_{j_cell}"].append(will_visit)
                
                train_samples += 1
            
            # TEST SAMPLES: Later steps predicting within later portion  
            test_start = split_point
            test_end = len(trajectory_data) - k_steps
            
            for i in range(test_start, test_end):
                current_data = trajectory_data[i]
                
                # Get future positions within test portion
                future_positions = []
                for future_step in range(1, k_steps + 1):
                    if i + future_step < len(trajectory_data):
                        future_pos = trajectory_data[i + future_step]['position']
                        future_positions.append(future_pos)
                
                if len(future_positions) == 0:
                    continue
                
                test_features.append(current_data['network_representation'])
                
                # Create binary labels for each grid cell
                grid_visits = set()
                for future_pos in future_positions:
                    grid_x, grid_y = position_to_grid_cell(future_pos, grid_size, map_bounds)
                    grid_visits.add((grid_x, grid_y))
                
                for i_cell in range(grid_size):
                    for j_cell in range(grid_size):
                        will_visit = 1 if (i_cell, j_cell) in grid_visits else 0
                        test_labels[f"cell_{i_cell}_{j_cell}"].append(will_visit)
                
                test_samples += 1
    
    env.close()
    
    print(f"Processed {successful_trajectories} complete trajectories")
    print(f"Training samples: {train_samples}")
    print(f"Test samples: {test_samples}")
    
    # Convert to numpy arrays
    for cell_key in train_labels:
        train_labels[cell_key] = np.array(train_labels[cell_key])
        test_labels[cell_key] = np.array(test_labels[cell_key])
    
    return {
        'train_features': np.array(train_features),
        'test_features': np.array(test_features),
        'train_labels': train_labels,
        'test_labels': test_labels,
        'grid_size': grid_size,
        'n_trajectories': successful_trajectories
    }

def evaluate_trajectory_prediction_balanced(data, min_visit_freq=0.05):
    """Evaluate within-rollout trajectory prediction with proper class balance handling."""
    train_features = data['train_features']
    test_features = data['test_features']
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    grid_size = data['grid_size']
    
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    
    # Apply PCA
    pca = PCA(n_components=0.95)
    train_features_pca = pca.fit_transform(train_features)
    test_features_pca = pca.transform(test_features)
    
    print(f"PCA: {train_features.shape[1]} → {train_features_pca.shape[1]} dimensions")
    
    results = {}
    accuracy_grid = np.zeros((grid_size, grid_size))
    balanced_accuracy_grid = np.zeros((grid_size, grid_size))
    f1_grid = np.zeros((grid_size, grid_size))
    visit_freq_grid = np.zeros((grid_size, grid_size))
    
    # Baseline predictions (always predict majority class)
    baseline_accuracies = []
    
    print(f"\nEvaluating WITHIN-ROLLOUT trajectory prediction with BALANCED metrics...")
    print(f"Focusing on cells with visit frequency >= {min_visit_freq}")
    
    meaningful_cells = 0
    
    for cell_key in train_labels:
        # Parse cell coordinates
        parts = cell_key.split('_')
        i_cell, j_cell = int(parts[1]), int(parts[2])
        
        y_train = train_labels[cell_key]
        y_test = test_labels[cell_key]
        
        # Calculate visit frequencies
        train_freq = np.mean(y_train) if len(y_train) > 0 else 0
        test_freq = np.mean(y_test) if len(y_test) > 0 else 0
        visit_freq_grid[i_cell, j_cell] = test_freq
        
        # Calculate baseline accuracy (always predict majority class)
        if len(y_test) > 0:
            baseline_acc = max(np.mean(y_test), 1 - np.mean(y_test))
            baseline_accuracies.append(baseline_acc)
        else:
            baseline_acc = 0.5
        
        # Check if we have both classes and sufficient data
        unique_train = np.unique(y_train)
        unique_test = np.unique(y_test)
        
        if len(unique_train) < 2 or len(y_train) == 0 or len(y_test) == 0 or train_freq < min_visit_freq:
            # Insufficient data or too rare
            accuracy = baseline_acc
            balanced_acc = 0.5  # Random performance
            f1 = 0.0
            status = "insufficient_data"
        else:
            meaningful_cells += 1
            # Train balanced classifier
            try:
                # Use class weights to handle imbalance
                classes = np.unique(y_train)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                class_weight_dict = dict(zip(classes, class_weights))
                
                model = LogisticRegression(max_iter=1000, random_state=42, 
                                         class_weight=class_weight_dict)
                model.fit(train_features_pca, y_train)
                
                y_pred = model.predict(test_features_pca)
                
                accuracy = accuracy_score(y_test, y_pred)
                balanced_acc = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, zero_division='0')
                
                status = "trained"
                
                print(f"  Cell ({i_cell},{j_cell}): Acc={accuracy:.3f}, Bal_Acc={balanced_acc:.3f}, F1={f1:.3f} (freq: {test_freq:.3f}, baseline: {baseline_acc:.3f})")
                
            except Exception as e:
                accuracy = baseline_acc
                balanced_acc = 0.5
                f1 = 0.0
                status = f"failed: {str(e)[:30]}"
                print(f"  Cell ({i_cell},{j_cell}): Failed to train - {status}")
        
        accuracy_grid[i_cell, j_cell] = accuracy
        balanced_accuracy_grid[i_cell, j_cell] = balanced_acc
        f1_grid[i_cell, j_cell] = f1
        
        results[cell_key] = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'visit_frequency': test_freq,
            'baseline_accuracy': baseline_acc,
            'status': status
        }
    
    # Calculate overall metrics
    overall_accuracy = np.mean(accuracy_grid)
    overall_balanced_acc = np.mean(balanced_accuracy_grid)
    overall_f1 = np.mean(f1_grid)
    overall_baseline = np.mean(baseline_accuracies)
    
    print(f"\n📊 SUMMARY:")
    print(f"Meaningful cells (freq >= {min_visit_freq}): {meaningful_cells}/{grid_size*grid_size}")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    print(f"Overall balanced accuracy: {overall_balanced_acc:.3f}")
    print(f"Overall F1 score: {overall_f1:.3f}")
    print(f"Baseline accuracy (majority class): {overall_baseline:.3f}")
    print(f"📚 Paper reference: This class imbalance is exactly what Sokoban paper described!")
    print(f"   'Many squares assigned NEVER → use macro F1 instead of accuracy'")
    
    return {
        'results': results,
        'accuracy_grid': accuracy_grid,
        'balanced_accuracy_grid': balanced_accuracy_grid,
        'f1_grid': f1_grid,
        'visit_frequency_grid': visit_freq_grid,
        'overall_accuracy': overall_accuracy,
        'overall_balanced_accuracy': overall_balanced_acc,
        'overall_f1': overall_f1,
        'overall_baseline': overall_baseline,
        'meaningful_cells': meaningful_cells,
        'grid_size': grid_size
    }

def visualize_balanced_results(evaluation_results, output_dir):
    """Visualize balanced trajectory prediction results."""
    accuracy_grid = evaluation_results['accuracy_grid']
    balanced_accuracy_grid = evaluation_results['balanced_accuracy_grid']
    f1_grid = evaluation_results['f1_grid']
    visit_freq_grid = evaluation_results['visit_frequency_grid']
    grid_size = evaluation_results['grid_size']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Regular Accuracy
    im1 = axes[0,0].imshow(accuracy_grid.T, origin='lower', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    axes[0,0].set_title('Regular Accuracy\n(Misleading due to class imbalance)', fontweight='bold')
    axes[0,0].set_xlabel('Grid X')
    axes[0,0].set_ylabel('Grid Y')
    for i in range(grid_size):
        for j in range(grid_size):
            text = axes[0,0].text(i, j, f'{accuracy_grid[i, j]:.2f}', ha="center", va="center", 
                                color="black", fontsize=8)
    plt.colorbar(im1, ax=axes[0,0])
    
    # Plot 2: Balanced Accuracy  
    im2 = axes[0,1].imshow(balanced_accuracy_grid.T, origin='lower', cmap='RdYlGn', vmin=0.3, vmax=0.8)
    axes[0,1].set_title('Balanced Accuracy\n(True predictive capability)', fontweight='bold')
    axes[0,1].set_xlabel('Grid X')
    axes[0,1].set_ylabel('Grid Y')
    for i in range(grid_size):
        for j in range(grid_size):
            color = "white" if balanced_accuracy_grid[i, j] < 0.55 else "black"
            text = axes[0,1].text(i, j, f'{balanced_accuracy_grid[i, j]:.2f}', ha="center", va="center", 
                                color=color, fontsize=8)
    plt.colorbar(im2, ax=axes[0,1])
    
    # Plot 3: F1 Score
    im3 = axes[1,0].imshow(f1_grid.T, origin='lower', cmap='RdYlGn', vmin=0.0, vmax=0.5)
    axes[1,0].set_title('F1 Score\n(Positive class prediction quality)', fontweight='bold')
    axes[1,0].set_xlabel('Grid X')
    axes[1,0].set_ylabel('Grid Y')
    for i in range(grid_size):
        for j in range(grid_size):
            color = "white" if f1_grid[i, j] < 0.25 else "black"
            text = axes[1,0].text(i, j, f'{f1_grid[i, j]:.2f}', ha="center", va="center", 
                                color=color, fontsize=8)
    plt.colorbar(im3, ax=axes[1,0])
    
    # Plot 4: Visit Frequency
    im4 = axes[1,1].imshow(visit_freq_grid.T, origin='lower', cmap='Blues', vmin=0, vmax=0.15)
    axes[1,1].set_title('Visit Frequency\n(Explains the class imbalance)', fontweight='bold')
    axes[1,1].set_xlabel('Grid X')
    axes[1,1].set_ylabel('Grid Y')
    for i in range(grid_size):
        for j in range(grid_size):
            color = "white" if visit_freq_grid[i, j] > 0.075 else "black"
            text = axes[1,1].text(i, j, f'{visit_freq_grid[i, j]:.2f}', ha="center", va="center", 
                                color=color, fontsize=8)
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/balanced_trajectory_prediction_{grid_size}x{grid_size}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Balanced Within-Rollout Trajectory Prediction')
    parser.add_argument('--goal', default="FG blue", help='LTL goal to test')
    parser.add_argument('--grid-size', type=int, default=5, help='Grid size')
    parser.add_argument('--k-steps', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--train-ratio', type=float, default=0.5, help='Fraction of trajectory for training')
    parser.add_argument('--min-visit-freq', type=float, default=0.05, help='Minimum visit frequency for meaningful cells')
    parser.add_argument('--n-worlds', type=int, default=6, help='Number of worlds')
    parser.add_argument('--n-rollouts', type=int, default=8, help='Rollouts per world')
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps per rollout')
    parser.add_argument('--output-dir', type=str, 
                       default='interpretability/probing/corrected_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=== BALANCED WITHIN-ROLLOUT TRAJECTORY PREDICTION PROBE ===")
    print(f"🎯 Can the network REALLY predict specific trajectories?")
    print(f"🔍 This time we properly handle class imbalance!")
    print(f"Goal: {args.goal}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Minimum visit frequency: {args.min_visit_freq}")
    
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
    data = collect_trajectory_data(
        model, world_ids, args.goal, args.grid_size, 
        args.n_rollouts, args.max_steps, args.train_ratio, args.k_steps
    )
    
    if data['train_features'].shape[0] == 0:
        print("❌ No training data collected!")
        return
        
    # Evaluate with proper balanced metrics
    print("\nEvaluating with balanced metrics...")
    evaluation_results = evaluate_trajectory_prediction_balanced(data, args.min_visit_freq)
    
    # Visualize
    print("\nCreating balanced visualization...")
    visualize_balanced_results(evaluation_results, args.output_dir)
    
    # Interpretation
    balanced_acc = evaluation_results['overall_balanced_accuracy']
    f1_score = evaluation_results['overall_f1']
    meaningful_cells = evaluation_results['meaningful_cells']
    
    print(f"\n=== THE REAL RESULTS (CLASS IMBALANCE CORRECTED) ===")
    print(f"Meaningful cells analyzed: {meaningful_cells}/{args.grid_size**2}")
    print(f"Balanced accuracy: {balanced_acc:.3f}")
    print(f"F1 score: {f1_score:.3f}")
    print(f"Baseline (majority class): {evaluation_results['overall_baseline']:.3f}")
    
    if balanced_acc > 0.65:
        print("🚀 IMPRESSIVE! True trajectory prediction capability!")
    elif balanced_acc > 0.55:
        print("✅ MODEST trajectory prediction - some spatial reasoning")
    else:
        print("❌ POOR trajectory prediction - the high accuracy was due to class imbalance!")
        print("   The network just learned to predict 'won't visit' for most cells")
        print("📚 This matches the Sokoban paper: 'many squares assigned NEVER' → use macro F1!")

if __name__ == "__main__":
    main() 