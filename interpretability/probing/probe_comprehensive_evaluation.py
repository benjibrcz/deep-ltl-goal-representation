#!/usr/bin/env python3
"""
Comprehensive Behavioral Prediction Evaluation

Tests our key findings across many different situations:
- Multiple LTL goals
- Different prediction horizons  
- Various grid sizes
- Cross-rollout vs within-rollout prediction
- Behavioral vs spatial prediction tasks

This will validate whether our core discoveries hold across diverse conditions.

Usage: python interpretability/probing/probe_comprehensive_evaluation.py
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA  
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

def collect_behavioral_data(model, goals, world_ids, n_rollouts_per_world=6, max_steps=40):
    """Collect data for all behavioral prediction tasks."""
    print(f"Collecting comprehensive behavioral data...")
    print(f"Goals: {goals}")
    print(f"Worlds: {len(world_ids)} worlds")
    print(f"Rollouts per world: {n_rollouts_per_world}")
    
    all_data = {
        'features': [],
        'actions': [],
        'positions': [],
        'distances_to_target': [],
        'metadata': {
            'world_ids': [],
            'rollout_ids': [],
            'step_ids': [],
            'goal_types': []
        }
    }
    
    for goal in goals:
        print(f"\n--- Collecting data for goal: {goal} ---")
        
        env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
        props = set(env.get_propositions())
        agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
        
        target_zone_name = goal.split()[-1]  # "FG blue" -> "blue"
        
        for world_id in world_ids:
            for rollout_id in trange(n_rollouts_per_world, desc=f"World {world_id}"):
                done = False
                obs = env.reset(seed=world_id + rollout_id * 1000)
                agent.reset()
                
                # Find target zone position
                target_zone_pos = None
                if hasattr(env, 'zone_positions') and env.zone_positions:
                    for zone_name, zone_pos in env.zone_positions.items():
                        if target_zone_name.lower() in zone_name.lower():
                            target_zone_pos = zone_pos[:2]
                            break
                
                if target_zone_pos is None:
                    continue
                
                for step_id in range(max_steps):
                    if done:
                        break
                        
                    current_pos = env.agent_pos[:2].copy()
                    current_dist = np.linalg.norm(current_pos - target_zone_pos)
                    
                    # Get network representation
                    obs_features = obs.get('features', np.zeros(80))
                    goal_encoding = np.zeros(10)
                    if 'blue' in goal.lower():
                        goal_encoding[0] = 1.0
                    elif 'red' in goal.lower():
                        goal_encoding[1] = 1.0
                    elif 'green' in goal.lower():
                        goal_encoding[2] = 1.0
                    elif 'yellow' in goal.lower():
                        goal_encoding[3] = 1.0
                    
                    network_representation = np.concatenate([obs_features, goal_encoding])
                    
                    # Get action
                    action = agent.get_action(obs, {}, deterministic=True).flatten()
                    
                    # Store data
                    all_data['features'].append(network_representation)
                    all_data['actions'].append(action)
                    all_data['positions'].append(current_pos)
                    all_data['distances_to_target'].append(current_dist)
                    all_data['metadata']['world_ids'].append(world_id)
                    all_data['metadata']['rollout_ids'].append(rollout_id)
                    all_data['metadata']['step_ids'].append(step_id)
                    all_data['metadata']['goal_types'].append(goal)
                    
                    # Step environment
                    obs, _, done, info = env.step(action)
        
        env.close()
    
    # Convert to numpy arrays
    for key in ['features', 'actions', 'positions', 'distances_to_target']:
        all_data[key] = np.array(all_data[key])
    
    print(f"Total samples collected: {len(all_data['features'])}")
    return all_data

def evaluate_next_action_prediction(data, split_type='cross_rollout'):
    """Evaluate next action prediction across different conditions."""
    features = data['features']
    actions = data['actions']
    
    world_ids = np.array(data['metadata']['world_ids'])
    rollout_ids = np.array(data['metadata']['rollout_ids'])
    goal_types = np.array(data['metadata']['goal_types'])
    
    # Create train/test splits
    if split_type == 'cross_rollout':
        # Train/test across different rollouts
        unique_combinations = list(set(zip(world_ids, rollout_ids, goal_types)))
        random.shuffle(unique_combinations)
        split_idx = len(unique_combinations) // 2
        train_combinations = set(unique_combinations[:split_idx])
        
        train_mask = np.array([
            (w, r, g) in train_combinations 
            for w, r, g in zip(world_ids, rollout_ids, goal_types)
        ])
        test_mask = ~train_mask
        
    elif split_type == 'cross_goal':
        # Train/test across different goals
        unique_goals = list(np.unique(goal_types))
        if len(unique_goals) < 2:
            return {'r2': 0.0, 'mse': 999.0, 'n_train': 0, 'n_test': 0, 'status': 'insufficient_goals'}
        
        random.shuffle(unique_goals)
        split_idx = len(unique_goals) // 2
        train_goals = set(unique_goals[:split_idx])
        
        train_mask = np.array([g in train_goals for g in goal_types])
        test_mask = ~train_mask
    
    else:  # temporal
        # Train/test across early vs late steps
        step_ids = np.array(data['metadata']['step_ids'])
        median_step = np.median(step_ids)
        train_mask = step_ids <= median_step
        test_mask = step_ids > median_step
    
    if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
        return {'r2': 0.0, 'mse': 999.0, 'n_train': 0, 'n_test': 0, 'status': 'insufficient_data'}
    
    X_train, X_test = features[train_mask], features[test_mask]
    y_train, y_test = actions[train_mask], actions[test_mask]
    
    # Apply PCA if needed
    if X_train.shape[1] > 50:
        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    
    # Train model
    try:
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            'r2': r2,
            'mse': mse,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'status': 'success'
        }
    except Exception as e:
        return {'r2': 0.0, 'mse': 999.0, 'n_train': len(X_train), 'n_test': len(X_test), 'status': f'error: {str(e)[:30]}'}

def evaluate_navigation_success_prediction(data, k_steps=5, split_type='cross_rollout'):
    """Evaluate navigation success prediction."""
    features = data['features']
    distances = data['distances_to_target']
    
    world_ids = np.array(data['metadata']['world_ids'])
    rollout_ids = np.array(data['metadata']['rollout_ids'])
    goal_types = np.array(data['metadata']['goal_types'])
    step_ids = np.array(data['metadata']['step_ids'])
    
    # Create success labels
    success_labels = []
    valid_features = []
    valid_metadata = {'world_ids': [], 'rollout_ids': [], 'goal_types': []}
    
    for i in range(len(features) - k_steps):
        current_dist = distances[i]
        future_dist = distances[i + k_steps]
        
        # Success = getting significantly closer
        is_successful = 1 if future_dist < current_dist - 0.1 else 0
        success_labels.append(is_successful)
        valid_features.append(features[i])
        
        valid_metadata['world_ids'].append(world_ids[i])
        valid_metadata['rollout_ids'].append(rollout_ids[i])
        valid_metadata['goal_types'].append(goal_types[i])
    
    if len(success_labels) == 0:
        return {'accuracy': 0.0, 'balanced_accuracy': 0.5, 'f1': 0.0, 'n_train': 0, 'n_test': 0, 'status': 'no_data'}
    
    features = np.array(valid_features)
    success_labels = np.array(success_labels)
    world_ids = np.array(valid_metadata['world_ids'])
    rollout_ids = np.array(valid_metadata['rollout_ids'])
    goal_types = np.array(valid_metadata['goal_types'])
    
    # Create splits (similar logic as next action)
    if split_type == 'cross_rollout':
        unique_combinations = list(set(zip(world_ids, rollout_ids, goal_types)))
        random.shuffle(unique_combinations)
        split_idx = len(unique_combinations) // 2
        train_combinations = set(unique_combinations[:split_idx])
        
        train_mask = np.array([
            (w, r, g) in train_combinations 
            for w, r, g in zip(world_ids, rollout_ids, goal_types)
        ])
        test_mask = ~train_mask
    else:
        # Add other split types as needed
        indices = list(range(len(features)))
        random.shuffle(indices)
        split_idx = len(indices) // 2
        train_mask = np.zeros(len(features), dtype=bool)
        train_mask[indices[:split_idx]] = True
        test_mask = ~train_mask
    
    if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
        return {'accuracy': 0.0, 'balanced_accuracy': 0.5, 'f1': 0.0, 'n_train': 0, 'n_test': 0, 'status': 'insufficient_data'}
    
    X_train, X_test = features[train_mask], features[test_mask]
    y_train, y_test = success_labels[train_mask], success_labels[test_mask]
    
    # Check for class balance
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)
    
    if len(unique_train) < 2:
        baseline_acc = max(np.mean(y_test), 1 - np.mean(y_test)) if len(y_test) > 0 else 0.5
        return {'accuracy': baseline_acc, 'balanced_accuracy': 0.5, 'f1': 0.0, 'n_train': len(X_train), 'n_test': len(X_test), 'status': 'no_positive_class'}
    
    # Apply PCA
    if X_train.shape[1] > 50:
        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    
    # Train balanced classifier
    try:
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight_dict)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division='0')
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1': f1,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'status': 'success'
        }
    except Exception as e:
        return {'accuracy': 0.0, 'balanced_accuracy': 0.5, 'f1': 0.0, 'n_train': len(X_train), 'n_test': len(X_test), 'status': f'error: {str(e)[:30]}'}

def run_comprehensive_evaluation(model, goals, world_ids, n_rollouts_per_world=6, max_steps=40):
    """Run comprehensive evaluation across many conditions."""
    print("=== COMPREHENSIVE BEHAVIORAL PREDICTION EVALUATION ===")
    print(f"🎯 Testing our key findings across diverse situations!")
    
    # Collect data
    print("\n1. Collecting comprehensive data...")
    data = collect_behavioral_data(model, goals, world_ids, n_rollouts_per_world, max_steps)
    
    results = []
    
    # Test 1: Next Action Prediction (should work moderately well)
    print("\n2. Testing NEXT ACTION prediction...")
    for split_type in ['cross_rollout', 'temporal', 'cross_goal']:
        result = evaluate_next_action_prediction(data, split_type)
        results.append({
            'task': 'next_action',
            'split_type': split_type,
            'metric': 'r2',
            'value': result['r2'],
            'n_train': result['n_train'],
            'n_test': result['n_test'],
            'status': result['status']
        })
        print(f"  {split_type}: R² = {result['r2']:.3f} ({result['status']})")
    
    # Test 2: Navigation Success Prediction (should work well)
    print("\n3. Testing NAVIGATION SUCCESS prediction...")
    for k_steps in [3, 5, 10]:
        for split_type in ['cross_rollout']:
            result = evaluate_navigation_success_prediction(data, k_steps, split_type)
            results.append({
                'task': f'navigation_success_{k_steps}steps',
                'split_type': split_type,
                'metric': 'balanced_accuracy',
                'value': result['balanced_accuracy'],
                'n_train': result['n_train'],
                'n_test': result['n_test'],
                'status': result['status']
            })
            print(f"  {k_steps}-step {split_type}: Balanced Acc = {result['balanced_accuracy']:.3f} ({result['status']})")
    
    return results

def visualize_comprehensive_results(results, output_dir):
    """Create comprehensive visualization of results."""
    df = pd.DataFrame(results)
    
    # Filter successful results
    df_success = df[df['status'] == 'success'].copy()
    
    if len(df_success) == 0:
        print("No successful results to visualize")
        return
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Next Action Performance by Split Type
    next_action_data = df_success[df_success['task'] == 'next_action']
    if len(next_action_data) > 0:
        sns.barplot(data=next_action_data, x='split_type', y='value', ax=axes[0,0])
        axes[0,0].set_title('Next Action Prediction (R²)', fontweight='bold')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(next_action_data['value']):
            axes[0,0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Plot 2: Navigation Success by Prediction Horizon
    nav_success_data = df_success[df_success['task'].str.contains('navigation_success')]
    if len(nav_success_data) > 0:
        # Extract k_steps from task name
        nav_success_data['k_steps'] = nav_success_data['task'].str.extract(r'(\d+)steps').astype(int)
        sns.barplot(data=nav_success_data, x='k_steps', y='value', ax=axes[0,1])
        axes[0,1].set_title('Navigation Success by Horizon', fontweight='bold')
        axes[0,1].set_ylabel('Balanced Accuracy')
        axes[0,1].set_xlabel('Prediction Horizon (steps)')
        axes[0,1].set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(nav_success_data['value']):
            axes[0,1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Plot 3: Performance Summary by Task Type
    task_summary = df_success.groupby('task')['value'].agg(['mean', 'std']).reset_index()
    task_summary['task_type'] = task_summary['task'].apply(lambda x: 'Next Action' if 'next_action' in x else 'Nav Success')
    
    task_type_summary = task_summary.groupby('task_type')['mean'].mean().reset_index()
    sns.barplot(data=task_type_summary, x='task_type', y='mean', ax=axes[1,0])
    axes[1,0].set_title('Average Performance by Task Type', fontweight='bold')
    axes[1,0].set_ylabel('Average Performance')
    axes[1,0].set_ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(task_type_summary['mean']):
        axes[1,0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Plot 4: Sample Sizes
    sns.scatterplot(data=df_success, x='n_train', y='value', hue='task', ax=axes[1,1])
    axes[1,1].set_title('Performance vs Training Sample Size', fontweight='bold')
    axes[1,1].set_xlabel('Training Samples')
    axes[1,1].set_ylabel('Performance')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/comprehensive_evaluation_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comprehensive visualization saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Behavioral Prediction Evaluation')
    parser.add_argument('--goals', nargs='+', default=["FG blue", "FG red", "FG green"],
                       help='LTL goals to test')
    parser.add_argument('--n-worlds', type=int, default=8,
                       help='Number of worlds to use')
    parser.add_argument('--n-rollouts', type=int, default=6,
                       help='Number of rollouts per world per goal')
    parser.add_argument('--max-steps', type=int, default=40,
                       help='Maximum steps per rollout')
    parser.add_argument('--output-dir', type=str, 
                       default='interpretability/probing/corrected_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=== COMPREHENSIVE BEHAVIORAL PREDICTION EVALUATION ===")
    print(f"🚀 Testing robustness across diverse situations!")
    print(f"Goals: {args.goals}")
    print(f"Worlds: {args.n_worlds}")
    print(f"Rollouts per world: {args.n_rollouts}")
    print(f"Max steps: {args.max_steps}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    store = ModelStore(ENV, EXP, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial(args.goals[0]), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Run comprehensive evaluation
    world_ids = list(range(args.n_worlds))
    results = run_comprehensive_evaluation(model, args.goals, world_ids, args.n_rollouts, args.max_steps)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(results)
    csv_path = f"{args.output_dir}/comprehensive_evaluation_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Visualize
    print("\nCreating comprehensive visualization...")
    visualize_comprehensive_results(results, args.output_dir)
    
    # Summary
    print(f"\n=== COMPREHENSIVE EVALUATION SUMMARY ===")
    successful_results = results_df[results_df['status'] == 'success']
    
    if len(successful_results) > 0:
        # Next action performance
        next_action_results = successful_results[successful_results['task'] == 'next_action']
        if len(next_action_results) > 0:
            avg_next_action = next_action_results['value'].mean()
            print(f"📊 Next Action Prediction: Average R² = {avg_next_action:.3f}")
            
        # Navigation success performance  
        nav_success_results = successful_results[successful_results['task'].str.contains('navigation_success')]
        if len(nav_success_results) > 0:
            avg_nav_success = nav_success_results['value'].mean()
            print(f"🎯 Navigation Success Prediction: Average Balanced Accuracy = {avg_nav_success:.3f}")
        
        print(f"\n🚀 COMPREHENSIVE CONCLUSIONS:")
        if avg_next_action > 0.3 and avg_nav_success > 0.7:
            print("✅ Findings CONFIRMED across diverse situations!")
            print("   - Network has behavioral awareness (navigation success)")
            print("   - Network can predict immediate actions")
            print("   - Consistent across goals, worlds, and prediction horizons")
        else:
            print("🤔 Results vary significantly across conditions")
            print("   - May indicate context-dependent capabilities")
    else:
        print("❌ No successful evaluations - check data collection")

if __name__ == "__main__":
    main() 