#!/usr/bin/env python3
"""
Behavioral Predictions Probing

Tests what the COMPLETE network can actually predict - behavioral outcomes
rather than geometric quantities like directions.

HYPOTHESIS: The associative navigation system should be excellent at predicting:
- Next actions
- Goal achievement  
- Zone transitions
- Navigation success
- Behavioral modes

Usage: python interpretability/probing/probe_behavioral_predictions.py
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
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

class BehavioralPredictionsProbe:
    """Test what behavioral outcomes the complete network can predict."""
    
    def __init__(self, model, prediction_targets=None, k_steps=5):
        self.model = model
        self.prediction_targets = prediction_targets or [
            'next_action', 'goal_achievement', 'zone_transition', 
            'obstacle_encounter', 'navigation_success'
        ]
        self.k_steps = k_steps
        
    def collect_data(self, world_ids, goals, n_rollouts_per_world=10, max_steps=200):
        """Collect complete network behavior and behavioral prediction targets."""
        print(f"Collecting BEHAVIORAL PREDICTION data")
        print(f"Prediction targets: {self.prediction_targets}")
        print(f"Prediction horizon: {self.k_steps} steps")
        print(f"Testing goals: {goals}")
        
        all_features = []
        all_targets = {target: [] for target in self.prediction_targets}
        all_metadata = {
            'world_ids': [],
            'rollout_ids': [],
            'step_ids': [],
            'goal_types': [],
            'current_zones': [],
            'distances_to_goal': []
        }
        
        total_samples = 0
        
        # Test each goal type
        for goal in goals:
            print(f"\n--- Testing goal: {goal} ---")
            
            env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
            props = set(env.get_propositions())
            agent = Agent(self.model, ExhaustiveSearch(self.model, props, num_loops=2),
                         propositions=props, verbose=False)
            
            # Extract target zone name
            target_zone_name = goal.split()[-1]  # "FG blue" -> "blue"
            
            for world_id in world_ids:
                print(f"Processing world {world_id} for goal {goal}...")
                for rollout_id in trange(n_rollouts_per_world, desc=f"Rollouts for world {world_id}"):
                    trajectory_data = []
                    
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
                        print(f"Warning: Could not find target zone '{target_zone_name}' in world {world_id}")
                        continue
                    
                    # Collect full trajectory first
                    for step_id in range(max_steps):
                        if done:
                            break
                            
                        # Get current state information
                        current_pos = env.agent_pos[:2].copy()
                        
                        # Determine current zone
                        current_zone = "none"
                        if hasattr(env, 'zone_positions') and env.zone_positions:
                            for zone_name, zone_pos in env.zone_positions.items():
                                dist_to_zone = np.linalg.norm(current_pos - zone_pos[:2])
                                if dist_to_zone < 0.5:  # Within zone radius
                                    current_zone = zone_name.lower()
                                    break
                        
                        # Distance to target zone
                        dist_to_target = np.linalg.norm(current_pos - target_zone_pos) if target_zone_pos is not None else 999
                        
                        # Get complete network representation
                        obs_features = obs.get('features', np.zeros(80))
                        
                        # Simple goal encoding
                        goal_encoding = np.zeros(10)
                        if 'blue' in goal.lower():
                            goal_encoding[0] = 1.0
                        elif 'red' in goal.lower():
                            goal_encoding[1] = 1.0
                        elif 'green' in goal.lower():
                            goal_encoding[2] = 1.0
                        
                        network_representation = np.concatenate([obs_features, goal_encoding])
                        
                        # Get action
                        action = agent.get_action(obs, {}, deterministic=True).flatten()
                        
                        # Store trajectory data
                        trajectory_data.append({
                            'step_id': step_id,
                            'network_representation': network_representation,
                            'action': action,
                            'position': current_pos,
                            'current_zone': current_zone,
                            'dist_to_target': dist_to_target,
                            'done': done
                        })
                        
                        # Step environment
                        obs, _, done, info = env.step(action)
                    
                    # Now create prediction targets from complete trajectory
                    valid_samples = len(trajectory_data) - self.k_steps
                    if valid_samples <= 0:
                        continue
                        
                    for i in range(valid_samples):
                        current_data = trajectory_data[i]
                        future_data = trajectory_data[i + self.k_steps] if i + self.k_steps < len(trajectory_data) else trajectory_data[-1]
                        
                        # Store features
                        all_features.append(current_data['network_representation'])
                        
                        # 1. NEXT ACTION PREDICTION (immediate next action)
                        if 'next_action' in self.prediction_targets:
                            next_action = trajectory_data[i + 1]['action'] if i + 1 < len(trajectory_data) else current_data['action']
                            all_targets['next_action'].append(next_action)
                        
                        # 2. GOAL ACHIEVEMENT PREDICTION (binary: will reach target zone?)
                        if 'goal_achievement' in self.prediction_targets:
                            will_reach_goal = 0
                            for j in range(i, min(i + self.k_steps + 1, len(trajectory_data))):
                                if target_zone_name.lower() in trajectory_data[j]['current_zone']:
                                    will_reach_goal = 1
                                    break
                            all_targets['goal_achievement'].append(will_reach_goal)
                        
                        # 3. ZONE TRANSITION PREDICTION (which zone will agent be in?)
                        if 'zone_transition' in self.prediction_targets:
                            future_zone = future_data['current_zone']
                            # Encode zones as integers
                            zone_encoding = {'none': 0, 'blue': 1, 'red': 2, 'green': 3, 'yellow': 4}
                            zone_id = zone_encoding.get(future_zone, 0)
                            all_targets['zone_transition'].append(zone_id)
                        
                        # 4. OBSTACLE ENCOUNTER PREDICTION (will hit wall?)
                        if 'obstacle_encounter' in self.prediction_targets:
                            # Simple heuristic: if agent doesn't move much, it might be hitting obstacles
                            current_pos = current_data['position']
                            future_pos = future_data['position']
                            movement_distance = np.linalg.norm(future_pos - current_pos)
                            will_hit_obstacle = 1 if movement_distance < 0.1 * self.k_steps else 0
                            all_targets['obstacle_encounter'].append(will_hit_obstacle)
                        
                        # 5. NAVIGATION SUCCESS PREDICTION (distance to target decreasing?)
                        if 'navigation_success' in self.prediction_targets:
                            current_dist = current_data['dist_to_target']
                            future_dist = future_data['dist_to_target']
                            is_successful = 1 if future_dist < current_dist else 0
                            all_targets['navigation_success'].append(is_successful)
                        
                        # Store metadata
                        all_metadata['world_ids'].append(world_id)
                        all_metadata['rollout_ids'].append(rollout_id)
                        all_metadata['step_ids'].append(current_data['step_id'])
                        all_metadata['goal_types'].append(goal)
                        all_metadata['current_zones'].append(current_data['current_zone'])
                        all_metadata['distances_to_goal'].append(current_data['dist_to_target'])
                        total_samples += 1
            
            env.close()
        
        if total_samples == 0:
            print("No samples collected!")
            return None
            
        print(f"Collected {total_samples} samples across {len(goals)} goals")
        
        return {
            'features': np.array(all_features),
            'targets': {target: np.array(values) for target, values in all_targets.items()},
            'metadata': all_metadata
        }
    
    def create_generalization_splits(self, data):
        """Create temporal, spatial, and environmental generalization splits."""
        features = data['features']
        world_ids = np.array(data['metadata']['world_ids'])
        rollout_ids = np.array(data['metadata']['rollout_ids'])
        step_ids = np.array(data['metadata']['step_ids'])
        goal_types = np.array(data['metadata']['goal_types'])
        
        splits = {}
        
        # 1. Temporal split (early vs late steps)
        median_step = np.median(step_ids)
        train_mask = step_ids <= median_step
        test_mask = step_ids > median_step
        
        splits['temporal'] = {
            'train_mask': train_mask,
            'test_mask': test_mask
        }
        
        # 2. Spatial split (different rollouts within same worlds)
        unique_combinations = list(set(zip(world_ids, rollout_ids, goal_types)))
        random.shuffle(unique_combinations)
        
        split_idx = len(unique_combinations) // 2
        train_combinations = set(unique_combinations[:split_idx])
        
        train_mask = np.array([
            (w, r, g) in train_combinations 
            for w, r, g in zip(world_ids, rollout_ids, goal_types)
        ])
        test_mask = ~train_mask
        
        splits['spatial'] = {
            'train_mask': train_mask,
            'test_mask': test_mask
        }
        
        # 3. Environmental split (different worlds)
        unique_worlds = list(np.unique(world_ids))
        random.shuffle(unique_worlds)
        
        split_idx = len(unique_worlds) // 2
        train_worlds = set(unique_worlds[:split_idx])
        
        train_mask = np.array([w in train_worlds for w in world_ids])
        test_mask = ~train_mask
        
        splits['environmental'] = {
            'train_mask': train_mask,
            'test_mask': test_mask
        }
        
        return splits
    
    def train_and_evaluate_probes(self, data, splits):
        """Train and evaluate behavioral prediction probes."""
        features = data['features']
        targets = data['targets']
        
        results = {}
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=0.95)  # 95% variance
        features_pca = pca.fit_transform(features)
        print(f"PCA: {features.shape[1]} → {features_pca.shape[1]} dimensions")
        
        for target_name in self.prediction_targets:
            if target_name not in targets:
                continue
                
            print(f"\n--- Evaluating {target_name.upper()} prediction ---")
            target_values = targets[target_name]
            
            # Determine if classification or regression
            is_classification = target_name in ['goal_achievement', 'obstacle_encounter', 'navigation_success', 'zone_transition']
            
            for split_name, split_data in splits.items():
                train_mask = split_data['train_mask']
                test_mask = split_data['test_mask']
                
                X_train = features_pca[train_mask]
                X_test = features_pca[test_mask]
                y_train = target_values[train_mask]
                y_test = target_values[test_mask]
                
                if len(X_train) == 0 or len(X_test) == 0:
                    print(f"  Skipping {split_name}: insufficient data")
                    continue
                
                # Train appropriate model
                if is_classification:
                    if target_name == 'zone_transition':
                        # Multi-class classification
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        print(f"  {split_name}: Accuracy={accuracy:.3f} (train: {len(X_train)}, test: {len(X_test)})")
                        
                        # Store results
                        if target_name not in results:
                            results[target_name] = {}
                        results[target_name][split_name] = {
                            'accuracy': accuracy,
                            'n_train': len(X_train),
                            'n_test': len(X_test)
                        }
                    else:
                        # Binary classification
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        print(f"  {split_name}: Accuracy={accuracy:.3f} (train: {len(X_train)}, test: {len(X_test)})")
                        
                        # Store results
                        if target_name not in results:
                            results[target_name] = {}
                        results[target_name][split_name] = {
                            'accuracy': accuracy,
                            'n_train': len(X_train),
                            'n_test': len(X_test)
                        }
                else:
                    # Regression (for next_action)
                    model = Ridge(alpha=1.0)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    print(f"  {split_name}: R²={r2:.3f}, MSE={mse:.3f} (train: {len(X_train)}, test: {len(X_test)})")
                    
                    # Store results
                    if target_name not in results:
                        results[target_name] = {}
                    results[target_name][split_name] = {
                        'r2': r2,
                        'mse': mse,
                        'n_train': len(X_train),
                        'n_test': len(X_test)
                    }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Behavioral Predictions Probing')
    parser.add_argument('--targets', nargs='+', 
                       default=['next_action', 'goal_achievement', 'zone_transition', 'obstacle_encounter', 'navigation_success'],
                       help='Prediction targets to test')
    parser.add_argument('--goals', nargs='+', default=["FG blue", "FG red"],
                       help='LTL goals to test')
    parser.add_argument('--k-steps', type=int, default=5,
                       help='Prediction horizon (k steps into future)')
    parser.add_argument('--n-worlds', type=int, default=6,
                       help='Number of worlds to use')
    parser.add_argument('--n-rollouts', type=int, default=8,
                       help='Number of rollouts per world per goal')
    parser.add_argument('--max-steps', type=int, default=50,
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=== BEHAVIORAL PREDICTIONS PROBING ===")
    print(f"🎯 HYPOTHESIS: Associative navigation should predict behavioral outcomes!")
    print(f"Prediction targets: {args.targets}")
    print(f"Goals to test: {args.goals}")
    print(f"Prediction horizon: {args.k_steps} steps")
    print(f"Worlds per goal: {args.n_worlds}")
    print(f"Rollouts per world: {args.n_rollouts}")
    
    # Load model
    print("\nLoading model...")
    store = ModelStore(ENV, EXP, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Create probe
    probe = BehavioralPredictionsProbe(model, args.targets, args.k_steps)
    
    # Collect data
    world_ids = list(range(args.n_worlds))
    data = probe.collect_data(world_ids, args.goals, args.n_rollouts, args.max_steps)
    
    if data is None:
        print("No data collected. Exiting.")
        return
    
    # Create generalization splits
    print("\nCreating generalization splits...")
    splits = probe.create_generalization_splits(data)
    
    # Train and evaluate probes
    print("\nTraining and evaluating behavioral prediction probes...")
    results = probe.train_and_evaluate_probes(data, splits)
    
    # Save results
    print("\nSaving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create summary
    print(f"\n=== BEHAVIORAL PREDICTIONS PROBE SUMMARY ===")
    for target_name, target_results in results.items():
        print(f"\n🎯 {target_name.upper()}:")
        for split_name, metrics in target_results.items():
            if 'accuracy' in metrics:
                print(f"  {split_name}: Accuracy = {metrics['accuracy']:.3f}")
            else:
                print(f"  {split_name}: R² = {metrics['r2']:.3f}")
    
    print(f"\n🎯 INTERPRETATION:")
    print("Look for high-performing predictions - these reveal what the network actually computes!")
    print("Expected successes: next_action, goal_achievement, navigation_success")
    print("These would prove the system uses learned behavioral associations! 🧠✨")

if __name__ == "__main__":
    main() 