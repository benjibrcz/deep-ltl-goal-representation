#!/usr/bin/env python3
"""
Complete Network Movement Delta Probing (Single Goal)

This script tests whether the COMPLETE network computation can predict future movement deltas
for a single goal, with full control over data collection parameters.

HYPOTHESIS: The complete integration should work since the system successfully navigates in practice.

The complete network processes:
1. Raw observations (80D) → env_net → spatial embeddings (64D)
2. LTL formula → ltl_net → goal embeddings (16D) 
3. Combined embeddings → actor → actions (2D)

We test multiple extraction points:
- Raw input features (80D + goal encoding)
- Env_net output (64D spatial embeddings)
- Combined embeddings (64D + 16D = 80D)
- Actor output (action logits/probabilities)

Usage: python interpretability/probing/probe_complete_network_movement_delta_single.py --method combined_embeddings --goal "FG blue" --k-steps 1 --n-worlds 5 --n-rollouts 10 --max-steps 100
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
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

class SingleGoalMovementDeltaProbe:
    """Test if the complete network can predict future movement deltas for a single goal."""
    
    def __init__(self, model, extraction_method="combined_embeddings", k_steps=1, goal="FG blue", n_components=None):
        self.model = model
        self.extraction_method = extraction_method
        self.k_steps = k_steps
        self.goal = goal
        self.n_components = n_components
        self.pca = None
        
        print(f"🔍 SINGLE GOAL MOVEMENT DELTA PROBE")
        print(f"Goal: {goal}")
        print(f"Extraction Method: {extraction_method}")
        print(f"Prediction Horizon: {k_steps} steps")
    
    def collect_data(self, world_ids, n_rollouts_per_world=10, max_steps=200):
        """Collect complete network representations and future movement deltas."""
        print(f"\n📊 COLLECTING DATA")
        print(f"Worlds: {len(world_ids)}, Rollouts/world: {n_rollouts_per_world}, Max steps: {max_steps}")
        
        all_features = []
        all_movement_deltas = []
        all_metadata = {
            'world_ids': [],
            'rollout_ids': [],
            'step_ids': []
        }
        
        env = make_env(ENV, FixedSampler.partial(self.goal), sequence=False, render_mode=None)
        props = set(env.get_propositions())
        agent = Agent(self.model, ExhaustiveSearch(self.model, props, num_loops=2),
                     propositions=props, verbose=False)
        
        total_samples = 0
        
        for world_id in world_ids:
            print(f"Processing world {world_id}...")
            for rollout_id in trange(n_rollouts_per_world, desc=f"Rollouts"):
                # Reset environment
                env.reset(seed=world_id + rollout_id * 1000)
                agent.reset()
                
                # Collect trajectory data
                trajectory_features = []
                trajectory_positions = []
                
                # Setup hooks for different extraction methods
                captured_data = {}
                
                def capture_env_net_output(module, input, output):
                    captured_data['env_output'] = output.detach().cpu().numpy().ravel()
                
                def capture_combined_embeddings(module, input, output):
                    # The input to actor contains combined embeddings
                    captured_data['combined_embeddings'] = input[0].detach().cpu().numpy().ravel()
                
                def capture_actor_output(module, input, output):
                    captured_data['actor_output'] = output.detach().cpu().numpy().ravel()
                
                # Register hooks based on extraction method
                hooks = []
                if self.extraction_method in ['env_output', 'combined_embeddings', 'actor_output']:
                    if self.extraction_method == 'env_output':
                        hooks.append(self.model.env_net.register_forward_hook(capture_env_net_output))
                    elif self.extraction_method == 'combined_embeddings':
                        hooks.append(self.model.actor.register_forward_hook(capture_combined_embeddings))
                    elif self.extraction_method == 'actor_output':
                        hooks.append(self.model.actor.register_forward_hook(capture_actor_output))
                
                done = False
                obs = env.reset(seed=world_id + rollout_id * 1000)
                
                for step_id in range(max_steps):
                    if done:
                        break
                        
                    # Store current position
                    current_pos = env.agent_pos[:2].copy()
                    trajectory_positions.append(current_pos)
                    
                    # Extract features based on method
                    if self.extraction_method == 'raw_input':
                        # Raw input features + goal encoding
                        obs_features = obs.get('features', np.zeros(80))
                        goal_encoding = self._encode_goal(self.goal)
                        features = np.concatenate([obs_features, goal_encoding])
                        
                        # Still need to get action for stepping
                        action = agent.get_action(obs, {}, deterministic=True).flatten()
                    
                    else:
                        # For network-based methods, we need to run forward pass
                        captured_data.clear()
                        
                        # Get action - this triggers the hooks
                        action = agent.get_action(obs, {}, deterministic=True).flatten()
                        
                        # Extract the appropriate features
                        if self.extraction_method == 'env_output':
                            features = captured_data.get('env_output', np.zeros(64))
                        elif self.extraction_method == 'combined_embeddings':
                            features = captured_data.get('combined_embeddings', np.zeros(80))
                        elif self.extraction_method == 'actor_output':
                            features = captured_data.get('actor_output', np.zeros(2))
                        else:
                            features = np.zeros(80)  # fallback
                    
                    trajectory_features.append(features)
                    
                    # Step environment
                    obs, _, done, info = env.step(action)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                # Convert to arrays
                trajectory_features = np.array(trajectory_features)
                trajectory_positions = np.array(trajectory_positions)
                
                # Create movement delta targets
                valid_samples = len(trajectory_positions) - self.k_steps
                if valid_samples <= 0:
                    continue
                    
                for i in range(valid_samples):
                    current_features = trajectory_features[i]
                    current_pos = trajectory_positions[i]
                    future_pos = trajectory_positions[i + self.k_steps]
                    
                    # Movement delta = future_pos - current_pos
                    movement_delta = future_pos - current_pos
                    
                    all_features.append(current_features)
                    all_movement_deltas.append(movement_delta)
                    all_metadata['world_ids'].append(world_id)
                    all_metadata['rollout_ids'].append(rollout_id)
                    all_metadata['step_ids'].append(i)
                    
                    total_samples += 1
        
        env.close()
        
        if total_samples == 0:
            raise ValueError("No valid samples collected!")
        
        print(f"\n✅ COLLECTED {total_samples} SAMPLES")
        
        # Convert to arrays
        features = np.array(all_features)
        movement_deltas = np.array(all_movement_deltas)
        
        print(f"Features shape: {features.shape}")
        print(f"Movement deltas shape: {movement_deltas.shape}")
        print(f"Movement delta stats: mean={np.mean(np.linalg.norm(movement_deltas, axis=1)):.4f}, "
              f"std={np.std(np.linalg.norm(movement_deltas, axis=1)):.4f}")
        
        return features, movement_deltas, all_metadata
    
    def _encode_goal(self, goal):
        """Encode goal as one-hot vector."""
        goal_encoding = np.zeros(10)
        if 'blue' in goal.lower():
            goal_encoding[0] = 1.0
        elif 'green' in goal.lower():
            goal_encoding[1] = 1.0
        elif 'yellow' in goal.lower():
            goal_encoding[2] = 1.0
        elif 'magenta' in goal.lower():
            goal_encoding[3] = 1.0
        return goal_encoding
    
    def train_and_evaluate(self, features, movement_deltas, metadata, split_method="steps"):
        """Train probe and evaluate performance."""
        print(f"\n🎯 TRAINING AND EVALUATION")
        print(f"Split method: {split_method}")
        
        # Apply PCA if specified
        if self.n_components is not None:
            print(f"Applying PCA: {features.shape[1]} → {self.n_components} dimensions")
            self.pca = PCA(n_components=self.n_components)
            features = self.pca.fit_transform(features)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Different train/test split methods
        if split_method == "steps":
            # Original method: split randomly across all steps (data leakage)
            print("⚠️  WARNING: Splitting by steps can cause data leakage!")
            n_train = int(0.7 * len(features))
            indices = list(range(len(features)))
            random.shuffle(indices)
            
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
            
        elif split_method == "rollouts":
            # Split by rollouts: train and test on different rollouts
            print("📊 Splitting by rollouts (no temporal leakage within rollouts)")
            unique_rollouts = list(set(zip(metadata['world_ids'], metadata['rollout_ids'])))
            random.shuffle(unique_rollouts)
            
            n_train_rollouts = int(0.7 * len(unique_rollouts))
            train_rollouts = set(unique_rollouts[:n_train_rollouts])
            test_rollouts = set(unique_rollouts[n_train_rollouts:])
            
            train_indices = [i for i, (wid, rid) in enumerate(zip(metadata['world_ids'], metadata['rollout_ids'])) 
                           if (wid, rid) in train_rollouts]
            test_indices = [i for i, (wid, rid) in enumerate(zip(metadata['world_ids'], metadata['rollout_ids'])) 
                          if (wid, rid) in test_rollouts]
            
            print(f"Train rollouts: {len(train_rollouts)}, Test rollouts: {len(test_rollouts)}")
            
        elif split_method == "worlds":
            # Split by worlds: train and test on completely different worlds
            print("🌍 Splitting by worlds (strongest generalization test)")
            unique_worlds = list(set(metadata['world_ids']))
            random.shuffle(unique_worlds)
            
            n_train_worlds = max(1, int(0.7 * len(unique_worlds)))
            train_worlds = set(unique_worlds[:n_train_worlds])
            test_worlds = set(unique_worlds[n_train_worlds:])
            
            train_indices = [i for i, wid in enumerate(metadata['world_ids']) if wid in train_worlds]
            test_indices = [i for i, wid in enumerate(metadata['world_ids']) if wid in test_worlds]
            
            print(f"Train worlds: {sorted(train_worlds)}, Test worlds: {sorted(test_worlds)}")
            
        else:
            raise ValueError(f"Unknown split_method: {split_method}")
        
        # Check we have valid splits
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError(f"Invalid split: {len(train_indices)} train, {len(test_indices)} test samples")
        
        X_train = features[train_indices]
        X_test = features[test_indices]
        y_train = movement_deltas[train_indices]
        y_test = movement_deltas[test_indices]
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train separate models for X and Y movement
        results = {}
        
        for dim, dim_name in enumerate(['X', 'Y']):
            print(f"\n--- Training {dim_name} Movement Prediction ---")
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train[:, dim])
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_mse = mean_squared_error(y_train[:, dim], y_pred_train)
            test_mse = mean_squared_error(y_test[:, dim], y_pred_test)
            train_r2 = r2_score(y_train[:, dim], y_pred_train)
            test_r2 = r2_score(y_test[:, dim], y_pred_test)
            
            results[dim_name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'model': model
            }
            
            print(f"Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
            print(f"Test MSE: {test_mse:.6f}, R²: {test_r2:.4f}")
        
        # Combined metrics
        y_pred_combined_test = np.column_stack([
            results['X']['model'].predict(X_test),
            results['Y']['model'].predict(X_test)
        ])
        
        # Euclidean distance metrics
        test_distances_true = np.linalg.norm(y_test, axis=1)
        test_distances_pred = np.linalg.norm(y_pred_combined_test, axis=1)
        
        distance_mse = mean_squared_error(test_distances_true, test_distances_pred)
        distance_r2 = r2_score(test_distances_true, test_distances_pred)
        
        results['Combined'] = {
            'distance_mse': distance_mse,
            'distance_r2': distance_r2,
            'mean_true_distance': np.mean(test_distances_true),
            'mean_pred_distance': np.mean(test_distances_pred)
        }
        
        print(f"\n--- Combined Movement Prediction ---")
        print(f"Distance MSE: {distance_mse:.6f}, R²: {distance_r2:.4f}")
        print(f"Mean true distance: {np.mean(test_distances_true):.4f}")
        print(f"Mean predicted distance: {np.mean(test_distances_pred):.4f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Single Goal Complete Network Movement Delta Probing')
    parser.add_argument('--method', required=True, 
                       choices=['raw_input', 'env_output', 'combined_embeddings', 'actor_output'],
                       help='Extraction method: raw_input, env_output, combined_embeddings, or actor_output')
    parser.add_argument('--goal', type=str, default="FG blue",
                       help='Goal formula to test (e.g., "FG blue", "FG green")')
    parser.add_argument('--k-steps', type=int, default=1,
                       help='Prediction horizon: predict movement delta k steps ahead')
    parser.add_argument('--n-worlds', type=int, default=5,
                       help='Number of worlds to use')
    parser.add_argument('--n-rollouts', type=int, default=10,
                       help='Number of rollouts per world')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum steps per rollout')
    parser.add_argument('--n-components', type=int, default=None,
                       help='Number of PCA components (None for no PCA)')
    parser.add_argument('--split-method', type=str, default='rollouts',
                       choices=['steps', 'rollouts', 'worlds'],
                       help='How to split train/test: steps (data leakage), rollouts (recommended), worlds (hardest)')
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
    
    print("=" * 80)
    print("SINGLE GOAL COMPLETE NETWORK MOVEMENT DELTA PROBING")
    print("=" * 80)
    print(f"Goal: {args.goal}")
    print(f"Method: {args.method}")
    print(f"Split Method: {args.split_method}")
    print(f"Prediction Horizon: {args.k_steps} steps ahead")
    print(f"Worlds: {args.n_worlds}, Rollouts/world: {args.n_rollouts}")
    print(f"Max steps: {args.max_steps}")
    
    method_descriptions = {
        'raw_input': '📥 Raw observations (80D) + goal encoding (10D) = 90D',
        'env_output': '🧠 Env_net output embeddings (64D spatial representations)',
        'combined_embeddings': '🔗 Combined env+ltl embeddings (80D) going into actor',
        'actor_output': '🎯 Actor output (2D action logits/probabilities)'
    }
    print(f"\n{method_descriptions[args.method]}")
    
    # Load model
    print(f"\n🔧 Loading model...")
    store = ModelStore(ENV, EXP, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial(args.goal), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    print(f"✅ Model loaded successfully")
    
    # Create probe
    probe = SingleGoalMovementDeltaProbe(
        model=model,
        extraction_method=args.method,
        k_steps=args.k_steps,
        goal=args.goal,
        n_components=args.n_components
    )
    
    # Collect data
    world_ids = list(range(args.n_worlds))
    features, movement_deltas, metadata = probe.collect_data(
        world_ids=world_ids,
        n_rollouts_per_world=args.n_rollouts,
        max_steps=args.max_steps
    )
    
    # Train and evaluate
    results = probe.train_and_evaluate(features, movement_deltas, metadata, args.split_method)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create summary DataFrame
    summary_data = [{
        'method': args.method,
        'goal': args.goal,
        'split_method': args.split_method,
        'k_steps': args.k_steps,
        'n_worlds': args.n_worlds,
        'n_rollouts': args.n_rollouts,
        'max_steps': args.max_steps,
        'n_samples': len(features),
        'x_train_r2': results['X']['train_r2'],
        'x_test_r2': results['X']['test_r2'],
        'y_train_r2': results['Y']['train_r2'],
        'y_test_r2': results['Y']['test_r2'],
        'distance_r2': results['Combined']['distance_r2'],
        'mean_true_distance': results['Combined']['mean_true_distance'],
        'mean_pred_distance': results['Combined']['mean_pred_distance'],
        'timestamp': timestamp
    }]
    
    # Save to CSV
    goal_clean = args.goal.replace(' ', '_')
    filename = f"single_goal_movement_delta_{args.method}_{args.split_method}_{goal_clean}_k{args.k_steps}_{timestamp}.csv"
    filepath = os.path.join(args.output_dir, filename)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(filepath, index=False)
    print(f"\n💾 Results saved to: {filepath}")
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n🎯 PERFORMANCE FOR {args.goal}:")
    print(f"X Movement R²: {results['X']['test_r2']:.4f}")
    print(f"Y Movement R²: {results['Y']['test_r2']:.4f}")
    print(f"Combined Distance R²: {results['Combined']['distance_r2']:.4f}")
    print(f"Total Samples: {len(features)}")
    
    # Performance interpretation
    avg_r2 = (results['X']['test_r2'] + results['Y']['test_r2']) / 2
    if avg_r2 > 0.3:
        print(f"\n🎉 GOOD PERFORMANCE: Average R² = {avg_r2:.4f}")
    elif avg_r2 > 0.1:
        print(f"\n⚠️  MODERATE PERFORMANCE: Average R² = {avg_r2:.4f}")
    else:
        print(f"\n❌ POOR PERFORMANCE: Average R² = {avg_r2:.4f}")

if __name__ == "__main__":
    main() 