#!/usr/bin/env python3
"""
Complete Network Movement Delta Probing

This script tests whether the COMPLETE network computation can predict future movement deltas
by probing the end-to-end process: raw observations → final actions.

HYPOTHESIS: Individual components (env_net, actor) failed to predict movement well,
but the complete integration should work since the system successfully navigates in practice.

The complete network processes:
1. Raw observations (80D) → env_net → spatial embeddings (64D)
2. LTL formula → ltl_net → goal embeddings (16D) 
3. Combined embeddings → actor → actions (2D)

We test multiple extraction points:
- Raw input features (80D + goal encoding)
- Env_net output (64D spatial embeddings)
- Combined embeddings (64D + 16D = 80D)
- Actor output (action logits/probabilities)

Usage: python interpretability/probing/probe_complete_network_movement_delta.py --method combined_embeddings --k-steps 1
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

class CompleteNetworkMovementDeltaProbe:
    """Test if the complete network can predict future movement deltas."""
    
    def __init__(self, model, extraction_method="combined_embeddings", k_steps=1, target_goals=None, n_components=None):
        self.model = model
        self.extraction_method = extraction_method
        self.k_steps = k_steps
        self.target_goals = target_goals or ["FG blue", "FG green", "FG yellow", "FG magenta"]
        self.n_components = n_components
        self.pca = None
        
        print(f"🔍 COMPLETE NETWORK MOVEMENT DELTA PROBE")
        print(f"Extraction Method: {extraction_method}")
        print(f"Prediction Horizon: {k_steps} steps")
        print(f"Target Goals: {target_goals}")
    
    def collect_data(self, world_ids, n_rollouts_per_world=8, max_steps=200):
        """Collect complete network representations and future movement deltas."""
        print(f"\n📊 COLLECTING DATA")
        print(f"Worlds: {len(world_ids)}, Rollouts/world: {n_rollouts_per_world}, Max steps: {max_steps}")
        
        all_features = []
        all_movement_deltas = []
        all_metadata = {
            'world_ids': [],
            'rollout_ids': [],
            'step_ids': [],
            'goals': []
        }
        
        total_samples = 0
        
        for goal in self.target_goals:
            print(f"\n--- Processing goal: {goal} ---")
            
            env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
            props = set(env.get_propositions())
            agent = Agent(self.model, ExhaustiveSearch(self.model, props, num_loops=2),
                         propositions=props, verbose=False)
            
            for world_id in world_ids:
                for rollout_id in trange(n_rollouts_per_world, desc=f"World {world_id}"):
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
                            goal_encoding = self._encode_goal(goal)
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
                        all_metadata['goals'].append(goal)
                        
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
    
    def train_and_evaluate(self, features, movement_deltas, metadata):
        """Train probe and evaluate performance."""
        print(f"\n🎯 TRAINING AND EVALUATION")
        
        # Apply PCA if specified
        if self.n_components is not None:
            print(f"Applying PCA: {features.shape[1]} → {self.n_components} dimensions")
            self.pca = PCA(n_components=self.n_components)
            features = self.pca.fit_transform(features)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Train/test split (70/30)
        n_train = int(0.7 * len(features))
        indices = list(range(len(features)))
        random.shuffle(indices)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
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
    
    def analyze_by_goal(self, features, movement_deltas, metadata):
        """Analyze performance broken down by goal."""
        print(f"\n📊 GOAL-SPECIFIC ANALYSIS")
        
        goal_results = {}
        
        for goal in self.target_goals:
            goal_mask = [g == goal for g in metadata['goals']]
            if not any(goal_mask):
                continue
                
            goal_features = features[goal_mask]
            goal_deltas = movement_deltas[goal_mask]
            
            print(f"\n--- {goal}: {len(goal_features)} samples ---")
            
            if len(goal_features) < 50:  # Need minimum samples
                print("❌ Insufficient samples for reliable analysis")
                continue
            
            # Quick train/test split for this goal
            n_train = int(0.7 * len(goal_features))
            indices = list(range(len(goal_features)))
            random.shuffle(indices)
            
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            X_train = goal_features[train_idx]
            X_test = goal_features[test_idx]
            y_train = goal_deltas[train_idx]
            y_test = goal_deltas[test_idx]
            
            # Train combined model
            model_x = Ridge(alpha=1.0)
            model_y = Ridge(alpha=1.0)
            
            model_x.fit(X_train, y_train[:, 0])
            model_y.fit(X_train, y_train[:, 1])
            
            # Predictions
            pred_x = model_x.predict(X_test)
            pred_y = model_y.predict(X_test)
            
            # Combined metrics
            y_pred_combined = np.column_stack([pred_x, pred_y])
            
            test_distances_true = np.linalg.norm(y_test, axis=1)
            test_distances_pred = np.linalg.norm(y_pred_combined, axis=1)
            
            distance_r2 = r2_score(test_distances_true, test_distances_pred)
            
            goal_results[goal] = {
                'n_samples': len(goal_features),
                'distance_r2': distance_r2,
                'mean_distance': np.mean(test_distances_true)
            }
            
            print(f"Distance R²: {distance_r2:.4f}")
            print(f"Mean movement distance: {np.mean(test_distances_true):.4f}")
        
        return goal_results

def main():
    parser = argparse.ArgumentParser(description='Complete Network Movement Delta Probing')
    parser.add_argument('--method', required=True, 
                       choices=['raw_input', 'env_output', 'combined_embeddings', 'actor_output'],
                       help='Extraction method: raw_input, env_output, combined_embeddings, or actor_output')
    parser.add_argument('--k-steps', type=int, default=1,
                       help='Prediction horizon: predict movement delta k steps ahead')
    parser.add_argument('--n-worlds', type=int, default=8,
                       help='Number of worlds to use')
    parser.add_argument('--n-rollouts', type=int, default=8,
                       help='Number of rollouts per world')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum steps per rollout')
    parser.add_argument('--n-components', type=int, default=None,
                       help='Number of PCA components (None for no PCA)')
    parser.add_argument('--goals', nargs='+', default=["FG blue", "FG green", "FG yellow", "FG magenta"],
                       help='Goals to test')
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
    print("COMPLETE NETWORK MOVEMENT DELTA PROBING")
    print("=" * 80)
    print(f"Method: {args.method}")
    print(f"Prediction Horizon: {args.k_steps} steps ahead")
    print(f"Goals: {args.goals}")
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
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    print(f"✅ Model loaded successfully")
    
    # Create probe
    probe = CompleteNetworkMovementDeltaProbe(
        model=model,
        extraction_method=args.method,
        k_steps=args.k_steps,
        target_goals=args.goals,
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
    overall_results = probe.train_and_evaluate(features, movement_deltas, metadata)
    goal_results = probe.analyze_by_goal(features, movement_deltas, metadata)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_summary = {
        'method': args.method,
        'k_steps': args.k_steps,
        'n_samples': len(features),
        'goals': args.goals,
        'overall_results': overall_results,
        'goal_results': goal_results,
        'timestamp': timestamp
    }
    
    # Save to CSV
    filename = f"complete_network_movement_delta_{args.method}_k{args.k_steps}_{timestamp}.csv"
    filepath = os.path.join(args.output_dir, filename)
    
    # Create summary DataFrame
    summary_data = []
    for goal, result in goal_results.items():
        summary_data.append({
            'method': args.method,
            'k_steps': args.k_steps,
            'goal': goal,
            'n_samples': result['n_samples'],
            'distance_r2': result['distance_r2'],
            'mean_distance': result['mean_distance']
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(filepath, index=False)
        print(f"\n💾 Results saved to: {filepath}")
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"X Movement R²: {overall_results['X']['test_r2']:.4f}")
    print(f"Y Movement R²: {overall_results['Y']['test_r2']:.4f}")
    print(f"Combined Distance R²: {overall_results['Combined']['distance_r2']:.4f}")
    
    print(f"\n📊 BY GOAL:")
    for goal, result in goal_results.items():
        print(f"{goal}: R²={result['distance_r2']:.4f} ({result['n_samples']} samples)")
    
    if goal_results:
        avg_r2 = np.mean([r['distance_r2'] for r in goal_results.values()])
        print(f"\n🏆 AVERAGE R² ACROSS GOALS: {avg_r2:.4f}")

if __name__ == "__main__":
    main() 