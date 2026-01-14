#!/usr/bin/env python3
"""
Complete Network Zone Direction Probing

This script tests whether the COMPLETE network computation can predict zone directions
by probing the end-to-end process: raw observations → final actions.

HYPOTHESIS: Individual components failed, but the complete integration should work
since the system successfully navigates to target zones in practice.

Usage: python interpretability/probing/probe_complete_network_zone_direction.py
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

class CompleteNetworkZoneDirectionProbe:
    """Test if the complete network can predict zone directions."""
    
    def __init__(self, model, probe_method="action_vectors", target_goals=None, n_components=None):
        self.model = model
        self.probe_method = probe_method  # "action_vectors" or "behavioral_analysis"
        self.target_goals = target_goals or ["FG blue"]
        self.n_components = n_components
        self.pca = None
        
    def collect_data(self, world_ids, n_rollouts_per_world=10, max_steps=200):
        """Collect complete network behavior and target zone directions."""
        print(f"Collecting COMPLETE NETWORK data for zone direction prediction")
        print(f"Method: {self.probe_method}")
        print(f"Testing goals: {self.target_goals}")
        
        all_features = []
        all_zone_directions = []
        all_actions = []
        all_metadata = {
            'world_ids': [],
            'rollout_ids': [],
            'step_ids': [],
            'goal_types': []
        }
        
        total_samples = 0
        
        # Test each goal type
        for goal in self.target_goals:
            print(f"\n--- Testing goal: {goal} ---")
            
            env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
            props = set(env.get_propositions())
            agent = Agent(self.model, ExhaustiveSearch(self.model, props, num_loops=2),
                         propositions=props, verbose=False)
            
            # Extract target zone name and find its position
            target_zone_name = goal.split()[-1]  # "FG blue" -> "blue"
            
            for world_id in world_ids:
                print(f"Processing world {world_id} for goal {goal}...")
                for rollout_id in trange(n_rollouts_per_world, desc=f"Rollouts for world {world_id}"):
                    done = False
                    obs = env.reset(seed=world_id + rollout_id * 1000)
                    agent.reset()
                    
                    # Find target zone position in this world
                    target_zone_pos = None
                    if hasattr(env, 'zone_positions') and env.zone_positions:
                        for zone_name, zone_pos in env.zone_positions.items():
                            if target_zone_name.lower() in zone_name.lower():
                                target_zone_pos = zone_pos[:2]  # Take x,y coordinates
                                break
                    
                    if target_zone_pos is None:
                        print(f"Warning: Could not find target zone '{target_zone_name}' in world {world_id}")
                        continue
                    
                    for step_id in range(max_steps):
                        if done:
                            break
                            
                        # Store current position and compute direction to target zone
                        current_pos = env.agent_pos[:2].copy()
                        direction_to_target = target_zone_pos - current_pos
                        
                        # Normalize direction (unit vector)
                        norm = np.linalg.norm(direction_to_target)
                        if norm > 0:
                            direction_to_target = direction_to_target / norm
                        else:
                            # Agent is exactly at target - use zero vector
                            direction_to_target = np.zeros(2)
                        
                        # Collect complete network information
                        if self.probe_method == "action_vectors":
                            # Method 1: Use actual action vectors as network output
                            action = agent.get_action(obs, {}, deterministic=True).flatten()
                            network_output = action  # 2D action vector
                            
                        elif self.probe_method == "behavioral_analysis":
                            # Method 2: Analyze the complete observation-to-decision process
                            # This captures the full context the network uses
                            obs_features = obs.get('features', np.zeros(80))
                            ltl_info = obs.get('goal', [])  # LTL sequence
                            
                            # Create a representation of the complete decision context
                            # Include spatial features + goal encoding
                            goal_encoding = np.zeros(10)  # Simple goal encoding
                            if ltl_info:
                                # Encode goal type (this is a simplified encoding)
                                if 'blue' in str(ltl_info).lower():
                                    goal_encoding[0] = 1.0
                                elif 'red' in str(ltl_info).lower():
                                    goal_encoding[1] = 1.0
                                elif 'green' in str(ltl_info).lower():
                                    goal_encoding[2] = 1.0
                            
                            network_output = np.concatenate([obs_features, goal_encoding])
                            
                            # Still need to call get_action to advance the agent state
                            _ = agent.get_action(obs, {}, deterministic=True)
                        
                        # Store data
                        all_features.append(network_output)
                        all_zone_directions.append(direction_to_target)
                        all_actions.append(action if self.probe_method == "action_vectors" else np.zeros(2))
                        all_metadata['world_ids'].append(world_id)
                        all_metadata['rollout_ids'].append(rollout_id)
                        all_metadata['step_ids'].append(step_id)
                        all_metadata['goal_types'].append(goal)
                        total_samples += 1
                        
                        # Step environment
                        obs, _, done, info = env.step(action if 'action' in locals() else agent.get_action(obs, {}, deterministic=True).flatten())
            
            env.close()
        
        if total_samples == 0:
            print("No samples collected!")
            return None
            
        print(f"Collected {total_samples} samples across {len(self.target_goals)} goals")
        
        return {
            'features': np.array(all_features),
            'zone_directions': np.array(all_zone_directions),
            'actions': np.array(all_actions),
            'metadata': all_metadata
        }
    
    def create_generalization_splits(self, data):
        """Create temporal, spatial, and environmental generalization splits."""
        features = data['features']
        zone_directions = data['zone_directions']
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
            'train': {'X': features[train_mask], 'y': zone_directions[train_mask]},
            'test': {'X': features[test_mask], 'y': zone_directions[test_mask]}
        }
        
        # 2. Spatial split (different rollouts within same worlds)
        unique_combinations = list(set(zip(world_ids, rollout_ids, goal_types)))
        random.shuffle(unique_combinations)
        
        split_idx = len(unique_combinations) // 2
        train_combinations = set(unique_combinations[:split_idx])
        test_combinations = set(unique_combinations[split_idx:])
        
        train_mask = np.array([
            (w, r, g) in train_combinations 
            for w, r, g in zip(world_ids, rollout_ids, goal_types)
        ])
        test_mask = ~train_mask
        
        splits['spatial'] = {
            'train': {'X': features[train_mask], 'y': zone_directions[train_mask]},
            'test': {'X': features[test_mask], 'y': zone_directions[test_mask]}
        }
        
        # 3. Environmental split (different worlds)
        unique_worlds = list(np.unique(world_ids))
        random.shuffle(unique_worlds)
        
        split_idx = len(unique_worlds) // 2
        train_worlds = set(unique_worlds[:split_idx])
        test_worlds = set(unique_worlds[split_idx:])
        
        train_mask = np.array([w in train_worlds for w in world_ids])
        test_mask = np.array([w in test_worlds for w in world_ids])
        
        splits['environmental'] = {
            'train': {'X': features[train_mask], 'y': zone_directions[train_mask]},
            'test': {'X': features[test_mask], 'y': zone_directions[test_mask]}
        }
        
        return splits
    
    def train_and_evaluate_probes(self, splits):
        """Train and evaluate complete network zone direction prediction probes."""
        results = {}
        
        for split_name, split_data in splits.items():
            X_train, y_train = split_data['train']['X'], split_data['train']['y']
            X_test, y_test = split_data['test']['X'], split_data['test']['y']
            
            if len(X_train) == 0 or len(X_test) == 0:
                print(f"  Skipping {split_name}: insufficient data")
                continue
            
            # Apply PCA if needed
            if self.n_components is not None:
                if self.pca is None:
                    self.pca = PCA(n_components=self.n_components)
                    X_train_pca = self.pca.fit_transform(X_train)
                else:
                    X_train_pca = self.pca.transform(X_train)
                X_test_pca = self.pca.transform(X_test)
            else:
                # Automatic PCA for high-dimensional data
                if X_train.shape[1] > 50:
                    if self.pca is None:
                        self.pca = PCA(n_components=0.95)  # 95% variance
                        X_train_pca = self.pca.fit_transform(X_train)
                    else:
                        X_train_pca = self.pca.transform(X_train)
                    X_test_pca = self.pca.transform(X_test)
                    n_components_used = X_train_pca.shape[1]
                else:
                    X_train_pca = X_train
                    X_test_pca = X_test
                    n_components_used = X_train.shape[1]
            
            # Train probe
            probe = Ridge(alpha=1.0)
            probe.fit(X_train_pca, y_train)
            
            # Evaluate
            y_pred = probe.predict(X_test_pca)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  {split_name}: R²={r2:.3f}, MSE={mse:.3f} (train: {len(X_train)}, test: {len(X_test)}, dims: {X_train.shape[1]}→{n_components_used})")
            
            results[split_name] = {
                'r2_test': r2,
                'mse_test': mse,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'input_dims': X_train.shape[1],
                'pca_dims': n_components_used
            }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Complete Network Zone Direction Probing')
    parser.add_argument('--method', choices=['action_vectors', 'behavioral_analysis'], 
                       default='action_vectors',
                       help='Probing method: action_vectors or behavioral_analysis')
    parser.add_argument('--goals', nargs='+', default=["FG blue"],
                       help='LTL goals to test (e.g., "FG blue")')
    parser.add_argument('--n-worlds', type=int, default=8,
                       help='Number of worlds to use')
    parser.add_argument('--n-rollouts', type=int, default=5,
                       help='Number of rollouts per world per goal')
    parser.add_argument('--max-steps', type=int, default=40,
                       help='Maximum steps per rollout')
    parser.add_argument('--n-components', type=int, default=None,
                       help='Number of PCA components (None for 95% variance)')
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
    
    print("=== COMPLETE NETWORK ZONE DIRECTION PROBING ===")
    print(f"🎯 HYPOTHESIS TEST: Can the complete network predict zone directions?")
    print(f"Method: {args.method}")
    print(f"Goals to test: {args.goals}")
    print(f"Worlds per goal: {args.n_worlds}")
    print(f"Rollouts per world: {args.n_rollouts}")
    print(f"Max steps: {args.max_steps}")
    
    if args.method == "action_vectors":
        print("📤 Analyzing: Final action vectors as network output")
        print("💡 If network computes directions, actions should encode them!")
    else:
        print("📊 Analyzing: Complete observation-to-decision context")
        print("💡 Testing full spatial + symbolic information integration!")
    
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
    probe = CompleteNetworkZoneDirectionProbe(model, args.method, args.goals, args.n_components)
    
    # Collect data
    world_ids = list(range(args.n_worlds))
    data = probe.collect_data(world_ids, args.n_rollouts, args.max_steps)
    
    if data is None:
        print("No data collected. Exiting.")
        return
    
    # Create generalization splits
    print("\nCreating generalization splits...")
    splits = probe.create_generalization_splits(data)
    
    # Train and evaluate probes
    print("\nTraining and evaluating complete network zone direction probes...")
    results = probe.train_and_evaluate_probes(splits)
    
    # Save results
    print("\nSaving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save CSV
    csv_data = []
    for split_name, metrics in results.items():
        csv_data.append({
            'timestamp': timestamp,
            'probe_type': f'complete_network_{args.method}',
            'goals': '|'.join(args.goals),
            'split_type': split_name,
            **metrics
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = f"{args.output_dir}/complete_network_zone_direction_{args.method}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to: {csv_path}")
    
    print(f"\n=== COMPLETE NETWORK ZONE DIRECTION PROBE SUMMARY ===")
    for split_name, metrics in results.items():
        print(f"{split_name}: R² = {metrics['r2_test']:.4f}, MSE = {metrics['mse_test']:.4f}")
    
    print(f"\n🎯 INTERPRETATION:")
    temporal_r2 = results.get('temporal', {}).get('r2_test', 'N/A')
    
    if isinstance(temporal_r2, float) and temporal_r2 > 0.5:
        print("✅ SUCCESS! Complete network CAN predict zone directions!")
        print("   This means spatial + symbolic integration works, just not in individual components.")
    elif isinstance(temporal_r2, float) and temporal_r2 > 0.1:
        print("🤔 MODERATE success - complete network has some directional capability")
    else:
        print("❌ FAILURE - even complete network cannot predict zone directions")
        print("   This suggests the system uses learned associations, not explicit direction computation")

if __name__ == "__main__":
    main() 