#!/usr/bin/env python3
"""
Corrected Movement Delta Probing

This script probes whether the environment network can predict future movement deltas:
- Given current network activations, predict agent_pos[t+k] - agent_pos[t]
- Tests different prediction horizons (k = 1, 2, 3, 4, 5 steps ahead)
- Uses corrected agent position extraction methodology
- Supports both input and output probing of env_net

Usage: python interpretability/probing/probe_movement_delta_corrected.py --probe-type input --k-steps 1
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

class MovementDeltaProbe:
    """Probe for predicting future movement deltas from current network activations."""
    
    def __init__(self, model, probe_type, k_steps, n_components=None):
        self.model = model
        self.probe_type = probe_type  # 'input' or 'output'
        self.k_steps = k_steps  # Prediction horizon
        self.n_components = n_components
        self.pca = None
        
    def collect_data(self, world_ids, n_rollouts_per_world=10, max_steps=200):
        """Collect network activations and future movement deltas."""
        print(f"Collecting {self.probe_type.upper()} data from {len(world_ids)} worlds for {self.k_steps}-step movement delta")
        
        all_activations = []
        all_movement_deltas = []
        all_metadata = {
            'world_ids': [],
            'rollout_ids': [],
            'step_ids': []
        }
        
        env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
        props = set(env.get_propositions())
        agent = Agent(self.model, ExhaustiveSearch(self.model, props, num_loops=2),
                     propositions=props, verbose=False)
        
        total_samples = 0
        
        for world_id in world_ids:
            print(f"Processing world {world_id}...")
            for rollout_id in trange(n_rollouts_per_world, desc=f"Rollouts for world {world_id}"):
                # Reset environment
                env.reset(seed=world_id + rollout_id * 1000)
                agent.reset()
                
                # Collect trajectory data
                trajectory_activations = []
                trajectory_positions = []
                
                # Hook to capture env_net input/output
                captured_data = []
                
                def capture_input_output(module, input, output):
                    if self.probe_type == 'input':
                        # Capture the input tensor (raw 80D features)
                        data = input[0].detach().cpu().numpy().ravel()
                    else:  # 'output'
                        # Capture the output tensor (final 64D embeddings)
                        data = output.detach().cpu().numpy().ravel()
                    captured_data.append(data)
                
                # Hook the entire env_net module
                hook = self.model.env_net.register_forward_hook(capture_input_output)
                
                done = False
                obs = env.reset(seed=world_id + rollout_id * 1000)
                
                for step_id in range(max_steps):
                     if done:
                         break
                         
                     # Store current position
                     current_pos = env.agent_pos[:2].copy()  # Current agent position
                     trajectory_positions.append(current_pos)
                     
                     # Get action - this triggers the hook to capture activations
                     action = agent.get_action(obs, {}, deterministic=True).flatten()
                     
                     # Store the captured activation
                     if len(captured_data) > step_id:
                         activation = captured_data[step_id]
                         trajectory_activations.append(activation)
                     
                     # Step environment
                     obs, _, done, info = env.step(action)
                
                # Remove hook
                hook.remove()
                
                # Convert to arrays
                trajectory_activations = np.array(trajectory_activations)
                trajectory_positions = np.array(trajectory_positions)
                
                # Create movement delta targets
                valid_samples = len(trajectory_positions) - self.k_steps
                if valid_samples <= 0:
                    continue
                    
                for i in range(valid_samples):
                    current_activation = trajectory_activations[i]
                    current_pos = trajectory_positions[i]
                    future_pos = trajectory_positions[i + self.k_steps]
                    
                    # Movement delta = future_pos - current_pos
                    movement_delta = future_pos - current_pos
                    
                    all_activations.append(current_activation)
                    all_movement_deltas.append(movement_delta)
                    all_metadata['world_ids'].append(world_id)
                    all_metadata['rollout_ids'].append(rollout_id)
                    all_metadata['step_ids'].append(i)
                    
                    total_samples += 1
        
        env.close()
        
        if total_samples == 0:
            print("No samples collected!")
            return None
            
        print(f"Collected {total_samples} samples")
        
        return {
            'activations': np.array(all_activations),
            'movement_deltas': np.array(all_movement_deltas),
            'metadata': all_metadata
        }
    
    def create_generalization_splits(self, data):
        """Create temporal, spatial, and environmental generalization splits."""
        activations = data['activations']
        movement_deltas = data['movement_deltas']
        world_ids = np.array(data['metadata']['world_ids'])
        rollout_ids = np.array(data['metadata']['rollout_ids'])
        step_ids = np.array(data['metadata']['step_ids'])
        
        splits = {}
        
        # 1. Temporal split (early vs late steps)
        median_step = np.median(step_ids)
        train_mask = step_ids <= median_step
        test_mask = step_ids > median_step
        
        splits['temporal'] = {
            'train': {'X': activations[train_mask], 'y': movement_deltas[train_mask]},
            'test': {'X': activations[test_mask], 'y': movement_deltas[test_mask]}
        }
        
        # 2. Spatial split (different rollouts within same worlds)
        unique_combinations = list(set(zip(world_ids, rollout_ids)))
        random.shuffle(unique_combinations)
        
        split_idx = len(unique_combinations) // 2
        train_combinations = set(unique_combinations[:split_idx])
        test_combinations = set(unique_combinations[split_idx:])
        
        train_mask = np.array([
            (w, r) in train_combinations 
            for w, r in zip(world_ids, rollout_ids)
        ])
        test_mask = ~train_mask
        
        splits['spatial'] = {
            'train': {'X': activations[train_mask], 'y': movement_deltas[train_mask]},
            'test': {'X': activations[test_mask], 'y': movement_deltas[test_mask]}
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
            'train': {'X': activations[train_mask], 'y': movement_deltas[train_mask]},
            'test': {'X': activations[test_mask], 'y': movement_deltas[test_mask]}
        }
        
        return splits
    
    def train_and_evaluate_probes(self, splits):
        """Train and evaluate movement delta prediction probes."""
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
    parser = argparse.ArgumentParser(description='Corrected Movement Delta Probing')
    parser.add_argument('--probe-type', required=True, choices=['input', 'output'],
                       help='What to probe: input (raw obs) or output (final embeddings)')
    parser.add_argument('--k-steps', type=int, required=True,
                       help='Prediction horizon: predict movement delta k steps ahead')
    parser.add_argument('--n-worlds', type=int, default=10,
                       help='Number of worlds to use')
    parser.add_argument('--n-rollouts', type=int, default=10,
                       help='Number of rollouts per world')
    parser.add_argument('--max-steps', type=int, default=100,
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
    
    print("=== CORRECTED MOVEMENT DELTA PROBING ===")
    print(f"Probe Type: {args.probe_type.upper()}")
    print(f"Prediction Horizon: {args.k_steps} steps ahead")
    print(f"Worlds: {args.n_worlds}")
    print(f"Rollouts per world: {args.n_rollouts}")
    print(f"Max steps: {args.max_steps}")
    
    if args.probe_type == 'input':
        print("📥 Probing: Raw observations (80D) going INTO env_net")
    else:
        print("📤 Probing: Final embeddings (64D) coming OUT of env_net")
    
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
    probe = MovementDeltaProbe(model, args.probe_type, args.k_steps, args.n_components)
    
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
    print("\nTraining and evaluating movement delta probes...")
    results = probe.train_and_evaluate_probes(splits)
    
    # Save results
    print("\nSaving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save CSV
    csv_data = []
    for split_name, metrics in results.items():
        csv_data.append({
            'timestamp': timestamp,
            'probe_type': args.probe_type,
            'k_steps': args.k_steps,
            'split_type': split_name,
            **metrics
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = f"{args.output_dir}/movement_delta_{args.probe_type}_k{args.k_steps}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to: {csv_path}")
    
    print(f"\n=== MOVEMENT DELTA PROBE SUMMARY (k={args.k_steps}) ===")
    for split_name, metrics in results.items():
        print(f"{split_name}: R² = {metrics['r2_test']:.4f}, MSE = {metrics['mse_test']:.4f}")

if __name__ == "__main__":
    main() 