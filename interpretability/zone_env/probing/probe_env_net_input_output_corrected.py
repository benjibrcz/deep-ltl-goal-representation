#!/usr/bin/env python3
"""
CORRECTED Environment Network True Input/Output Probing

This script correctly probes the ACTUAL input and output of the environment network:
- INPUT: Raw observations (obs['features'] - 80D) that go INTO env_net
- OUTPUT: Final embeddings (env_embedding - 64D) that come OUT of env_net

Key corrections:
1. Extract sensor data from the actual 80D features vector
2. Properly compute zone distances from agent position
3. Use actual observation data instead of placeholder zeros

Usage: python interpretability/probing/probe_env_net_input_output_corrected.py --target agent_pos --probe-type input
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
N_ROLLOUTS = 10
MAX_STEPS = 200

class CorrectedEnvNetProbe:
    """Corrected probe for the actual input and output of the environment network."""
    
    def __init__(self, model, probe_type, target_feature, n_components=None):
        self.model = model
        self.probe_type = probe_type  # 'input' or 'output'
        self.target_feature = target_feature
        self.n_components = n_components
        self.pca = None
        
    def collect_data(self, world_ids, n_rollouts_per_world=10, max_steps=200):
        """Collect input/output data and target features."""
        print(f"Collecting CORRECTED {self.probe_type.upper()} data from {len(world_ids)} worlds for target: {self.target_feature}")
        
        all_activations = []
        all_targets = []
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
        
        try:
            for world_id in world_ids:
                print(f"Processing world {world_id}...")
                
                for rollout_idx in trange(n_rollouts_per_world, desc=f"Rollouts for world {world_id}"):
                    # Try different seeds until we find a valid starting position
                    max_attempts = 10
                    success = False
                    
                    for attempt in range(max_attempts):
                        try:
                            obs = env.reset(seed=SEED + world_id * 1000 + rollout_idx * max_attempts + attempt)
                            success = True
                            break
                        except AssertionError as e:
                            if "World has starting cost" in str(e):
                                continue
                            else:
                                raise e
                    
                    if not success:
                        print(f"  Skipping rollout {rollout_idx} after {max_attempts} failed attempts")
                        continue
                        
                    agent.reset()
                    
                    done = False
                    step_count = 0
                    
                    for step in range(max_steps):
                        if done:
                            break
                            
                        # Store current state before taking action
                        current_obs = obs.copy() if isinstance(obs, dict) else obs
                        
                        # Extract target feature CORRECTLY from current observation/environment
                        target_value = self._extract_target_feature_corrected(env, current_obs)
                        if target_value is None:
                            continue
                        
                        # Take action - this will trigger the hook and capture data
                        action = agent.get_action(obs, {}, deterministic=True).flatten()
                        obs, _, done, info = env.step(action)
                        
                        # Store activation and target from the same time step
                        if len(captured_data) > step_count:
                            activation = captured_data[step_count]
                            all_activations.append(activation)
                            all_targets.append(target_value)
                            all_metadata['world_ids'].append(world_id)
                            all_metadata['rollout_ids'].append(rollout_idx)
                            all_metadata['step_ids'].append(step)
                            total_samples += 1
                            step_count += 1
                    
                    # Clear captured data for next rollout
                    captured_data.clear()
                    
        finally:
            hook.remove()
            env.close()
        
        if total_samples == 0:
            print("No samples collected!")
            return None
            
        print(f"Collected {total_samples} samples")
        
        return {
            'activations': np.array(all_activations),
            'targets': np.array(all_targets),
            'metadata': all_metadata
        }
    
    def _extract_target_feature_corrected(self, env, obs):
        """CORRECTED: Extract the target feature from environment/observation."""
        try:
            if self.target_feature == 'agent_pos':
                # Agent position - use environment attribute (this was already correct)
                return env.agent_pos[:2].copy()
                
            elif self.target_feature == 'zone_distances':
                # CORRECTED: Compute actual distances from agent to zone centers
                agent_pos = env.agent_pos[:2]
                if hasattr(env, 'zone_positions') and env.zone_positions:
                    distances = []
                    # Sort zone names for consistency
                    for zone_name in sorted(env.zone_positions.keys()):
                        zone_pos = env.zone_positions[zone_name]
                        dist = np.linalg.norm(agent_pos - zone_pos[:2])
                        distances.append(dist)
                    return np.array(distances)
                else:
                    return np.zeros(8)  # 8 zones based on what we saw
                    
            elif self.target_feature == 'zone_directions':
                # CORRECTED: Compute actual direction vectors from agent to zone centers
                agent_pos = env.agent_pos[:2]
                if hasattr(env, 'zone_positions') and env.zone_positions:
                    directions = []
                    # Sort zone names for consistency
                    for zone_name in sorted(env.zone_positions.keys()):
                        zone_pos = env.zone_positions[zone_name]
                        direction = zone_pos[:2] - agent_pos
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction = direction / norm
                        directions.extend(direction)  # Add x, y components
                    return np.array(directions)
                else:
                    return np.zeros(16)  # 8 zones × 2D = 16D
                    
            elif self.target_feature == 'raw_features':
                # NEW: Probe the entire raw features vector
                return obs['features'].copy()
                
            elif self.target_feature == 'features_subset':
                # NEW: Probe a subset of features that might be sensor-related
                features = obs['features']
                # Based on our exploration, try the last part that had non-zero values
                return features[-16:].copy()  # Last 16 elements
                
            elif self.target_feature == 'velocity_features':
                # NEW: Probe what might be velocity/acceleration (first few elements)
                features = obs['features']
                return features[:6].copy()  # First 6 elements
                
            elif self.target_feature == 'zone_lidar_approx':
                # NEW: Try to extract zone lidar from features
                features = obs['features']
                # Based on observation, try middle section
                return features[16:32].copy()  # Elements 16-31 (16 lidar bins?)
                
            elif self.target_feature == 'wall_sensor':
                # Wall sensor readings - extract from features if not separate key
                if 'wall_sensor' in obs:
                    return obs['wall_sensor']
                else:
                    # Try to extract from features (assuming specific positions)
                    features = obs['features']
                    return features[32:36].copy()  # Guess: elements 32-35 (4D wall sensor)
                    
            elif self.target_feature == 'wall_lidar':
                # Wall lidar readings - extract from features if not separate key  
                if 'walls_lidar' in obs or 'wall_lidar' in obs:
                    return obs.get('walls_lidar', obs.get('wall_lidar'))
                else:
                    # Try to extract from features (assuming specific positions)
                    features = obs['features']
                    return features[36:52].copy()  # Guess: elements 36-51 (16D wall lidar)
                    
            elif self.target_feature == 'agent_sensors':
                # Agent sensor readings (accelerometer, velocimeter, gyro)
                sensors = []
                for sensor_name in ['accelerometer', 'velocimeter', 'gyro']:
                    if sensor_name in obs:
                        sensors.extend(obs[sensor_name])
                if sensors:
                    return np.array(sensors)
                else:
                    # Try to extract from features (assuming specific positions)
                    features = obs['features']
                    return features[:9].copy()  # Guess: first 9 elements (3 sensors × 3D)
                    
            elif self.target_feature == 'joint_positions':
                # Joint positions and velocities
                joint_data = []
                for key in obs.keys():
                    if 'joint' in key.lower() or 'hinge' in key.lower():
                        joint_data.extend(obs[key])
                if joint_data:
                    return np.array(joint_data)
                else:
                    # Try to extract from features (assuming specific positions)
                    features = obs['features']
                    return features[52:58].copy()  # Guess: elements 52-57 (6D joint data)
                
            else:
                raise ValueError(f"Unknown target feature: {self.target_feature}")
                
        except Exception as e:
            print(f"Error extracting target feature {self.target_feature}: {e}")
            return None
    
    def create_generalization_splits(self, data, split_method="temporal"):
        """Create the specified generalization split."""
        activations = data['activations']
        targets = data['targets']
        world_ids = np.array(data['metadata']['world_ids'])
        rollout_ids = np.array(data['metadata']['rollout_ids'])
        step_ids = np.array(data['metadata']['step_ids'])
        
        splits = {}
        
        if split_method == "temporal":
            # Temporal split: early vs late steps
            print("📊 Creating temporal split (early vs late steps)")
            median_step = np.median(step_ids)
            train_mask = step_ids <= median_step
            test_mask = step_ids > median_step
            
            splits['temporal'] = {
                'train': {'X': activations[train_mask], 'y': targets[train_mask]},
                'test': {'X': activations[test_mask], 'y': targets[test_mask]}
            }
            
        elif split_method == "spatial":
            # Spatial split: different rollouts
            print("🌍 Creating spatial split (different rollouts)")
            unique_world_rollout_pairs = list(set(zip(world_ids, rollout_ids)))
            random.shuffle(unique_world_rollout_pairs)
            
            n_train_pairs = int(0.7 * len(unique_world_rollout_pairs))
            train_pairs = set(unique_world_rollout_pairs[:n_train_pairs])
            test_pairs = set(unique_world_rollout_pairs[n_train_pairs:])
            
            train_mask = np.array([((w, r) in train_pairs) for w, r in zip(world_ids, rollout_ids)])
            test_mask = np.array([((w, r) in test_pairs) for w, r in zip(world_ids, rollout_ids)])
            
            splits['spatial'] = {
                'train': {'X': activations[train_mask], 'y': targets[train_mask]},
                'test': {'X': activations[test_mask], 'y': targets[test_mask]}
            }
            
        elif split_method == "environmental":
            # Environmental split: different worlds
            print("🌎 Creating environmental split (different worlds)")
            unique_worlds = list(np.unique(world_ids))
            random.shuffle(unique_worlds)
            
            n_train_worlds = max(1, int(0.7 * len(unique_worlds)))
            train_worlds = set(unique_worlds[:n_train_worlds])
            test_worlds = set(unique_worlds[n_train_worlds:])
            
            train_mask = np.isin(world_ids, list(train_worlds))
            test_mask = np.isin(world_ids, list(test_worlds))
            
            print(f"Train worlds: {sorted(train_worlds)}, Test worlds: {sorted(test_worlds)}")
            
            splits['environmental'] = {
                'train': {'X': activations[train_mask], 'y': targets[train_mask]},
                'test': {'X': activations[test_mask], 'y': targets[test_mask]}
            }
            
        else:
            raise ValueError(f"Unknown split_method: {split_method}")
        
        return splits
    
    def apply_dimensionality_reduction(self, X_train, X_test=None):
        """Apply PCA for dimensionality reduction."""
        if self.n_components is None:
            self.pca = PCA(n_components=0.95)
        else:
            self.pca = PCA(n_components=self.n_components)
        
        X_train_reduced = self.pca.fit_transform(X_train)
        
        if X_test is not None:
            X_test_reduced = self.pca.transform(X_test)
            return X_train_reduced, X_test_reduced
        
        return X_train_reduced
    
    def train_and_evaluate_probes(self, splits):
        """Train and evaluate probes for each generalization split."""
        results = {}
        
        for split_name, split_data in splits.items():
            X_train = split_data['train']['X']
            y_train = split_data['train']['y']
            X_test = split_data['test']['X']
            y_test = split_data['test']['y']
            
            if len(X_train) == 0 or len(X_test) == 0:
                print(f"  {split_name}: Insufficient data (train: {len(X_train)}, test: {len(X_test)})")
                continue
            
            # Apply PCA
            X_train_reduced, X_test_reduced = self.apply_dimensionality_reduction(X_train, X_test)
            
            # Train probe
            probe = Ridge(alpha=1.0)
            probe.fit(X_train_reduced, y_train)
            
            # Evaluate
            y_train_pred = probe.predict(X_train_reduced)
            y_test_pred = probe.predict(X_test_reduced)
            
            # Calculate metrics
            if y_train.ndim > 1:
                r2_train = np.mean([r2_score(y_train[:, i], y_train_pred[:, i]) 
                                  for i in range(y_train.shape[1])])
                r2_test = np.mean([r2_score(y_test[:, i], y_test_pred[:, i]) 
                                 for i in range(y_test.shape[1])])
                mse_train = np.mean([mean_squared_error(y_train[:, i], y_train_pred[:, i]) 
                                   for i in range(y_train.shape[1])])
                mse_test = np.mean([mean_squared_error(y_test[:, i], y_test_pred[:, i]) 
                                  for i in range(y_test.shape[1])])
            else:
                r2_train = r2_score(y_train, y_train_pred)
                r2_test = r2_score(y_test, y_test_pred)
                mse_train = mean_squared_error(y_train, y_train_pred)
                mse_test = mean_squared_error(y_test, y_test_pred)
            
            results[split_name] = {
                'r2_train': r2_train,
                'r2_test': r2_test,
                'mse_train': mse_train,
                'mse_test': mse_test,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_components': X_train_reduced.shape[1] if self.pca else X_train.shape[1],
                'original_dims': X_train.shape[1]
            }
            
            print(f"  {split_name}: R²={r2_test:.3f}, MSE={mse_test:.3f} "
                  f"(train: {len(X_train)}, test: {len(X_test)}, dims: {X_train.shape[1]}→{X_train_reduced.shape[1]})")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='CORRECTED Environment Network True Input/Output Probing')
    parser.add_argument('--probe-type', required=True, choices=['input', 'output'],
                       help='What to probe: input (raw obs) or output (final embeddings)')
    parser.add_argument('--target', required=True,
                       choices=['agent_pos', 'zone_distances', 'zone_directions', 
                               'raw_features', 'features_subset', 'velocity_features', 
                               'zone_lidar_approx', 'wall_sensor', 'wall_lidar', 'agent_sensors', 'joint_positions'],
                       help='Target feature to predict')
    parser.add_argument('--n-worlds', type=int, default=5,
                       help='Number of worlds to use')
    parser.add_argument('--n-rollouts', type=int, default=5,
                       help='Number of rollouts per world')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum steps per rollout')
    parser.add_argument('--n-components', type=int, default=None,
                       help='Number of PCA components (None for 95% variance)')
    parser.add_argument('--split-method', type=str, default='temporal',
                       choices=['temporal', 'spatial', 'environmental'],
                       help='Split method: temporal (early vs late steps), spatial (different rollouts), environmental (different worlds)')
    parser.add_argument('--output-dir', type=str, 
                       default='interpretability/probing/corrected_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=== CORRECTED Environment Network True Input/Output Probing ===")
    print(f"Probe Type: {args.probe_type.upper()}")
    print(f"Target: {args.target}")
    print(f"Split Method: {args.split_method}")
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
    probe = CorrectedEnvNetProbe(model, args.probe_type, args.target, args.n_components)
    
    # Collect data
    world_ids = list(range(args.n_worlds))
    data = probe.collect_data(world_ids, args.n_rollouts, args.max_steps)
    
    if data is None:
        print("No data collected. Exiting.")
        return
    
    # Create generalization splits
    print(f"\nCreating {args.split_method} split...")
    splits = probe.create_generalization_splits(data, args.split_method)
    
    # Train and evaluate probes
    print("\nTraining and evaluating probes...")
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
            'split_method': args.split_method,
            'split_type': split_name,
            'target_feature': args.target,
            **metrics
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = f"{args.output_dir}/corrected_env_net_{args.probe_type}_{args.split_method}_{args.target}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to: {csv_path}")
    
    print(f"\n=== CORRECTED {args.probe_type.upper()} PROBE SUMMARY ===")
    print(f"Split Method: {args.split_method}")
    for split_name, metrics in results.items():
        print(f"{split_name}: R² = {metrics['r2_test']:.4f}, MSE = {metrics['mse_test']:.4f}")


if __name__ == "__main__":
    main() 