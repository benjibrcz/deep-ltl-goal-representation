#!/usr/bin/env python3
"""
Comprehensive Environment Network Probing

This script probes multiple components of the environment network and various features:
1. Environment network input
2. Environment network MLP layers  
3. Environment network output
4. Zone lidars
5. Zone differences 
6. Agent position

Usage: python interpretability/probing/comprehensive_env_net_probe.py --target zone_lidar --layer env_net.mlp.0
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
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
WORLD_DIR = f"eval_datasets/{ENV}/worlds"

class ComprehensiveEnvNetProbe:
    """Comprehensive probe for environment network components and environmental features."""
    
    def __init__(self, model, layer_name, target_feature, n_components=None):
        self.model = model
        self.layer_name = layer_name
        self.target_feature = target_feature
        self.n_components = n_components
        
        # Get the module to probe
        if layer_name in dict(model.named_modules()):
            self.module = dict(model.named_modules())[layer_name]
        else:
            raise ValueError(f"Layer {layer_name} not found in model")
            
        self.pca = None
        self.probe = None
        
    def collect_data(self, world_ids, n_rollouts_per_world=10, max_steps=200):
        """Collect activations and target features from specified worlds."""
        print(f"Collecting data from {len(world_ids)} worlds for target: {self.target_feature}")
        
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
                
                # Hook to capture activations
                activations = []
                def capture_activations(module, input, output):
                    if isinstance(output, tuple) and len(output) > 0:
                        x = output[1] if len(output) > 1 else output[0]
                    else:
                        x = output
                    if hasattr(x, 'detach'):
                        activations.append(x.detach().cpu().numpy().ravel())
                
                hook = self.module.register_forward_hook(capture_activations)
                
                done = False
                step_count = 0
                
                for step in range(max_steps):
                    if done:
                        break
                        
                    # Store current state before taking action
                    current_obs = obs.copy() if isinstance(obs, dict) else obs
                    
                    # Extract target feature from current observation
                    target_value = self._extract_target_feature(env, current_obs)
                    if target_value is None:
                        continue
                    
                    # Take action to generate activation
                    action = agent.get_action(obs, {}, deterministic=True).flatten()
                    step_result = env.step(action)
                    if len(step_result) == 4:
                        obs, _, done, info = step_result
                        truncated = False
                    else:
                        obs, _, done, truncated, info = step_result
                    done = done or truncated
                    
                    # Store activation and target from the same time step
                    if len(activations) > step_count:
                        activation = activations[step_count]
                        all_activations.append(activation)
                        all_targets.append(target_value)
                        all_metadata['world_ids'].append(world_id)
                        all_metadata['rollout_ids'].append(rollout_idx)
                        all_metadata['step_ids'].append(step)
                        total_samples += 1
                        step_count += 1
                
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
    
    def _extract_target_feature(self, env, obs):
        """Extract the target feature from environment/observation."""
        try:
            if self.target_feature == 'agent_pos':
                # Agent position
                return env.agent_pos[:2].copy()
                
            elif self.target_feature == 'zone_lidar':
                # Zone lidar readings - extract from observation
                zone_keys = [k for k in obs.keys() if k.endswith('_zones_lidar')]
                if zone_keys:
                    # Concatenate all zone lidar readings
                    zone_lidars = []
                    for key in sorted(zone_keys):  # Sort for consistency
                        zone_lidars.extend(obs[key])
                    return np.array(zone_lidars)
                else:
                    return np.zeros(16)  # Default lidar size
                    
            elif self.target_feature == 'zone_differences':
                # Differences between zone lidars
                zone_keys = [k for k in obs.keys() if k.endswith('_zones_lidar')]
                if len(zone_keys) >= 2:
                    zone_keys = sorted(zone_keys)
                    zone1 = obs[zone_keys[0]]
                    zone2 = obs[zone_keys[1]]
                    return zone1 - zone2
                else:
                    return np.zeros(16)  # Default size
                    
            elif self.target_feature == 'wall_sensor':
                # Wall sensor readings
                if 'wall_sensor' in obs:
                    return obs['wall_sensor']
                else:
                    return np.zeros(4)  # Default wall sensor size
                    
            elif self.target_feature == 'wall_lidar':
                # Wall lidar readings
                if 'walls_lidar' in obs:
                    return obs['walls_lidar']
                else:
                    return np.zeros(16)  # Default lidar size
                    
            elif self.target_feature == 'agent_sensors':
                # Agent sensor readings (accelerometer, velocimeter, gyro)
                sensors = []
                for sensor_name in ['accelerometer', 'velocimeter', 'gyro']:
                    if sensor_name in obs:
                        sensors.extend(obs[sensor_name])
                if sensors:
                    return np.array(sensors)
                else:
                    return np.zeros(9)  # 3 sensors × 3 dimensions
                    
            elif self.target_feature == 'joint_positions':
                # Joint positions and velocities
                joint_data = []
                for key in obs.keys():
                    if 'joint' in key.lower() or 'hinge' in key.lower():
                        joint_data.extend(obs[key])
                if joint_data:
                    return np.array(joint_data)
                else:
                    return np.zeros(6)  # Default joint data size
                    
            else:
                raise ValueError(f"Unknown target feature: {self.target_feature}")
                
        except Exception as e:
            print(f"Error extracting target feature {self.target_feature}: {e}")
            return None
    
    def create_generalization_splits(self, data):
        """Create temporal, spatial, and environmental generalization splits."""
        activations = data['activations']
        targets = data['targets']
        world_ids = np.array(data['metadata']['world_ids'])
        rollout_ids = np.array(data['metadata']['rollout_ids'])
        step_ids = np.array(data['metadata']['step_ids'])
        
        splits = {}
        
        # 1. Temporal split: train on early steps, test on later steps
        median_step = np.median(step_ids)
        train_mask = step_ids <= median_step
        test_mask = step_ids > median_step
        
        splits['temporal'] = {
            'train': {'X': activations[train_mask], 'y': targets[train_mask]},
            'test': {'X': activations[test_mask], 'y': targets[test_mask]}
        }
        
        # 2. Spatial split: train on some rollouts, test on others (same worlds)
        unique_world_rollout_pairs = list(set(zip(world_ids, rollout_ids)))
        random.shuffle(unique_world_rollout_pairs)
        
        n_train_pairs = len(unique_world_rollout_pairs) // 2
        train_pairs = set(unique_world_rollout_pairs[:n_train_pairs])
        test_pairs = set(unique_world_rollout_pairs[n_train_pairs:])
        
        train_mask = np.array([((w, r) in train_pairs) for w, r in zip(world_ids, rollout_ids)])
        test_mask = np.array([((w, r) in test_pairs) for w, r in zip(world_ids, rollout_ids)])
        
        splits['spatial'] = {
            'train': {'X': activations[train_mask], 'y': targets[train_mask]},
            'test': {'X': activations[test_mask], 'y': targets[test_mask]}
        }
        
        # 3. Environmental split: train on some worlds, test on others
        unique_worlds = list(np.unique(world_ids))
        random.shuffle(unique_worlds)
        
        n_train_worlds = len(unique_worlds) // 2
        train_worlds = set(unique_worlds[:n_train_worlds])
        test_worlds = set(unique_worlds[n_train_worlds:])
        
        train_mask = np.isin(world_ids, list(train_worlds))
        test_mask = np.isin(world_ids, list(test_worlds))
        
        splits['environmental'] = {
            'train': {'X': activations[train_mask], 'y': targets[train_mask]},
            'test': {'X': activations[test_mask], 'y': targets[test_mask]}
        }
        
        return splits
    
    def apply_dimensionality_reduction(self, X_train, X_test=None):
        """Apply PCA for dimensionality reduction."""
        if self.n_components is None:
            # Use 95% variance explained
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
                results[split_name] = {
                    'r2_train': 0, 'r2_test': 0,
                    'mse_train': float('inf'), 'mse_test': float('inf'),
                    'n_train': len(X_train), 'n_test': len(X_test)
                }
                continue
            
            # Apply PCA
            X_train_reduced, X_test_reduced = self.apply_dimensionality_reduction(X_train, X_test)
            
            # Train probe
            probe = Ridge(alpha=1.0)
            probe.fit(X_train_reduced, y_train)
            
            # Evaluate
            y_train_pred = probe.predict(X_train_reduced)
            y_test_pred = probe.predict(X_test_reduced)
            
            # Calculate metrics for multi-dimensional targets
            if y_train.ndim > 1:
                # Multi-dimensional target
                r2_train = np.mean([r2_score(y_train[:, i], y_train_pred[:, i]) 
                                  for i in range(y_train.shape[1])])
                r2_test = np.mean([r2_score(y_test[:, i], y_test_pred[:, i]) 
                                 for i in range(y_test.shape[1])])
                mse_train = np.mean([mean_squared_error(y_train[:, i], y_train_pred[:, i]) 
                                   for i in range(y_train.shape[1])])
                mse_test = np.mean([mean_squared_error(y_test[:, i], y_test_pred[:, i]) 
                                  for i in range(y_test.shape[1])])
            else:
                # 1D target
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
                'n_components': X_train_reduced.shape[1] if self.pca else X_train.shape[1]
            }
            
            print(f"  {split_name}: R²={r2_test:.3f}, MSE={mse_test:.3f} "
                  f"(train: {len(X_train)}, test: {len(X_test)})")
        
        return results
    
    def save_results(self, results, output_dir, timestamp):
        """Save results to CSV and create visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        csv_data = []
        for split_name, metrics in results.items():
            csv_data.append({
                'timestamp': timestamp,
                'split_type': split_name,
                'target_feature': self.target_feature,
                'layer': self.layer_name,
                **metrics
            })
        
        df = pd.DataFrame(csv_data)
        layer_safe = self.layer_name.replace('.', '_')
        csv_path = f"{output_dir}/comprehensive_probe_{self.target_feature}_{layer_safe}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Create visualization
        self._create_visualization(results, output_dir, timestamp)
        
        # Save summary report
        self._save_summary_report(results, output_dir, timestamp)
        
        return csv_path
    
    def _create_visualization(self, results, output_dir, timestamp):
        """Create visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Comprehensive Probe: {self.target_feature} from {self.layer_name}', fontsize=14)
        
        split_names = list(results.keys())
        r2_scores = [results[split]['r2_test'] for split in split_names]
        mse_scores = [results[split]['mse_test'] for split in split_names]
        n_train = [results[split]['n_train'] for split in split_names]
        n_test = [results[split]['n_test'] for split in split_names]
        
        # R² scores
        axes[0, 0].bar(split_names, r2_scores, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 0].set_title('R² Scores by Split Type')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MSE scores
        axes[0, 1].bar(split_names, mse_scores, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 1].set_title('MSE Scores by Split Type')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sample sizes
        x_pos = np.arange(len(split_names))
        axes[1, 0].bar(x_pos - 0.2, n_train, 0.4, label='Train', color='lightblue')
        axes[1, 0].bar(x_pos + 0.2, n_test, 0.4, label='Test', color='lightcoral')
        axes[1, 0].set_title('Sample Sizes')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(split_names, rotation=45)
        axes[1, 0].legend()
        
        # R² vs MSE scatter
        axes[1, 1].scatter(mse_scores, r2_scores, c=['skyblue', 'lightgreen', 'salmon'], s=100)
        for i, split in enumerate(split_names):
            axes[1, 1].annotate(split, (mse_scores[i], r2_scores[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('MSE')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].set_title('R² vs MSE')
        
        plt.tight_layout()
        
        layer_safe = self.layer_name.replace('.', '_')
        plot_path = f"{output_dir}/comprehensive_probe_{self.target_feature}_{layer_safe}_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _save_summary_report(self, results, output_dir, timestamp):
        """Save a human-readable summary report."""
        layer_safe = self.layer_name.replace('.', '_')
        report_path = f"{output_dir}/summary_{self.target_feature}_{layer_safe}_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Comprehensive Environment Network Probe Summary\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Target Feature: {self.target_feature}\n")
            f.write(f"Layer: {self.layer_name}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write("Results by Split Type:\n")
            f.write("-" * 25 + "\n")
            
            for split_name, metrics in results.items():
                f.write(f"\n{split_name.upper()} SPLIT:\n")
                f.write(f"  R² (test): {metrics['r2_test']:.4f}\n")
                f.write(f"  MSE (test): {metrics['mse_test']:.4f}\n")
                f.write(f"  Train samples: {metrics['n_train']}\n")
                f.write(f"  Test samples: {metrics['n_test']}\n")
                if 'n_components' in metrics:
                    f.write(f"  PCA components: {metrics['n_components']}\n")
            
            f.write(f"\nGeneralization Performance:\n")
            f.write(f"-" * 25 + "\n")
            
            # Calculate generalization gaps
            if 'temporal' in results and 'environmental' in results:
                temporal_r2 = results['temporal']['r2_test']
                env_r2 = results['environmental']['r2_test']
                gap = temporal_r2 - env_r2
                f.write(f"Temporal-Environmental Gap: {gap:.4f}\n")
            
            # Best performing split
            best_split = max(results.keys(), key=lambda x: results[x]['r2_test'])
            f.write(f"Best performing split: {best_split} (R² = {results[best_split]['r2_test']:.4f})\n")
        
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Environment Network Probing')
    parser.add_argument('--layer', required=True, 
                       help='Layer to probe (e.g., env_net.mlp.0, env_net.mlp.2)')
    parser.add_argument('--target', required=True,
                       choices=['agent_pos', 'zone_lidar', 'zone_differences', 'wall_sensor', 
                               'wall_lidar', 'agent_sensors', 'joint_positions'],
                       help='Target feature to predict')
    parser.add_argument('--n-worlds', type=int, default=10,
                       help='Number of worlds to use')
    parser.add_argument('--n-rollouts', type=int, default=N_ROLLOUTS,
                       help='Number of rollouts per world')
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS,
                       help='Maximum steps per rollout')
    parser.add_argument('--n-components', type=int, default=None,
                       help='Number of PCA components (None for 95% variance)')
    parser.add_argument('--output-dir', type=str, 
                       default='interpretability/probing/comprehensive_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=== Comprehensive Environment Network Probing ===")
    print(f"Layer: {args.layer}")
    print(f"Target: {args.target}")
    print(f"Worlds: {args.n_worlds}")
    print(f"Rollouts per world: {args.n_rollouts}")
    print(f"Max steps: {args.max_steps}")
    print(f"PCA components: {args.n_components}")
    
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
    probe = ComprehensiveEnvNetProbe(model, args.layer, args.target, args.n_components)
    
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
    print("\nTraining and evaluating probes...")
    results = probe.train_and_evaluate_probes(splits)
    
    # Save results
    print("\nSaving results...")
    csv_path = probe.save_results(results, args.output_dir, timestamp)
    print(f"Results saved to: {csv_path}")
    
    print("\n=== Summary ===")
    for split_name, metrics in results.items():
        print(f"{split_name}: R² = {metrics['r2_test']:.4f}, MSE = {metrics['mse_test']:.4f}")


if __name__ == "__main__":
    main() 