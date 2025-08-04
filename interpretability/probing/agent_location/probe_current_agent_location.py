#!/usr/bin/env python3
"""
Robust Agent Location Probing with Proper Evaluation

This script implements a comprehensive agent location probe that addresses overfitting issues
by incorporating:
1. Cross-world generalization testing
2. Temporal generalization testing  
3. Dimensionality reduction to prevent feature overfitting
4. Baseline comparisons
5. Proper train/test splits with adequate sample sizes
6. Multiple evaluation metrics

Usage: python interpretability/probing/agent_location/probe_current_agent_location.py --layer env_net.mlp.2 --n-train-worlds 8 --n-test-worlds 2 --max-steps 200
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

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
N_TRAIN_WORLDS = 8
N_TEST_WORLDS = 2
N_ROLLOUTS_PER_WORLD = 10
MAX_STEPS = 200
WORLD_DIR = f"eval_datasets/{ENV}/worlds"

class RobustAgentLocationProbe:
    """Robust agent location probe with proper evaluation methodology."""
    
    def __init__(self, model, layer_name, n_components=None):
        self.model = model
        self.layer_name = layer_name
        self.n_components = n_components
        self.module = dict(model.named_modules())[layer_name]
        self.pca = None
        self.probe = None
        self.baseline_probe = None
        
    def collect_data(self, world_ids, n_rollouts_per_world=10, max_steps=200):
        """Collect activations and positions from specified worlds."""
        print(f"Collecting data from {len(world_ids)} worlds...")
        
        all_activations = []
        all_positions = []
        all_world_ids = []
        all_rollout_ids = []
        all_step_ids = []
        
        env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
        props = set(env.get_propositions())
        agent = Agent(self.model, ExhaustiveSearch(self.model, props, num_loops=2),
                     propositions=props, verbose=False)
        
        total_samples = 0
        
        for world_idx, world_id in enumerate(world_ids):
            world_file = f"{WORLD_DIR}/world_info_{world_id}.pkl"
            if not os.path.exists(world_file):
                print(f"World file not found: {world_file}, skipping.")
                continue
                
            env.load_world_info(world_file)
            
            for rollout_idx in trange(n_rollouts_per_world, desc=f"World {world_id}"):
                # Try different seeds for different starting positions
                max_attempts = 10
                for attempt in range(max_attempts):
                    try:
                        obs = env.reset(seed=SEED + world_id * 1000 + rollout_idx * max_attempts + attempt)
                        break
                    except AssertionError as e:
                        if "World has starting cost" in str(e) and attempt < max_attempts - 1:
                            continue
                        else:
                            print(f"Failed to reset world {world_id}, rollout {rollout_idx}: {e}")
                            break
                
                if attempt == max_attempts - 1:
                    continue
                    
                agent.reset()
                
                # Collect activations and positions
                activations = []
                positions = []
                
                def grab_activation(m, inp, out):
                    x = out[1] if isinstance(out, tuple) else out
                    activations.append(x.detach().cpu().numpy().ravel())
                
                hook = self.module.register_forward_hook(grab_activation)
                
                done = False
                for step in range(max_steps):
                    if done:
                        break
                    
                    # Record position BEFORE taking action
                    positions.append(env.agent_pos[:2].copy())
                    
                    # Take action and record activation
                    action = agent.get_action(obs, {}, deterministic=True).flatten()
                    obs, _, done, _ = env.step(action)
                    
                    if len(activations) < len(positions):
                        activations.append(activations[-1] if activations else np.zeros(activations[0].shape))
                
                hook.remove()
                
                # Align activations and positions
                if len(activations) > len(positions):
                    activations = activations[:len(positions)]
                
                if len(activations) > 0:
                    all_activations.extend(activations)
                    all_positions.extend(positions)
                    all_world_ids.extend([world_id] * len(positions))
                    all_rollout_ids.extend([rollout_idx] * len(positions))
                    all_step_ids.extend(list(range(len(positions))))
                    total_samples += len(positions)
        
        env.close()
        
        print(f"Collected {total_samples} samples from {len(world_ids)} worlds")
        
        return {
            'activations': np.array(all_activations),
            'positions': np.array(all_positions),
            'world_ids': np.array(all_world_ids),
            'rollout_ids': np.array(all_rollout_ids),
            'step_ids': np.array(all_step_ids)
        }
    
    def apply_dimensionality_reduction(self, X_train, X_test=None):
        """Apply PCA for dimensionality reduction to prevent overfitting."""
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
    
    def train_baseline_probes(self, X_train, Y_train, X_test, Y_test):
        """Train baseline probes for comparison."""
        baselines = {}
        
        # Baseline 1: Previous position (if available)
        if len(Y_train) > 1:
            Y_train_prev = np.roll(Y_train, 1, axis=0)
            Y_train_prev[0] = Y_train[0]  # First position same as current
            Y_test_prev = np.roll(Y_test, 1, axis=0)
            Y_test_prev[0] = Y_test[0]
            
            prev_mse = mean_squared_error(Y_test, Y_test_prev)
            prev_r2 = r2_score(Y_test, Y_test_prev)
            baselines['previous_position'] = {'mse': prev_mse, 'r2': prev_r2}
        
        # Baseline 2: Mean position
        mean_pos = np.mean(Y_train, axis=0)
        Y_test_mean = np.tile(mean_pos, (len(Y_test), 1))
        mean_mse = mean_squared_error(Y_test, Y_test_mean)
        mean_r2 = r2_score(Y_test, Y_test_mean)
        baselines['mean_position'] = {'mse': mean_mse, 'r2': mean_r2}
        
        # Baseline 3: Linear extrapolation (using last 2 positions)
        Y_test_linear = np.zeros_like(Y_test)
        Y_test_linear[0] = Y_test[0]  # First position same as current
        for i in range(1, len(Y_test)):
            if i >= 2:
                # Linear extrapolation: current = previous + (previous - two_ago)
                Y_test_linear[i] = Y_test[i-1] + (Y_test[i-1] - Y_test[i-2])
            else:
                Y_test_linear[i] = Y_test[i-1]  # Fall back to previous position
        linear_mse = mean_squared_error(Y_test, Y_test_linear)
        linear_r2 = r2_score(Y_test, Y_test_linear)
        baselines['linear_extrapolation'] = {'mse': linear_mse, 'r2': linear_r2}
        
        # Baseline 4: Random position within observed range
        pos_range = np.ptp(Y_test, axis=0)  # Peak-to-peak range
        pos_min = np.min(Y_test, axis=0)
        Y_test_random = np.random.uniform(pos_min, pos_min + pos_range, size=Y_test.shape)
        random_mse = mean_squared_error(Y_test, Y_test_random)
        random_r2 = r2_score(Y_test, Y_test_random)
        baselines['random_position'] = {'mse': random_mse, 'r2': random_r2}
        
        # Baseline 5: Linear regression on raw features (no PCA)
        self.baseline_probe = Ridge(alpha=1.0)
        self.baseline_probe.fit(X_train, Y_train)
        Y_pred_baseline = self.baseline_probe.predict(X_test)
        baseline_mse = mean_squared_error(Y_test, Y_pred_baseline)
        baseline_r2 = r2_score(Y_test, Y_pred_baseline)
        baselines['ridge_raw_features'] = {'mse': baseline_mse, 'r2': baseline_r2}
        
        return baselines
    
    def train_probe(self, X_train, Y_train, X_test, Y_test):
        """Train the main probe with dimensionality reduction."""
        # Apply dimensionality reduction
        X_train_reduced, X_test_reduced = self.apply_dimensionality_reduction(X_train, X_test)
        
        print(f"Feature dimensionality: {X_train.shape[1]} -> {X_train_reduced.shape[1]}")
        if self.pca is not None:
            print(f"Variance explained: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Train probe
        self.probe = Ridge(alpha=1.0)
        self.probe.fit(X_train_reduced, Y_train)
        
        # Evaluate
        Y_pred = self.probe.predict(X_test_reduced)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        
        return {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'predictions': Y_pred,
            'n_components': X_train_reduced.shape[1]
        }
    
    def evaluate_temporal_generalization(self, data, train_steps=100, test_steps=50):
        """Evaluate temporal generalization: train on early steps, test on later steps."""
        print(f"\nEvaluating temporal generalization (train: 0-{train_steps}, test: {train_steps}-{train_steps+test_steps})...")
        
        # Filter data by step
        train_mask = data['step_ids'] < train_steps
        test_mask = (data['step_ids'] >= train_steps) & (data['step_ids'] < train_steps + test_steps)
        
        if not np.any(train_mask) or not np.any(test_mask):
            print("Insufficient data for temporal evaluation")
            return None
        
        X_train = data['activations'][train_mask]
        Y_train = data['positions'][train_mask]
        X_test = data['activations'][test_mask]
        Y_test = data['positions'][test_mask]
        
        print(f"Temporal split: {len(X_train)} train samples, {len(X_test)} test samples")
        
        # Train and evaluate
        X_train_reduced, X_test_reduced = self.apply_dimensionality_reduction(X_train, X_test)
        probe_temp = Ridge(alpha=1.0)
        probe_temp.fit(X_train_reduced, Y_train)
        
        Y_pred_train = probe_temp.predict(X_train_reduced)
        train_mse = mean_squared_error(Y_train, Y_pred_train)
        train_r2 = r2_score(Y_train, Y_pred_train)
        train_mae = mean_absolute_error(Y_train, Y_pred_train)
        
        Y_pred = probe_temp.predict(X_test_reduced)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        
        print(f"Train performance: R²={train_r2:.4f}, MSE={train_mse:.6f}, MAE={train_mae:.4f}")
        print(f"Test performance:  R²={r2:.4f}, MSE={mse:.6f}, MAE={mae:.4f}")
        
        return {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'predictions': Y_pred,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_mse': train_mse,
            'train_r2': train_r2,
            'train_mae': train_mae
        }
    
    def create_visualizations(self, results, data, output_dir):
        """Create comprehensive visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Only plot temporal generalization if in single rollout mode
        if results['data_info'].get('mode', '') == 'single_rollout_temporal':
            if results['temporal'] is not None:
                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(data['test_positions'][:, 0], data['test_positions'][:, 1], 
                           alpha=0.6, label='True', s=10)
                plt.scatter(results['temporal']['predictions'][:, 0], results['temporal']['predictions'][:, 1], 
                           alpha=0.6, label='Temporal Predicted', s=10)
                plt.title('Temporal Generalization: True vs Predicted')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.legend()
                plt.axis('equal')
                
                plt.subplot(1, 2, 2)
                temporal_errors = np.linalg.norm(data['test_positions'] - results['temporal']['predictions'], axis=1)
                plt.hist(temporal_errors, bins=30, alpha=0.7, edgecolor='black')
                plt.title('Temporal Prediction Error Distribution')
                plt.xlabel('Euclidean Error')
                plt.ylabel('Frequency')
                plt.axvline(np.mean(temporal_errors), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(temporal_errors):.3f}')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'temporal_generalization.png'), dpi=150, bbox_inches='tight')
                plt.close()
            else:
                print("No temporal results available for visualization")
            return
        
        # 1. Performance comparison plot (main/baseline mode)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² comparison
        methods = list(results['baselines'].keys()) + ['pca_probe']
        r2_scores = [results['baselines'][m]['r2'] for m in results['baselines']] + [results['main']['r2']]
        
        axes[0, 0].bar(methods, r2_scores, color=['lightblue'] * len(results['baselines']) + ['orange'])
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MSE comparison
        mse_scores = [results['baselines'][m]['mse'] for m in results['baselines']] + [results['main']['mse']]
        axes[0, 1].bar(methods, mse_scores, color=['lightblue'] * len(results['baselines']) + ['orange'])
        axes[0, 1].set_title('MSE Comparison')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # True vs Predicted scatter
        axes[1, 0].scatter(data['test_positions'][:, 0], data['test_positions'][:, 1], 
                         alpha=0.6, label='True', s=10)
        axes[1, 0].scatter(results['main']['predictions'][:, 0], results['main']['predictions'][:, 1], 
                         alpha=0.6, label='Predicted', s=10)
        axes[1, 0].set_title('True vs Predicted Positions')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].legend()
        axes[1, 0].set_aspect('equal')
        
        # Error distribution
        errors = np.linalg.norm(data['test_positions'] - results['main']['predictions'], axis=1)
        axes[1, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Prediction Error Distribution')
        axes[1, 1].set_xlabel('Euclidean Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Temporal generalization plot (main/baseline mode)
        if results['temporal'] is not None:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(data['test_positions'][:, 0], data['test_positions'][:, 1], 
                       alpha=0.6, label='True', s=10)
            plt.scatter(results['temporal']['predictions'][:, 0], results['temporal']['predictions'][:, 1], 
                       alpha=0.6, label='Temporal Predicted', s=10)
            plt.title('Temporal Generalization: True vs Predicted')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.axis('equal')
            
            plt.subplot(1, 2, 2)
            temporal_errors = np.linalg.norm(data['test_positions'] - results['temporal']['predictions'], axis=1)
            plt.hist(temporal_errors, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Temporal Prediction Error Distribution')
            plt.xlabel('Euclidean Error')
            plt.ylabel('Frequency')
            plt.axvline(np.mean(temporal_errors), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(temporal_errors):.3f}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'temporal_generalization.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. PCA variance explained plot
        if self.pca is not None:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
            plt.title('PCA Cumulative Variance Explained')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Variance Explained')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            n_components = min(20, len(self.pca.explained_variance_ratio_))
            plt.bar(range(1, n_components + 1), 
                   self.pca.explained_variance_ratio_[:n_components])
            plt.title('PCA Individual Component Variance')
            plt.xlabel('Component')
            plt.ylabel('Variance Explained')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pca_analysis.png'), dpi=150, bbox_inches='tight')
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Robust Agent Location Probing')
    parser.add_argument('--layer', required=True, help='Layer to probe')
    parser.add_argument('--n-train-worlds', type=int, default=N_TRAIN_WORLDS, 
                       help='Number of worlds for training')
    parser.add_argument('--n-test-worlds', type=int, default=N_TEST_WORLDS, 
                       help='Number of worlds for testing')
    parser.add_argument('--n-rollouts', type=int, default=N_ROLLOUTS_PER_WORLD,
                       help='Number of rollouts per world')
    parser.add_argument('--single-world', action='store_true',
                       help='Train and test on different rollouts from the same world')
    parser.add_argument('--world-id', type=int, default=0,
                       help='World ID to use for single-world testing')
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS,
                       help='Maximum steps per rollout')
    parser.add_argument('--n-components', type=int, default=None,
                       help='Number of PCA components (None for 95% variance)')
    parser.add_argument('--output-dir', type=str, default='interpretability/probing/agent_location/results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=== Robust Agent Location Probing ===")
    print(f"Layer: {args.layer}")
    if args.single_world:
        print(f"Single world testing: World {args.world_id}")
        print(f"Rollouts per world: {args.n_rollouts}")
    else:
        print(f"Train worlds: {args.n_train_worlds}, Test worlds: {args.n_test_worlds}")
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
    probe = RobustAgentLocationProbe(model, args.layer, args.n_components)
    
    if args.single_world:
        # Single world testing: collect all data from one world, then split by rollouts
        print(f"\nCollecting data from world {args.world_id}...")
        all_data = probe.collect_data([args.world_id], args.n_rollouts, args.max_steps)
        
        # Split rollouts into train/test
        unique_rollouts = np.unique(all_data['rollout_ids'])
        n_rollouts = len(unique_rollouts)
        
        if n_rollouts < 2:
            print(f"Warning: Only {n_rollouts} successful rollouts. Proceeding with temporal generalization only.")
            # For single rollout, we'll only do temporal generalization
            train_data = all_data
            test_data = all_data  # Same data, but will be split by time
            single_rollout_mode = True
        else:
            single_rollout_mode = False
            
        # Ensure at least 1 rollout in each split
        train_rollout_indices = unique_rollouts[:max(1, n_rollouts // 2)]  # At least 1 for training
        test_rollout_indices = unique_rollouts[max(1, n_rollouts // 2):]  # Remaining for testing
        
        # Filter data by rollout indices
        train_mask = np.isin(all_data['rollout_ids'], train_rollout_indices)
        test_mask = np.isin(all_data['rollout_ids'], test_rollout_indices)
        
        train_data = {
            'activations': all_data['activations'][train_mask],
            'positions': all_data['positions'][train_mask],
            'step_ids': all_data['step_ids'][train_mask],
            'rollout_ids': all_data['rollout_ids'][train_mask]
        }
        test_data = {
            'activations': all_data['activations'][test_mask],
            'positions': all_data['positions'][test_mask],
            'step_ids': all_data['step_ids'][test_mask],
            'rollout_ids': all_data['rollout_ids'][test_mask]
        }
        
        print(f"Train rollouts: {train_rollout_indices}")
        print(f"Test rollouts: {test_rollout_indices}")
    else:
        # Cross-world testing: use different worlds for train/test
        all_worlds = list(range(20))  # Use first 20 worlds
        train_worlds = all_worlds[:args.n_train_worlds]
        test_worlds = all_worlds[args.n_train_worlds:args.n_train_worlds + args.n_test_worlds]
        
        print(f"Train worlds: {train_worlds}")
        print(f"Test worlds: {test_worlds}")
        
        # Collect training data
        print("\nCollecting training data...")
        train_data = probe.collect_data(train_worlds, args.n_rollouts, args.max_steps)
        
        # Collect test data
        print("\nCollecting test data...")
        test_data = probe.collect_data(test_worlds, args.n_rollouts, args.max_steps)
    
    if not single_rollout_mode:
        # Train/test split for main evaluation (cross-rollout)
        X_train = train_data['activations']
        Y_train = train_data['positions']
        X_test = test_data['activations']
        Y_test = test_data['positions']
        
        print(f"\nData summary:")
        print(f"Train: {len(X_train)} samples, {X_train.shape[1]} features")
        print(f"Test: {len(X_test)} samples, {X_test.shape[1]} features")
        if len(X_train) > 0:
            print(f"Feature-to-sample ratio: {X_train.shape[1]/len(X_train):.2f}")
        else:
            print("No training data available!")
            return
        
        # Train baseline probes
        print("\nTraining baseline probes...")
        baselines = probe.train_baseline_probes(X_train, Y_train, X_test, Y_test)
        
        # Train main probe
        print("\nTraining main probe...")
        main_results = probe.train_probe(X_train, Y_train, X_test, Y_test)
    else:
        # Single rollout mode - skip cross-rollout evaluation
        print(f"\nSingle rollout mode - skipping cross-rollout evaluation")
        print(f"Total samples: {len(train_data['activations'])}")
        baselines = {}
        main_results = {}
    
    # Evaluate temporal generalization
    print("\nEvaluating temporal generalization...")
    if single_rollout_mode:
        # Use the single rollout data directly
        temporal_data = train_data
    else:
        # Combine train and test data
        temporal_data = {
            'activations': np.concatenate([train_data['activations'], test_data['activations']]),
            'positions': np.concatenate([train_data['positions'], test_data['positions']]),
            'step_ids': np.concatenate([train_data['step_ids'], test_data['step_ids']])
        }
    temporal_results = probe.evaluate_temporal_generalization(temporal_data)
    
    # Compile results
    if single_rollout_mode:
        results = {
            'main': {},
            'baselines': {},
            'temporal': temporal_results,
            'data_info': {
                'total_samples': len(temporal_data['activations']),
                'feature_dim': temporal_data['activations'].shape[1],
                'mode': 'single_rollout_temporal'
            }
        }
    else:
        results = {
            'main': main_results,
            'baselines': baselines,
            'temporal': temporal_results,
            'data_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_dim': X_train.shape[1],
                'reduced_dim': main_results['n_components']
            }
        }
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if single_rollout_mode:
        print(f"\nSingle Rollout Mode - Temporal Generalization Only:")
        if temporal_results is not None:
            print(f"  R² Score: {temporal_results['r2']:.4f}")
            print(f"  MSE: {temporal_results['mse']:.6f}")
            print(f"  MAE: {temporal_results['mae']:.4f}")
            print(f"  Train samples: {temporal_results['n_train']}")
            print(f"  Test samples: {temporal_results['n_test']}")
            if 'train_r2' in temporal_results:
                print(f"  Train R²: {temporal_results['train_r2']:.4f}")
                print(f"  Train MSE: {temporal_results['train_mse']:.6f}")
                print(f"  Train MAE: {temporal_results['train_mae']:.4f}")
        else:
            print("  No temporal results available")
    else:
        print(f"\nMain Probe (PCA + Ridge):")
        print(f"  R² Score: {main_results['r2']:.4f}")
        print(f"  MSE: {main_results['mse']:.6f}")
        print(f"  MAE: {main_results['mae']:.4f}")
        print(f"  Components: {main_results['n_components']}")
        
        print(f"\nBaseline Comparisons:")
        for name, metrics in baselines.items():
            print(f"  {name}: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.6f}")
        
        if temporal_results is not None:
            print(f"\nTemporal Generalization:")
            print(f"  R² Score: {temporal_results['r2']:.4f}")
            print(f"  MSE: {temporal_results['mse']:.6f}")
            print(f"  MAE: {temporal_results['mae']:.4f}")
            print(f"  Train samples: {temporal_results['n_train']}")
            print(f"  Test samples: {temporal_results['n_test']}")
            if 'train_r2' in temporal_results:
                print(f"  Train R²: {temporal_results['train_r2']:.4f}")
                print(f"  Train MSE: {temporal_results['train_mse']:.6f}")
                print(f"  Train MAE: {temporal_results['train_mae']:.4f}")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    if single_rollout_mode:
        # For single rollout, use temporal test data
        if temporal_results is not None:
            probe.create_visualizations(results, {
                'test_positions': temporal_data['positions'][temporal_data['step_ids'] >= 100][:50],  # Test steps
                'test_activations': temporal_data['activations'][temporal_data['step_ids'] >= 100][:50]
            }, args.output_dir)
        else:
            print("No temporal results available for visualization")
    else:
        probe.create_visualizations(results, {
            'test_positions': Y_test,
            'test_activations': X_test
        }, args.output_dir)
    
    # Save results
    if single_rollout_mode:
        # Only save temporal generalization results
        if temporal_results is not None:
            results_df = pd.DataFrame([{
                'method': 'temporal_generalization',
                'r2': temporal_results['r2'],
                'mse': temporal_results['mse'],
                'mae': temporal_results['mae'],
                'train_r2': temporal_results['train_r2'],
                'train_mse': temporal_results['train_mse'],
                'train_mae': temporal_results['train_mae'],
                'n_train': temporal_results['n_train'],
                'n_test': temporal_results['n_test']
            }])
            results_df.to_csv(os.path.join(args.output_dir, 'results_summary.csv'), index=False)
            print(f"\nResults saved to: {args.output_dir}")
            print("="*60)
        else:
            print("No temporal results to save.")
    else:
        results_df = pd.DataFrame([
            {
                'method': 'main_probe',
                'r2': main_results['r2'],
                'mse': main_results['mse'],
                'mae': main_results['mae'],
                'n_components': main_results['n_components']
            }
        ] + [
            {
                'method': name,
                'r2': metrics['r2'],
                'mse': metrics['mse'],
                'mae': np.nan,
                'n_components': np.nan
            }
            for name, metrics in baselines.items()
        ])
        if temporal_results is not None:
            temporal_df = pd.DataFrame([{
                'method': 'temporal_generalization',
                'r2': temporal_results['r2'],
                'mse': temporal_results['mse'],
                'mae': temporal_results['mae'],
                'n_components': np.nan
            }])
            results_df = pd.concat([results_df, temporal_df], ignore_index=True)
        results_df.to_csv(os.path.join(args.output_dir, 'results_summary.csv'), index=False)
        print(f"\nResults saved to: {args.output_dir}")
        print("="*60)

if __name__ == '__main__':
    main() 