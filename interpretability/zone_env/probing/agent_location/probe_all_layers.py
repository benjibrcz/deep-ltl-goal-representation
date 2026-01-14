#!/usr/bin/env python3
"""
Systematic Layer Probing for Agent Location

This script probes all available layers in the model to find which ones
encode spatial information about the agent's position.

Usage: python interpretability/probing/agent_location/probe_all_layers.py --world-id 0 --n-rollouts 1 --max-steps 200
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..", "src")))

from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from ltl import FixedSampler
from envs import make_env
from sequence.search import ExhaustiveSearch
from model.agent import Agent

# Constants
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0
WORLD_DIR = f"eval_datasets/{ENV}/worlds"

def get_all_layers(model):
    """Get all named layers in the model that can be probed."""
    layers = []
    for name, module in model.named_modules():
        # Skip very small layers and non-parametric layers
        if hasattr(module, 'weight') and module.weight.numel() > 100:
            layers.append(name)
    return sorted(layers)

def collect_data_for_layer(model, layer_name, world_id, n_rollouts=1, max_steps=200):
    """Collect activations and positions for a specific layer."""
    env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                 propositions=props, verbose=False)
    
    # Get the module to probe
    module = dict(model.named_modules())[layer_name]
    
    world_file = f"{WORLD_DIR}/world_info_{world_id}.pkl"
    if not os.path.exists(world_file):
        print(f"World file not found: {world_file}")
        env.close()
        return None
    
    env.load_world_info(world_file)
    
    all_activations = []
    all_positions = []
    all_step_ids = []
    
    for rollout_idx in range(n_rollouts):
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
        
        hook = module.register_forward_hook(grab_activation)
        
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
            all_step_ids.extend(list(range(len(positions))))
    
    env.close()
    
    if len(all_activations) == 0:
        return None
    
    return {
        'activations': np.array(all_activations),
        'positions': np.array(all_positions),
        'step_ids': np.array(all_step_ids)
    }

def evaluate_temporal_generalization(data, train_steps=100, test_steps=50, n_components=None):
    """Evaluate temporal generalization for a layer."""
    # Filter data by step
    train_mask = data['step_ids'] < train_steps
    test_mask = (data['step_ids'] >= train_steps) & (data['step_ids'] < train_steps + test_steps)
    
    if not np.any(train_mask) or not np.any(test_mask):
        return None
    
    X_train = data['activations'][train_mask]
    Y_train = data['positions'][train_mask]
    X_test = data['activations'][test_mask]
    Y_test = data['positions'][test_mask]
    
    # Apply dimensionality reduction
    if n_components is None:
        pca = PCA(n_components=0.95)
    else:
        pca = PCA(n_components=n_components)
    
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    
    # Train probe
    probe = Ridge(alpha=1.0)
    probe.fit(X_train_reduced, Y_train)
    
    # Evaluate train performance
    Y_pred_train = probe.predict(X_train_reduced)
    train_mse = mean_squared_error(Y_train, Y_pred_train)
    train_r2 = r2_score(Y_train, Y_pred_train)
    train_mae = mean_absolute_error(Y_train, Y_pred_train)
    
    # Evaluate test performance
    Y_pred_test = probe.predict(X_test_reduced)
    test_mse = mean_squared_error(Y_test, Y_pred_test)
    test_r2 = r2_score(Y_test, Y_pred_test)
    test_mae = mean_absolute_error(Y_test, Y_pred_test)
    
    return {
        'train_r2': train_r2,
        'train_mse': train_mse,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'n_components': X_train_reduced.shape[1],
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_dim': X_train.shape[1]
    }

def main():
    parser = argparse.ArgumentParser(description='Probe all layers for agent location')
    parser.add_argument('--world-id', type=int, default=0, help='World ID to use')
    parser.add_argument('--n-rollouts', type=int, default=1, help='Number of rollouts per layer')
    parser.add_argument('--max-steps', type=int, default=200, help='Maximum steps per rollout')
    parser.add_argument('--n-components', type=int, default=None, help='Number of PCA components')
    parser.add_argument('--output-dir', type=str, default='interpretability/probing/agent_location/all_layers_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=== Systematic Layer Probing for Agent Location ===")
    print(f"World ID: {args.world_id}")
    print(f"Rollouts per layer: {args.n_rollouts}")
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
    
    # Get all layers
    layers = get_all_layers(model)
    print(f"\nFound {len(layers)} layers to probe:")
    for i, layer in enumerate(layers):
        print(f"  {i+1:2d}. {layer}")
    
    # Probe each layer
    results = []
    
    for layer_name in tqdm(layers, desc="Probing layers"):
        print(f"\nProbing layer: {layer_name}")
        
        # Collect data
        data = collect_data_for_layer(model, layer_name, args.world_id, args.n_rollouts, args.max_steps)
        
        if data is None:
            print(f"  No data collected for {layer_name}")
            continue
        
        print(f"  Collected {len(data['activations'])} samples")
        
        # Evaluate temporal generalization
        eval_results = evaluate_temporal_generalization(data, n_components=args.n_components)
        
        if eval_results is None:
            print(f"  Insufficient data for evaluation")
            continue
        
        # Store results
        layer_result = {
            'layer': layer_name,
            **eval_results
        }
        results.append(layer_result)
        
        print(f"  Train R²: {eval_results['train_r2']:.4f}, Test R²: {eval_results['test_r2']:.4f}")
    
    # Create results DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        df.to_csv(os.path.join(args.output_dir, 'layer_probing_results.csv'), index=False)
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Train R² by layer
        axes[0, 0].bar(range(len(df)), df['train_r2'], alpha=0.7)
        axes[0, 0].set_title('Train R² by Layer')
        axes[0, 0].set_ylabel('Train R²')
        axes[0, 0].set_xticks(range(len(df)))
        axes[0, 0].set_xticklabels([layer.split('.')[-1] for layer in df['layer']], rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test R² by layer
        axes[0, 1].bar(range(len(df)), df['test_r2'], alpha=0.7)
        axes[0, 1].set_title('Test R² by Layer')
        axes[0, 1].set_ylabel('Test R²')
        axes[0, 1].set_xticks(range(len(df)))
        axes[0, 1].set_xticklabels([layer.split('.')[-1] for layer in df['layer']], rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Train vs Test R² scatter
        axes[1, 0].scatter(df['train_r2'], df['test_r2'], alpha=0.7)
        axes[1, 0].set_xlabel('Train R²')
        axes[1, 0].set_ylabel('Test R²')
        axes[1, 0].set_title('Train vs Test R²')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature dimensions
        axes[1, 1].bar(range(len(df)), df['feature_dim'], alpha=0.7)
        axes[1, 1].set_title('Feature Dimensions by Layer')
        axes[1, 1].set_ylabel('Feature Dimension')
        axes[1, 1].set_xticks(range(len(df)))
        axes[1, 1].set_xticklabels([layer.split('.')[-1] for layer in df['layer']], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'layer_probing_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print(f"\n" + "="*60)
        print("LAYER PROBING SUMMARY")
        print("="*60)
        
        # Sort by test R²
        df_sorted = df.sort_values('test_r2', ascending=False)
        
        print(f"\nTop 5 layers by Test R²:")
        for i, (_, row) in enumerate(df_sorted.head().iterrows()):
            print(f"  {i+1}. {row['layer']}: Test R²={row['test_r2']:.4f}, Train R²={row['train_r2']:.4f}")
        
        print(f"\nBottom 5 layers by Test R²:")
        for i, (_, row) in enumerate(df_sorted.tail().iterrows()):
            print(f"  {i+1}. {row['layer']}: Test R²={row['test_r2']:.4f}, Train R²={row['train_r2']:.4f}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*60)
        
    else:
        print("No results obtained from any layer.")

if __name__ == '__main__':
    main() 