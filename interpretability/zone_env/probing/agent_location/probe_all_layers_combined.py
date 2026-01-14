#!/usr/bin/env python3
"""
Combined Layer Probing for Agent Location

This script trains a single probe that uses activations from ALL layers
simultaneously as input features to predict agent location.

Usage: python interpretability/probing/agent_location/probe_all_layers_combined.py --world-id 0 --n-rollouts 1 --max-steps 200
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
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

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

def collect_combined_data(model, layer_names, world_ids, n_rollouts_per_world=1, max_steps=200):
    """Collect activations from all layers simultaneously across multiple worlds."""
    env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                 propositions=props, verbose=False)
    
    # Get modules to probe
    modules = {name: dict(model.named_modules())[name] for name in layer_names}
    
    all_activations = []
    all_positions = []
    all_step_ids = []
    all_world_ids = []
    
    for world_id in world_ids:
        world_file = f"{WORLD_DIR}/world_info_{world_id}.pkl"
        if not os.path.exists(world_file):
            print(f"World file not found: {world_file}, skipping.")
            continue
        
        env.load_world_info(world_file)
        
        for rollout_idx in range(n_rollouts_per_world):
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
            
            # Collect activations from all layers
            layer_activations = {name: [] for name in layer_names}
            positions = []
            
            # Register hooks for all layers
            hooks = []
            for name, module in modules.items():
                def grab_activation(layer_name):
                    def hook(m, inp, out):
                        x = out[1] if isinstance(out, tuple) else out
                        layer_activations[layer_name].append(x.detach().cpu().numpy().ravel())
                    return hook
                
                hook = module.register_forward_hook(grab_activation(name))
                hooks.append(hook)
            
            done = False
            for step in range(max_steps):
                if done:
                    break
                
                # Record position BEFORE taking action
                positions.append(env.agent_pos[:2].copy())
                
                # Take action and record activations from all layers
                action = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, _ = env.step(action)
                
                # Ensure all layers have the same number of activations
                min_len = min(len(acts) for acts in layer_activations.values())
                for name in layer_names:
                    if len(layer_activations[name]) < min_len:
                        layer_activations[name].append(layer_activations[name][-1] if layer_activations[name] else np.zeros(layer_activations[name][0].shape))
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Align activations and positions
            min_acts = min(len(acts) for acts in layer_activations.values())
            if min_acts > len(positions):
                min_acts = len(positions)
            
            # Truncate all to same length
            for name in layer_names:
                layer_activations[name] = layer_activations[name][:min_acts]
            positions = positions[:min_acts]
            
            # Only use steps where all layers have valid activations and correct shape
            n_skipped = 0
            expected_shapes = {name: layer_activations[name][0].shape for name in layer_names if len(layer_activations[name]) > 0}
            for step_idx in range(min_acts):
                valid = True
                for name in layer_names:
                    act = layer_activations[name][step_idx]
                    if act is None or not isinstance(act, np.ndarray) or act.shape != expected_shapes[name]:
                        valid = False
                        break
                if valid:
                    combined_activation = np.concatenate([layer_activations[name][step_idx] for name in layer_names])
                    all_activations.append(combined_activation)
                    all_positions.append(positions[step_idx])
                    all_step_ids.append(step_idx)
                    all_world_ids.append(world_id)
                else:
                    n_skipped += 1
            if n_skipped > 0:
                print(f"Warning: Skipped {n_skipped} steps in world {world_id}, rollout {rollout_idx} due to missing or mismatched activations.")
    
    env.close()
    
    if len(all_activations) == 0:
        return None
    
    all_activations = np.array(all_activations, dtype=float)
    all_positions = np.array(all_positions, dtype=float)
    all_step_ids = np.array(all_step_ids, dtype=int)
    all_world_ids = np.array(all_world_ids, dtype=int)
    
    return {
        'activations': all_activations,
        'positions': all_positions,
        'step_ids': all_step_ids,
        'world_ids': all_world_ids,
        'layer_names': layer_names,
        'layer_dims': {name: layer_activations[name][0].shape[0] if layer_activations[name] else 0 for name in layer_names}
    }

def evaluate_temporal_generalization(data, train_steps=100, test_steps=50, n_components=None):
    """Evaluate temporal generalization using combined layer activations."""
    # Filter data by step
    train_mask = data['step_ids'] < train_steps
    test_mask = (data['step_ids'] >= train_steps) & (data['step_ids'] < train_steps + test_steps)
    
    if not np.any(train_mask) or not np.any(test_mask):
        return None
    
    X_train = data['activations'][train_mask]
    Y_train = data['positions'][train_mask]
    X_test = data['activations'][test_mask]
    Y_test = data['positions'][test_mask]
    
    print(f"Combined feature dimension: {X_train.shape[1]}")
    print(f"Layer contributions:")
    start_idx = 0
    for name in data['layer_names']:
        layer_dim = data['layer_dims'][name]
        print(f"  {name}: {layer_dim} features (indices {start_idx}-{start_idx + layer_dim - 1})")
        start_idx += layer_dim
    
    # Apply dimensionality reduction
    if n_components is None:
        pca = PCA(n_components=0.95)
    else:
        pca = PCA(n_components=n_components)
    
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    
    print(f"Reduced to {X_train_reduced.shape[1]} components (variance explained: {pca.explained_variance_ratio_.sum():.3f})")
    
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
        'feature_dim': X_train.shape[1],
        'pca': pca,
        'probe': probe
    }

def main():
    parser = argparse.ArgumentParser(description='Probe all layers combined for agent location')
    parser.add_argument('--world-ids', type=str, default='0-10', 
                       help='World IDs to use (e.g., "0-10" for range, "0,1,2" for specific worlds)')
    parser.add_argument('--n-rollouts-per-world', type=int, default=1, help='Number of rollouts per world')
    parser.add_argument('--max-steps', type=int, default=200, help='Maximum steps per rollout')
    parser.add_argument('--n-components', type=int, default=None, help='Number of PCA components')
    parser.add_argument('--output-dir', type=str, default='interpretability/probing/agent_location/combined_layers_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Parse world IDs
    if '-' in args.world_ids:
        # Range format: "0-10"
        start, end = map(int, args.world_ids.split('-'))
        world_ids = list(range(start, end + 1))
    else:
        # List format: "0,1,2"
        world_ids = [int(x.strip()) for x in args.world_ids.split(',')]
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=== Combined Layer Probing for Agent Location ===")
    print(f"World IDs: {world_ids}")
    print(f"Rollouts per world: {args.n_rollouts_per_world}")
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
    layer_names = get_all_layers(model)
    print(f"\nFound {len(layer_names)} layers to combine:")
    for i, layer in enumerate(layer_names):
        print(f"  {i+1:2d}. {layer}")
    
    # Collect combined data
    print(f"\nCollecting data from all layers across {len(world_ids)} worlds...")
    data = collect_combined_data(model, layer_names, world_ids, args.n_rollouts_per_world, args.max_steps)
    
    if data is None:
        print("No data collected")
        return
    
    print(f"Collected {len(data['activations'])} samples")
    
    # Evaluate temporal generalization
    print(f"\nEvaluating temporal generalization...")
    eval_results = evaluate_temporal_generalization(data, n_components=args.n_components)
    
    if eval_results is None:
        print("Insufficient data for evaluation")
        return
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create results DataFrame
    results_df = pd.DataFrame([{
        'method': 'combined_layers',
        'train_r2': eval_results['train_r2'],
        'train_mse': eval_results['train_mse'],
        'train_mae': eval_results['train_mae'],
        'test_r2': eval_results['test_r2'],
        'test_mse': eval_results['test_mse'],
        'test_mae': eval_results['test_mae'],
        'n_components': eval_results['n_components'],
        'feature_dim': eval_results['feature_dim'],
        'n_layers': len(layer_names),
        'n_worlds': len(world_ids)
    }])
    
    results_df.to_csv(os.path.join(args.output_dir, 'combined_layers_results.csv'), index=False)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Train vs Test performance
    axes[0, 0].bar(['Train', 'Test'], [eval_results['train_r2'], eval_results['test_r2']], alpha=0.7)
    axes[0, 0].set_title('Train vs Test R²')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE comparison
    axes[0, 1].bar(['Train', 'Test'], [eval_results['train_mse'], eval_results['test_mse']], alpha=0.7)
    axes[0, 1].set_title('Train vs Test MSE')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA variance explained
    if eval_results['pca'] is not None:
        axes[1, 0].plot(np.cumsum(eval_results['pca'].explained_variance_ratio_))
        axes[1, 0].set_title('PCA Cumulative Variance Explained')
        axes[1, 0].set_xlabel('Number of Components')
        axes[1, 0].set_ylabel('Cumulative Variance Explained')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Layer dimensions
    layer_dims = list(data['layer_dims'].values())
    layer_names_short = [name.split('.')[-1] for name in data['layer_names']]
    axes[1, 1].bar(range(len(layer_dims)), layer_dims, alpha=0.7)
    axes[1, 1].set_title('Feature Dimensions by Layer')
    axes[1, 1].set_ylabel('Feature Dimension')
    axes[1, 1].set_xticks(range(len(layer_dims)))
    axes[1, 1].set_xticklabels(layer_names_short, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'combined_layers_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n" + "="*60)
    print("COMBINED LAYERS PROBING SUMMARY")
    print("="*60)
    print(f"Number of layers combined: {len(layer_names)}")
    print(f"Number of worlds: {len(world_ids)}")
    print(f"Total feature dimension: {eval_results['feature_dim']}")
    print(f"Reduced to {eval_results['n_components']} components")
    print(f"Train R²: {eval_results['train_r2']:.4f}")
    print(f"Test R²: {eval_results['test_r2']:.4f}")
    print(f"Train MSE: {eval_results['train_mse']:.6f}")
    print(f"Test MSE: {eval_results['test_mse']:.6f}")
    print(f"Train MAE: {eval_results['train_mae']:.4f}")
    print(f"Test MAE: {eval_results['test_mae']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*60)

if __name__ == '__main__':
    main() 