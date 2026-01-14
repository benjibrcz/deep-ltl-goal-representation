#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..", "src")))

from utils.model_store    import ModelStore
from model.model          import build_model
from config               import model_configs
from ltl                  import FixedSampler
from envs                 import make_env
from sequence.search      import ExhaustiveSearch
from model.agent          import Agent

# ─── defaults ─────────────────────────────────────────────────────────────────
ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
N_ROLLOUTS = 10  # Number of different starting positions per world
WORLD_ID   = -1   # Which world to use (or -1 for multiple worlds)
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

def collect_activations_and_features(model, layer_name, sampler, n_rollouts=10, max_steps=200, world_ids=None):
    """
    Collect activations and corresponding input features for analysis.
    """
    if world_ids is None:
        # Use worlds 0-9 by default
        world_ids = list(range(10))
    
    all_activations = []
    all_features = []
    all_feature_components: dict[str, list] = {
        'agent_sensors': [],      # First 3 values: accelerometer, velocimeter, gyro
        'zone_lidar': [],         # Next 16 values: lidar readings for zones
        'wall_lidar': [],         # Next 16 values: lidar readings for walls  
        'wall_sensor': [],        # Next 4 values: wall sensor
        'joint_positions': [],    # Remaining values: joint positions/velocities
        'agent_pos': [],          # Agent position (not in features, but available)
        'zone_distances': [],     # Distances to zones
        'zone_directions': [],    # Directions to zones
        'propositions': []        # Active propositions
    }
    
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    total_successful_rollouts = 0
    
    for world_id in world_ids:
        # Skip world file loading to allow proper seeding/randomization
        # world_dir_path = f"{WORLD_DIR}"
        # world_file = f"{world_dir_path}/world_info_{world_id}.pkl"
        # if not os.path.exists(world_file):
        #     print(f"World file not found: {world_file}, skipping.")
        #     continue
        # env.load_world_info(world_file)
        print(f"Processing world_id {world_id} with random world generation")
        zone_pos = dict(env.zone_positions) if hasattr(env, 'zone_positions') else {}

        successful_rollouts = 0
        max_attempts_per_rollout = 10
        
        for rollout_idx in trange(n_rollouts, desc=f"Rollouts for world {world_id}"):
            # Try different seeds until we find a valid starting position
            for attempt in range(max_attempts_per_rollout):
                try:
                    # Reset with different seed to get different starting position
                    obs = env.reset(seed=SEED + world_id * 1000 + rollout_idx * max_attempts_per_rollout + attempt)
                    break
                except AssertionError as e:
                    if "World has starting cost" in str(e):
                        if attempt == max_attempts_per_rollout - 1:
                            print(f"  Skipping rollout {rollout_idx} after {max_attempts_per_rollout} failed attempts")
                            continue
                        else:
                            continue
                    else:
                        raise e
            
            # If we couldn't find a valid starting position, skip this rollout
            if attempt == max_attempts_per_rollout - 1:
                continue
                
            agent.reset()

            feats = []
            def grab(m, inp, out):
                x = out[1] if isinstance(out, tuple) else out
                feats.append(x.detach().cpu().numpy().ravel())
            h = module.register_forward_hook(grab)
            
            done = False
            for step in range(max_steps):
                if done:
                    break
                
                # Store current observation and agent position BEFORE taking action
                current_obs = obs.copy() if isinstance(obs, dict) else obs
                agent_pos = env.agent_pos[:2].copy()
                
                # Take action and record activation
                a = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, _ = env.step(a)
                
                # Store activation and corresponding features from the SAME time step
                if len(feats) > 0:
                    activation = feats[-1]
                    features = current_obs['features']  # Features from step t (before action)
                    
                    all_activations.append(activation)
                    all_features.append(features)
                    
                    # Break down features into components (all from step t)
                    all_feature_components['agent_sensors'].append(features[0:3])
                    all_feature_components['zone_lidar'].append(features[3:19])
                    all_feature_components['wall_lidar'].append(features[19:35])
                    all_feature_components['wall_sensor'].append(features[35:39])
                    all_feature_components['joint_positions'].append(features[39:])
                    all_feature_components['agent_pos'].append(agent_pos)  # Position from step t
                    
                    # Calculate zone distances and directions (from step t)
                    zone_dists = []
                    zone_dirs = []
                    if isinstance(zone_pos, dict):
                        for zone_name, zpos in zone_pos.items():
                            dist = np.linalg.norm(agent_pos - zpos[:2])
                            direction = (zpos[:2] - agent_pos) / (dist + 1e-8)
                            zone_dists.append(dist)
                            zone_dirs.extend(direction)
                    all_feature_components['zone_distances'].append(zone_dists)
                    all_feature_components['zone_directions'].append(zone_dirs)
                    all_feature_components['propositions'].append(list(current_obs.get('propositions', [])))
            
            h.remove()
            successful_rollouts += 1
            total_successful_rollouts += 1
        
        print(f"World {world_id}: {successful_rollouts}/{n_rollouts} successful rollouts")
    
    env.close()
    print(f"Total successfully collected data from {total_successful_rollouts} rollouts across {len(world_ids)} worlds")
    
    # Convert to arrays
    feature_components_arrays: dict[str, np.ndarray] = {}
    for key in all_feature_components:
        if len(all_feature_components[key]) > 0:
            feature_components_arrays[key] = np.array(all_feature_components[key])
        else:
            feature_components_arrays[key] = np.array([])
    
    return np.array(all_activations), np.array(all_features), feature_components_arrays

def train_probes_for_components(activations, feature_components, test_size=0.2):
    """
    Train probes for each component of the input features.
    """
    results = {}
    
    # Split data
    n_samples = len(activations)
    indices = np.arange(n_samples)
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=SEED)
    
    X_train = activations[train_indices]
    X_test = activations[test_indices]
    
    print(f"\nTraining probes for {len(feature_components)} feature components...")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    for component_name, component_data in feature_components.items():
        if len(component_data) == 0:
            print(f"  {component_name}: No data available")
            continue
        
        # Skip non-numeric components (like propositions)
        if component_name == 'propositions':
            print(f"  {component_name}: Skipping non-numeric component")
            continue
            
        if component_data.ndim == 1:
            # Single value per sample
            Y_train = component_data[train_indices]
            Y_test = component_data[test_indices]
        else:
            # Multiple values per sample
            Y_train = component_data[train_indices]
            Y_test = component_data[test_indices]
        
        # Train probe
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, Y_train)
        
        # Evaluate
        Y_pred_train = probe.predict(X_train)
        Y_pred_test = probe.predict(X_test)
        
        # Calculate metrics
        if Y_train.ndim == 1:
            mse_train = mean_squared_error(Y_train, Y_pred_train)
            r2_train = r2_score(Y_train, Y_pred_train)
            mse_test = mean_squared_error(Y_test, Y_pred_test)
            r2_test = r2_score(Y_test, Y_pred_test)
        else:
            # For multi-dimensional targets, calculate per-dimension then average
            mse_train = np.mean([mean_squared_error(Y_train[:, i], Y_pred_train[:, i]) 
                                for i in range(Y_train.shape[1])])
            r2_train = np.mean([r2_score(Y_train[:, i], Y_pred_train[:, i]) 
                               for i in range(Y_train.shape[1])])
            mse_test = np.mean([mean_squared_error(Y_test[:, i], Y_pred_test[:, i]) 
                               for i in range(Y_test.shape[1])])
            r2_test = np.mean([r2_score(Y_test[:, i], Y_pred_test[:, i]) 
                              for i in range(Y_test.shape[1])])
        
        results[component_name] = {
            'mse_train': mse_train,
            'r2_train': r2_train,
            'mse_test': mse_test,
            'r2_test': r2_test,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'feature_dim': X_train.shape[1],
            'target_dim': Y_train.shape[1] if Y_train.ndim > 1 else 1,
            'probe': probe
        }
        
        print(f"  {component_name}: R²={r2_test:.3f}, MSE={mse_test:.4f}, "
              f"dim={Y_train.shape[1] if Y_train.ndim > 1 else 1}")
    
    return results

def create_component_analysis_plots(results, save_path=None):
    """
    Create plots comparing probe performance across different input components.
    """
    if not results:
        print("No results to plot")
        return
    
    component_names = list(results.keys())
    r2_scores = [results[name]['r2_test'] for name in component_names]
    mse_scores = [results[name]['mse_test'] for name in component_names]
    target_dims = [results[name]['target_dim'] for name in component_names]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # R² scores
    bars1 = ax1.bar(range(len(component_names)), r2_scores, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Feature Component')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Probe Performance by Input Component (R²)')
    ax1.set_xticks(range(len(component_names)))
    ax1.set_xticklabels(component_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{score:.3f}', 
                ha='center', va='bottom')
    
    # MSE scores
    bars2 = ax2.bar(range(len(component_names)), mse_scores, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Feature Component')
    ax2.set_ylabel('MSE Score')
    ax2.set_title('Probe Performance by Input Component (MSE)')
    ax2.set_xticks(range(len(component_names)))
    ax2.set_xticklabels(component_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    for bar, score in zip(bars2, mse_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{score:.4f}', 
                ha='center', va='bottom')
    
    # Target dimensions
    bars3 = ax3.bar(range(len(component_names)), target_dims, color='lightgreen', alpha=0.7)
    ax3.set_xlabel('Feature Component')
    ax3.set_ylabel('Target Dimension')
    ax3.set_title('Target Dimensions by Component')
    ax3.set_xticks(range(len(component_names)))
    ax3.set_xticklabels(component_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    for bar, dim in zip(bars3, target_dims):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{dim}', 
                ha='center', va='bottom')
    
    # R² vs MSE scatter
    ax4.scatter(mse_scores, r2_scores, s=100, alpha=0.7)
    for i, name in enumerate(component_names):
        ax4.annotate(name, (mse_scores[i], r2_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax4.set_xlabel('MSE Score')
    ax4.set_ylabel('R² Score')
    ax4.set_title('R² vs MSE for Each Component')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved component analysis plot to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True)
    p.add_argument('--n-rollouts',  type=int, default=N_ROLLOUTS)
    p.add_argument('--max-steps',   type=int, default=200)
    p.add_argument('--world-id',    type=int, default=WORLD_ID, help='World ID to use (or -1 for multiple worlds)')
    p.add_argument('--out',         type=str)
    args = p.parse_args()

    # seeds & sampler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = "FG blue"
    sampler = FixedSampler.partial(formula)

    # ── load model ───────────────────────────────────────────────────────────────
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[ENV]
    dummy  = make_env(ENV, sampler, sequence=False, render_mode=None)
    model  = build_model(dummy, status, cfg).eval()
    dummy.close()

    # ── collect data ─────────────────────────────────────────────────────────────
    if args.world_id == -1:
        world_ids = list(range(10))
        print(f"Using multiple worlds: {world_ids}")
    else:
        world_ids = [args.world_id]
        print(f"Using single world: {args.world_id}")
    
    activations, features, feature_components = collect_activations_and_features(
        model, args.layer, sampler, n_rollouts=args.n_rollouts, max_steps=args.max_steps, world_ids=world_ids)
    
    if len(activations) == 0:
        print("Could not collect data for probe training. Exiting.")
        return
    
    print(f"Collected {len(activations)} samples")
    print(f"Activation shape: {activations.shape}")
    print(f"Feature shape: {features.shape}")
    
    # Print feature component shapes
    print("\nFeature component shapes:")
    for name, data in feature_components.items():
        if len(data) > 0:
            print(f"  {name}: {data.shape}")
    
    # ── train probes for each component ─────────────────────────────────────────
    results = train_probes_for_components(activations, feature_components)
    
    # ── create summary plots ───────────────────────────────────────────────────
    if results:
        plot_file = args.out or f'input_component_probes_{args.layer}_world{args.world_id}.png'
        create_component_analysis_plots(results, save_path=plot_file)
        
        # Save results to CSV
        df = pd.DataFrame([
            {
                'component': name,
                'r2_train': results[name]['r2_train'],
                'r2_test': results[name]['r2_test'],
                'mse_train': results[name]['mse_train'],
                'mse_test': results[name]['mse_test'],
                'n_train_samples': results[name]['n_train_samples'],
                'n_test_samples': results[name]['n_test_samples'],
                'feature_dim': results[name]['feature_dim'],
                'target_dim': results[name]['target_dim']
            }
            for name in sorted(results.keys())
        ])
        csv_file = f'input_component_probes_{args.layer}_world{args.world_id}.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved results to {csv_file}")
        
        # Print summary statistics
        r2_scores = [results[key]['r2_test'] for key in results.keys()]
        mse_scores = [results[key]['mse_test'] for key in results.keys()]
        print(f"\nSummary Statistics:")
        print(f"  Total components analyzed: {len(results)}")
        print(f"  Average R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
        print(f"  Average MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
        print(f"  Best R²: {np.max(r2_scores):.3f} ({list(results.keys())[np.argmax(r2_scores)]})")
        print(f"  Worst R²: {np.min(r2_scores):.3f} ({list(results.keys())[np.argmin(r2_scores)]})")
        
        # Find best performing components
        best_components = sorted(results.items(), key=lambda x: x[1]['r2_test'], reverse=True)
        print(f"\nTop 3 best performing components:")
        for i, (name, result) in enumerate(best_components[:3]):
            print(f"  {i+1}. {name}: R²={result['r2_test']:.3f}, MSE={result['mse_test']:.4f}")
    else:
        print("No probes could be trained (insufficient data).")

if __name__ == '__main__':
    main() 