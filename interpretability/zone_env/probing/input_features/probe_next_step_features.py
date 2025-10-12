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

def collect_activations_and_next_features(model, layer_name, sampler, n_rollouts=10, max_steps=200, world_ids=None):
    """
    Collect activations from step t and corresponding features from step t+1.
    This probes whether the model can predict future observations.
    """
    if world_ids is None:
        # Use worlds 0-9 by default
        world_ids = list(range(10))
    
    # Store data separately by rollout to avoid temporal leakage
    rollout_data = []  # List of (activations, next_features, next_feature_components) for each rollout
    all_next_feature_components = {
        'next_agent_sensors': [],      # Next step: accelerometer, velocimeter, gyro
        'next_zone_lidar': [],         # Next step: lidar readings for zones
        'next_wall_lidar': [],         # Next step: lidar readings for walls  
        'next_wall_sensor': [],        # Next step: wall sensor
        'next_joint_positions': [],    # Next step: joint positions/velocities
        'next_agent_pos': [],          # Next step: agent position
        'next_zone_distances': [],     # Next step: distances to zones
        'next_zone_directions': [],    # Next step: directions to zones
        'next_propositions': []        # Next step: active propositions
    }
    
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    total_successful_rollouts = 0
    total_next_step_pairs = 0
    
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
            rollout_pairs = 0
            
            # Store this rollout's data separately
            rollout_activations = []
            rollout_next_features = []
            rollout_next_components = {
                'next_agent_sensors': [],
                'next_zone_lidar': [],
                'next_wall_lidar': [],
                'next_wall_sensor': [],
                'next_joint_positions': [],
                'next_agent_pos': [],
                'next_zone_distances': [],
                'next_zone_directions': [],
                'next_propositions': []
            }
            
            for step in range(max_steps):
                if done:
                    break
                
                # Store current agent position (step t)
                current_agent_pos = env.agent_pos[:2].copy()
                
                # Take action and record activation from step t
                a = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, _ = env.step(a)
                
                # Now we're at step t+1
                # Store activation from step t and features from step t+1
                if len(feats) > 0 and not done:  # Don't collect if episode ended
                    activation_t = feats[-1]  # Activation from step t
                    next_features = obs['features']  # Features from step t+1
                    next_agent_pos = env.agent_pos[:2].copy()  # Position at step t+1
                    
                    rollout_activations.append(activation_t)
                    rollout_next_features.append(next_features)
                    
                    # Break down NEXT step features into components
                    rollout_next_components['next_agent_sensors'].append(next_features[0:3])
                    rollout_next_components['next_zone_lidar'].append(next_features[3:19])
                    rollout_next_components['next_wall_lidar'].append(next_features[19:35])
                    rollout_next_components['next_wall_sensor'].append(next_features[35:39])
                    rollout_next_components['next_joint_positions'].append(next_features[39:])
                    rollout_next_components['next_agent_pos'].append(next_agent_pos)
                    
                    # Calculate NEXT step zone distances and directions
                    next_zone_dists = []
                    next_zone_dirs = []
                    if isinstance(zone_pos, dict):
                        for zone_name, zpos in zone_pos.items():
                            dist = np.linalg.norm(next_agent_pos - zpos[:2])
                            direction = (zpos[:2] - next_agent_pos) / (dist + 1e-8)
                            next_zone_dists.append(dist)
                            next_zone_dirs.extend(direction)
                    rollout_next_components['next_zone_distances'].append(next_zone_dists)
                    rollout_next_components['next_zone_directions'].append(next_zone_dirs)
                    rollout_next_components['next_propositions'].append(list(obs.get('propositions', [])))
                    
                    rollout_pairs += 1
                    total_next_step_pairs += 1
            
            # Store this rollout's data if we collected any
            if rollout_pairs > 0:
                rollout_data.append({
                    'world_id': world_id,
                    'rollout_idx': rollout_idx,
                    'activations': np.array(rollout_activations),
                    'next_features': np.array(rollout_next_features),
                    'next_components': rollout_next_components
                })
            
            h.remove()
            if rollout_pairs > 0:
                successful_rollouts += 1
                total_successful_rollouts += 1
        
        print(f"World {world_id}: {successful_rollouts}/{n_rollouts} successful rollouts, {rollout_pairs} step pairs")
    
    env.close()
    print(f"Total successfully collected {total_next_step_pairs} step pairs from {total_successful_rollouts} rollouts across {len(world_ids)} worlds")
    
    return rollout_data

def train_next_step_probes(rollout_data, test_size=0.2):
    """
    Train probes to predict next step features from current activations.
    Uses proper train/test split by rollout to avoid temporal leakage.
    """
    results = {}
    
    if len(rollout_data) == 0:
        print("No rollout data available")
        return results
    
    # Split by rollout, not by individual samples
    n_rollouts = len(rollout_data)
    rollout_indices = np.arange(n_rollouts)
    train_rollout_indices, test_rollout_indices = train_test_split(
        rollout_indices, test_size=test_size, random_state=SEED)
    
    # Collect train data from train rollouts
    train_activations = []
    train_next_components = {key: [] for key in rollout_data[0]['next_components'].keys()}
    
    for rollout_idx in train_rollout_indices:
        rollout = rollout_data[rollout_idx]
        train_activations.extend(rollout['activations'])
        for key, values in rollout['next_components'].items():
            train_next_components[key].extend(values)
    
    # Collect test data from test rollouts  
    test_activations = []
    test_next_components = {key: [] for key in rollout_data[0]['next_components'].keys()}
    
    for rollout_idx in test_rollout_indices:
        rollout = rollout_data[rollout_idx]
        test_activations.extend(rollout['activations'])
        for key, values in rollout['next_components'].items():
            test_next_components[key].extend(values)
    
    X_train = np.array(train_activations)
    X_test = np.array(test_activations)
    
    print(f"\nTraining next-step probes for {len(train_next_components)} feature components...")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train rollouts: {len(train_rollout_indices)}, Test rollouts: {len(test_rollout_indices)}")
    
    for component_name in train_next_components.keys():
        train_component_data = np.array(train_next_components[component_name])
        test_component_data = np.array(test_next_components[component_name])
        
        if len(train_component_data) == 0 or len(test_component_data) == 0:
            print(f"  {component_name}: No data available")
            continue
        
        # Skip non-numeric components (like propositions)
        if component_name == 'next_propositions':
            print(f"  {component_name}: Skipping non-numeric component")
            continue
            
        Y_train = train_component_data
        Y_test = test_component_data
        
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

def create_next_step_analysis_plots(results, save_path=None):
    """
    Create plots comparing next-step probe performance across different input components.
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
    bars1 = ax1.bar(range(len(component_names)), r2_scores, color='lightcoral', alpha=0.7)
    ax1.set_xlabel('Next-Step Feature Component')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Next-Step Prediction Performance by Component (R²)')
    ax1.set_xticks(range(len(component_names)))
    ax1.set_xticklabels([name.replace('next_', '') for name in component_names], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{score:.3f}', 
                ha='center', va='bottom')
    
    # MSE scores
    bars2 = ax2.bar(range(len(component_names)), mse_scores, color='lightblue', alpha=0.7)
    ax2.set_xlabel('Next-Step Feature Component')
    ax2.set_ylabel('MSE Score')
    ax2.set_title('Next-Step Prediction Performance by Component (MSE)')
    ax2.set_xticks(range(len(component_names)))
    ax2.set_xticklabels([name.replace('next_', '') for name in component_names], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    for bar, score in zip(bars2, mse_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{score:.4f}', 
                ha='center', va='bottom')
    
    # Target dimensions
    bars3 = ax3.bar(range(len(component_names)), target_dims, color='lightgreen', alpha=0.7)
    ax3.set_xlabel('Next-Step Feature Component')
    ax3.set_ylabel('Target Dimension')
    ax3.set_title('Target Dimensions by Component')
    ax3.set_xticks(range(len(component_names)))
    ax3.set_xticklabels([name.replace('next_', '') for name in component_names], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    for bar, dim in zip(bars3, target_dims):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{dim}', 
                ha='center', va='bottom')
    
    # R² vs MSE scatter
    ax4.scatter(mse_scores, r2_scores, s=100, alpha=0.7, color='orange')
    for i, name in enumerate(component_names):
        ax4.annotate(name.replace('next_', ''), (mse_scores[i], r2_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax4.set_xlabel('MSE Score')
    ax4.set_ylabel('R² Score')
    ax4.set_title('R² vs MSE for Next-Step Prediction')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved next-step analysis plot to {save_path}")
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
    
    rollout_data = collect_activations_and_next_features(
        model, args.layer, sampler, n_rollouts=args.n_rollouts, max_steps=args.max_steps, world_ids=world_ids)
    
    if len(rollout_data) == 0:
        print("Could not collect data for probe training. Exiting.")
        return
    
    # Calculate total samples
    total_samples = sum(len(rollout['activations']) for rollout in rollout_data)
    print(f"Collected {total_samples} step pairs from {len(rollout_data)} rollouts")
    
    # Print sample rollout info
    if len(rollout_data) > 0:
        sample_rollout = rollout_data[0]
        print(f"Sample rollout activation shape: {sample_rollout['activations'].shape}")
        print(f"Sample rollout next features shape: {sample_rollout['next_features'].shape}")
        
        # Print next feature component shapes from first rollout
        print("\nNext-step feature component shapes (from first rollout):")
        for name, data in sample_rollout['next_components'].items():
            if len(data) > 0:
                print(f"  {name}: {np.array(data).shape}")
    
    # ── train probes for each next-step component ─────────────────────────────────────────
    results = train_next_step_probes(rollout_data)
    
    # ── create summary plots ───────────────────────────────────────────────────
    if results:
        plot_file = args.out or f'next_step_probes_{args.layer.replace(".", "_")}_world{args.world_id}.png'
        create_next_step_analysis_plots(results, save_path=plot_file)
        
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
        csv_file = f'next_step_probes_{args.layer.replace(".", "_")}_world{args.world_id}.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved results to {csv_file}")
        
        # Print summary statistics
        r2_scores = [results[key]['r2_test'] for key in results.keys()]
        mse_scores = [results[key]['mse_test'] for key in results.keys()]
        print(f"\nNext-Step Prediction Summary:")
        print(f"  Total components analyzed: {len(results)}")
        print(f"  Average R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
        print(f"  Average MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
        print(f"  Best R²: {np.max(r2_scores):.3f} ({list(results.keys())[np.argmax(r2_scores)]})")
        print(f"  Worst R²: {np.min(r2_scores):.3f} ({list(results.keys())[np.argmin(r2_scores)]})")
        
        # Find best performing components
        best_components = sorted(results.items(), key=lambda x: x[1]['r2_test'], reverse=True)
        print(f"\nTop 3 best next-step prediction components:")
        for i, (name, result) in enumerate(best_components[:3]):
            print(f"  {i+1}. {name}: R²={result['r2_test']:.3f}, MSE={result['mse_test']:.4f}")
    else:
        print("No next-step probes could be trained (insufficient data).")

if __name__ == '__main__':
    main() 