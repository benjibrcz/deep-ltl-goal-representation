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
from datetime import datetime

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
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

def collect_activations_and_next_features_by_world(model, layer_name, sampler, n_rollouts=10, max_steps=200, world_ids=None):
    """
    Collect activations from step t and corresponding features from step t+1 for comprehensive next-step generalization testing.
    Returns: {world_id: {'rollouts': [{'activations': [...], 'next_features': [...], 'next_positions': [...]}], 'zone_pos': {...}}}
    """
    if world_ids is None:
        world_ids = list(range(10))
    
    world_data = {}
    env = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    for world_id in world_ids:
        # Skip world file loading to allow proper seeding/randomization
        # world_file = f"{WORLD_DIR}/world_info_{world_id}.pkl"
        # if not os.path.exists(world_file):
        #     print(f"World file not found: {world_file}, skipping.")
        #     continue
        # env.load_world_info(world_file)
        print(f"Processing world_id {world_id} with random world generation")
        zone_pos = dict(env.zone_positions) if hasattr(env, 'zone_positions') else {}
        
        world_rollouts = []
        successful_rollouts = 0
        max_attempts_per_rollout = 10
        
        for rollout_idx in trange(n_rollouts, desc=f"World {world_id} rollouts"):
            # Try different seeds until we find a valid starting position
            for attempt in range(max_attempts_per_rollout):
                try:
                    obs = env.reset(seed=SEED + world_id * 1000 + rollout_idx * max_attempts_per_rollout + attempt)
                    break
                except AssertionError as e:
                    if "World has starting cost" in str(e):
                        if attempt == max_attempts_per_rollout - 1:
                            continue
                    else:
                        raise e
            
            if attempt == max_attempts_per_rollout - 1:
                continue
                
            agent.reset()
            
            rollout_activations = []
            rollout_next_features = []
            rollout_next_positions = []
            rollout_next_zone_distances = []
            rollout_next_zone_directions = []
            
            def grab(m, inp, out):
                x = out[1] if isinstance(out, tuple) else out
                rollout_activations.append(x.detach().cpu().numpy().ravel())
            
            h = module.register_forward_hook(grab)
            
            done = False
            step_activations = []  # Track activations by step
            step_next_features = []
            step_next_positions = []
            
            for step in range(max_steps):
                if done:
                    break
                
                # Record activation count before taking action
                activations_before = len(rollout_activations)
                
                # Take action (this records activation from step t)
                a = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, _ = env.step(a)
                
                # Get the activation that was just recorded for this step
                if len(rollout_activations) > activations_before:
                    step_activation = rollout_activations[-1]  # Latest activation
                    step_activations.append(step_activation)
                    
                    # Store NEXT step data (step t+1) to pair with activation from step t
                    next_agent_pos = env.agent_pos[:2].copy()
                    next_features = obs['features'] if isinstance(obs, dict) else obs
                    
                    step_next_features.append(next_features)
                    step_next_positions.append(next_agent_pos)
                    
                    # Calculate next step zone distances and directions
                    zone_dists = []
                    zone_dirs = []
                    if isinstance(zone_pos, dict) and len(zone_pos) > 0:
                        for zone_name, zpos in zone_pos.items():
                            dist = np.linalg.norm(next_agent_pos - zpos[:2])
                            direction = (zpos[:2] - next_agent_pos) / (dist + 1e-8)
                            zone_dists.append(dist)
                            zone_dirs.extend(direction)
                    rollout_next_zone_distances.append(zone_dists if zone_dists else [0.0])
                    rollout_next_zone_directions.append(zone_dirs if zone_dirs else [0.0, 0.0])
            
            h.remove()
            
            # Only keep rollouts with sufficient data (need activation[t] -> next_feature[t+1] pairs)
            if len(step_activations) >= 2 and len(step_next_features) >= 2:
                # Ensure arrays are same length - now properly aligned by step
                min_len = min(len(step_activations), len(step_next_features), len(step_next_positions))
                world_rollouts.append({
                    'activations': np.array(step_activations[:min_len]),
                    'next_features': np.array(step_next_features[:min_len]),
                    'next_positions': np.array(step_next_positions[:min_len]),
                    'next_zone_distances': rollout_next_zone_distances[:min_len],
                    'next_zone_directions': rollout_next_zone_directions[:min_len],
                    'rollout_idx': rollout_idx
                })
                successful_rollouts += 1
        
        world_data[world_id] = {
            'rollouts': world_rollouts,
            'zone_pos': zone_pos
        }
        
        print(f"World {world_id}: {successful_rollouts}/{n_rollouts} successful rollouts")
    
    env.close()
    return world_data

def create_comprehensive_next_step_train_test_splits(world_data, target_feature='agent_pos'):
    """
    Create three types of train/test splits for next-step prediction:
    1. Same rollout: Train on first half of time steps, test on second half (same rollouts)
    2. Same world: Train on some rollouts, test on different rollouts (same worlds)  
    3. Different world: Train on some worlds, test on different worlds
    """
    splits = {
        'test_same_rollout': {'train': {'X': [], 'y': []}, 'test': {'X': [], 'y': []}},
        'test_same_world': {'train': {'X': [], 'y': []}, 'test': {'X': [], 'y': []}},
        'test_different_world': {'train': {'X': [], 'y': []}, 'test': {'X': [], 'y': []}}
    }
    
    world_ids = list(world_data.keys())
    if len(world_ids) < 2:
        print("Need at least 2 worlds for environmental split")
        return splits
    
    # Different world split: use first half of worlds for train, second half for test
    train_worlds = world_ids[:len(world_ids)//2]
    test_worlds = world_ids[len(world_ids)//2:]
    
    print(f"Next-step Different World split: Train worlds {train_worlds}, Test worlds {test_worlds}")
    
    # Process each world
    for world_id, world_info in world_data.items():
        rollouts = world_info['rollouts']
        if len(rollouts) < 2:
            continue
            
        # Same world split: use rollouts from same world
        train_rollouts = rollouts[:len(rollouts)//2]
        test_rollouts = rollouts[len(rollouts)//2:]
        
        # Process rollouts for same world and same rollout splits
        for rollout_set, split_type in [(train_rollouts, 'train'), (test_rollouts, 'test')]:
            for rollout in rollout_set:
                activations = rollout['activations']
                next_features = rollout['next_features'] 
                next_positions = rollout['next_positions']
                
                if target_feature == 'agent_pos':
                    targets = next_positions
                elif target_feature == 'wall_sensor':
                    targets = next_features[:, 35:39]  # Wall sensor is features 35-39
                elif target_feature == 'zone_lidar':
                    targets = next_features[:, 3:19]   # Zone lidar is features 3-19
                elif target_feature == 'agent_sensors':
                    targets = next_features[:, 0:3]    # Agent sensors are features 0-3
                elif target_feature == 'wall_lidar':
                    targets = next_features[:, 19:35]  # Wall lidar is features 19-35
                elif target_feature == 'joint_positions':
                    targets = next_features[:, 39:]    # Joint positions are features 39+
                elif target_feature == 'zone_distances':
                    # Convert list of distance lists to numpy array
                    zone_dist_data = rollout['next_zone_distances']
                    if len(zone_dist_data) > 0 and len(zone_dist_data[0]) > 0:
                        targets = np.array([np.array(dists) for dists in zone_dist_data])
                    else:
                        targets = np.zeros((len(next_positions), 1))  # Fallback
                elif target_feature == 'zone_directions':
                    # Convert list of direction lists to numpy array
                    zone_dir_data = rollout['next_zone_directions']
                    if len(zone_dir_data) > 0 and len(zone_dir_data[0]) > 0:
                        targets = np.array([np.array(dirs) for dirs in zone_dir_data])
                    else:
                        targets = np.zeros((len(next_positions), 2))  # Fallback
                else:
                    targets = next_positions  # Default to positions
                
                # Same rollout split: same rollout, different time steps
                if len(activations) >= 4:  # Need enough steps to split
                    mid_point = len(activations) // 2
                    
                    # First half for same rollout train, second half for same rollout test
                    splits['test_same_rollout']['train']['X'].extend(activations[:mid_point].tolist())
                    splits['test_same_rollout']['train']['y'].extend(targets[:mid_point].tolist())
                    splits['test_same_rollout']['test']['X'].extend(activations[mid_point:].tolist())
                    splits['test_same_rollout']['test']['y'].extend(targets[mid_point:].tolist())
                
                # Same world split: add all data from this rollout to appropriate set
                splits['test_same_world'][split_type]['X'].extend(activations.tolist())
                splits['test_same_world'][split_type]['y'].extend(targets.tolist())
                
                # Different world split: add based on which world this is
                env_split_type = 'train' if world_id in train_worlds else 'test'
                splits['test_different_world'][env_split_type]['X'].extend(activations.tolist())
                splits['test_different_world'][env_split_type]['y'].extend(targets.tolist())
    
    # Convert to numpy arrays
    for split_name in splits:
        for subset in ['train', 'test']:
            if splits[split_name][subset]['X']:
                splits[split_name][subset]['X'] = np.array(splits[split_name][subset]['X'])
                splits[split_name][subset]['y'] = np.array(splits[split_name][subset]['y'])
            else:
                splits[split_name][subset]['X'] = np.array([])
                splits[split_name][subset]['y'] = np.array([])
    
    return splits

def train_and_evaluate_next_step_probes(splits, target_feature='agent_pos'):
    """
    Train and evaluate next-step prediction probes for each type of generalization.
    """
    results = {}
    
    for split_name, split_data in splits.items():
        X_train = split_data['train']['X']
        y_train = split_data['train']['y']
        X_test = split_data['test']['X']
        y_test = split_data['test']['y']
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  {split_name}: Insufficient data (train: {len(X_train)}, test: {len(X_test)})")
            continue
        
        # Train probe
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = probe.predict(X_train)
        y_pred_test = probe.predict(X_test)
        
        # Calculate metrics
        if y_train.ndim == 1:
            mse_train = mean_squared_error(y_train, y_pred_train)
            r2_train = r2_score(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)
        else:
            # Multi-dimensional targets
            mse_train = np.mean([mean_squared_error(y_train[:, i], y_pred_train[:, i]) 
                                for i in range(y_train.shape[1])])
            r2_train = np.mean([r2_score(y_train[:, i], y_pred_train[:, i]) 
                               for i in range(y_train.shape[1])])
            mse_test = np.mean([mean_squared_error(y_test[:, i], y_pred_test[:, i]) 
                               for i in range(y_test.shape[1])])
            r2_test = np.mean([r2_score(y_test[:, i], y_pred_test[:, i]) 
                              for i in range(y_test.shape[1])])
        
        results[split_name] = {
            'mse_train': mse_train,
            'r2_train': r2_train, 
            'mse_test': mse_test,
            'r2_test': r2_test,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'feature_dim': X_train.shape[1],
            'target_dim': y_train.shape[1] if y_train.ndim > 1 else 1,
            'probe': probe
        }
        
        print(f"  {split_name}: R²={r2_test:.3f}, MSE={mse_test:.4f}, "
              f"train_n={len(X_train)}, test_n={len(X_test)}")
    
    return results

def create_next_step_generalization_comparison_plot(results, target_feature, save_path=None):
    """
    Create a comprehensive plot comparing the three types of next-step generalization.
    """
    if not results:
        print("No results to plot")
        return
    
    split_names = list(results.keys())
    r2_scores = [results[name]['r2_test'] for name in split_names]
    mse_scores = [results[name]['mse_test'] for name in split_names]
    n_train = [results[name]['n_train_samples'] for name in split_names]
    n_test = [results[name]['n_test_samples'] for name in split_names]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Colors for each test type
    colors = {'test_same_rollout': 'skyblue', 'test_same_world': 'lightgreen', 'test_different_world': 'lightcoral'}
    split_colors = [colors.get(name, 'gray') for name in split_names]
    
    # R² comparison
    bars1 = ax1.bar(split_names, r2_scores, color=split_colors, alpha=0.7)
    ax1.set_ylabel('R² Score')
    ax1.set_title(f'Next-Step Prediction Performance: {target_feature} (R²)')
    ax1.grid(True, alpha=0.3)
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{score:.3f}', 
                ha='center', va='bottom')
    
    # MSE comparison
    bars2 = ax2.bar(split_names, mse_scores, color=split_colors, alpha=0.7)
    ax2.set_ylabel('MSE Score') 
    ax2.set_title(f'Next-Step Prediction Performance: {target_feature} (MSE)')
    ax2.grid(True, alpha=0.3)
    for bar, score in zip(bars2, mse_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{score:.4f}', 
                ha='center', va='bottom')
    
    # Sample sizes
    x_pos = np.arange(len(split_names))
    ax3.bar(x_pos - 0.2, n_train, 0.4, label='Train', color='lightblue', alpha=0.7)
    ax3.bar(x_pos + 0.2, n_test, 0.4, label='Test', color='orange', alpha=0.7)
    ax3.set_ylabel('Number of Samples')
    ax3.set_title('Sample Sizes by Test Type')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(split_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # R² vs MSE scatter with annotations
    ax4.scatter(mse_scores, r2_scores, c=[colors.get(name, 'gray') for name in split_names], 
               s=100, alpha=0.7)
    for i, name in enumerate(split_names):
        ax4.annotate(name, (mse_scores[i], r2_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax4.set_xlabel('MSE Score')
    ax4.set_ylabel('R² Score')
    ax4.set_title('R² vs MSE by Next-Step Test Type')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved next-step generalization comparison plot to {save_path}")
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True)
    p.add_argument('--n-rollouts',  type=int, default=N_ROLLOUTS)
    p.add_argument('--max-steps',   type=int, default=200)
    p.add_argument('--target-feature', type=str, default='agent_pos', 
                   choices=['agent_pos', 'wall_sensor', 'zone_lidar', 'agent_sensors', 
                           'wall_lidar', 'joint_positions', 'zone_distances', 'zone_directions'],
                   help='Which next-step feature to predict')
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

    # ── collect data organized by world and rollout ─────────────────────────────
    world_ids = list(range(10))  # Use first 10 worlds
    print(f"Using worlds: {world_ids}")
    
    world_data = collect_activations_and_next_features_by_world(
        model, args.layer, sampler, n_rollouts=args.n_rollouts, 
        max_steps=args.max_steps, world_ids=world_ids)
    
    if not world_data:
        print("Could not collect data for next-step probe training. Exiting.")
        return
    
    total_rollouts = sum(len(world_info['rollouts']) for world_info in world_data.values())
    print(f"Collected data from {total_rollouts} rollouts across {len(world_data)} worlds")
    
    # ── create comprehensive train/test splits ──────────────────────────────────
    print(f"\nCreating comprehensive next-step train/test splits for {args.target_feature}...")
    splits = create_comprehensive_next_step_train_test_splits(world_data, target_feature=args.target_feature)
    
    # Print split statistics
    print("\nNext-step split statistics:")
    for split_name, split_data in splits.items():
        train_size = len(split_data['train']['X']) if len(split_data['train']['X']) > 0 else 0
        test_size = len(split_data['test']['X']) if len(split_data['test']['X']) > 0 else 0
        print(f"  {split_name}: {train_size} train, {test_size} test samples")
    
    # ── train and evaluate next-step probes ─────────────────────────────────────
    print(f"\nTraining next-step prediction probes for {args.target_feature}...")
    results = train_and_evaluate_next_step_probes(splits, target_feature=args.target_feature)
    
    # ── create comparison plots ─────────────────────────────────────────────────
    if results:
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        layer_name = args.layer.replace(".", "_")
        
        # Use results directory relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results")
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate unique filenames
        plot_file = args.out or f'{results_dir}/next_step_generalization_{args.target_feature}_{layer_name}_{timestamp}.png'
        csv_file = f'{results_dir}/next_step_generalization_{args.target_feature}_{layer_name}_{timestamp}.csv'
        summary_file = f'{results_dir}/next_step_summary_{args.target_feature}_{layer_name}_{timestamp}.txt'
        
        create_next_step_generalization_comparison_plot(results, args.target_feature, save_path=plot_file)
        
        # Save results to CSV
        df = pd.DataFrame([
            {
                'timestamp': timestamp,
                'split_type': name,
                'target_feature': args.target_feature,
                'layer': args.layer,
                'prediction_type': 'next_step',
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
        df.to_csv(csv_file, index=False)
        print(f"Saved results to {csv_file}")
        
        # Save detailed summary
        with open(summary_file, 'w') as f:
            f.write(f"Comprehensive Next-Step Prediction Analysis\n")
            f.write(f"===========================================\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Layer: {args.layer}\n")
            f.write(f"Target Feature: {args.target_feature}\n")
            f.write(f"Prediction Type: Next Step (t → t+1)\n")
            f.write(f"Number of Rollouts per World: {args.n_rollouts}\n")
            f.write(f"Max Steps per Rollout: {args.max_steps}\n")
            f.write(f"Total Rollouts Collected: {total_rollouts}\n")
            f.write(f"Total Worlds Used: {len(world_data)}\n\n")
            
            f.write("Split Statistics:\n")
            f.write("-" * 50 + "\n")
            for split_name, split_data in splits.items():
                train_size = len(split_data['train']['X']) if len(split_data['train']['X']) > 0 else 0
                test_size = len(split_data['test']['X']) if len(split_data['test']['X']) > 0 else 0
                f.write(f"{split_name:20}: {train_size:4d} train, {test_size:4d} test samples\n")
            
            f.write("\nNext-Step Prediction Performance:\n")
            f.write("-" * 50 + "\n")
            for test_name in ['test_same_rollout', 'test_same_world', 'test_different_world']:
                if test_name in results:
                    r2 = results[test_name]['r2_test']
                    mse = results[test_name]['mse_test']
                    n_train = results[test_name]['n_train_samples']
                    n_test = results[test_name]['n_test_samples']
                    f.write(f"{test_name:20}: R²={r2:.3f}, MSE={mse:.4f} (train={n_train}, test={n_test})\n")
                else:
                    f.write(f"{test_name:20}: No data available\n")
            
            if len(results) > 0:
                r2_scores = [(name, results[name]['r2_test']) for name in results.keys()]
                best_test = max(r2_scores, key=lambda x: x[1])
                worst_test = min(r2_scores, key=lambda x: x[1])
                f.write(f"\nBest next-step prediction:  {best_test[0]} (R²={best_test[1]:.3f})\n")
                f.write(f"Worst next-step prediction: {worst_test[0]} (R²={worst_test[1]:.3f})\n")
                
                # Calculate generalization gaps
                f.write("\nNext-Step Prediction Gaps (R² differences):\n")
                f.write("-" * 50 + "\n")
                if 'test_same_rollout' in results and 'test_same_world' in results:
                    rollout_world_gap = results['test_same_rollout']['r2_test'] - results['test_same_world']['r2_test']
                    f.write(f"Same Rollout → Same World Gap:     {rollout_world_gap:+.3f}\n")
                if 'test_same_world' in results and 'test_different_world' in results:
                    world_diff_gap = results['test_same_world']['r2_test'] - results['test_different_world']['r2_test']
                    f.write(f"Same World → Different World Gap:  {world_diff_gap:+.3f}\n")
                if 'test_same_rollout' in results and 'test_different_world' in results:
                    rollout_diff_gap = results['test_same_rollout']['r2_test'] - results['test_different_world']['r2_test']
                    f.write(f"Same Rollout → Different World Gap: {rollout_diff_gap:+.3f}\n")
        
        print(f"Saved detailed summary to {summary_file}")
        
        # Print summary
        print(f"\nComprehensive Next-Step Prediction Results for {args.target_feature}:")
        for test_name in ['test_same_rollout', 'test_same_world', 'test_different_world']:
            if test_name in results:
                r2 = results[test_name]['r2_test']
                mse = results[test_name]['mse_test']
                print(f"  {test_name:20}: R²={r2:.3f}, MSE={mse:.4f}")
        
        # Identify best and worst test scenarios
        r2_scores = [(name, results[name]['r2_test']) for name in results.keys()]
        best_test = max(r2_scores, key=lambda x: x[1])
        worst_test = min(r2_scores, key=lambda x: x[1])
        print(f"\nBest next-step prediction:  {best_test[0]} (R²={best_test[1]:.3f})")
        print(f"Worst next-step prediction: {worst_test[0]} (R²={worst_test[1]:.3f})")
        
    else:
        print("No next-step prediction probes could be trained (insufficient data).")

if __name__ == '__main__':
    main() 