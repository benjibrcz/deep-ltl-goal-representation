#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
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
from visualize.zones      import FancyAxes, draw_zones, draw_path, draw_diamond, setup_axis

# ─── defaults ─────────────────────────────────────────────────────────────────
ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
N_ROLLOUTS = 10
MAX_STEPS  = 200
# ───────────────────────────────────────────────────────────────────────────────

def collect_movement_vector_data_by_world(model, layer_name, sampler, k_steps=1, n_rollouts=10, max_steps=200, world_ids=None):
    """
    Collect activations and k-step movement vectors organized by world and rollout.
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
            rollout_positions = []
            
            def grab(m, inp, out):
                x = out[1] if isinstance(out, tuple) else out
                rollout_activations.append(x.detach().cpu().numpy().ravel())
            
            h = module.register_forward_hook(grab)
            
            done = False
            for step in range(max_steps):
                if done:
                    break
                
                # Record current position
                current_pos = env.agent_pos[:2].copy()
                rollout_positions.append(current_pos)
                
                # Take action (this records activation)
                a = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, _ = env.step(a)
            
            h.remove()
            
            # Calculate k-step movement vectors
            rollout_movement_vectors = []
            if len(rollout_positions) >= k_steps + 1 and len(rollout_activations) >= k_steps + 1:
                for i in range(len(rollout_positions) - k_steps):
                    current_pos = rollout_positions[i]
                    future_pos = rollout_positions[i + k_steps]
                    movement_vector = future_pos - current_pos
                    rollout_movement_vectors.append(movement_vector)
                
                # Ensure arrays are same length
                min_len = min(len(rollout_activations), len(rollout_movement_vectors))
                if min_len >= 2:
                    world_rollouts.append({
                        'activations': np.array(rollout_activations[:min_len]),
                        'movement_vectors': np.array(rollout_movement_vectors[:min_len]),
                        'positions': np.array(rollout_positions[:min_len]),  # For visualization
                        'rollout_idx': rollout_idx,
                        'k_steps': k_steps
                    })
                    successful_rollouts += 1
        
        world_data[world_id] = {
            'rollouts': world_rollouts,
            'zone_pos': zone_pos
        }
        
        print(f"World {world_id}: {successful_rollouts}/{n_rollouts} successful rollouts")
    
    env.close()
    return world_data

def create_movement_vector_splits(world_data):
    """
    Create three types of train/test splits for movement vector prediction.
    """
    splits = {
        'test_same_rollout': {'train': {'X': [], 'y': []}, 'test': {'X': [], 'y': []}},
        'test_same_world': {'train': {'X': [], 'y': []}, 'test': {'X': [], 'y': []}},
        'test_different_world': {'train': {'X': [], 'y': []}, 'test': {'X': [], 'y': []}}
    }
    
    world_ids = list(world_data.keys())
    print(f"Processing {len(world_ids)} worlds for generalization splits")
    
    # Set up environmental split only if we have enough worlds
    can_do_environmental_split = len(world_ids) >= 2
    if can_do_environmental_split:
        # Environmental split: use first half of worlds for train, second half for test
        train_worlds = world_ids[:len(world_ids)//2]
        test_worlds = world_ids[len(world_ids)//2:]
        print(f"Different World split: Train worlds {train_worlds}, Test worlds {test_worlds}")
    else:
        print("Only 1 world available - skipping environmental split, but same_rollout should still work")
    
    # Process each world
    for world_id, world_info in world_data.items():
        rollouts = world_info['rollouts']
        if len(rollouts) == 0:
            continue
            
        # Same world split: use rollouts from same world (only if we have multiple rollouts)
        can_do_same_world_split = len(rollouts) >= 2
        if can_do_same_world_split:
            train_rollouts = rollouts[:len(rollouts)//2]
            test_rollouts = rollouts[len(rollouts)//2:]
        else:
            # With only 1 rollout, we can't do same_world split, but can still do same_rollout
            train_rollouts = []
            test_rollouts = []
        
        # Process rollouts for same_rollout splits (always) and same_world splits (if possible)
        
        # 1. Same rollout splits: process all rollouts for temporal splitting
        for rollout in rollouts:
            activations = rollout['activations']
            movement_vectors = rollout['movement_vectors']
            
            # Same rollout split: same rollout, different time steps
            if len(activations) >= 4:  # Need enough steps to split
                mid_point = len(activations) // 2
                
                # First half for same rollout train, second half for same rollout test
                splits['test_same_rollout']['train']['X'].extend(activations[:mid_point].tolist())
                splits['test_same_rollout']['train']['y'].extend(movement_vectors[:mid_point].tolist())
                splits['test_same_rollout']['test']['X'].extend(activations[mid_point:].tolist())
                splits['test_same_rollout']['test']['y'].extend(movement_vectors[mid_point:].tolist())
        
        # 2. Same world splits: only if we have multiple rollouts
        if can_do_same_world_split:
            for rollout_set, split_type in [(train_rollouts, 'train'), (test_rollouts, 'test')]:
                for rollout in rollout_set:
                    activations = rollout['activations']
                    movement_vectors = rollout['movement_vectors']
                    
                    # Same world split: add all data from this rollout to appropriate set
                    splits['test_same_world'][split_type]['X'].extend(activations.tolist())
                    splits['test_same_world'][split_type]['y'].extend(movement_vectors.tolist())
        
        # 3. Different world splits: only if we have multiple worlds
        if can_do_environmental_split:
            env_split_type = 'train' if world_id in train_worlds else 'test'
            for rollout in rollouts:
                activations = rollout['activations']
                movement_vectors = rollout['movement_vectors']
                splits['test_different_world'][env_split_type]['X'].extend(activations.tolist())
                splits['test_different_world'][env_split_type]['y'].extend(movement_vectors.tolist())
    
    # Convert to numpy arrays - create new dict to avoid type issues
    for split_name in splits:
        for subset in ['train', 'test']:
            if splits[split_name][subset]['X']:
                X_array = np.array(splits[split_name][subset]['X'])
                y_array = np.array(splits[split_name][subset]['y'])
                splits[split_name][subset] = {'X': X_array, 'y': y_array}  # type: ignore
            else:
                splits[split_name][subset] = {'X': np.array([]), 'y': np.array([])}  # type: ignore
    
    return splits

def train_and_evaluate_movement_probes(splits, k_steps=1):
    """
    Train and evaluate movement vector probes for each type of generalization.
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
        
        # Calculate metrics for 2D vectors
        mse_train = np.mean([mean_squared_error(y_train[:, i], y_pred_train[:, i]) 
                            for i in range(y_train.shape[1])])
        r2_train = np.mean([r2_score(y_train[:, i], y_pred_train[:, i]) 
                           for i in range(y_train.shape[1])])
        mse_test = np.mean([mean_squared_error(y_test[:, i], y_pred_test[:, i]) 
                           for i in range(y_test.shape[1])])
        r2_test = np.mean([r2_score(y_test[:, i], y_pred_test[:, i]) 
                          for i in range(y_test.shape[1])])
        
        # Calculate movement magnitude metrics
        train_mag_actual = np.linalg.norm(y_train, axis=1)
        train_mag_pred = np.linalg.norm(y_pred_train, axis=1)
        test_mag_actual = np.linalg.norm(y_test, axis=1)
        test_mag_pred = np.linalg.norm(y_pred_test, axis=1)
        
        mag_r2_train = r2_score(train_mag_actual, train_mag_pred)
        mag_r2_test = r2_score(test_mag_actual, test_mag_pred)
        
        results[split_name] = {
            'mse_train': mse_train,
            'r2_train': r2_train, 
            'mse_test': mse_test,
            'r2_test': r2_test,
            'mag_r2_train': mag_r2_train,
            'mag_r2_test': mag_r2_test,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'feature_dim': X_train.shape[1],
            'probe': probe,
            'k_steps': k_steps,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
        
        print(f"  {split_name}: R²={r2_test:.3f}, Mag-R²={mag_r2_test:.3f}, MSE={mse_test:.6f}, "
              f"train_n={len(X_train)}, test_n={len(X_test)}")
    
    return results

def create_movement_vector_visualization(results, k_steps, save_path=None):
    """
    Create comprehensive visualization of movement vector predictions.
    """
    if not results:
        print("No results to visualize")
        return
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'{k_steps}-Step Movement Vector Predictions: Generalization Analysis', 
                fontsize=20, fontweight='bold', y=0.95)
    
    split_names = list(results.keys())
    split_titles = {
        'test_same_rollout': 'Same Rollout (Temporal)',
        'test_same_world': 'Same World (Spatial)', 
        'test_different_world': 'Different World (Environmental)'
    }
    
    # 1. R² Comparison
    ax1 = fig.add_subplot(2, 3, 1)
    r2_scores = [results[name]['r2_test'] for name in split_names]
    mag_r2_scores = [results[name]['mag_r2_test'] for name in split_names]
    
    x = np.arange(len(split_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, r2_scores, width, label='Vector R²', alpha=0.7, color='skyblue')
    bars2 = ax1.bar(x + width/2, mag_r2_scores, width, label='Magnitude R²', alpha=0.7, color='lightcoral')
    
    ax1.set_ylabel('R² Score')
    ax1.set_title(f'{k_steps}-Step Movement Vector Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels([split_titles.get(name, name) or name for name in split_names], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{score:.3f}', 
                ha='center', va='bottom')
    for bar, score in zip(bars2, mag_r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{score:.3f}', 
                ha='center', va='bottom')
    
    # 2-4. Movement vector scatter plots for each split
    for i, split_name in enumerate(split_names):
        ax = fig.add_subplot(2, 3, i + 2)
        
        if split_name in results:
            y_test = results[split_name]['y_test']
            y_pred = results[split_name]['y_pred_test']
            
            # Scatter plot of actual vs predicted movements
            ax.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.6, s=20, label='X-movement', color='blue')
            ax.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.6, s=20, label='Y-movement', color='red')
            
            # Perfect prediction lines
            all_vals = np.concatenate([y_test.flatten(), y_pred.flatten()])
            min_val, max_val = np.min(all_vals), np.max(all_vals)
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')
            
            ax.set_xlabel('Actual Movement')
            ax.set_ylabel('Predicted Movement') 
            ax.set_title(f'{split_titles.get(split_name, split_name)}\nR²={results[split_name]["r2_test"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 5. Movement magnitude comparison
    ax5 = fig.add_subplot(2, 3, 5)
    
    for i, split_name in enumerate(split_names):
        if split_name in results:
            y_test = results[split_name]['y_test']
            y_pred = results[split_name]['y_pred_test']
            
            mag_actual = np.linalg.norm(y_test, axis=1)
            mag_pred = np.linalg.norm(y_pred, axis=1)
            
            colors = ['blue', 'red', 'green']
            ax5.scatter(mag_actual, mag_pred, alpha=0.6, s=20, 
                       color=colors[i % len(colors)], label=split_titles.get(split_name, split_name))
    
    # Perfect prediction line for magnitudes
    if results:
        all_mags = []
        for split_name in split_names:
            if split_name in results:
                y_test = results[split_name]['y_test']
                y_pred = results[split_name]['y_pred_test']
                all_mags.extend(np.linalg.norm(y_test, axis=1))
                all_mags.extend(np.linalg.norm(y_pred, axis=1))
        
        if all_mags:
            min_mag, max_mag = np.min(all_mags), np.max(all_mags)
            ax5.plot([min_mag, max_mag], [min_mag, max_mag], 'k--', alpha=0.5, label='Perfect')
    
    ax5.set_xlabel('Actual Movement Magnitude')
    ax5.set_ylabel('Predicted Movement Magnitude')
    ax5.set_title(f'{k_steps}-Step Movement Magnitude Prediction')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""{k_steps}-STEP MOVEMENT VECTOR RESULTS

Generalization Performance (R²):
"""
    
    for split_name in split_names:
        if split_name in results:
            r2 = results[split_name]['r2_test']
            mag_r2 = results[split_name]['mag_r2_test']
            n_test = results[split_name]['n_test_samples']
            stats_text += f"  {split_titles.get(split_name, split_name):<20}: {r2:.3f} (mag: {mag_r2:.3f}, n={n_test})\n"
    
    if len(results) > 0:
        r2_scores = [(name, results[name]['r2_test']) for name in results.keys()]
        best_split = max(r2_scores, key=lambda x: x[1])
        worst_split = min(r2_scores, key=lambda x: x[1])
        
        stats_text += f"""
Best generalization:  {split_titles.get(best_split[0], best_split[0])} (R²={best_split[1]:.3f})
Worst generalization: {split_titles.get(worst_split[0], worst_split[0])} (R²={worst_split[1]:.3f})

Movement Vector Analysis:
  - Predicting {k_steps}-step movement vectors
  - Translation-invariant representation
  - Tests spatial planning capabilities
"""
        
        # Calculate sample movement statistics
        sample_result = next(iter(results.values()))
        sample_movements = sample_result['y_test']
        avg_movement = np.mean(np.linalg.norm(sample_movements, axis=1))
        max_movement = np.max(np.linalg.norm(sample_movements, axis=1))
        
        stats_text += f"""
Sample Movement Statistics:
  Average {k_steps}-step movement: {avg_movement:.4f}
  Maximum {k_steps}-step movement: {max_movement:.4f}
"""
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved movement vector analysis to {save_path}")
    
    plt.show()
    return fig

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True, help='Layer to probe')
    p.add_argument('--k-steps',     type=int, default=1, help='Number of steps ahead to predict movement (1=next step, 5=5 steps ahead)')
    p.add_argument('--n-rollouts',  type=int, default=N_ROLLOUTS, help='Number of rollouts per world')
    p.add_argument('--n-worlds',    type=int, default=10, help='Number of worlds to use (default: 10)')
    p.add_argument('--max-steps',   type=int, default=MAX_STEPS, help='Maximum steps per rollout')
    p.add_argument('--out',         type=str, help='Output file path')
    args = p.parse_args()

    # seeds & sampler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = "FG blue"
    sampler = FixedSampler.partial(formula)

    # ── load model ───────────────────────────────────────────────────────────────
    print("Loading model...")
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[ENV]
    dummy  = make_env(ENV, sampler, sequence=False, render_mode=None)
    model  = build_model(dummy, status, cfg).eval()
    dummy.close()

    # ── collect movement vector data organized by world and rollout ─────────────
    world_ids = list(range(args.n_worlds))
    print(f"Using {args.n_worlds} worlds: {world_ids}")
    print(f"Predicting {args.k_steps}-step movement vectors")
    
    world_data = collect_movement_vector_data_by_world(
        model, args.layer, sampler, k_steps=args.k_steps, n_rollouts=args.n_rollouts, 
        max_steps=args.max_steps, world_ids=world_ids)
    
    if not world_data:
        print("Could not collect data for probe training. Exiting.")
        return
    
    total_rollouts = sum(len(world_info['rollouts']) for world_info in world_data.values())
    print(f"Collected data from {total_rollouts} rollouts across {len(world_data)} worlds")
    
    # ── create comprehensive train/test splits ──────────────────────────────────
    print(f"\nCreating movement vector splits for {args.k_steps}-step prediction...")
    splits = create_movement_vector_splits(world_data)
    
    # Print split statistics
    print("\nSplit statistics:")
    for split_name, split_data in splits.items():
        train_size = len(split_data['train']['X']) if len(split_data['train']['X']) > 0 else 0
        test_size = len(split_data['test']['X']) if len(split_data['test']['X']) > 0 else 0
        print(f"  {split_name}: {train_size} train, {test_size} test samples")
    
    # ── train and evaluate probes ───────────────────────────────────────────────
    print(f"\nTraining {args.k_steps}-step movement vector probes...")
    results = train_and_evaluate_movement_probes(splits, k_steps=args.k_steps)
    
    # ── create visualization ────────────────────────────────────────────────────
    if results:
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        layer_name = args.layer.replace(".", "_")
        
        # Use results directory relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate unique filename
        plot_file = args.out or f'{results_dir}/movement_vectors_{args.k_steps}step_{layer_name}_{timestamp}.png'
        
        create_movement_vector_visualization(results, args.k_steps, save_path=plot_file)
        
        # Print summary
        print(f"\n{args.k_steps}-Step Movement Vector Prediction Results:")
        for test_name in ['test_same_rollout', 'test_same_world', 'test_different_world']:
            if test_name in results:
                r2 = results[test_name]['r2_test']
                mag_r2 = results[test_name]['mag_r2_test']
                print(f"  {test_name:20}: R²={r2:.3f}, Magnitude-R²={mag_r2:.3f}")
        
        # Identify best and worst test scenarios
        r2_scores = [(name, results[name]['r2_test']) for name in results.keys()]
        best_test = max(r2_scores, key=lambda x: x[1])
        worst_test = min(r2_scores, key=lambda x: x[1])
        print(f"\nBest test scenario:  {best_test[0]} (R²={best_test[1]:.3f})")
        print(f"Worst test scenario: {worst_test[0]} (R²={worst_test[1]:.3f})")
        
    else:
        print("No probes could be trained (insufficient data).")

if __name__ == '__main__':
    main() 