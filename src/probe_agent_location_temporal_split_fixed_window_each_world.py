#!/usr/bin/env python3
"""
This script trains a linear probe to predict the agent's next position using temporal train/test split.

Key features:
- Uses early steps (0-train_steps) for training
- Uses later steps (test_start-test_end) for testing
- Tests temporal generalization: can the probe predict future behavior from early behavior?
- Avoids data leakage by temporally separating train and test sets

Usage: python src/probe_agent_location_temporal_split_fixed_window_each_world.py --layer env_net.mlp.2 --train-steps 200 --test-start 200 --test-end 250 --world-id -1
"""
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store    import ModelStore
from model.model          import build_model
from config               import model_configs
from ltl                  import FixedSampler
from envs                 import make_env
from sequence.search      import ExhaustiveSearch
from model.agent          import Agent
from visualize.zones      import draw_zones, draw_diamond, setup_axis

ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
N_ROLLOUTS = 10
WORLD_ID   = 0
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"

def collect_hidden_and_next_positions_temporal_split(model, layer_name, sampler, n_rollouts=10, max_steps=500, world_ids=None):
    if world_ids is None:
        world_ids = list(range(10))
    world_data = {}
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]
    total_successful_rollouts = 0
    for world_id in world_ids:
        world_dir_path = f"{WORLD_DIR}"
        world_file = f"{world_dir_path}/world_info_{world_id}.pkl"
        if not os.path.exists(world_file):
            print(f"World file not found: {world_file}, skipping.")
            continue
        env.load_world_info(world_file)
        zone_pos = dict(env.zone_positions)
        successful_rollouts = 0
        max_attempts_per_rollout = 10
        world_activations = []
        world_positions = []
        rollout_lengths = []
        for rollout_idx in trange(n_rollouts, desc=f"Rollouts for world {world_id}"):
            for attempt in range(max_attempts_per_rollout):
                try:
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
            if attempt == max_attempts_per_rollout - 1:
                continue
            agent.reset()
            feats = []
            positions = []
            def grab(m, inp, out):
                x = out[1] if isinstance(out, tuple) else out
                feats.append(x.detach().cpu().numpy().ravel())
            h = module.register_forward_hook(grab)
            done = False
            for step in range(max_steps):
                if done:
                    break
                positions.append(env.agent_pos[:2].copy())
                a = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, _ = env.step(a)
                if len(feats) < len(positions):
                    feats.append(feats[-1])
            # For next-state prediction: align feats[t] with positions[t+1]
            if len(positions) > 1 and len(feats) > 1:
                feats = feats[:-1]      # activations at t
                positions = positions[1:]  # positions at t+1
            h.remove()
            if len(feats) > len(positions):
                feats = feats[:len(positions)]
            if len(positions) > 0:
                world_activations.append(np.stack(feats))
                world_positions.append(np.stack(positions))
                rollout_lengths.append(len(positions))
                successful_rollouts += 1
                total_successful_rollouts += 1
        world_data[world_id] = {
            'activations': world_activations,
            'positions': world_positions,
            'rollout_lengths': rollout_lengths,
            'zone_pos': zone_pos
        }
        print(f"World {world_id}: {successful_rollouts}/{n_rollouts} successful rollouts")
    env.close()
    print(f"Total successfully collected data from {total_successful_rollouts} rollouts across {len(world_ids)} worlds")
    return world_data

def create_temporal_split_features(activations, positions, train_steps=200, test_start=200, test_end=250):
    """
    Create features for temporal split:
    - Training: activations[0:train_steps] -> positions[1:train_steps+1]
    - Testing: activations[test_start:test_end] -> positions[test_start+1:test_end+1]
    """
    X_train_all, Y_train_all = [], []
    X_test_all, Y_test_all = [], []
    
    for rollout_idx, (rollout_activations, rollout_positions) in enumerate(zip(activations, positions)):
        rollout_length = len(rollout_positions)
        
        # Training data: early steps
        train_end = min(train_steps, rollout_length - 1)
        if train_end > 0:
            for t in range(train_end):
                X_train_all.append(rollout_activations[t])
                Y_train_all.append(rollout_positions[t])
        
        # Testing data: later steps
        test_start_idx = min(test_start, rollout_length - 1)
        test_end_idx = min(test_end, rollout_length - 1)
        if test_end_idx > test_start_idx:
            for t in range(test_start_idx, test_end_idx):
                X_test_all.append(rollout_activations[t])
                Y_test_all.append(rollout_positions[t])
    
    if len(X_train_all) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    X_train_array = np.stack(X_train_all)
    Y_train_array = np.stack(Y_train_all)
    X_test_array = np.stack(X_test_all) if len(X_test_all) > 0 else np.array([])
    Y_test_array = np.stack(Y_test_all) if len(Y_test_all) > 0 else np.array([])
    
    return X_train_array, Y_train_array, X_test_array, Y_test_array

def train_temporal_split_probes_by_world(world_data, train_steps=200, test_start=200, test_end=250):
    world_probes = {}
    world_results = {}
    print(f"\nTraining temporal-split next-state probes for each world (train: 0-{train_steps}, test: {test_start}-{test_end})...")
    for world_id in sorted(world_data.keys()):
        world_info = world_data[world_id]
        activations = world_info['activations']
        positions = world_info['positions']
        if len(activations) == 0:
            print(f"  World {world_id}: No data available")
            continue
        
        X_train_world, Y_train_world, X_test_world, Y_test_world = create_temporal_split_features(
            activations, positions, train_steps=train_steps, test_start=test_start, test_end=test_end)
        
        if len(X_train_world) < 4:
            print(f"  World {world_id}: Insufficient training data ({len(X_train_world)} samples)")
            continue
        
        if len(X_test_world) < 2:
            print(f"  World {world_id}: Insufficient test data ({len(X_test_world)} samples)")
            continue
        
        probe = Ridge().fit(X_train_world, Y_train_world)
        Y_pred_train = probe.predict(X_train_world)
        Y_pred_test = probe.predict(X_test_world)
        mse_train = mean_squared_error(Y_train_world, Y_pred_train)
        r2_train = r2_score(Y_train_world, Y_pred_train)
        mse_test = mean_squared_error(Y_test_world, Y_pred_test)
        r2_test = r2_score(Y_test_world, Y_pred_test)
        world_probes[world_id] = probe
        world_results[world_id] = {
            'mse_train': mse_train,
            'r2_train': r2_train,
            'mse_test': mse_test,
            'r2_test': r2_test,
            'n_train_samples': len(X_train_world),
            'n_test_samples': len(X_test_world),
            'feature_dim': X_train_world.shape[1]
        }
        print(f"  World {world_id}: R²={r2_test:.3f}, MSE={mse_test:.4f}, train_n={len(X_train_world)}, test_n={len(X_test_world)}, dim={X_train_world.shape[1]}")
    return world_probes, world_results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True)
    p.add_argument('--n-rollouts',  type=int, default=N_ROLLOUTS)
    p.add_argument('--max-steps',   type=int, default=500)
    p.add_argument('--train-steps', type=int, default=200, help='Number of steps to use for training (0 to train_steps)')
    p.add_argument('--test-start',  type=int, default=200, help='Start step for testing')
    p.add_argument('--test-end',    type=int, default=250, help='End step for testing')
    p.add_argument('--world-id',    type=int, default=WORLD_ID, help='World ID to use (or -1 for multiple worlds)')
    p.add_argument('--out',         type=str)
    args = p.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = "FG blue"
    sampler = FixedSampler.partial(formula)

    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[ENV]
    dummy  = make_env(ENV, sampler, sequence=False, render_mode=None)
    model  = build_model(dummy, status, cfg).eval()
    dummy.close()

    if args.world_id == -1:
        world_ids = list(range(10))
        print(f"Using multiple worlds: {world_ids}")
    else:
        world_ids = [args.world_id]
        print(f"Using single world: {args.world_id}")
    world_data = collect_hidden_and_next_positions_temporal_split(
        model, args.layer, sampler, n_rollouts=args.n_rollouts, max_steps=args.max_steps, world_ids=world_ids)
    if not world_data:
        print("Could not collect data for probe training. Exiting.")
        return
    world_probes, world_results = train_temporal_split_probes_by_world(world_data, 
                                                                      train_steps=args.train_steps, 
                                                                      test_start=args.test_start, 
                                                                      test_end=args.test_end)
    if world_results:
        worlds = sorted(world_results.keys())
        r2_scores = [world_results[world_id]['r2_test'] for world_id in worlds]
        mse_scores = [world_results[world_id]['mse_test'] for world_id in worlds]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        bars1 = ax1.bar(worlds, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_xlabel('World ID')
        ax1.set_ylabel('R² Score')
        ax1.set_title(f'Temporal Split Next-State Prediction R² by World (train: 0-{args.train_steps}, test: {args.test_start}-{args.test_end})')
        ax1.grid(True, alpha=0.3)
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{score:.3f}', ha='center', va='bottom')
        bars2 = ax2.bar(worlds, mse_scores, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('World ID')
        ax2.set_ylabel('MSE Score')
        ax2.set_title('Temporal Split Next-State Prediction MSE by World')
        ax2.grid(True, alpha=0.3)
        for bar, score in zip(bars2, mse_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{score:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        plot_file = args.out or f'temporal_split_probes_world{args.world_id}_train{args.train_steps}_test{args.test_start}-{args.test_end}.png'
        plt.savefig(plot_file, dpi=150)
        print(f"\nSaved temporal split probe analysis to {plot_file}")
        df = pd.DataFrame([
            {
                'world_id': world_id,
                'r2_train': world_results[world_id]['r2_train'],
                'r2_test': world_results[world_id]['r2_test'],
                'mse_train': world_results[world_id]['mse_train'],
                'mse_test': world_results[world_id]['mse_test'],
                'n_train_samples': world_results[world_id]['n_train_samples'],
                'n_test_samples': world_results[world_id]['n_test_samples'],
                'feature_dim': world_results[world_id]['feature_dim']
            }
            for world_id in sorted(world_results.keys())
        ])
        csv_file = f'temporal_split_probes_world{args.world_id}_train{args.train_steps}_test{args.test_start}-{args.test_end}.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved temporal split results to {csv_file}")
        
        # Create trajectory plots showing temporal split
        print(f"\nCreating temporal split trajectory plots for each world...")
        for world_id in sorted(world_results.keys()):
            world_info = world_data[world_id]
            activations = world_info['activations']
            positions = world_info['positions']
            zone_pos = world_info['zone_pos']
            probe = world_probes[world_id]
            
            # Find a rollout that has enough steps for both train and test
            selected_rollout = None
            for rollout_idx, (rollout_activations, rollout_positions) in enumerate(zip(activations, positions)):
                if len(rollout_positions) >= args.test_end:
                    selected_rollout = rollout_idx
                    break
            
            if selected_rollout is not None:
                rollout_activations = activations[selected_rollout]
                rollout_positions = positions[selected_rollout]
                
                # Generate predictions for the full rollout
                rollout_length = len(rollout_positions)
                predicted_positions = []
                for t in range(rollout_length):
                    pred_pos = probe.predict([rollout_activations[t]])[0]
                    predicted_positions.append(pred_pos)
                
                # Create plot showing train/test split
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                setup_axis(ax)
                draw_zones(ax, zone_pos)
                
                # Plot actual trajectory
                ax.plot(rollout_positions[:, 0], rollout_positions[:, 1], '-o', 
                       color='blue', markersize=3, alpha=0.7, label='Actual')
                
                # Plot predicted positions
                pred_pos_array = np.array(predicted_positions)
                ax.plot(pred_pos_array[:, 0], pred_pos_array[:, 1], '-o', 
                       color='red', markersize=4, alpha=0.8, label='Predicted')
                
                # Highlight train and test regions
                train_end = min(args.train_steps, rollout_length - 1)
                test_start = min(args.test_start, rollout_length - 1)
                test_end = min(args.test_end, rollout_length - 1)
                
                # Train region (green)
                ax.plot(rollout_positions[:train_end, 0], rollout_positions[:train_end, 1], 
                       'o', color='green', markersize=6, alpha=0.8, label='Train Region')
                
                # Test region (orange)
                if test_end > test_start:
                    ax.plot(rollout_positions[test_start:test_end, 0], rollout_positions[test_start:test_end, 1], 
                           'o', color='orange', markersize=6, alpha=0.8, label='Test Region')
                
                # Mark start and end points
                ax.plot(rollout_positions[0, 0], rollout_positions[0, 1], 'go', markersize=10, label='Start')
                ax.plot(rollout_positions[-1, 0], rollout_positions[-1, 1], 'ro', markersize=10, label='End')
                
                ax.set_title(f'World {world_id}, Temporal Split (Train: 0-{train_end}, Test: {test_start}-{test_end})')
                ax.legend()
                plt.tight_layout()
                trajectory_file = f'temporal_split_trajectories_world{world_id}_train{args.train_steps}_test{args.test_start}-{args.test_end}.png'
                plt.savefig(trajectory_file, dpi=150)
                print(f"  Saved temporal split trajectory plot for world {world_id} to {trajectory_file}")
                plt.close()
        
        r2_scores = [world_results[key]['r2_test'] for key in world_results.keys()]
        mse_scores = [world_results[key]['mse_test'] for key in world_results.keys()]
        print(f"\nSummary Statistics:")
        print(f"  Total worlds analyzed: {len(world_results)}")
        print(f"  Average R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
        print(f"  Average MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
        print(f"  Best R²: {np.max(r2_scores):.3f}")
        print(f"  Worst R²: {np.min(r2_scores):.3f}")
        feature_dims = [world_results[key]['feature_dim'] for key in world_results.keys()]
        print(f"  Feature dimensions: {np.mean(feature_dims):.0f} ± {np.std(feature_dims):.0f}")
    else:
        print("No temporal split probes could be trained (insufficient data).")

if __name__ == '__main__':
    main() 