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
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store    import ModelStore
from model.model          import build_model
from config               import model_configs
from ltl                  import FixedSampler
from envs                 import make_env
from sequence.search      import ExhaustiveSearch
from model.agent          import Agent
from visualize.zones      import draw_zones, draw_diamond, setup_axis

# ─── defaults ─────────────────────────────────────────────────────────────────
ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
N_ROLLOUTS = 10  # Number of different starting positions per world
WORLD_ID   = 0   # Which world to use (or -1 for multiple worlds)
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

def collect_hidden_and_positions_multiple_worlds(model, layer_name, sampler, n_rollouts=10, max_steps=200, world_ids=None):
    """
    Collect data from multiple worlds for cumulative step prediction.
    Returns activations and positions for each step.
    """
    if world_ids is None:
        # Use worlds 0-9 by default
        world_ids = list(range(10))
    
    world_data = {}  # {world_id: {'activations': [...], 'positions': [...], 'rollout_lengths': [...]}}
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

        # Load the world
        env.load_world_info(world_file)
        zone_pos = dict(env.zone_positions)

        successful_rollouts = 0
        max_attempts_per_rollout = 10
        
        world_activations = []
        world_positions = []
        rollout_lengths = []
        
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
            positions = []
            def grab(m, inp, out):
                x = out[1] if isinstance(out, tuple) else out
                feats.append(x.detach().cpu().numpy().ravel())
            h = module.register_forward_hook(grab)
            done = False
            for step in range(max_steps):
                if done:
                    break
                a = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, _ = env.step(a)
                positions.append(env.agent_pos[:2].copy())
                if len(feats) < len(positions):
                    feats.append(feats[-1])
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

def create_fixed_window_features(activations, positions, window_size=10):
    """
    For each step >= window_size, use activations from the previous window_size steps to predict the agent's position at the next step.
    """
    X_all, Y_all, step_info = [], [], []
    for rollout_idx, (rollout_activations, rollout_positions) in enumerate(zip(activations, positions)):
        rollout_length = len(rollout_positions)
        for t in range(window_size, rollout_length):
            # Use activations from steps [t-window_size, ..., t-1] to predict position at step t
            window_activations = rollout_activations[t-window_size:t]
            target_position = rollout_positions[t]  # Position at step t
            # Flatten and concatenate
            window_features = np.concatenate([act.flatten() for act in window_activations])
            X_all.append(window_features)
            Y_all.append(target_position)
            step_info.append({
                'rollout_idx': rollout_idx,
                'target_step': t,
                'window_size': window_size,
                'feature_dim': len(window_features)
            })
    if len(X_all) == 0:
        return np.array([]), np.array([]), []
    X_array = np.stack(X_all)
    Y_array = np.stack(Y_all)
    return X_array, Y_array, step_info

def train_fixed_window_probes_by_world(world_data, window_size=10):
    world_probes = {}
    world_results = {}
    print(f"\nTraining fixed-window probes for each world (window size = {window_size})...")
    for world_id in sorted(world_data.keys()):
        world_info = world_data[world_id]
        activations = world_info['activations']
        positions = world_info['positions']
        if len(activations) == 0:
            print(f"  World {world_id}: No data available")
            continue
        X_world, Y_world, step_info = create_fixed_window_features(activations, positions, window_size=window_size)
        if len(X_world) < 4:
            print(f"  World {world_id}: Insufficient data ({len(X_world)} samples)")
            continue
        X_train, X_test, Y_train, Y_test = train_test_split(X_world, Y_world, test_size=0.2, random_state=SEED)
        probe = Ridge().fit(X_train, Y_train)
        Y_pred_train = probe.predict(X_train)
        Y_pred_test = probe.predict(X_test)
        mse_train = mean_squared_error(Y_train, Y_pred_train)
        r2_train = r2_score(Y_train, Y_pred_train)
        mse_test = mean_squared_error(Y_test, Y_pred_test)
        r2_test = r2_score(Y_test, Y_pred_test)
        world_probes[world_id] = probe
        world_results[world_id] = {
            'mse_train': mse_train,
            'r2_train': r2_train,
            'mse_test': mse_test,
            'r2_test': r2_test,
            'n_samples': len(X_world),
            'feature_dim': X_world.shape[1],
            'step_info': step_info
        }
        print(f"  World {world_id}: R²={r2_test:.3f}, MSE={mse_test:.4f}, n={len(X_world)}, dim={X_world.shape[1]}")
    return world_probes, world_results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True)
    p.add_argument('--n-rollouts',  type=int, default=N_ROLLOUTS)
    p.add_argument('--max-steps',   type=int, default=200)
    p.add_argument('--window-size', type=int, default=10, help='Number of previous steps to use')
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
    world_data = collect_hidden_and_positions_multiple_worlds(
        model, args.layer, sampler, n_rollouts=args.n_rollouts, max_steps=args.max_steps, world_ids=world_ids)
    if not world_data:
        print("Could not collect data for probe training. Exiting.")
        return

    # ── train fixed-window probes for each world ─────────────────────────────
    world_probes, world_results = train_fixed_window_probes_by_world(
        world_data, window_size=args.window_size)

    # ── create summary plots ───────────────────────────────────────────────────
    if world_results:
        worlds = sorted(world_results.keys())
        r2_scores = [world_results[world_id]['r2_test'] for world_id in worlds]
        mse_scores = [world_results[world_id]['mse_test'] for world_id in worlds]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        bars1 = ax1.bar(worlds, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_xlabel('World ID')
        ax1.set_ylabel('R² Score')
        ax1.set_title(f'Fixed-Window Prediction R² by World (window size = {args.window_size})')
        ax1.grid(True, alpha=0.3)
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom')
        bars2 = ax2.bar(worlds, mse_scores, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('World ID')
        ax2.set_ylabel('MSE Score')
        ax2.set_title('Fixed-Window Prediction MSE by World')
        ax2.grid(True, alpha=0.3)
        for bar, score in zip(bars2, mse_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        plot_file = args.out or f'fixed_window_probes_world{args.world_id}_window{args.window_size}.png'
        plt.savefig(plot_file, dpi=150)
        print(f"\nSaved fixed-window probe analysis to {plot_file}")
        df = pd.DataFrame([
            {
                'world_id': world_id,
                'r2_train': world_results[world_id]['r2_train'],
                'r2_test': world_results[world_id]['r2_test'],
                'mse_train': world_results[world_id]['mse_train'],
                'mse_test': world_results[world_id]['mse_test'],
                'n_samples': world_results[world_id]['n_samples'],
                'feature_dim': world_results[world_id]['feature_dim']
            }
            for world_id in sorted(world_results.keys())
        ])
        csv_file = f'fixed_window_probes_world{args.world_id}_window{args.window_size}.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved fixed-window probe results to {csv_file}")
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
        print("No fixed-window probes could be trained (insufficient data).")

if __name__ == '__main__':
    main() 