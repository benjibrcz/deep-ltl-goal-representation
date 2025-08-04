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

def collect_hidden_and_kth_next_positions_multiple_worlds(model, layer_name, sampler, n_rollouts=10, max_steps=200, k=1, world_ids=None):
    """
    Collect data from multiple worlds to get more samples for step-specific probes.
    """
    if world_ids is None:
        # Use worlds 0-9 by default
        world_ids = list(range(10))
    
    X_all, Y_all, zone_poss, world_trajs, step_idx_all, world_ids_all = [], [], [], [], [], []
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
            # For k-th next-state prediction, X = feats[:-k], Y = positions[k:], step_idx = np.arange(len(positions)-k)
            if len(positions) > k:
                X_all.append(np.stack(feats[:-k]))
                Y_all.append(np.stack(positions[k:]))
                step_idx_all.append(np.arange(len(positions)-k))
                world_trajs.append(np.stack(positions))
                world_ids_all.append(world_id)  # Track which world this rollout came from
                successful_rollouts += 1
                total_successful_rollouts += 1
        
        print(f"World {world_id}: {successful_rollouts}/{n_rollouts} successful rollouts")
    
    env.close()
    print(f"Total successfully collected data from {total_successful_rollouts} rollouts across {len(world_ids)} worlds")
    return X_all, Y_all, zone_pos, world_trajs, step_idx_all, world_ids_all

def train_step_specific_probes_by_world(X_all, Y_all, step_idx_all, world_ids_all, k=1):
    """
    Train separate probes for each (step, world) combination.
    Returns a dictionary mapping (step, world) -> trained probe model.
    """
    step_world_probes = {}
    step_world_results = {}
    
    # Group data by (step, world) combination
    step_world_data = {}
    
    # Group data by (step, world) using the world_ids_all list
    for i, (X_seq, Y_seq, step_indices) in enumerate(zip(X_all, Y_all, step_idx_all)):
        world_id = world_ids_all[i]  # Get the world ID for this rollout
        for x, y, step in zip(X_seq, Y_seq, step_indices):
            key = (step, world_id)
            if key not in step_world_data:
                step_world_data[key] = {'X': [], 'Y': []}
            step_world_data[key]['X'].append(x)
            step_world_data[key]['Y'].append(y)
    
    print(f"\nTraining step-specific probes for each world...")
    print(f"Total successful rollouts: {len(X_all)}")
    print(f"Step-world combinations with data: {len(step_world_data)}")
    
    for (step, world_id) in sorted(step_world_data.keys()):
        X_step_world = np.array(step_world_data[(step, world_id)]['X'])
        Y_step_world = np.array(step_world_data[(step, world_id)]['Y'])
        
        if len(X_step_world) < 2:  # Need at least 2 samples for train/test
            print(f"  Step {step}, World {world_id}: Skipping (only {len(X_step_world)} samples)")
            continue
            
        # Train/test split for this step-world combination
        if len(X_step_world) >= 4:
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_step_world, Y_step_world, test_size=0.2, random_state=SEED)
        else:
            # If we have very few samples, use all data for training and skip test evaluation
            X_train, Y_train = X_step_world, Y_step_world
            X_test, Y_test = X_step_world[:1], Y_step_world[:1]  # Dummy for evaluation
        
        # Train probe for this step-world combination
        probe = Ridge().fit(X_train, Y_train)
        Y_pred_train = probe.predict(X_train)
        Y_pred_test = probe.predict(X_test)
        
        # Evaluate
        mse_train = mean_squared_error(Y_train, Y_pred_train)
        r2_train = r2_score(Y_train, Y_pred_train)
        mse_test = mean_squared_error(Y_test, Y_pred_test)
        r2_test = r2_score(Y_test, Y_pred_test)
        
        step_world_probes[(step, world_id)] = probe
        step_world_results[(step, world_id)] = {
            'mse_train': mse_train,
            'r2_train': r2_train,
            'mse_test': mse_test,
            'r2_test': r2_test,
            'n_samples': len(X_step_world)
        }
        
        print(f"  Step {step}, World {world_id}: R²={r2_test:.3f}, MSE={mse_test:.4f}, n={len(X_step_world)}")
    
    return step_world_probes, step_world_results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True)
    p.add_argument('--n-rollouts',  type=int, default=N_ROLLOUTS)
    p.add_argument('--max-steps',   type=int, default=200)
    p.add_argument('--k',           type=int, default=1, help='Prediction horizon (k-th next step)')
    p.add_argument('--world-id',    type=int, default=WORLD_ID, help='World ID to use (or -1 for multiple worlds)')
    p.add_argument('--out',         type=str)
    args = p.parse_args()

    # seeds & sampler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = "FG blue"
    sampler = FixedSampler.partial(formula)
    k = args.k

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
        # Use multiple worlds
        world_ids = list(range(10))  # Use worlds 0-9
        print(f"Using multiple worlds: {world_ids}")
    else:
        # Use single world
        world_ids = [args.world_id]
        print(f"Using single world: {args.world_id}")
    
    X_all, Y_all, zone_pos, world_trajs, step_idx_all, world_ids_all = collect_hidden_and_kth_next_positions_multiple_worlds(
        model, args.layer, sampler, n_rollouts=args.n_rollouts, max_steps=args.max_steps, k=k, world_ids=world_ids)
    if X_all is None or Y_all is None or zone_pos is None or world_trajs is None or step_idx_all is None or world_ids_all is None:
        print("Could not collect data for probe training. Exiting.")
        return

    # ── train step-specific probes for each world ─────────────────────────────
    step_world_probes, step_world_results = train_step_specific_probes_by_world(X_all, Y_all, step_idx_all, world_ids_all, k=k)

    # ── create summary plots ───────────────────────────────────────────────────
    if step_world_results:
        # Create a heatmap of R² scores by step and world
        steps = sorted(set(key[0] for key in step_world_results.keys()))
        worlds = sorted(set(key[1] for key in step_world_results.keys()))
        
        # Create R² matrix
        r2_matrix = np.full((len(steps), len(worlds)), np.nan)
        for (step, world_id) in step_world_results.keys():
            step_idx = steps.index(step)
            world_idx = worlds.index(world_id)
            r2_matrix[step_idx, world_idx] = step_world_results[(step, world_id)]['r2_test']
        
        # Create MSE matrix
        mse_matrix = np.full((len(steps), len(worlds)), np.nan)
        for (step, world_id) in step_world_results.keys():
            step_idx = steps.index(step)
            world_idx = worlds.index(world_id)
            mse_matrix[step_idx, world_idx] = step_world_results[(step, world_id)]['mse_test']
        
        # Plot heatmaps
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # R² heatmap
        im1 = ax1.imshow(r2_matrix, cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
        ax1.set_xlabel('World ID')
        ax1.set_ylabel('Step')
        ax1.set_title(f'R² Scores by Step and World (k={k})')
        ax1.set_xticks(range(len(worlds)))
        ax1.set_xticklabels(worlds)
        ax1.set_yticks(range(0, len(steps), 10))
        ax1.set_yticklabels([steps[i] for i in range(0, len(steps), 10)])
        plt.colorbar(im1, ax=ax1)
        
        # MSE heatmap
        im2 = ax2.imshow(mse_matrix, cmap='Reds', aspect='auto')
        ax2.set_xlabel('World ID')
        ax2.set_ylabel('Step')
        ax2.set_title('MSE Scores by Step and World')
        ax2.set_xticks(range(len(worlds)))
        ax2.set_xticklabels(worlds)
        ax2.set_yticks(range(0, len(steps), 10))
        ax2.set_yticklabels([steps[i] for i in range(0, len(steps), 10)])
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plot_file = args.out or f'step_world_specific_probes_k{k}_world{args.world_id}.png'
        plt.savefig(plot_file, dpi=150)
        print(f"\nSaved step-world specific probe analysis to {plot_file}")
        
        # Save results to CSV
        df = pd.DataFrame([
            {
                'step': step,
                'world_id': world_id,
                'r2_train': step_world_results[(step, world_id)]['r2_train'],
                'r2_test': step_world_results[(step, world_id)]['r2_test'],
                'mse_train': step_world_results[(step, world_id)]['mse_train'],
                'mse_test': step_world_results[(step, world_id)]['mse_test'],
                'n_samples': step_world_results[(step, world_id)]['n_samples']
            }
            for (step, world_id) in sorted(step_world_results.keys())
        ])
        csv_file = f'step_world_specific_probes_k{k}_world{args.world_id}.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved step-world specific results to {csv_file}")
        
        # Print summary statistics
        r2_scores = [step_world_results[key]['r2_test'] for key in step_world_results.keys()]
        mse_scores = [step_world_results[key]['mse_test'] for key in step_world_results.keys()]
        
        print(f"\nSummary Statistics:")
        print(f"  Total step-world combinations analyzed: {len(step_world_results)}")
        print(f"  Average R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
        print(f"  Average MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
        print(f"  Best R²: {np.max(r2_scores):.3f}")
        print(f"  Worst R²: {np.min(r2_scores):.3f}")
        
        # Print world-specific statistics
        print(f"\nWorld-specific statistics:")
        for world_id in worlds:
            world_r2 = [step_world_results[key]['r2_test'] for key in step_world_results.keys() if key[1] == world_id]
            if world_r2:
                print(f"  World {world_id}: R² = {np.mean(world_r2):.3f} ± {np.std(world_r2):.3f}")
    else:
        print("No step-world specific probes could be trained (insufficient data).")

if __name__ == '__main__':
    main() 