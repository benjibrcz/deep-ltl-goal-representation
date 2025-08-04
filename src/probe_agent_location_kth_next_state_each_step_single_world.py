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
N_ROLLOUTS = 50  # Number of different starting positions
WORLD_ID   = 0   # Which world to use
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

def collect_hidden_and_kth_next_positions_single_world(model, layer_name, sampler, n_rollouts=50, max_steps=200, k=1, world_id=0):
    X_all, Y_all, zone_poss, world_trajs, step_idx_all = [], [], [], [], []
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    world_dir_path = f"{WORLD_DIR}"
    world_file = f"{world_dir_path}/world_info_{world_id}.pkl"
    if not os.path.exists(world_file):
        print(f"World file not found: {world_file}, skipping data collection.")
        env.close()
        return None, None, None, None, None

    # Load the single world
    env.load_world_info(world_file)
    zone_pos = dict(env.zone_positions)

    successful_rollouts = 0
    max_attempts_per_rollout = 10
    
    for rollout_idx in trange(n_rollouts, desc=f"Rollouts for world {world_id}"):
        # Try different seeds until we find a valid starting position
        for attempt in range(max_attempts_per_rollout):
            try:
                # Reset with different seed to get different starting position
                obs = env.reset(seed=SEED + rollout_idx * max_attempts_per_rollout + attempt)
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
            successful_rollouts += 1
    env.close()
    print(f"Successfully collected data from {successful_rollouts}/{n_rollouts} rollouts")
    return X_all, Y_all, zone_pos, world_trajs, step_idx_all

def train_step_specific_probes(X_all, Y_all, step_idx_all, k=1):
    """
    Train separate probes for each step index.
    Returns a dictionary mapping step -> trained probe model.
    """
    step_probes = {}
    step_results = {}
    
    # Group data by step index
    step_data = {}
    for X_seq, Y_seq, step_indices in zip(X_all, Y_all, step_idx_all):
        for i, (x, y, step) in enumerate(zip(X_seq, Y_seq, step_indices)):
            if step not in step_data:
                step_data[step] = {'X': [], 'Y': []}
            step_data[step]['X'].append(x)
            step_data[step]['Y'].append(y)
    
    print(f"\nTraining separate probes for each step...")
    for step in sorted(step_data.keys()):
        X_step = np.array(step_data[step]['X'])
        Y_step = np.array(step_data[step]['Y'])
        
        if len(X_step) < 10:  # Need enough samples for train/test split
            print(f"  Step {step}: Skipping (only {len(X_step)} samples)")
            continue
            
        # Train/test split for this step
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_step, Y_step, test_size=0.2, random_state=SEED)
        
        # Train probe for this step
        probe = Ridge().fit(X_train, Y_train)
        Y_pred_train = probe.predict(X_train)
        Y_pred_test = probe.predict(X_test)
        
        # Evaluate
        mse_train = mean_squared_error(Y_train, Y_pred_train)
        r2_train = r2_score(Y_train, Y_pred_train)
        mse_test = mean_squared_error(Y_test, Y_pred_test)
        r2_test = r2_score(Y_test, Y_pred_test)
        
        step_probes[step] = probe
        step_results[step] = {
            'mse_train': mse_train,
            'r2_train': r2_train,
            'mse_test': mse_test,
            'r2_test': r2_test,
            'n_samples': len(X_step)
        }
        
        print(f"  Step {step}: R²={r2_test:.3f}, MSE={mse_test:.4f}, n={len(X_step)}")
    
    return step_probes, step_results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True)
    p.add_argument('--n-rollouts',  type=int, default=N_ROLLOUTS)
    p.add_argument('--max-steps',   type=int, default=200)
    p.add_argument('--k',           type=int, default=1, help='Prediction horizon (k-th next step)')
    p.add_argument('--world-id',    type=int, default=WORLD_ID, help='World ID to use')
    p.add_argument('--out',         type=str)
    args = p.parse_args()

    # seeds & sampler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = "GF blue & GF green"
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
    X_all, Y_all, zone_pos, world_trajs, step_idx_all = collect_hidden_and_kth_next_positions_single_world(
        model, args.layer, sampler, n_rollouts=args.n_rollouts, max_steps=args.max_steps, k=k, world_id=args.world_id)
    if X_all is None or Y_all is None or zone_pos is None or world_trajs is None or step_idx_all is None:
        print("Could not collect data for probe training. Exiting.")
        return

    # ── train step-specific probes ─────────────────────────────────────────────
    step_probes, step_results = train_step_specific_probes(X_all, Y_all, step_idx_all, k=k)

    # ── create summary plots ───────────────────────────────────────────────────
    if step_results:
        # Plot R² by step
        steps = sorted(step_results.keys())
        r2_scores = [step_results[step]['r2_test'] for step in steps]
        mse_scores = [step_results[step]['mse_test'] for step in steps]
        n_samples = [step_results[step]['n_samples'] for step in steps]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # R² plot
        ax1.plot(steps, r2_scores, 'o-', color='blue')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Test R²')
        ax1.set_title(f'Step-Specific Probe Performance (k={k}, world {args.world_id})')
        ax1.grid(True, alpha=0.3)
        
        # MSE plot
        ax2.plot(steps, mse_scores, 'o-', color='red')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Test MSE')
        ax2.set_title('Mean Squared Error by Step')
        ax2.grid(True, alpha=0.3)
        
        # Sample count plot
        ax3.plot(steps, n_samples, 'o-', color='green')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Number of Samples')
        ax3.set_title('Number of Samples per Step')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = args.out or f'step_specific_probes_k{k}_world{args.world_id}.png'
        plt.savefig(plot_file, dpi=150)
        print(f"\nSaved step-specific probe analysis to {plot_file}")
        
        # Save results to CSV
        df = pd.DataFrame([
            {
                'step': step,
                'r2_train': step_results[step]['r2_train'],
                'r2_test': step_results[step]['r2_test'],
                'mse_train': step_results[step]['mse_train'],
                'mse_test': step_results[step]['mse_test'],
                'n_samples': step_results[step]['n_samples']
            }
            for step in sorted(step_results.keys())
        ])
        csv_file = f'step_specific_probes_k{k}_world{args.world_id}.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved step-specific results to {csv_file}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Total steps analyzed: {len(step_results)}")
        print(f"  Average R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
        print(f"  Average MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
        print(f"  Best R² at step {steps[np.argmax(r2_scores)]}: {np.max(r2_scores):.3f}")
        print(f"  Worst R² at step {steps[np.argmin(r2_scores)]}: {np.min(r2_scores):.3f}")
    else:
        print("No step-specific probes could be trained (insufficient data).")

if __name__ == '__main__':
    main() 