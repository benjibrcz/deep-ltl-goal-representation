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

def collect_hidden_and_kth_next_positions_multi_layer(model, layer_names, sampler, n_rollouts=50, max_steps=200, k=1, world_id=0):
    # Initialize data structures for each layer
    layer_data = {layer: {'X_all': [], 'Y_all': [], 'step_idx_all': []} for layer in layer_names}
    zone_pos = None
    world_trajs = []
    
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    
    # Get modules for each layer
    modules = {}
    for layer_name in layer_names:
        try:
            modules[layer_name] = dict(model.named_modules())[layer_name]
        except KeyError:
            print(f"Warning: Layer '{layer_name}' not found in model")
            continue

    world_dir_path = f"{WORLD_DIR}"
    world_file = f"{world_dir_path}/world_info_{world_id}.pkl"
    if not os.path.exists(world_file):
        print(f"World file not found: {world_file}, skipping data collection.")
        env.close()
        return None, None, None

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

        # Initialize data collection for this rollout
        rollout_data = {layer: {'feats': [], 'positions': []} for layer in layer_names}
        
        # Set up hooks for all layers
        hooks = {}
        def make_hook(layer_name):
            def hook(m, inp, out):
                if layer_name == "ltl_rnn":
                    if isinstance(out, tuple):
                        h_n = out[1]  # Final hidden state
                        arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy()
                    else:
                        arr = out.detach().squeeze().cpu().numpy()
                else:
                    if hasattr(out, 'detach'):
                        arr = out.detach().squeeze().cpu().numpy()
                    else:
                        arr = out.squeeze().cpu().numpy()
                rollout_data[layer_name]['feats'].append(arr)
            return hook
        
        for layer_name in modules:
            hooks[layer_name] = modules[layer_name].register_forward_hook(make_hook(layer_name))
        
        done = False
        for step in range(max_steps):
            if done:
                break
            a = agent.get_action(obs, {}, deterministic=True).flatten()
            obs, _, done, _ = env.step(a)
            pos = env.agent_pos[:2].copy()
            for layer_name in layer_names:
                rollout_data[layer_name]['positions'].append(pos)
                if len(rollout_data[layer_name]['feats']) < len(rollout_data[layer_name]['positions']):
                    if len(rollout_data[layer_name]['feats']) > 0:
                        rollout_data[layer_name]['feats'].append(rollout_data[layer_name]['feats'][-1])
                    else:
                        # If no features yet, create a zero array
                        rollout_data[layer_name]['feats'].append(np.zeros(modules[layer_name].output_shape[0] if hasattr(modules[layer_name], 'output_shape') else 64))
        
        # Remove hooks
        for hook in hooks.values():
            hook.remove()
        
        # Process data for each layer
        for layer_name in layer_names:
            feats = rollout_data[layer_name]['feats']
            positions = rollout_data[layer_name]['positions']
            
            if len(feats) > len(positions):
                feats = feats[:len(positions)]
            
            # For k-th next-state prediction, X = feats[:-k], Y = positions[k:], step_idx = np.arange(len(positions)-k)
            if len(positions) > k:
                layer_data[layer_name]['X_all'].append(np.stack(feats[:-k]))
                layer_data[layer_name]['Y_all'].append(np.stack(positions[k:]))
                layer_data[layer_name]['step_idx_all'].append(np.arange(len(positions)-k))
        
        world_trajs.append(np.stack(positions))
        successful_rollouts += 1
    
    env.close()
    print(f"Successfully collected data from {successful_rollouts}/{n_rollouts} rollouts")
    return layer_data, zone_pos, world_trajs

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers',      nargs='+', default=['env_net', 'policy_encoder', 'ltl_rnn'], help='Layers to probe')
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
    layer_data, zone_pos, world_trajs = collect_hidden_and_kth_next_positions_multi_layer(
        model, args.layers, sampler, n_rollouts=args.n_rollouts, max_steps=args.max_steps, k=k, world_id=args.world_id)
    if layer_data is None or zone_pos is None or world_trajs is None:
        print("Could not collect data for probe training. Exiting.")
        return

    # Train probes for each layer
    results = {}
    for layer_name in args.layers:
        if layer_name not in layer_data or len(layer_data[layer_name]['X_all']) == 0:
            print(f"Skipping {layer_name} - no data collected")
            continue
            
        print(f"\n=== Training probe for {layer_name} ===")
        
        # Flatten data for this layer
        X = np.concatenate([x for x in layer_data[layer_name]['X_all'] if len(x) > 0], axis=0)
        Y = np.concatenate([y for y in layer_data[layer_name]['Y_all'] if len(y) > 0], axis=0)
        step_indices = np.concatenate([s for s in layer_data[layer_name]['step_idx_all'] if len(s) > 0], axis=0)

        # Train/test split
        X_train, X_test, Y_train, Y_test, step_train, step_test = train_test_split(
            X, Y, step_indices, test_size=0.2, random_state=SEED)

        # Train probe
        ridge = Ridge().fit(X_train, Y_train)
        Y_pred_train = ridge.predict(X_train)
        Y_pred_test = ridge.predict(X_test)
        mse_train = mean_squared_error(Y_train, Y_pred_train)
        r2_train = r2_score(Y_train, Y_pred_train)
        mse_test = mean_squared_error(Y_test, Y_pred_test)
        r2_test = r2_score(Y_test, Y_pred_test)
        
        print(f"  Train MSE: {mse_train:.4f}  Train R^2: {r2_train:.4f}")
        print(f"  Test  MSE: {mse_test:.4f}  Test  R^2: {r2_test:.4f}")
        
        # Per-step analysis
        per_step_results = []
        unique_steps = np.unique(step_test)
        for step in unique_steps:
            idx = (step_test == step)
            if np.sum(idx) < 2:
                continue
            mse = mean_squared_error(Y_test[idx], Y_pred_test[idx])
            r2 = r2_score(Y_test[idx], Y_pred_test[idx])
            per_step_results.append({'step': int(step), 'mse': mse, 'r2': r2, 'n_test': int(np.sum(idx))})
        
        results[layer_name] = {
            'mse_train': mse_train,
            'r2_train': r2_train,
            'mse_test': mse_test,
            'r2_test': r2_test,
            'per_step': per_step_results,
            'probe': ridge
        }

    # Print summary
    print(f"\n{'='*60}")
    print("LAYER COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Layer':<15} {'Train R²':<12} {'Test R²':<12} {'Test MSE':<12}")
    print("-" * 60)
    for layer_name, result in results.items():
        print(f"{layer_name:<15} {result['r2_train']:<12.4f} {result['r2_test']:<12.4f} {result['mse_test']:<12.4f}")

    # Plot comparison
    if len(results) > 1:
        plt.figure(figsize=(10, 6))
        layers = list(results.keys())
        r2_values = [results[layer]['r2_test'] for layer in layers]
        mse_values = [results[layer]['mse_test'] for layer in layers]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # R² comparison
        bars1 = ax1.bar(layers, r2_values, color=['blue', 'green', 'red', 'orange', 'purple'][:len(layers)])
        ax1.set_ylabel('Test R²')
        ax1.set_title('Layer Performance Comparison (R²)')
        ax1.set_ylim(min(r2_values) - 0.1, max(r2_values) + 0.1)
        for bar, value in zip(bars1, r2_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.3f}', 
                    ha='center', va='bottom')
        
        # MSE comparison
        bars2 = ax2.bar(layers, mse_values, color=['blue', 'green', 'red', 'orange', 'purple'][:len(layers)])
        ax2.set_ylabel('Test MSE')
        ax2.set_title('Layer Performance Comparison (MSE)')
        for bar, value in zip(bars2, mse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{value:.4f}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = args.out or f'agent_location_kth_next_state_multi_layer_k{k}_world{args.world_id}.png'
        plt.savefig(plot_file, dpi=150)
        print(f"\nSaved layer comparison plot to {plot_file}")

    # Save per-step results for each layer
    for layer_name, result in results.items():
        if result['per_step']:
            df = pd.DataFrame(result['per_step'])
            df = df.sort_values('step')
            csv_file = f'agent_location_kth_next_state_multi_layer_{layer_name}_k{k}_world{args.world_id}.csv'
            df.to_csv(csv_file, index=False)
            print(f"Saved per-step results for {layer_name} to {csv_file}")

if __name__ == '__main__':
    main() 