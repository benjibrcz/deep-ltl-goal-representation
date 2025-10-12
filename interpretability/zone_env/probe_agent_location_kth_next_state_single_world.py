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

    # Flatten for probe training
    X = np.concatenate([x for x in X_all if len(x) > 0], axis=0)
    Y = np.concatenate([y for y in Y_all if len(y) > 0], axis=0)
    step_indices = np.concatenate([s for s in step_idx_all if len(s) > 0], axis=0)

    # ── train/test split ───────────────────────────────────────────────────────
    X_train, X_test, Y_train, Y_test, step_train, step_test = train_test_split(
        X, Y, step_indices, test_size=0.2, random_state=SEED)

    # ── train probe ──────────────────────────────────────────────────────────────
    ridge = Ridge().fit(X_train, Y_train)
    Y_pred_train = ridge.predict(X_train)
    Y_pred_test = ridge.predict(X_test)
    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)
    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2_test = r2_score(Y_test, Y_pred_test)
    print(f"\nAgent k-th next location probe results (k={k}, world {args.world_id}):")
    print(f"  Train MSE: {mse_train:.4f}  Train R^2: {r2_train:.4f}")
    print(f"  Test  MSE: {mse_test:.4f}  Test  R^2: {r2_test:.4f}")

    # ── plot per-rollout trajectories ─────────────────────────────────────────────
    import math
    n_rollouts = len(X_all)
    cols = 4 if n_rollouts > 4 else n_rollouts
    rows = math.ceil(n_rollouts / cols)
    fig = plt.figure(figsize=(5*cols, 5*rows))
    for i, (X_seq, Y_seq, true_traj) in enumerate(zip(X_all, Y_all, world_trajs)):
        if len(X_seq) == 0 or len(Y_seq) == 0:
            continue
        Y_pred_seq = ridge.predict(X_seq)
        ax = fig.add_subplot(rows, cols, i+1)
        setup_axis(ax)
        draw_zones(ax, zone_pos)
        draw_diamond(ax, true_traj[0], color='orange')
        # True k-th next-state trajectory (green)
        ax.plot(Y_seq[:,0], Y_seq[:,1], '-o', color='green', markersize=4, label=f'True Next (k={k})')
        # Predicted k-th next-state trajectory (red)
        ax.plot(Y_pred_seq[:,0], Y_pred_seq[:,1], '-o', color='red', markersize=4, alpha=0.7, label=f'Predicted Next (k={k})')
        # Actual agent trajectory (blue, faded)
        ax.plot(true_traj[:,0], true_traj[:,1], '--', color='blue', alpha=0.3, label='Agent Trajectory')
        ax.set_title(f'Rollout {i}')
        ax.legend()
    plt.tight_layout()
    out_file = args.out or f'agent_location_kth_next_state_single_world_k{k}_world{args.world_id}.png'
    plt.savefig(out_file, dpi=150)
    print(f"\nSaved agent k-th next location trajectory probe plot to {out_file}")

    # ── plot R^2 by step index ─────────────────────────────────────────────
    per_step_results = []
    unique_steps = np.unique(step_test)
    for step in unique_steps:
        idx = (step_test == step)
        if np.sum(idx) < 2:
            continue
        mse = mean_squared_error(Y_test[idx], Y_pred_test[idx])
        r2 = r2_score(Y_test[idx], Y_pred_test[idx])
        per_step_results.append({'step': int(step), 'mse': mse, 'r2': r2, 'n_test': int(np.sum(idx))})
    df = pd.DataFrame(per_step_results)
    df = df.sort_values('step')
    plot_file = f'agent_location_kth_next_state_single_world_r2_by_step_k{k}_world{args.world_id}.png'
    # Clip R^2 values for plotting
    df['r2_clipped'] = df['r2'].clip(lower=-1)
    plt.figure(figsize=(8,4))
    plt.plot(df['step'], df['r2_clipped'], marker='o')
    plt.xlabel('Step')
    plt.ylabel('Test R^2 (clipped at -1)')
    plt.title(f'Single World Probe R^2 by Step (k={k}, world {args.world_id})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    print(f"Saved R^2 by step plot to {plot_file}")
    csv_file = f'agent_location_kth_next_state_single_world_r2_by_step_k{k}_world{args.world_id}.csv'
    df.to_csv(csv_file, index=False)
    print(f"Saved per-step results to {csv_file}")

if __name__ == '__main__':
    main() 