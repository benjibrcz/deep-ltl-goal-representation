#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

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
N_WORLDS   = 50
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

def collect_hidden_and_next_positions(model, layer_name, sampler, n_worlds=10, max_steps=200):
    X, Y = [], []
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    world_dir_path = f"{WORLD_DIR}"
    if not os.path.exists(world_dir_path):
        print(f"World directory not found: {world_dir_path}, skipping data collection.")
        env.close()
        return None, None

    for i in trange(n_worlds, desc="Collect worlds"):
        world_file = f"{world_dir_path}/world_info_{i}.pkl"
        if not os.path.exists(world_file):
            continue
        env.load_world_info(world_file)
        obs = env.reset(seed=SEED+i)
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
        # For next-state prediction, X = feats[:-1], Y = positions[1:]
        if len(feats) > len(positions):
            feats = feats[:len(positions)]
        if len(positions) > 1:
            X.extend(feats[:-1])
            Y.extend(positions[1:])
    env.close()
    return np.stack(X), np.stack(Y)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True)
    p.add_argument('--n-worlds',    type=int, default=10)
    p.add_argument('--max-steps',   type=int, default=200)
    p.add_argument('--out',         type=str, default='agent_location_next_state_probe.png')
    args = p.parse_args()

    # seeds & sampler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = "GF blue & GF green"
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
    X, Y = collect_hidden_and_next_positions(model, args.layer, sampler, n_worlds=args.n_worlds, max_steps=args.max_steps)
    if X is None:
        print("Could not collect data for probe training. Exiting.")
        return

    # ── train/test split ───────────────────────────────────────────────────────
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)

    # ── train probe ──────────────────────────────────────────────────────────────
    ridge = Ridge().fit(X_train, Y_train)
    Y_pred_train = ridge.predict(X_train)
    Y_pred_test = ridge.predict(X_test)
    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)
    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2_test = r2_score(Y_test, Y_pred_test)
    print(f"\nAgent next location probe results:")
    print(f"  Train MSE: {mse_train:.4f}  Train R^2: {r2_train:.4f}")
    print(f"  Test  MSE: {mse_test:.4f}  Test  R^2: {r2_test:.4f}")

    # Ensure Y_test and Y_pred_test are numpy arrays for plotting
    Y_test = np.array(Y_test)
    Y_pred_test = np.array(Y_pred_test)

    # ── plot true vs predicted positions (test set) ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(Y_test[:,0], Y_test[:,1], c='k', s=8, label='True (Test)')
    axes[0].scatter(Y_pred_test[:,0], Y_pred_test[:,1], c='r', s=8, alpha=0.5, label='Predicted (Test)')
    axes[0].set_title('Scatter: True vs Predicted Next Agent Positions (Test)')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # Trajectory plot for a random test trajectory
    n = min(len(Y_test), args.max_steps)
    axes[1].plot(Y_test[:n,0], Y_test[:n,1], '-o', color='k', markersize=3, label='True Trajectory (Test)')
    axes[1].plot(Y_pred_test[:n,0], Y_pred_test[:n,1], '-o', color='r', markersize=3, alpha=0.7, label='Predicted Trajectory (Test)')
    axes[1].set_title('Trajectory: True vs Predicted Next (Test)')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved agent next location probe plot to {args.out}")

if __name__ == '__main__':
    main() 