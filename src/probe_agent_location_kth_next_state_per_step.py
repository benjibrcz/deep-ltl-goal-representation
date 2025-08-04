#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
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

# ─── defaults ─────────────────────────────────────────────────────────────────
ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
N_WORLDS   = 10
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

def collect_hidden_and_kth_next_positions_per_world(model, layer_name, sampler, n_worlds=10, max_steps=200, k=1):
    X_steps = []  # List of lists: X_steps[step] = [hidden at step across worlds]
    Y_steps = []  # List of lists: Y_steps[step] = [k-th next pos at step across worlds]
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
        if len(feats) > len(positions):
            feats = feats[:len(positions)]
        # For k-th next-state prediction, for each step, store (feat, pos) if possible
        for step in range(len(positions) - k):
            while len(X_steps) <= step:
                X_steps.append([])
                Y_steps.append([])
            X_steps[step].append(feats[step])
            Y_steps[step].append(positions[step + k])
    env.close()
    return X_steps, Y_steps

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True)
    p.add_argument('--n-worlds',    type=int, default=N_WORLDS)
    p.add_argument('--max-steps',   type=int, default=200)
    p.add_argument('--k',           type=int, default=1, help='Prediction horizon (k-th next step)')
    p.add_argument('--out',         type=str, default=None)
    args = p.parse_args()

    # seeds & sampler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = "FG blue" # "GF blue & GF green"
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
    X_steps, Y_steps = collect_hidden_and_kth_next_positions_per_world(
        model, args.layer, sampler, n_worlds=args.n_worlds, max_steps=args.max_steps, k=k)
    if X_steps is None or Y_steps is None:
        print("Could not collect data for probe training. Exiting.")
        return

    results = []
    for step, (X, Y) in enumerate(zip(X_steps, Y_steps)):
        X = np.array(X)
        Y = np.array(Y)
        if len(X) < 10:
            continue
        # Train/test split
        n = len(X)
        split = int(0.8 * n)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        ridge = Ridge().fit(X_train, Y_train)
        Y_pred = ridge.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        results.append({'step': step, 'mse': mse, 'r2': r2, 'n_test': len(X_test)})
        print(f"Step {step}: Test MSE={mse:.4f}, Test R^2={r2:.4f} (n_test={len(X_test)})")

    # Save results to CSV
    df = pd.DataFrame(results)
    out_file = args.out or f'agent_location_kth_next_state_per_step_k{k}.csv'
    df.to_csv(out_file, index=False)
    print(f"\nSaved per-step probe results to {out_file}")

    # Optionally plot R^2 over steps
    if len(results) > 0:
        plt.figure(figsize=(8,4))
        plt.plot(df['step'], df['r2'], marker='o')
        plt.xlabel('Step')
        plt.ylabel('Test R^2')
        plt.title(f'Per-step Probe R^2 (k={k})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = out_file.replace('.csv', '.png')
        plt.savefig(plot_file, dpi=150)
        print(f"Saved R^2 plot to {plot_file}")

if __name__ == '__main__':
    main() 