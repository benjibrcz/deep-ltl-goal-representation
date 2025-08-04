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
from visualize.zones      import draw_zones, draw_diamond, setup_axis

# ─── defaults ─────────────────────────────────────────────────────────────────
ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
N_WORLDS   = 5
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

def collect_hidden_and_next_positions_per_world(model, layer_name, sampler, n_worlds=10, max_steps=200):
    X_all, Y_all, zone_poss, world_trajs = [], [], [], []
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    world_dir_path = f"{WORLD_DIR}"
    if not os.path.exists(world_dir_path):
        print(f"World directory not found: {world_dir_path}, skipping data collection.")
        env.close()
        return None, None, None, None

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
        if len(positions) > 1:
            X_all.append(np.stack(feats[:-1]))
            Y_all.append(np.stack(positions[1:]))
            zone_poss.append(dict(env.zone_positions))
            world_trajs.append(np.stack(positions))
    env.close()
    return X_all, Y_all, zone_poss, world_trajs

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True)
    p.add_argument('--n-worlds',    type=int, default=N_WORLDS)
    p.add_argument('--max-steps',   type=int, default=500)
    p.add_argument('--out',         type=str, default='agent_location_next_state_trajectories.png')
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
    X_all, Y_all, zone_poss, world_trajs = collect_hidden_and_next_positions_per_world(
        model, args.layer, sampler, n_worlds=args.n_worlds, max_steps=args.max_steps)
    if X_all is None or Y_all is None or zone_poss is None or world_trajs is None:
        print("Could not collect data for probe training. Exiting.")
        return

    # Flatten for probe training
    X = np.concatenate([x for x in X_all if len(x) > 0], axis=0)
    Y = np.concatenate([y for y in Y_all if len(y) > 0], axis=0)

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

    # ── plot per-world trajectories ─────────────────────────────────────────────
    import math
    n_worlds = len(X_all)
    cols = 4 if n_worlds > 4 else n_worlds
    rows = math.ceil(n_worlds / cols)
    fig = plt.figure(figsize=(5*cols, 5*rows))
    for i, (X_seq, Y_seq, zone_pos, true_traj) in enumerate(zip(X_all, Y_all, zone_poss, world_trajs)):
        if len(X_seq) == 0 or len(Y_seq) == 0:
            continue
        # Predict next positions for this world
        Y_pred_seq = ridge.predict(X_seq)
        ax = fig.add_subplot(rows, cols, i+1)
        setup_axis(ax)
        draw_zones(ax, zone_pos)
        draw_diamond(ax, true_traj[0], color='orange')
        # True next-state trajectory (green)
        ax.plot(Y_seq[:,0], Y_seq[:,1], '-o', color='green', markersize=4, label='True Next')
        # Predicted next-state trajectory (red)
        ax.plot(Y_pred_seq[:,0], Y_pred_seq[:,1], '-o', color='red', markersize=4, alpha=0.7, label='Predicted Next')
        # Optionally, plot the actual agent trajectory (blue, faded)
        ax.plot(true_traj[:,0], true_traj[:,1], '--', color='blue', alpha=0.3, label='Agent Trajectory')
        ax.set_title(f'World {i}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved agent next location trajectory probe plot to {args.out}")

if __name__ == '__main__':
    main() 