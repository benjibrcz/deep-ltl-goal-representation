#!/usr/bin/env python3
import os, sys, random
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))
from utils.model_store    import ModelStore
from model.model          import build_model
from config               import model_configs
from ltl                  import FixedSampler
from envs                 import make_env
from sequence.search      import ExhaustiveSearch
from model.agent          import Agent
from envs.flatworld       import FlatWorld

ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
N_WORLDS   = 10
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"

LAYER_NAMES = [
    'mlp.0',   # first linear
    'mlp.2',   # second linear
    'mlp',     # full MLP output
    '',        # env_net output (same as mlp)
]

LABELS = [
    'env_net.mlp.0',
    'env_net.mlp.2',
    'env_net.mlp',
    'env_net',
]

ZONE_NAMES = [c.color for c in FlatWorld.CIRCLES]
ZONE_CENTERS = [c.center for c in FlatWorld.CIRCLES]
N_ZONES = len(ZONE_NAMES)


def collect_hidden_and_targets(model, layer_name, sampler, n_worlds=10, max_steps=200):
    X, D, U = [], [], []
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.env_net.named_modules())[layer_name] if layer_name else model.env_net
    world_dir_path = f"{WORLD_DIR}"
    for i in trange(n_worlds, desc=f"Collect worlds ({layer_name or 'env_net'})"):
        world_file = f"{world_dir_path}/world_info_{i}.pkl"
        if not os.path.exists(world_file):
            continue
        env.load_world_info(world_file)
        obs = env.reset(seed=SEED+i)
        agent.reset()
        feats = []
        dists = []
        dirs = []
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
            pos = env.agent_pos[:2].copy()
            # Compute distances and directions to each zone
            d = [np.linalg.norm(pos - zc) for zc in ZONE_CENTERS]
            u = [(zc - pos) / (np.linalg.norm(zc - pos) + 1e-8) for zc in ZONE_CENTERS]
            dists.append(d)
            dirs.append(np.concatenate(u))
            if len(feats) < len(dists):
                feats.append(feats[-1])
        h.remove()
        if len(feats) > len(dists):
            feats = feats[:len(dists)]
        X.extend(feats)
        D.extend(dists)
        U.extend(dirs)
    env.close()
    return np.stack(X), np.stack(D), np.stack(U)

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = "GF blue & GF green"
    sampler = FixedSampler.partial(formula)
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[ENV]
    dummy  = make_env(ENV, sampler, sequence=False, render_mode=None)
    model  = build_model(dummy, status, cfg).eval()
    dummy.close()

    results_dist = []
    results_dir = []
    for layer_name, label in zip(LAYER_NAMES, LABELS):
        print(f"\n=== Probing {label} ===")
        X, D, U = collect_hidden_and_targets(model, layer_name, sampler, n_worlds=N_WORLDS, max_steps=200)
        # Distance probe
        ridge_d = Ridge().fit(X, D)
        D_pred = ridge_d.predict(X)
        mse_d = mean_squared_error(D, D_pred)
        r2_d = r2_score(D, D_pred)
        print(f"  [Distance]   MSE: {mse_d:.4f}   R^2: {r2_d:.4f}")
        results_dist.append((label, mse_d, r2_d))
        # Barplot of weights (distance)
        Wd = ridge_d.coef_
        fig, ax = plt.subplots(figsize=(12, 4))
        for i, zn in enumerate(ZONE_NAMES):
            weights = Wd[i]
            ax.bar(np.arange(len(weights)) + i*0.1, weights, width=0.1, alpha=0.7, label=zn)
        ax.set_ylabel('Probe weight (distance)')
        ax.set_xlabel('Hidden state dimension')
        ax.set_title(f'{label}: Distance probe weights')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"zone_distance_probe_weights_{label.replace('.', '_')}.png", dpi=150)
        plt.close(fig)
        # Direction probe
        ridge_u = Ridge().fit(X, U)
        U_pred = ridge_u.predict(X)
        mse_u = mean_squared_error(U, U_pred)
        r2_u = r2_score(U, U_pred)
        print(f"  [Direction]  MSE: {mse_u:.4f}   R^2: {r2_u:.4f}")
        results_dir.append((label, mse_u, r2_u))
        # Barplot of weights (direction)
        Wu = ridge_u.coef_
        fig, ax = plt.subplots(figsize=(12, 4))
        for i, zn in enumerate(ZONE_NAMES):
            weights = Wu[2*i:2*i+2].ravel()
            ax.bar(np.arange(len(weights)) + i*0.1, weights, width=0.1, alpha=0.7, label=zn)
        ax.set_ylabel('Probe weight (direction)')
        ax.set_xlabel('Hidden state dimension')
        ax.set_title(f'{label}: Direction probe weights')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"zone_direction_probe_weights_{label.replace('.', '_')}.png", dpi=150)
        plt.close(fig)
    # Print summary tables
    print("\n=== Summary: Distance ===")
    print(f"{'Layer':<20} | {'MSE':>8} | {'R^2':>8}")
    print("-"*42)
    for label, mse, r2 in results_dist:
        print(f"{label:<20} | {mse:8.4f} | {r2:8.4f}")
    print("\n=== Summary: Direction ===")
    print(f"{'Layer':<20} | {'MSE':>8} | {'R^2':>8}")
    print("-"*42)
    for label, mse, r2 in results_dir:
        print(f"{label:<20} | {mse:8.4f} | {r2:8.4f}")

if __name__ == '__main__':
    main() 