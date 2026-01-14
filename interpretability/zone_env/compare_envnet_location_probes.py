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


def collect_hidden_and_positions(model, layer_name, sampler, n_worlds=10, max_steps=200):
    X, Y = [], []
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
        X.extend(feats)
        Y.extend(positions)
    env.close()
    return np.stack(X), np.stack(Y)

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

    results = []
    for layer_name, label in zip(LAYER_NAMES, LABELS):
        print(f"\n=== Probing {label} ===")
        X, Y = collect_hidden_and_positions(model, layer_name, sampler, n_worlds=N_WORLDS, max_steps=200)
        ridge = Ridge().fit(X, Y)
        Y_pred = ridge.predict(X)
        mse = mean_squared_error(Y, Y_pred)
        r2 = r2_score(Y, Y_pred)
        print(f"  MSE: {mse:.4f}")
        print(f"  R^2: {r2:.4f}")
        results.append((label, mse, r2))
        # Barplot of weights
        W = ridge.coef_
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for i, coord in enumerate(['x', 'y']):
            weights = W[i]
            axes[i].bar(np.arange(len(weights)), weights, color='gray', alpha=0.7)
            topk_idx = np.argsort(np.abs(weights))[-5:][::-1]
            axes[i].bar(topk_idx, weights[topk_idx], color='red', alpha=0.9, label='Top 5')
            axes[i].set_ylabel(f'Probe weight for {coord}')
            axes[i].legend()
            axes[i].set_title(f'{label}: {coord}-coordinate')
        axes[1].set_xlabel('Hidden state dimension')
        plt.tight_layout()
        plt.savefig(f"agent_location_probe_weights_{label.replace('.', '_')}.png", dpi=150)
        plt.close(fig)
    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Layer':<20} | {'MSE':>8} | {'R^2':>8}")
    print("-"*42)
    for label, mse, r2 in results:
        print(f"{label:<20} | {mse:8.4f} | {r2:8.4f}")

if __name__ == '__main__':
    main() 