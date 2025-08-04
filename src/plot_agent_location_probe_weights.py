#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

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

def collect_hidden_and_positions(model, layer_name, sampler, n_worlds=10, max_steps=200):
    X, Y = [], []
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]
    world_dir_path = f"{WORLD_DIR}"
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
        X.extend(feats)
        Y.extend(positions)
    env.close()
    return np.stack(X), np.stack(Y)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True)
    p.add_argument('--n-worlds',    type=int, default=10)
    p.add_argument('--max-steps',   type=int, default=200)
    p.add_argument('--topk',        type=int, default=5)
    p.add_argument('--out',         type=str, default='agent_location_probe_weights.png')
    args = p.parse_args()

    # seeds & sampler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = "GF blue & GF green"
    sampler = FixedSampler.partial(formula)

    # ── load model and collect data ─────────────────────────────────────────────
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[ENV]
    dummy  = make_env(ENV, sampler, sequence=False, render_mode=None)
    model  = build_model(dummy, status, cfg).eval()
    dummy.close()

    X, Y = collect_hidden_and_positions(model, args.layer, sampler, n_worlds=args.n_worlds, max_steps=args.max_steps)
    if X is None:
        print("Could not collect data for probe training. Exiting.")
        return

    # ── train probe ──────────────────────────────────────────────────────────────
    ridge = Ridge().fit(X, Y)
    W = ridge.coef_  # shape (2, hidden_dim)
    b = ridge.intercept_

    # ── plot barplot of weights ─────────────────────────────────────────────────-
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i, label in enumerate(['x', 'y']):
        weights = W[i]
        axes[i].bar(np.arange(len(weights)), weights, color='gray', alpha=0.7)
        # Highlight top-k largest-magnitude weights
        topk_idx = np.argsort(np.abs(weights))[-args.topk:][::-1]
        axes[i].bar(topk_idx, weights[topk_idx], color='red', alpha=0.9, label=f'Top {args.topk}')
        axes[i].set_ylabel(f'Probe weight for {label}')
        axes[i].legend()
        axes[i].set_title(f'Agent Location Probe Weights: {label}-coordinate')
    axes[1].set_xlabel('Hidden state dimension')
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved probe weights barplot to {args.out}")

if __name__ == '__main__':
    main() 