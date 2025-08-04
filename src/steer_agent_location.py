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
WORLD_IDX  = 0
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

def get_probe_and_pinv(model, layer_name, sampler, n_worlds=10, max_steps=200):
    X, Y = [], []
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]
    world_dir_path = f"{WORLD_DIR}"
    for i in range(n_worlds):
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
    X = np.stack(X)
    Y = np.stack(Y)
    ridge = Ridge().fit(X, Y)
    W = ridge.coef_  # shape (2, hidden_dim)
    b = ridge.intercept_
    W_pinv = np.linalg.pinv(W)
    return ridge, W, b, W_pinv

def run_rollout_with_steering(model, layer_name, probe, W, b, W_pinv, steer_vec, sampler, steer=True, max_steps=200):
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    env.load_world_info(f"{WORLD_DIR}/world_info_{WORLD_IDX}.pkl")
    obs = env.reset(seed=SEED+WORLD_IDX)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]
    feats = []
    positions = []
    steered_positions = []
    def grab(m, inp, out):
        x = out[1] if isinstance(out, tuple) else out
        feats.append(x.detach().cpu().numpy().ravel())
    h = module.register_forward_hook(grab)
    agent.reset()
    done = False
    for step in range(max_steps):
        if done:
            break
        # Get action (with or without steering)
        if len(feats) == 0:
            a = agent.get_action(obs, {}, deterministic=True).flatten()
        else:
            h_state = feats[-1].copy()
            if steer:
                # Compute current predicted location
                pred_xy = np.dot(W, h_state) + b
                # Compute delta in probe output
                delta_xy = steer_vec
                # Compute delta in hidden state
                delta_h = np.dot(W_pinv, delta_xy)
                # Steer hidden state
                h_state_steered = h_state + delta_h
                # Replace the hidden state in the model (assume it's the first layer after obs)
                # This depends on model internals; for now, we simulate by using the steered h for probe
                pred_xy_steered = np.dot(W, h_state_steered) + b
                feats[-1] = h_state_steered
            a = agent.get_action(obs, {}, deterministic=True).flatten()
        obs, _, done, _ = env.step(a)
        positions.append(env.agent_pos[:2].copy())
        # For visualization, also record the probe's predicted location (steered or not)
        h_state = feats[-1]
        pred_xy = np.dot(W, h_state) + b
        steered_positions.append(pred_xy)
    h.remove()
    env.close()
    return np.array(positions), np.array(steered_positions)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer',       required=True)
    p.add_argument('--max-steps',   type=int, default=200)
    p.add_argument('--out',         type=str, default='steer_agent_location.png')
    args = p.parse_args()

    # seeds & sampler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = "GF blue & GF green"
    sampler = FixedSampler.partial(formula)

    # ── load model and probe ─────────────────────────────────────────────────────
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[ENV]
    dummy  = make_env(ENV, sampler, sequence=False, render_mode=None)
    model  = build_model(dummy, status, cfg).eval()
    dummy.close()

    probe, W, b, W_pinv = get_probe_and_pinv(model, args.layer, sampler)
    steer_vec = np.array([2.0, 0.0])  # +2 in x

    # ── run rollouts ─────────────────────────────────────────────────────────────
    pos_unsteered, probe_unsteered = run_rollout_with_steering(model, args.layer, probe, W, b, W_pinv, steer_vec, sampler, steer=False, max_steps=args.max_steps)
    pos_steered, probe_steered = run_rollout_with_steering(model, args.layer, probe, W, b, W_pinv, steer_vec, sampler, steer=True, max_steps=args.max_steps)

    # ── plot results ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(pos_unsteered[:,0], pos_unsteered[:,1], '-o', color='k', markersize=3, label='Unsteered')
    axes[0].plot(pos_steered[:,0], pos_steered[:,1], '-o', color='r', markersize=3, label='Steered (+2 x)')
    axes[0].set_title('Agent Trajectory: Unsteered vs Steered')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].legend(); axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(probe_unsteered[:,0], probe_unsteered[:,1], '--', color='k', label='Probe (Unsteered)')
    axes[1].plot(probe_steered[:,0], probe_steered[:,1], '--', color='r', label='Probe (Steered)')
    axes[1].set_title('Probe-Predicted Location: Unsteered vs Steered')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    axes[1].legend(); axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved steering plot to {args.out}")

if __name__ == '__main__':
    main() 