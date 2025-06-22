#!/usr/bin/env python3
import os
import sys
import random
import argparse

import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent


def parse_args():
    p = argparse.ArgumentParser(description="Probe layer → zone centers (one step)")
    p.add_argument('--env',        type=str, default='PointLtl2-v0')
    p.add_argument('--exp',        type=str, default='big_test')
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--layer',      type=str, required=True,
                   help="e.g. actor.enc.3")
    p.add_argument('--step',       type=int, default=0,
                   help="Timestep to sample activations at")
    p.add_argument('--num-worlds', type=int, default=50,
                   help="How many distinct worlds to sample")
    p.add_argument('--out',        type=str, default='zone_true_pred.png')
    return p.parse_args()


def read_first_task(env_name):
    fn = os.path.join("eval_datasets", env_name, "tasks.txt")
    with open(fn) as f:
        return f.readline().strip()


def load_model(env_name, exp, seed, formula):
    store = ModelStore(env_name, exp, seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[env_name]
    sampler = FixedSampler.partial(formula)
    dummy = make_env(env_name, sampler)
    model = build_model(dummy, status, cfg)
    model.eval()
    return model


def collect_worlds(env_name, num_worlds, seed):
    """
    Returns array of shape (num_worlds, num_zones, 2)
    """
    first = read_first_task(env_name)
    sampler = FixedSampler.partial(first)
    env = make_env(env_name, sampler)

    all_z = []
    for i in range(num_worlds):
        env.load_world_info(
            f"eval_datasets/{env_name}/worlds/world_info_{i}.pkl"
        )
        out = env.reset(seed=seed + i)
        # we only need env.zone_positions
        zp = env.zone_positions
        if isinstance(zp, dict):
            keys = sorted(zp.keys())
            arr = np.stack([zp[k] for k in keys])
        else:
            arr = np.array(zp)
        all_z.append(arr)
    env.close()
    return np.stack(all_z)  # (W, Z, 2)


def collect_activations(model, layer_name, env_name, step, num_worlds, seed):
    """
    Returns X of shape (num_worlds, feat_dim)
    """
    first = read_first_task(env_name)
    sampler = FixedSampler.partial(first)
    env = make_env(env_name, sampler)
    props  = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent  = Agent(model, search=search, propositions=props, verbose=False)

    layer = dict(model.named_modules())[layer_name]
    feats = []

    # hook captures the final hidden state for RNNs or the output otherwise
    def _hook(m, inp, out):
        v = out[1] if isinstance(out, tuple) else out
        feats.append(v.detach().cpu().squeeze(0).numpy())

    handle = layer.register_forward_hook(_hook)

    for i in range(num_worlds):
        env.load_world_info(
            f"eval_datasets/{env_name}/worlds/world_info_{i}.pkl"
        )
        out = env.reset(seed=seed + i)
        # obs might be (obs,info) or obs
        obs = out[0] if isinstance(out, tuple) else out

        # take `step` random actions
        for _ in range(step):
            nxt = env.step(env.action_space.sample())
            obs = nxt[0] if isinstance(nxt, tuple) else nxt

        # trigger the hook via the agent
        agent.reset()
        try:
            _ = agent.get_action(obs, {}, deterministic=True)
        except:
            # sometimes the search can fail—activation may still have been captured
            pass

    handle.remove()
    env.close()

    arr = np.stack(feats)
    if arr.shape[0] < num_worlds:
        raise RuntimeError(
            f"Too few activations at step={step}: got {arr.shape[0]}"
        )
    return arr[:num_worlds]


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("⏳ Loading model…")
    first = read_first_task(args.env)
    model = load_model(args.env, args.exp, args.seed, first)
    print(f"✅ Model loaded. Probing layer `{args.layer}` at step {args.step}…")

    # 1) get true zone centers
    WZ = collect_worlds(args.env, args.num_worlds, args.seed)
    # WZ: (W, Z, 2)
    num_zones = WZ.shape[1]

    # 2) get activations
    X = collect_activations(
        model, args.layer,
        args.env, args.step,
        args.num_worlds, args.seed
    )  # (W, D)

    # 3) fit ridge probe
    Y = WZ.reshape(args.num_worlds, num_zones * 2)  # (W, Z*2)
    reg = Ridge().fit(X, Y)
    Yp  = reg.predict(X)

    # 4) plot true vs. pred for each zone center
    fig, axes = plt.subplots(
        num_zones, 1,
        figsize=(5, 3 * num_zones),
        squeeze=False
    )
    for k in range(num_zones):
        tx = Y[:, 2*k]
        ty = Y[:, 2*k + 1]
        px = Yp[:, 2*k]
        py = Yp[:, 2*k + 1]

        ax = axes[k][0]
        ax.scatter(tx, ty,   label="true", alpha=0.6, s=20)
        ax.scatter(px, py, marker='x', label="pred", alpha=0.8, s=20)
        ax.set_title(f"Zone {k} center (step={args.step})")
        ax.set_aspect('equal', 'box')
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.show()
    print(f"▶ Saved figure to {args.out}")


if __name__ == '__main__':
    main()
