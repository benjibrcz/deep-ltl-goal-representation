#!/usr/bin/env python3
import os, sys, argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store    import ModelStore
from model.model         import build_model
from config               import model_configs
from ltl                  import FixedSampler
from envs                 import make_env
from sequence.search      import ExhaustiveSearch
from model.agent          import Agent

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot true vs predicted zone‐center maps over time"
    )
    p.add_argument('--env',         type=str,   default='PointLtl2-v0')
    p.add_argument('--exp',         type=str,   default='big_test')
    p.add_argument('--seed',        type=int,   default=0)
    p.add_argument('--layer',       type=str,   required=True,
                   help="Layer to probe, e.g. `actor.enc.3`")
    p.add_argument('--snaps',       nargs='+',  type=int, required=True,
                   help="Timesteps to sample at (e.g. `0 5 10 15 20`)")
    p.add_argument('--num-worlds',  type=int,   default=50,
                   help="How many distinct worlds to sample")
    p.add_argument('--out',         type=str,   default='zone_true_pred.png',
                   help="Where to save the figure")
    return p.parse_args()

def read_first_task(env_name):
    fn = os.path.join("eval_datasets", env_name, "tasks.txt")
    with open(fn) as f:
        return f.readline().strip()

def load_model(env, exp, seed, formula):
    store = ModelStore(env, exp, seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[env]
    sampler = FixedSampler.partial(formula)
    dummy = make_env(env, sampler)
    m = build_model(dummy, status, cfg)
    m.eval()
    return m

def collect_worlds(env_name, num_worlds, seed):
    """
    Returns:
      world_zones: np.array of shape (W, Z, 2)
      zone_keys:   list of length Z, the sorted dict-keys (colors)
    """
    first = read_first_task(env_name)
    sampler = FixedSampler.partial(first)
    env = make_env(env_name, sampler)
    all_z = []
    zone_keys = None

    for i in range(num_worlds):
        env.load_world_info(f"eval_datasets/{env_name}/worlds/world_info_{i}.pkl")
        _ = env.reset(seed=seed + i)
        zp = env.zone_positions
        if isinstance(zp, dict):
            if zone_keys is None:
                zone_keys = sorted(zp.keys())
            arr = np.stack([zp[k] for k in zone_keys])
        else:
            # fallback if list
            if zone_keys is None:
                zone_keys = list(range(len(zp)))
            arr = np.array(zp)
        all_z.append(arr)

    env.close()
    return np.stack(all_z), zone_keys  # (W, Z, 2), [Z]

def collect_activations(model, layer_name, env_name, snaps, num_worlds, seed):
    """
    Returns X of shape (T, W, D) where
      T = len(snaps), W = num_worlds, D = feature-dim at that layer.
    """
    first = read_first_task(env_name)
    sampler = FixedSampler.partial(first)
    env = make_env(env_name, sampler)
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)

    layer = dict(model.named_modules())[layer_name]
    X_list = []

    for t in snaps:
        feats = []

        def hook_fn(m, inp, out):
            h = out[1] if isinstance(out, tuple) else out
            feats.append(h.detach().cpu().squeeze(0).numpy())

        handle = layer.register_forward_hook(hook_fn)

        for i in range(num_worlds):
            env.load_world_info(f"eval_datasets/{env_name}/worlds/world_info_{i}.pkl")
            obs = env.reset(seed=seed + i)
            # take t random steps
            for _ in range(t):
                obs, _, done, _ = env.step(env.action_space.sample())
                if done:
                    break
            agent.reset()
            try:
                _ = agent.get_action(obs, {}, deterministic=True)
            except:
                pass

        handle.remove()
        arr = np.stack(feats)
        if arr.shape[0] < num_worlds:
            raise RuntimeError(f"Too few activations at t={t}: got {arr.shape[0]}")
        X_list.append(arr[:num_worlds])

    env.close()
    return np.stack(X_list)  # (T, W, D)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1) load model (zero‐shot formula comes from the very first task)
    formula = read_first_task(args.env)
    print(f"⏳ Loading model on `{formula}` …")
    model = load_model(args.env, args.exp, args.seed, formula)

    # 2) collect activations at each snapshot
    print(f"⏳ Collecting activations from `{args.layer}` …")
    X = collect_activations(
        model, args.layer, args.env,
        args.snaps, args.num_worlds, args.seed
    )  # (T, W, D)

    # 3) collect true zone‐centers
    print("⏳ Collecting true zone positions …")
    world_zones, zone_keys = collect_worlds(
        args.env, args.num_worlds, args.seed
    )  # shape (W, Z, 2)

    W, Z, _ = world_zones.shape
    T = len(args.snaps)

    # 4) fit & predict, then plot for world 0 only
    fig, axes = plt.subplots(T, 2,
                             figsize=(6*2, 4*T),
                             squeeze=False)

    for ti, t in enumerate(args.snaps):
        feats = X[ti]                        # (W, D)
        Y_true = world_zones.reshape(W, Z*2) # (W, 2Z)
        reg = Ridge().fit(feats, Y_true)
        Y_hat = reg.predict(feats).reshape(W, Z, 2)

        true_pos = world_zones[0]  # for world index 0
        pred_pos = Y_hat[0]

        # left: true
        ax0 = axes[ti][0]
        for zi, (x, y) in enumerate(true_pos):
            c = zone_keys[zi] if isinstance(zone_keys[zi], str) else f"C{zi}"
            ax0.scatter(x, y, s=100, color=c, label=zone_keys[zi] if ti==0 else "")
        ax0.set_title(f"t={t} TRUE")
        ax0.set_aspect('equal')
        ax0.set_xticks([]); ax0.set_yticks([])

        # right: predicted
        ax1 = axes[ti][1]
        for zi, (x, y) in enumerate(pred_pos):
            c = zone_keys[zi] if isinstance(zone_keys[zi], str) else f"C{zi}"
            ax1.scatter(x, y, s=100, marker='x', color=c)
        ax1.set_title(f"t={t} PRED")
        ax1.set_aspect('equal')
        ax1.set_xticks([]); ax1.set_yticks([])

    # shared legend at bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               ncol=len(zone_keys),
               frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(args.out, dpi=150)
    plt.show()
    print(f"✅ Saved to {args.out}")

if __name__ == '__main__':
    main()
