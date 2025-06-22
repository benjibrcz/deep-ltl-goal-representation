#!/usr/bin/env python3
import os
import sys
import random
import argparse

import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ENV       = "PointLtl2-v0"
EXP       = "big_test"
SEED      = 0
N_WORLDS  = 50
WORLD_DIR = f"eval_datasets/{ENV}/worlds"
FORMULA   = "GF blue & GF yellow"   # used only to sample worlds

DEVICE    = "cpu"
# ────────────────────────────────────────────────────────────────────────────────

def collect_data_for_layer(layer_name):
    """
    Collect one activation per world for `layer_name`, plus true zone centers.
    Returns:
      X    : (N_WORLDS, hidden_dim)
      Ypos : (N_WORLDS, 2*num_zones)
    """
    X_list, Ypos = [], []

    sampler = FixedSampler.partial(FORMULA)
    env     = make_env(ENV, sampler, sequence=False, render_mode=None)
    props   = set(env.get_propositions())
    search  = ExhaustiveSearch(model, props, num_loops=2)
    agent   = Agent(model, search=search, propositions=props, verbose=False)

    for i in trange(N_WORLDS, desc=f"Sampling {layer_name}"):
        env.load_world_info(os.path.join(WORLD_DIR, f"world_info_{i}.pkl"))
        obs = env.reset(seed=SEED + i)
        agent.reset()

        # Hook the layer once
        feats = []
        module = dict(model.named_modules()).get(layer_name)
        if module is None:
            raise ValueError(f"Layer {layer_name} not found")
        handle = module.register_forward_hook(
            lambda m, inp, out: feats.append(out.detach().cpu().squeeze(0))
        )

        # Fire the policy exactly once
        try:
            _ = agent.get_action(obs, {}, deterministic=True)
        except:
            pass
        handle.remove()

        if not feats:
            env.close()
            raise RuntimeError(f"Layer {layer_name} never fired on world {i}")

        X_list.append(feats[0].numpy())

        # Extract true zone centers
        layout = env.task.world_info.layout
        zs = []
        for key, val in layout.items():
            for color in ["blue", "green", "yellow", "pink"]:
                if color in key:
                    zs.append(np.asarray(val, float).tolist())
                    break
        # sort by key to keep consistent order
        keys = [k for k in layout if any(c in k for c in ["blue","green","yellow","pink"])]
        order = sorted(range(len(zs)), key=lambda j: keys[j])
        zs = [zs[j] for j in order]

        Ypos.append(np.array(zs).flatten())

    env.close()
    return np.stack(X_list), np.stack(Ypos)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Probe error over time snapshots for one layer")
    p.add_argument("--layer",         required=True, help="Module to probe, e.g. actor.enc.3")
    p.add_argument("--steps-per-world", type=int, default=20, help="Max rollout length")
    p.add_argument("--snapshots",       type=int, default=5,  help="Number of time‐points")
    args = p.parse_args()

    # seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # load model
    sampler   = FixedSampler.partial(FORMULA)
    store     = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status    = store.load_training_status(map_location=DEVICE)
    cfg       = model_configs[ENV]
    dummy_env = make_env(ENV, sampler, sequence=False, render_mode=None)
    model     = build_model(dummy_env, status, cfg).to(DEVICE).eval()
    dummy_env.close()

    # 1) collect training data and fit position probe
    print(f"Collecting train activations for layer {args.layer} …")
    X_train, Ypos_train = collect_data_for_layer(args.layer)
    pos_reg = Ridge().fit(X_train, Ypos_train)

    # 2) prepare time‐points
    steps = args.steps_per_world
    snaps = np.linspace(0, steps-1, args.snapshots, dtype=int)

    # 3) sample errors at each snapshot
    errors = {t: [] for t in snaps}

    # rebuild env+agent
    sampler = FixedSampler.partial(FORMULA)
    env     = make_env(ENV, sampler, sequence=False, render_mode=None)
    props   = set(env.get_propositions())
    search  = ExhaustiveSearch(model, props, num_loops=2)
    agent   = Agent(model, search=search, propositions=props, verbose=False)

    for i in trange(N_WORLDS, desc="Eval snapshots"):
        env.load_world_info(os.path.join(WORLD_DIR, f"world_info_{i}.pkl"))
        obs = env.reset(seed=SEED + i)
        agent.reset()

        for t in range(steps):
            # record at snapshot times
            if t in snaps:
                # hook & fire
                feats = []
                module = dict(model.named_modules())[args.layer]
                handle = module.register_forward_hook(
                    lambda m, inp, out: feats.append(out.detach().cpu().squeeze(0))
                )
                try:
                    _ = agent.get_action(obs, {}, deterministic=True)
                except:
                    pass
                handle.remove()

                if feats:
                    feat = feats[0].unsqueeze(0).numpy()  # shape (1, hid)
                    pred = pos_reg.predict(feat)[0]      # shape (2*num_zones,)
                    true = Ypos_train[i]
                    errors[t].append(mean_squared_error(true, pred))

            # step forward
            try:
                a = agent.get_action(obs, {}, deterministic=True).flatten()
            except:
                a = env.action_space.sample()
            obs, _, done, _ = env.step(a)
            if done:
                break

    env.close()

    # 4) compute and plot mean error per snapshot
    means = [np.mean(errors[t]) for t in snaps]
    plt.figure(figsize=(6,4))
    plt.plot(snaps, means, marker='o')
    plt.xlabel("Timestep")
    plt.ylabel("Mean MSE of zone‐center probe")
    plt.title(f"Layer {args.layer} probe error over time")
    plt.xticks(snaps)
    plt.tight_layout()
    plt.savefig("probe_error_snapshots.png", dpi=150)
    plt.show()
