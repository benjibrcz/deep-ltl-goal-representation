#!/usr/bin/env python3
import os
import sys
import random
import argparse

import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import Ridge, LogisticRegression

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

# ─── color maps ────────────────────────────────────────────────────────────────
CMAP_RGB   = {0:"#4C72B0",1:"#55A868",2:"#C44E52",3:"#8172B2"}
COLOR_NAMES= ["blue","green","yellow","pink"]
# ────────────────────────────────────────────────────────────────────────────────

def read_first_task(env_name):
    fn = os.path.join("eval_datasets", env_name, "tasks.txt")
    return open(fn).readline().strip()

def decode_true_layout(env):
    """Return (Z×2 array of centers, list of color‐idx)."""
    layout = env.task.world_info.layout
    zs, cs, keys = [], [], []
    for k,v in layout.items():
        for idx,name in enumerate(COLOR_NAMES):
            if name in k:
                zs.append(np.asarray(v, float).tolist())
                cs.append(idx)
                keys.append(k)
                break
    order = sorted(range(len(zs)), key=lambda j: keys[j])
    return np.array([zs[j] for j in order]), [cs[j] for j in order]

def collect_probe_data(model, sampler, layer_name, n_worlds, seed):
    """Run n_worlds, hook layer→ activations + collect true map. Returns X, Ypos, Ycol."""
    env    = make_env(args.env, sampler, sequence=False, render_mode=None)
    props  = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent  = Agent(model, search=search, propositions=props, verbose=False)

    layer  = dict(model.named_modules())[layer_name]
    X_list, Ypos, Ycol = [], [], []

    for i in trange(n_worlds, desc="Probe-worlds"):
        env.load_world_info(f"eval_datasets/{args.env}/worlds/world_info_{i}.pkl")
        obs = env.reset(seed=seed+i)
        agent.reset()

        feats = []
        h = layer.register_forward_hook(
            lambda m, inp, out: feats.append(
                (out[1] if isinstance(out,tuple) else out)
                .detach().cpu().numpy().ravel()
            )
        )
        try:
            _ = agent.get_action(obs, {}, deterministic=True)
        except:
            pass
        h.remove()

        X_list.append(feats[0])

        zpos,zcol = decode_true_layout(env)
        Ypos.append(zpos.flatten())
        Ycol.append(zcol)

    env.close()
    return (np.stack(X_list),
            np.stack(Ypos),
            np.stack(Ycol))

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env",        default="PointLtl2-v0")
    p.add_argument("--exp",        default="big_test")
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--layer",      required=True)
    p.add_argument("--formula",    default=None)
    p.add_argument("--probe-worlds",type=int,  default=50)
    p.add_argument("--world-idx",  type=int,   default=0)
    p.add_argument("--warmup",     type=int,   default=0,
                   help="skip first N steps")
    p.add_argument("--snaps",      nargs="+",  type=int, default=[0,5,10,20,50])
    p.add_argument("--out",        default="dynamics_probe_online.png")
    args = p.parse_args()

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # formula + sampler
    formula = args.formula or read_first_task(args.env)
    sampler = FixedSampler.partial(formula)

    # load model
    store   = ModelStore(args.env, args.exp, args.seed)
    store.load_vocab()
    status  = store.load_training_status(map_location="cpu")
    cfg     = model_configs[args.env]
    dummy   = make_env(args.env, sampler, sequence=False, render_mode=None)
    model   = build_model(dummy, status, cfg).eval()
    dummy.close()

    # 1) collect probe data & train
    X, Ypos, Ycol = collect_probe_data(
        model, sampler, args.layer,
        args.probe_worlds, args.seed
    )
    ridge = Ridge().fit(X, Ypos)
    color_clfs = []
    for z in range(Ycol.shape[1]):
        ys = Ycol[:,z]
        if len(np.unique(ys))>1:
            clf = LogisticRegression(multi_class="multinomial", max_iter=300)
            clf.fit(X, ys)
        else:
            clf = None
        color_clfs.append(clf)

    # 2) rollout one world while capturing activations + trajectory
    env    = make_env(args.env, sampler, render_mode=None, max_steps=1000)
    env.load_world_info(f"eval_datasets/{args.env}/worlds/world_info_{args.world_idx}.pkl")
    obs    = env.reset(seed=args.seed+args.world_idx)
    true_zpos,true_zcol = decode_true_layout(env)

    props  = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent  = Agent(model, search=search, propositions=props, verbose=False)

    layer  = dict(model.named_modules())[args.layer]
    acts, traj = [], []
    h = layer.register_forward_hook(
        lambda m,inp,out: acts.append(
            (out[1] if isinstance(out,tuple) else out)
            .detach().cpu().numpy().ravel()
        )
    )

    agent.reset()
    done=False
    traj.append(env.agent_pos[:2].copy())
    steps=0
    while not done and steps<1000:
        a, *_ = agent.get_action(obs, {}, deterministic=True)
        obs, *_ , info = env.step(a.flatten())
        traj.append(env.agent_pos[:2].copy())
        done = info.get("terminated", info.get("done", False))
        steps+=1

    h.remove()
    env.close()

    acts = np.stack(acts)     # (T, D)
    traj = np.stack(traj)     # (T+1, 2)

    # 3) plot
    S = [s for s in args.snaps if s<len(acts)]
    fig, axes = plt.subplots(1, len(S),
                             figsize=(4*len(S),4),
                             squeeze=False)

    for j,s in enumerate(S):
        a_s = acts[s:s+1]
        ppos = ridge.predict(a_s).reshape(-1,2)
        pcol = [
            (clf.predict(a_s)[0] if clf else true_zcol[z])
            for z,clf in enumerate(color_clfs)
        ]

        ax = axes[0][j]
        ax.set_title(f"t={s}")
        ax.set_aspect("equal")
        ax.set_xlim(-2.5,2.5); ax.set_ylim(-2.5,2.5)
        ax.grid(True, linestyle="--", alpha=0.3)

        # true zones
        for (zx,zy),c in zip(true_zpos,true_zcol):
            ax.add_patch(patches.Circle((zx,zy),0.3,
                facecolor=CMAP_RGB[c], alpha=0.2,
                edgecolor="none"))

        # predicted zones
        for (zx,zy),c in zip(ppos,pcol):
            ax.add_patch(patches.Circle((zx,zy),0.3,
                facecolor="none",
                edgecolor=CMAP_RGB[c],
                linestyle="--", linewidth=2))

        # trajectory so far
        seg = traj[:s+1]
        ax.plot(seg[:,0], seg[:,1], "-k", marker="o", markersize=3)

    # legend
    handles = [patches.Patch(facecolor=CMAP_RGB[i],
               edgecolor="k", label=COLOR_NAMES[i])
               for i in CMAP_RGB]
    axes[0][-1].legend(handles=handles, title="zone color",
                       bbox_to_anchor=(1.05,1), loc="upper left")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.show()
