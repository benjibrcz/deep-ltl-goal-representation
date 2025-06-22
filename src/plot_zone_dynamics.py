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
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

# ─── colour maps ───────────────────────────────────────────────────────────────
# map colour names → indices
CMAP       = {"blue":0, "green":1, "yellow":2, "pink":3}
# a nicer blue/green/yellow/pink palette
CMAP_RGB   = {
    0: "#4C72B0",   # blue
    1: "#55A868",   # green
    2: "#E1C027",   # a true yellow
    3: "#BB78A5",   # pink
}
COLOR_NAMES = ["blue","green","yellow","pink"]
# ────────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Plot zone‐dynamics + probe predictions")
    p.add_argument('--env',          type=str,   default='PointLtl2-v0')
    p.add_argument('--exp',          type=str,   default='big_test')
    p.add_argument('--seed',         type=int,   default=0)
    p.add_argument('--formula',      type=str,   default=None,
                   help="LTL task (defaults to first line of tasks.txt)")
    p.add_argument('--layer',        type=str,   required=True,
                   help="Module to hook, e.g. env_net.mlp.0")
    p.add_argument('--train-worlds', type=int,   default=50,
                   help="How many worlds to sample for training the probe")
    p.add_argument('--world-idx',    type=int,   default=0,
                   help="Which world to visualize dynamically")
    p.add_argument('--warmup',       type=int,   default=0,
                   help="Do N random steps before recording dynamics")
    p.add_argument('--snaps',        nargs='+',  type=int,
                   default=[0,5,10,15,20],
                   help="Timesteps at which to snapshot")
    p.add_argument('--out',          type=str,   default='zone_dynamics.png')
    return p.parse_args()

def read_first_task(env):
    fn = os.path.join("eval_datasets", env, "tasks.txt")
    return open(fn).readline().strip()

def load_model(env,exp,seed,formula):
    store   = ModelStore(env,exp,seed)
    store.load_vocab()
    status  = store.load_training_status(map_location='cpu')
    cfg     = model_configs[env]
    sampler = FixedSampler.partial(formula)
    dummy   = make_env(env, sampler, sequence=False)
    m       = build_model(dummy, status, cfg)
    m.eval()
    dummy.close()
    return m

def sample_probe_data(args, model):
    X_list, Ypos, Ycol = [], [], []
    sampler = FixedSampler.partial(args.formula)
    env     = make_env(args.env, sampler, sequence=False)
    props   = set(env.get_propositions())
    search  = ExhaustiveSearch(model, props, num_loops=2)
    agent   = Agent(model, search=search, propositions=props, verbose=False)
    layer   = dict(model.named_modules())[args.layer]

    for i in trange(args.train_worlds, desc="Train‐probe data"):
        #env.load_world_info(f"eval_datasets/{args.env}/worlds/world_info_{i}.pkl")
        res = env.reset(seed=args.seed + i)
        obs = res[0] if isinstance(res, tuple) else res
        agent.reset()

        feats = []
        h = layer.register_forward_hook(
            lambda m,inp,out: feats.append(out.detach().cpu().squeeze(0).numpy())
        )
        try:
            _ = agent.get_action(obs, {}, deterministic=True)
        except:
            pass
        h.remove()

        X_list.append(feats[0])

        layout = env.task.world_info.layout
        zs, cs = [], []
        for key,val in sorted(layout.items()):
            for name,idx in CMAP.items():
                if name in key:
                    arr = np.array(val, float).flatten()
                    zs.append(arr[:2].tolist())
                    cs.append(idx)
                    break

        Ypos.append(np.array(zs).flatten())
        Ycol.append(cs)

    env.close()
    return np.stack(X_list), np.stack(Ypos), np.stack(Ycol)

def rollout_and_hook(args, model):
    sampler = FixedSampler.partial(args.formula)
    env     = make_env(args.env, sampler, sequence=False)
    props   = set(env.get_propositions())
    search  = ExhaustiveSearch(model, props, num_loops=2)
    agent   = Agent(model, search=search, propositions=props, verbose=False)

    #env.load_world_info(f"eval_datasets/{args.env}/worlds/world_info_{args.world_idx}.pkl")
    res = env.reset(seed=args.seed + args.world_idx)
    obs = res[0] if isinstance(res, tuple) else res
    agent.reset()

    for _ in range(args.warmup):
        try:
            a = agent.get_action(obs, {}, deterministic=True).flatten().cpu().numpy()
        except:
            a = env.action_space.sample()
        step = env.step(a)
        obs = step[0] if isinstance(step, tuple) else step
        # ignore done for warm‐up

    layer = dict(model.named_modules())[args.layer]
    feats_dyn, traj = [], []
    h = layer.register_forward_hook(
        lambda m,inp,out: feats_dyn.append(out.detach().cpu().squeeze(0).numpy())
    )

    traj.append(env.agent_pos[:2].copy())
    max_snap = max(args.snaps)

    for t in range(max_snap+1):
        try:
            a = agent.get_action(obs, {}, deterministic=True).flatten().cpu().numpy()
        except:
            a = env.action_space.sample()

        if t < max_snap:
            step = env.step(a)
            obs,_,done,_ = step if len(step)==4 else (*step[0:3], step[3])
            traj.append(env.agent_pos[:2].copy())
            if done:
                # pad
                for _ in range(t+1, max_snap+1):
                    feats_dyn.append(feats_dyn[-1])
                break

    h.remove()
    env.close()
    return np.stack(feats_dyn), np.stack(traj)

def main():
    args = parse_args()
    if args.formula is None:
        args.formula = read_first_task(args.env)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("⏳ Loading model…")
    model = load_model(args.env, args.exp, args.seed, args.formula)
    print(f"✅ Model loaded, probing `{args.layer}`\n")

    # 1) train probes
    X, Ypos, Ycol = sample_probe_data(args, model)
    pos_reg       = Ridge().fit(X, Ypos)
    zone_clfs     = []
    for z in range(Ycol.shape[1]):
        y = Ycol[:,z]
        if len(np.unique(y))>1:
            clf = LogisticRegression(max_iter=500).fit(X, y)
        else:
            clf = None
        zone_clfs.append(clf)

    # 2) roll out one world
    feats_dyn, traj = rollout_and_hook(args, model)
    true_zs = Ypos[args.world_idx].reshape(-1,2)
    true_cs = Ycol[args.world_idx]

    # 3) plot
    snaps = [s for s in args.snaps if s < len(feats_dyn)]
    fig, axes = plt.subplots(1, len(snaps), figsize=(4*len(snaps),4), squeeze=False)
    for ax,s in zip(axes[0], snaps):
        ax.set_title(f"t = {s}")
        ax.set_aspect("equal")
        ax.set_xlim(-2.5,2.5); ax.set_ylim(-2.5,2.5)
        ax.grid(True,linestyle="--",color="gray",alpha=0.3)

        # true zones: light fill
        for (zx,zy),c in zip(true_zs,true_cs):
            ax.add_patch(patches.Circle(
                (zx,zy),0.3,facecolor=CMAP_RGB[c],
                edgecolor='none',alpha=0.2
            ))

        # predicted zones: dashed outlines in the correct colour
        x = feats_dyn[s:s+1]
        pred_pos = pos_reg.predict(x).reshape(-1,2)
        pred_col = [
            (zone_clfs[i].predict(x)[0] if zone_clfs[i] else true_cs[i])
            for i in range(len(true_cs))
        ]
        for (zx,zy),c in zip(pred_pos,pred_col):
            ax.add_patch(patches.Circle(
                (zx,zy),0.3,facecolor='none',
                edgecolor=CMAP_RGB[c],linestyle='--',linewidth=2
            ))

        # agent trajectory
        seg = traj[:s+1]
        ax.plot(seg[:,0],seg[:,1],'-o',color='k',markersize=4)

    # legend on last panel
    handles = [
        patches.Patch(facecolor=CMAP_RGB[i],edgecolor='k',label=COLOR_NAMES[i])
        for i in CMAP_RGB
    ] + [
        plt.Line2D([0],[0],color='k',marker='o',linestyle='-',markersize=4,label="agent"),
        plt.Line2D([0],[0],color='k',marker='x',linestyle='--',markersize=8,label="pred zone")
    ]
    axes[0,-1].legend(handles=handles,bbox_to_anchor=(1.05,1),loc="upper left")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.show()

if __name__=="__main__":
    main()
