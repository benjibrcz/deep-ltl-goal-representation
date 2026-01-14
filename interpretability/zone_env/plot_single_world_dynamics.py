#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store    import ModelStore
from model.model          import build_model
from config               import model_configs
from ltl                  import FixedSampler
from envs                 import make_env
from sequence.search      import ExhaustiveSearch
from model.agent          import Agent
from visualize.zones      import draw_trajectories

# ─── defaults ─────────────────────────────────────────────────────────────────
ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
N_WORLDS   = 50
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ────────────────────────────────────────────────────────────────────────────────

CMAP       = {"blue":0, "green":1, "yellow":2, "pink":3}
CMAP_RGB   = {
    0: "#4C72B0",   # blue
    1: "#55A868",   # green
    2: "#E3CF57",   # yellow
    3: "#CC79A7",   # pink
}
COLOR_NAMES= ["blue","green","yellow","pink"]
# ────────────────────────────────────────────────────────────────────────────────

def fig_to_array(fig):
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[...,:3]

def read_first_task(env_name):
    return open(f"eval_datasets/{env_name}/tasks.txt").readline().strip()

def collect_data_for_layer(model, layer_name, sampler):
    X, Ypos, Ycol = [], [], []
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    for i in trange(N_WORLDS, desc="Collect worlds"):
        env.load_world_info(f"{WORLD_DIR}/world_info_{i}.pkl")
        obs = env.reset(seed=SEED+i)
        agent.reset()

        feats = []
        def grab(m, inp, out):
            x = out[1] if isinstance(out, tuple) else out
            feats.append(x.detach().cpu().numpy().ravel())
        h = module.register_forward_hook(grab)
        try: agent.get_action(obs, {}, deterministic=True)
        except: pass
        h.remove()

        X.append(feats[0])

        layout = env.task.world_info.layout
        zs, cs, keys = [], [], []
        for k,v in layout.items():
            for name,idx in CMAP.items():
                if name in k:
                    zs.append(np.asarray(v, float).tolist())
                    cs.append(idx); keys.append(k)
                    break
        order = sorted(range(len(zs)), key=lambda j: keys[j])
        Ypos.append(np.array([zs[j] for j in order]).flatten())
        Ycol.append([cs[j] for j in order])

    env.close()
    return np.stack(X), np.stack(Ypos), np.stack(Ycol)

def decode_true_layout(env):
    layout = env.task.world_info.layout
    zs, cs, keys = [], [], []
    for k,v in layout.items():
        for name,idx in CMAP.items():
            if name in k:
                zs.append(np.asarray(v, float).tolist())
                cs.append(idx); keys.append(k)
                break
    order = sorted(range(len(zs)), key=lambda j: keys[j])
    return np.array([zs[j] for j in order]), [cs[j] for j in order]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--formula',     type=str, default=None)
    p.add_argument('--layer',       required=True)
    p.add_argument('--snaps',       nargs='+', type=int,
                   default=[0,10,20,50,100])
    p.add_argument('--warmup',      type=int, default=10)
    p.add_argument('--world-idx',   type=int, default=0)
    p.add_argument('--show-colors', nargs='+', choices=COLOR_NAMES,
                   default=COLOR_NAMES)
    p.add_argument('--gif',         action='store_true')
    p.add_argument('--fps',         type=int, default=2)
    p.add_argument('--gif-dt',      type=int, default=10)
    p.add_argument('--out',         type=str, default='dynamics.png')
    args = p.parse_args()

    # seeds & sampler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = args.formula or read_first_task(ENV)
    sampler = FixedSampler.partial(formula)

    # ── load model ───────────────────────────────────────────────────────────────
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[ENV]
    dummy  = make_env(ENV, sampler, sequence=False, render_mode=None)
    model  = build_model(dummy, status, cfg).eval()
    dummy.close()

    # ── train probes ─────────────────────────────────────────────────────────────
    X, Ypos, Ycol = collect_data_for_layer(model, args.layer, sampler)
    ridge = Ridge().fit(X, Ypos)
    clfs  = []
    for z in range(Ycol.shape[1]):
        y = Ycol[:,z]
        clf = None
        if len(np.unique(y))>1:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(multi_class="multinomial", max_iter=500)
            clf.fit(X, y)
        clfs.append(clf)

    # ── rollout the selected world ───────────────────────────────────────────────
    env = make_env(ENV, sampler, sequence=False, render_mode=None)
    env.load_world_info(f"{WORLD_DIR}/world_info_{args.world_idx}.pkl")
    obs = env.reset(seed=SEED+args.world_idx)
    raw_zones = env.zone_positions
    true_zpos, true_zcol = decode_true_layout(env)

    props  = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)

    feats, traj = [], []
    module = dict(model.named_modules())[args.layer]
    def grab(m, inp, out):
        x = out[1] if isinstance(out, tuple) else out
        feats.append(x.detach().cpu().numpy().ravel())
    h = module.register_forward_hook(grab)

    agent.reset()
    traj.append(env.agent_pos[:2].copy())
    done=False
    while not done:
        a = agent.get_action(obs, {}, deterministic=True).flatten()
        obs, _, done, _ = env.step(a)
        traj.append(env.agent_pos[:2].copy())
    h.remove()
    env.close()

    feats = np.stack(feats)
    traj  = np.stack(traj)

    # ── figure: snapshots + full rollout ────────────────────────────────────────
    snaps = [s for s in args.snaps if args.warmup <= s < len(feats)]
    full_fig = draw_trajectories([raw_zones], [traj], num_cols=1, num_rows=1)
    full_img = fig_to_array(full_fig)

    n = len(snaps) + 1
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)
    fig.suptitle(f"Formula: {formula}", y=1.05)

    # select only those zone‐indices whose true color is in show-colors
    selected = [
      i for i,c in enumerate(true_zcol)
      if COLOR_NAMES[c] in args.show_colors
    ]

    for i, s in enumerate(snaps):
        ax = axes[0,i]
        ax.set_title(f"t = {s}")
        ax.set_aspect("equal")
        ax.set_xlim(-2.5,2.5); ax.set_ylim(-2.5,2.5)
        ax.grid(True, ls="--", color="lightgray", alpha=0.5)

        # true zones (light fill)
        for (zx,zy),c in zip(true_zpos,true_zcol):
            ax.add_patch(patches.Circle((zx,zy),0.3,
                facecolor=CMAP_RGB[c], edgecolor='none', alpha=0.2))

        # predictions for selected zones
        x = feats[s:s+1]
        ppos = ridge.predict(x).reshape(-1,2)
        # clamp predicted centers to be inside the grid bounds
        xmin, xmax = -2.5 + 0.3, 2.5 - 0.3   # the 0.3 is your circle radius
        ymin, ymax = -2.5 + 0.3, 2.5 - 0.3
        ppos[:,0] = np.clip(ppos[:,0], xmin, xmax)
        ppos[:,1] = np.clip(ppos[:,1], ymin, ymax)
        for z in selected:
            col_idx = (clfs[z].predict(x)[0]
                       if clfs[z]
                       else int(np.unique(Ycol[:,z])[0]))
            zx, zy = ppos[z]
            ax.add_patch(patches.Circle((zx,zy),0.3,
                facecolor="none",
                edgecolor=CMAP_RGB[col_idx],
                linestyle='--', linewidth=2))

        seg = traj[:s+1]
        ax.plot(seg[:,0], seg[:,1], "-o", color="k", markersize=3)

    # final: full rollout
    axf = axes[0,-1]
    axf.axis("off")
    axf.imshow(full_img)
    handles = [patches.Patch(facecolor=CMAP_RGB[i],
                             edgecolor='k',
                             label=COLOR_NAMES[i])
               for i in CMAP_RGB]
    axf.legend(handles=handles, title="zone color",
               loc="upper right",
               bbox_to_anchor=(0.95,0.95))

    plt.tight_layout()
    plt.savefig(args.out.replace('.gif','.png'), dpi=150, bbox_inches="tight")

    # ── make GIF if requested ────────────────────────────────────────────────────
    if args.gif:
        anim_fig, anim_ax = plt.subplots(figsize=(4,4), dpi=100)
        anim_ax.set_xlim(-2.5,2.5); anim_ax.set_ylim(-2.5,2.5)
        anim_ax.set_aspect("equal")
        anim_ax.grid(True, ls="--", color="lightgray", alpha=0.5)

        # draw true zones once
        for (zx,zy),c in zip(true_zpos,true_zcol):
            anim_ax.add_patch(patches.Circle((zx,zy),0.3,
                facecolor=CMAP_RGB[c], edgecolor='none', alpha=0.2))

        traj_line, = anim_ax.plot([],[], '-o', color='k', markersize=3)
        # one patch per selected zone
        pred_patches = []
        for z in selected:
            p = patches.Circle((0,0),0.3,
                facecolor='none',
                edgecolor='k', linestyle='--', lw=2)
            anim_ax.add_patch(p)
            pred_patches.append((z,p))

        def init():
            traj_line.set_data([],[])
            for _,p in pred_patches:
                p.set_center((0,0))
            return [traj_line] + [p for _,p in pred_patches]

        frame_idxs = list(range(args.warmup, len(feats), args.gif_dt))
        def update(i):
            idx = frame_idxs[i]
            seg = traj[:idx+1]
            traj_line.set_data(seg[:,0], seg[:,1])
            x = feats[idx:idx+1]
            ppos = ridge.predict(x).reshape(-1,2)
            # clamp predicted centers to be inside the grid bounds
            xmin, xmax = -2.5 + 0.3, 2.5 - 0.3   # the 0.3 is your circle radius
            ymin, ymax = -2.5 + 0.3, 2.5 - 0.3
            ppos[:,0] = np.clip(ppos[:,0], xmin, xmax)
            ppos[:,1] = np.clip(ppos[:,1], ymin, ymax)
            for z,p in pred_patches:
                col_idx = (clfs[z].predict(x)[0]
                           if clfs[z]
                           else int(np.unique(Ycol[:,z])[0]))
                p.set_edgecolor(CMAP_RGB[col_idx])
                p.set_center(ppos[z])
            return [traj_line] + [p for _,p in pred_patches]

        anim = FuncAnimation(anim_fig, update, init_func=init,
                             frames=len(frame_idxs), blit=True)
        gif_path = args.out.replace('.png','.gif')
        anim.save(gif_path, writer=PillowWriter(fps=args.fps))
        plt.close(anim_fig)

    plt.show()

if __name__=="__main__":
    main()
