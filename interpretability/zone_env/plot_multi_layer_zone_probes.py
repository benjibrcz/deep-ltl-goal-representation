#!/usr/bin/env python3
import os, sys, random

import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics       import mean_squared_error, accuracy_score
import matplotlib.pyplot    as plt
import matplotlib.patches   as patches

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

# ─── USER CONFIG ───────────────────────────────────────────────────────────────
ENV       = "PointLtl2-v0"
EXP       = "big_test"
SEED      = 0
N_WORLDS  = 50
WORLD_DIR = f"eval_datasets/{ENV}/worlds"
FORMULA   = "GF blue & GF yellow"

# layers to probe
LAYERS = [
    "actor.enc.0",
    "actor.mu.0",
    "ltl_net.set_network.mlp.0",
    "ltl_net.set_network.mlp.3",
    "env_net.mlp.0",
]

# map color names → int
CMAP      = {"blue":0, "green":1, "yellow":2, "pink":3}
CMAP_RGB  = {0:"#4C72B0", 1:"#55A868", 2:"#C44E52", 3:"#8172B2"}
COLOR_NAMES = ["blue","green","yellow","pink"]

DEVICE    = "cpu"
# ────────────────────────────────────────────────────────────────────────────────

def collect_data_for_layer(model, layer_name):
    """
    Returns X, Ypos, Ycol for one layer:
      X    : (N_WORLDS, D)
      Ypos : (N_WORLDS, 2*num_zones)
      Ycol : (N_WORLDS, num_zones)
    """
    X_list, Ypos, Ycol = [], [], []

    sampler = FixedSampler.partial(FORMULA)
    env     = make_env(ENV, sampler, sequence=False, render_mode=None)
    props   = set(env.get_propositions())
    search  = ExhaustiveSearch(model, props, num_loops=2)
    agent   = Agent(model, search=search, propositions=props, verbose=False)

    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Layer {layer_name!r} not found.")
    module = modules[layer_name]

    for i in trange(N_WORLDS, desc=f"Sampling {layer_name}"):
        env.load_world_info(os.path.join(WORLD_DIR, f"world_info_{i}.pkl"))
        obs = env.reset(seed=SEED + i)
        agent.reset()

        feats = []
        def grab(m, inp, out):
            # pick the tensor
            x = out[1] if isinstance(out, tuple) else out
            arr = x.detach().cpu().numpy()
            feats.append(arr.ravel())    # flatten everything

        handle = module.register_forward_hook(grab)
        try:
            _ = agent.get_action(obs, {}, deterministic=True)
        except:
            pass
        handle.remove()

        if not feats:
            env.close()
            return None, None, None

        X_list.append(feats[0])

        # read true zone layout
        layout = env.task.world_info.layout
        zs, cs = [], []
        for key, val in layout.items():
            for name, idx in CMAP.items():
                if name in key:
                    arr = np.asarray(val, float)
                    zs.append(arr.tolist())
                    cs.append(idx)
                    break
        if not zs:
            raise RuntimeError(f"No zones in layout: {list(layout)}")

        # sort by key so ordering is consistent
        keys  = [k for k in layout if any(c in k for c in CMAP)]
        order = sorted(range(len(zs)), key=lambda j: keys[j])
        zs = [zs[j] for j in order]
        cs = [cs[j] for j in order]

        Ypos.append(np.array(zs).flatten())
        Ycol.append(cs)

    env.close()
    X    = np.stack(X_list)
    Ypos = np.stack(Ypos)
    Ycol = np.stack(Ycol)
    return X, Ypos, Ycol

if __name__ == "__main__":
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

    results = []
    for layer_name in LAYERS:
        X, Ypos, Ycol = collect_data_for_layer(model, layer_name)
        if X is None:
            print(f"⚠️  {layer_name} never fired; skipping.")
            continue

        # position probe
        pos_reg   = Ridge().fit(X, Ypos)
        Ypos_pred = pos_reg.predict(X)
        mse       = mean_squared_error(Ypos, Ypos_pred)

        # color probe
        preds, accs = [], []
        for z in range(Ycol.shape[1]):
            y = Ycol[:,z]
            classes = np.unique(y)
            if len(classes)==1:
                preds.append(int(classes[0]))
                accs.append(1.0)
            else:
                clf = LogisticRegression(multi_class="multinomial", max_iter=500)
                clf.fit(X,y)
                p = clf.predict(X)
                preds.append(int(p[0]))
                accs.append(accuracy_score(y,p))
        avg_acc = np.mean(accs)

        print(f"{layer_name:25s} → MSE {mse:.4f}, avg color acc {avg_acc:.1%}")

        # world 0 for plotting
        true_ctr = Ypos [0].reshape(-1,2)
        pred_ctr = Ypos_pred[0].reshape(-1,2)
        true_col = list(Ycol[0])
        pred_col = preds

        results.append((layer_name, true_ctr, true_col, pred_ctr, pred_col))

    # plot
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4*n))
    for i, (layer_name, t_ctr,t_col, p_ctr,p_col) in enumerate(results):
        for j,(ctrs,cols,title) in enumerate((
            (t_ctr, t_col, f"{layer_name}\nTrue"),
            (p_ctr, p_col, f"{layer_name}\nPred"),
        )):
            ax = axes[i,j]
            ax.set_aspect("equal")
            ax.set_xlim(-2.5,2.5); ax.set_ylim(-2.5,2.5)
            ax.set_xticks(range(-2,3)); ax.set_yticks(range(-2,3))
            ax.grid(True, linestyle="--", color="lightgray", alpha=0.5)
            ax.set_title(title)
            for (x,y), c in zip(ctrs, cols):
                circ = patches.Circle((x,y), radius=0.3,
                                      facecolor=CMAP_RGB[c],
                                      edgecolor="k", alpha=0.6)
                ax.add_patch(circ)

    # legend
    handles = [patches.Patch(facecolor=CMAP_RGB[i], edgecolor="k", label=COLOR_NAMES[i])
               for i in CMAP_RGB]
    axes[0,1].legend(handles=handles, title="Zone color",
                     bbox_to_anchor=(1.05,1), loc="upper left")

    plt.tight_layout()
    plt.savefig("all_layers_zone_probes.png", dpi=150)
    plt.show()
