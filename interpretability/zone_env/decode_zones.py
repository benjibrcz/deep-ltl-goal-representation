#!/usr/bin/env python3
import os, sys, random
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
FORMULA    = "F yellow"        # only for how the world is sampled
HOOK_LAYER = "env_net.mlp.3"
DEVICE     = "cpu"
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
N_WORLDS   = 50
# ────────────────────────────────────────────────────────────────────────────────

# 0) seed everything
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1) load the model (we only need env_net)
sampler   = FixedSampler.partial(FORMULA)
store     = ModelStore(ENV, EXP, SEED)
store.load_vocab()
status    = store.load_training_status(map_location=DEVICE)
cfg       = model_configs[ENV]
dummy_env = make_env(ENV, sampler, sequence=False, render_mode=None)
model     = build_model(dummy_env, status, cfg).to(DEVICE).eval()

# 2) prepare the plain env
env = make_env(ENV, sampler, sequence=False, render_mode=None)

# 3) hook into the env‐encoder
layer = dict(model.named_modules())[HOOK_LAYER]
embs = []

def grab_activation(module, inp, out):
    # out is a Tensor [1, feat_dim]
    embs.append(out.detach().cpu().numpy().squeeze(0))

hook = layer.register_forward_hook(grab_activation)

# 4) define a mapping for colors
cmap_str = {"blue":0, "green":1, "yellow":2, "pink":3, "magenta":4}

# 5) collect embeddings + ground‐truth zone info
true_pos = []
true_col = []

for i in trange(N_WORLDS, desc="Collect worlds"):
    path = os.path.join(WORLD_DIR, f"world_info_{i}.pkl")
    env.load_world_info(path)
    obs = env.reset(seed=SEED + i)

    # 5a) grab the embedding
    raw = obs["features"]
    with torch.no_grad():
        inp = torch.tensor(raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        _   = model.env_net(inp)

    # 5b) extract *only* the zone entries
    layout = env.task.world_info.layout
    zone_items = sorted(
        [(k,v) for k,v in layout.items() if 'zone' in k],
        key=lambda kv: kv[0]
    )

    positions = []
    colors    = []
    for key, v in zone_items:
        # Case 1: a dict with pos & color
        if isinstance(v, dict):
            pos = v["pos"]
            col_name = v["color"]
            if col_name not in cmap_str:
                raise RuntimeError(f"Unknown color `{col_name}` in key `{key}`")
            col = cmap_str[col_name]

        # Case 2: a numpy array of shape (2,) → just position, infer color from key
        elif isinstance(v, np.ndarray) and v.ndim==1 and v.size==2:
            pos = [float(v[0]), float(v[1])]
            # infer color by substring in the key
            matched = [c for c in cmap_str if c in key]
            if not matched:
                raise RuntimeError(f"No color name in key '{key}'")
            col = cmap_str[matched[0]]

        # Case 3: custom object with .pos/.color attributes
        else:
            pos_attr = getattr(v, "pos", None)
            col_attr = getattr(v, "color", None)
            if pos_attr is None or col_attr is None:
                raise RuntimeError(f"Can’t decode layout entry {key}: {type(v)}")
            pos = pos_attr
            if isinstance(col_attr, str):
                if col_attr not in cmap_str:
                    raise RuntimeError(f"Unknown color `{col_attr}` in key `{key}`")
                col = cmap_str[col_attr]
            else:
                col = int(col_attr)

        positions.append(pos)
        colors.append(col)

    # record
    true_pos.append(np.array(positions).flatten())  # shape (2*num_zones,)
    true_col.append(np.array(colors))               # shape (num_zones,)

hook.remove()
env.close()

# 6) stack into arrays
X      = np.stack(embs)        # (N_WORLDS, feat_dim)
YPOS   = np.stack(true_pos)    # (N_WORLDS, 2*num_zones)
YCOL   = np.stack(true_col)    # (N_WORLDS, num_zones)

# 7) decode zone‐centers with Ridge regression
reg   = Ridge()
reg.fit(X, YPOS)
YPred = reg.predict(X)
mse   = mean_squared_error(YPOS, YPred)
print(f"Zone‐center MSE: {mse:.4f}")

# 8) decode zone‐colors with multinomial logistic
accs = []
for z in range(YCOL.shape[1]):
    clf = LogisticRegression(multi_class="multinomial", max_iter=500)
    y   = YCOL[:, z]
    if len(np.unique(y)) < 2:
        # skip if only one class present
        accs.append(float(y[0] == y[0]))
        continue
    clf.fit(X, y)
    p   = clf.predict(X)
    accs.append(accuracy_score(y, p))
print("Zone‐color decode accuracies:",
      ["%.2f"%a for a in accs])

# 9) visualize the first few zones’ true vs. predicted centers
num_z = YPOS.shape[1] // 2
fig, axes = plt.subplots(num_z, 1, figsize=(6, 2*num_z), squeeze=False)
axes = axes.flatten()
for k, ax in enumerate(axes):
    tx, ty = YPOS[:,2*k],   YPOS[:,2*k+1]
    px, py = YPred[:,2*k],  YPred[:,2*k+1]
    ax.scatter(tx, ty, c="C0", label="true", s=10, alpha=0.6)
    ax.scatter(px, py, c="C1", marker="x", label="pred", s=10, alpha=0.6)
    ax.set_title(f"Zone {k} center (true vs pred)")
    ax.legend()
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()
