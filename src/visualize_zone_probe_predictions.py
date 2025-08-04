#!/usr/bin/env python3
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))
from utils.model_store    import ModelStore
from model.model          import build_model
from config               import model_configs
from ltl                  import FixedSampler
from envs                 import make_env
from sequence.search      import ExhaustiveSearch
from model.agent          import Agent
from envs.flatworld       import FlatWorld

ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
WORLD_IDX  = 0
MAX_STEPS  = 200
LAYER_NAME = 'mlp.0'

ZONE_NAMES = [c.color for c in FlatWorld.CIRCLES]
ZONE_CENTERS = [c.center for c in FlatWorld.CIRCLES]
N_ZONES = len(ZONE_NAMES)

# 1. Load model and build probe on env_net.mlp.0
random = np.random
formula = "GF blue & GF green"
sampler = FixedSampler.partial(formula)
store = ModelStore(ENV, EXP, SEED)
store.load_vocab()
status = store.load_training_status(map_location='cpu')
cfg    = model_configs[ENV]
dummy  = make_env(ENV, sampler, sequence=False, render_mode=None)
model  = build_model(dummy, status, cfg).eval()
dummy.close()

# 2. Collect features and targets from a single rollout
build_env = make_env(ENV, sampler, sequence=False, render_mode=None)
props = set(build_env.get_propositions())
agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
module = dict(model.env_net.named_modules())[LAYER_NAME]
world_file = f"eval_datasets/{ENV}/worlds/world_info_{WORLD_IDX}.pkl"
build_env.load_world_info(world_file)
obs = build_env.reset(seed=SEED)
agent.reset()
feats = []
dists = []
dirs = []
pos_list = []
def grab(m, inp, out):
    x = out[1] if isinstance(out, tuple) else out
    feats.append(x.detach().cpu().numpy().ravel())
h = module.register_forward_hook(grab)
done = False
for step in range(MAX_STEPS):
    if done:
        break
    a = agent.get_action(obs, {}, deterministic=True).flatten()
    obs, _, done, _ = build_env.step(a)
    pos = build_env.agent_pos[:2].copy()
    pos_list.append(pos)
    d = [np.linalg.norm(pos - zc) for zc in ZONE_CENTERS]
    u = [(zc - pos) / (np.linalg.norm(zc - pos) + 1e-8) for zc in ZONE_CENTERS]
    dists.append(d)
    dirs.append(np.concatenate(u))
    if len(feats) < len(dists):
        feats.append(feats[-1])
h.remove()
if len(feats) > len(dists):
    feats = feats[:len(dists)]
X = np.stack(feats)
D = np.stack(dists)
U = np.stack(dirs)
pos_arr = np.stack(pos_list)
build_env.close()

# 3. Fit probes on this rollout (for visualization)
ridge_d = Ridge().fit(X, D)
D_pred = ridge_d.predict(X)
ridge_u = Ridge().fit(X, U)
U_pred = ridge_u.predict(X)

# 4. Plot true vs predicted distance for each zone
fig, axes = plt.subplots(N_ZONES, 1, figsize=(8, 2*N_ZONES), sharex=True)
for i, zn in enumerate(ZONE_NAMES):
    axes[i].plot(D[:, i], label='True', color='black')
    axes[i].plot(D_pred[:, i], label='Pred', color='red', alpha=0.7)
    axes[i].set_ylabel(f'{zn} dist')
    axes[i].legend()
axes[-1].set_xlabel('Step')
plt.tight_layout()
plt.savefig('zone_distance_true_vs_pred.png', dpi=150)
plt.close(fig)

# 5. Plot true vs predicted direction (quiver) for each zone
for i, zn in enumerate(ZONE_NAMES):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f'{zn}: True (black) vs Pred (red) directions')
    # Plot agent trajectory
    ax.plot(pos_arr[:, 0], pos_arr[:, 1], color='gray', alpha=0.5, label='Trajectory')
    # Plot zone location
    ax.scatter([ZONE_CENTERS[i][0]], [ZONE_CENTERS[i][1]], color='blue', s=80, label='Zone')
    # Subsample every 10th step
    idx = np.arange(0, len(pos_arr), 10)
    # Quiver: true
    ax.quiver(pos_arr[idx, 0], pos_arr[idx, 1], U[idx, 2*i], U[idx, 2*i+1], color='black', scale=10, width=0.005, label='True')
    # Quiver: pred
    ax.quiver(pos_arr[idx, 0], pos_arr[idx, 1], U_pred[idx, 2*i], U_pred[idx, 2*i+1], color='red', scale=10, width=0.005, label='Pred')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'zone_direction_true_vs_pred_{zn}.png', dpi=150)
    plt.close(fig) 