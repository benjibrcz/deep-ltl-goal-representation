#!/usr/bin/env python3
import os, sys, random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env

# ─── ARGUMENTS ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("World-model PCA with optional zone overlay")
parser.add_argument('--env',         default='PointLtl2-v0')
parser.add_argument('--exp',         default='big_test')
parser.add_argument('--seed',        type=int,   default=0)
parser.add_argument('--formula',     default='F yellow', help='LTL formula (for sampler)')
parser.add_argument('--steps',       type=int,   default=500,  help='Number of steps to collect')
parser.add_argument('--hook-layer',  default='env_net.mlp.3', help='Module to hook')
parser.add_argument('--world-info',  default=None, help='Path to world_info_{i}.pkl to load zones')
parser.add_argument('--device',      default='cpu')
args = parser.parse_args()

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ENV         = args.env
EXP         = args.exp
SEED        = args.seed
FORMULA     = args.formula
STEPS       = args.steps
HOOK_LAYER  = args.hook_layer
DEVICE      = args.device
WORLD_INFO  = args.world_info
# ────────────────────────────────────────────────────────────────────────────────

# 1) Seed all RNGs
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 2) Build the model (for env_net weights & vocab)
sampler   = FixedSampler.partial(FORMULA)
store     = ModelStore(ENV, EXP, SEED)
store.load_vocab()
status    = store.load_training_status(map_location=DEVICE)
cfg       = model_configs[ENV]
dummy_env = make_env(ENV, sampler, sequence=False, render_mode=None)
model     = build_model(dummy_env, status, cfg).to(DEVICE).eval()

# 3) Create env; optionally load a fixed world for zone data
env = make_env(ENV, sampler, sequence=False, render_mode=None)
zone_centers = None
zone_radius  = None
if WORLD_INFO:
    env.load_world_info(WORLD_INFO)
    try:
        zone_centers = np.array(env.zone_positions)
        zone_radius  = getattr(env, 'zone_radius', None)
    except Exception as e:
        print(f"Warning: could not load zones: {e}")

# 4) Hook into the env_net layer
layer = dict(model.named_modules())[HOOK_LAYER]
acts = []
coords = []
zone_ids = []
def grab_activation(module, inp, out):
    # out may be tensor or tuple; take tensor
    tensor_out = out[0] if isinstance(out, tuple) else out
    acts.append(tensor_out.detach().cpu().numpy().squeeze(0))
hook = layer.register_forward_hook(grab_activation)

# 5) Rollout: random policy
obs = env.reset(seed=SEED)
for _ in trange(STEPS, desc="Collecting embeddings"):
    # record position
    pos = env.agent_pos[:2].copy()
    coords.append(pos)
    # record zone id if centers loaded
    if zone_centers is not None:
        dists = np.linalg.norm(zone_centers - pos, axis=1)
        zone_ids.append(int(dists.argmin()))
    # extract features
    assert isinstance(obs, dict)
    raw = obs.get('features')
    if raw is None:
        raise KeyError(f"'features' missing in obs keys: {list(obs.keys())}")
    # trigger hook
    with torch.no_grad():
        inp = torch.tensor(raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        _   = model.env_net(inp)
    # take random action and step
    step_result = env.step(env.action_space.sample())
    # flexibly unpack
    if len(step_result) == 5:
        obs, _, terminated, truncated, _ = step_result
        done = terminated or truncated
    else:
        obs, _, done, _ = step_result
    if done:
        obs = env.reset(seed=SEED)

hook.remove()
env.close()

# 6) Prepare arrays
X      = np.stack(acts)
coords = np.array(coords)
if zone_centers is not None:
    zone_ids = np.array(zone_ids)

# 7) PCA
pca = PCA(n_components=2)
Z   = pca.fit_transform(X)

# 8) Plot side-by-side (2 or 3 panels)
num_panels = 3 if zone_centers is not None else 2
fig, axes = plt.subplots(1, num_panels, figsize=(6*num_panels, 5))
# PCA x
dot0 = axes[0].scatter(Z[:,0], Z[:,1], c=coords[:,0], cmap='viridis', s=5)
fig.colorbar(dot0, ax=axes[0], label='agent x')
axes[0].set_title('PCA colored by x')
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
# PCA y
dot1 = axes[1].scatter(Z[:,0], Z[:,1], c=coords[:,1], cmap='plasma', s=5)
fig.colorbar(dot1, ax=axes[1], label='agent y')
axes[1].set_title('PCA colored by y')
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
# Zones
if zone_centers is not None:
    sc = axes[2].scatter(coords[:,0], coords[:,1], c=zone_ids, cmap='tab10', s=5)
    fig.colorbar(sc, ax=axes[2], label='zone id')
    axes[2].scatter(zone_centers[:,0], zone_centers[:,1], marker='X', c='k', s=100)
    if zone_radius is not None:
        import matplotlib.patches as patches
        for (cx, cy) in zone_centers:
            circle = patches.Circle((cx,cy), zone_radius, fill=False, edgecolor='k')
            axes[2].add_patch(circle)
    axes[2].set_title('Positions by zone')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')

plt.tight_layout()
plt.show()
