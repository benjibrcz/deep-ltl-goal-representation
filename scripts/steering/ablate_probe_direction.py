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
ZONE_NAME  = 'blue'  # Probe for direction to blue zone
TOP_K      = 10

ZONE_NAMES = [c.color for c in FlatWorld.CIRCLES]
ZONE_CENTERS = [c.center for c in FlatWorld.CIRCLES]
ZONE_IDX = ZONE_NAMES.index(ZONE_NAME)

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
    u = (ZONE_CENTERS[ZONE_IDX] - pos) / (np.linalg.norm(ZONE_CENTERS[ZONE_IDX] - pos) + 1e-8)
    dirs.append(u)
    if len(feats) < len(dirs):
        feats.append(feats[-1])
h.remove()
if len(feats) > len(dirs):
    feats = feats[:len(dirs)]
X = np.stack(feats)
U = np.stack(dirs)
pos_arr = np.stack(pos_list)
build_env.close()

# 3. Fit direction probe for blue zone
ridge_u = Ridge().fit(X, U)
probe_weights = ridge_u.coef_  # shape (2, hidden_dim)
# For direction, combine x and y weights by L2 norm
weight_norms = np.linalg.norm(probe_weights, axis=0)
topk_idx = np.argsort(np.abs(weight_norms))[-TOP_K:][::-1]

# 4. Run ablated rollout
build_env = make_env(ENV, sampler, sequence=False, render_mode=None)
build_env.load_world_info(world_file)
obs = build_env.reset(seed=SEED)
agent.reset()
ablated_pos = []
feats_ablated = []
module = dict(model.env_net.named_modules())[LAYER_NAME]

# Helper: run forward with ablation
class AblateHook:
    def __init__(self, topk_idx):
        self.topk_idx = topk_idx
        self.last = None
    def __call__(self, m, inp, out):
        x = out[1] if isinstance(out, tuple) else out
        x_np = x.clone().detach().cpu().numpy().copy()
        x_np[..., self.topk_idx] = 0.0
        x_new = torch.from_numpy(x_np).to(x.device)
        self.last = x_new
        return (out[0], x_new) if isinstance(out, tuple) else x_new

ablate_hook = AblateHook(topk_idx)
h = module.register_forward_hook(ablate_hook)
done = False
for step in range(MAX_STEPS):
    if done:
        break
    a = agent.get_action(obs, {}, deterministic=True).flatten()
    obs, _, done, _ = build_env.step(a)
    ablated_pos.append(build_env.agent_pos[:2].copy())
    feats_ablated.append(ablate_hook.last.cpu().numpy().ravel())
h.remove()
ablated_pos = np.stack(ablated_pos)

# 5. Plot original vs ablated trajectory
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(pos_arr[:, 0], pos_arr[:, 1], label='Original', color='black')
ax.plot(ablated_pos[:, 0], ablated_pos[:, 1], label='Ablated', color='red', linestyle='dashed')
ax.scatter([ZONE_CENTERS[ZONE_IDX][0]], [ZONE_CENTERS[ZONE_IDX][1]], color='blue', s=80, label='Zone')
ax.set_title(f'Trajectory: original vs ablated (top {TOP_K} dir dims zeroed)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('ablation_trajectory_direction_blue.png', dpi=150)
plt.close(fig) 