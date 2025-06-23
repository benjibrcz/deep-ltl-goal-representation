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
TOP_K      = 10

ZONE_NAMES = [c.color for c in FlatWorld.CIRCLES]
ZONE_CENTERS = [c.center for c in FlatWorld.CIRCLES]
N_ZONES = len(ZONE_NAMES)

# 1. Load model
formula = "GF blue & GF green"
sampler = FixedSampler.partial(formula)
store = ModelStore(ENV, EXP, SEED)
store.load_vocab()
status = store.load_training_status(map_location='cpu')
cfg    = model_configs[ENV]
dummy  = make_env(ENV, sampler, sequence=False, render_mode=None)
model  = build_model(dummy, status, cfg).eval()
dummy.close()

# 2. Collect features and targets for all zones from a single rollout
build_env = make_env(ENV, sampler, sequence=False, render_mode=None)
props = set(build_env.get_propositions())
agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
module = dict(model.env_net.named_modules())[LAYER_NAME]
world_file = f"eval_datasets/{ENV}/worlds/world_info_{WORLD_IDX}.pkl"
build_env.load_world_info(world_file)
obs = build_env.reset(seed=SEED)
agent.reset()
feats = []
dirs = [[] for _ in range(N_ZONES)]
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
    for zi, zc in enumerate(ZONE_CENTERS):
        u = (zc - pos) / (np.linalg.norm(zc - pos) + 1e-8)
        dirs[zi].append(u)
    if len(feats) < len(dirs[0]):
        feats.append(feats[-1])
h.remove()
if len(feats) > len(dirs[0]):
    feats = feats[:len(dirs[0])]
X = np.stack(feats)
U = [np.stack(d) for d in dirs]
pos_arr = np.stack(pos_list)
build_env.close()

# 3. Fit direction probe for each zone
ridge_us = []
topk_idxs = []
for zi in range(N_ZONES):
    ridge_u = Ridge().fit(X, U[zi])
    probe_weights = ridge_u.coef_  # shape (2, hidden_dim)
    weight_norms = np.linalg.norm(probe_weights, axis=0)
    topk_idx = np.argsort(np.abs(weight_norms))[-TOP_K:][::-1]
    ridge_us.append(ridge_u)
    topk_idxs.append(topk_idx)

# 4. Run ablated rollout, ablating dims for current goal zone at each step
def get_current_goal_zone(agent):
    seq = getattr(agent, "sequence", None)
    if seq and len(seq) > 0:
        goal_set = seq[0][0]
        if len(goal_set) == 1:
            assignment = next(iter(goal_set))
            true_props = {p for p, v in assignment.assignment if v}
            if len(true_props) == 1:
                prop = next(iter(true_props))
                if prop in ZONE_NAMES:
                    return ZONE_NAMES.index(prop)
    return None

build_env = make_env(ENV, sampler, sequence=False, render_mode=None)
build_env.load_world_info(world_file)
obs = build_env.reset(seed=SEED)
agent.reset()
ablated_pos = []
feats_ablated = []
module = dict(model.env_net.named_modules())[LAYER_NAME]

class DynamicAblateHook:
    def __init__(self, topk_idxs, agent):
        self.topk_idxs = topk_idxs
        self.agent = agent
        self.last = None
        self.step = 0
    def __call__(self, m, inp, out):
        x = out[1] if isinstance(out, tuple) else out
        x_np = x.clone().detach().cpu().numpy().copy()
        # Determine current goal zone
        goal_zi = get_current_goal_zone(self.agent)
        if goal_zi is not None:
            x_np[..., self.topk_idxs[goal_zi]] = 0.0
        x_new = torch.from_numpy(x_np).to(x.device)
        self.last = x_new
        self.step += 1
        return (out[0], x_new) if isinstance(out, tuple) else x_new

dablate_hook = DynamicAblateHook(topk_idxs, agent)
h = module.register_forward_hook(dablate_hook)
done = False
for step in range(MAX_STEPS):
    if done:
        break
    a = agent.get_action(obs, {}, deterministic=True).flatten()
    obs, _, done, _ = build_env.step(a)
    ablated_pos.append(build_env.agent_pos[:2].copy())
    feats_ablated.append(dablate_hook.last.cpu().numpy().ravel())
h.remove()
ablated_pos = np.stack(ablated_pos)

# 5. Plot original vs ablated trajectory
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(pos_arr[:, 0], pos_arr[:, 1], label='Original', color='black')
ax.plot(ablated_pos[:, 0], ablated_pos[:, 1], label='Ablated', color='red', linestyle='dashed')
for zi, zc in enumerate(ZONE_CENTERS):
    ax.scatter([zc[0]], [zc[1]], color='blue', s=40, label=ZONE_NAMES[zi] if zi==0 else None)
ax.set_title(f'Trajectory: original vs ablated (top {TOP_K} dir dims for current goal zeroed)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('ablation_trajectory_direction_current_goal.png', dpi=150)
plt.close(fig) 