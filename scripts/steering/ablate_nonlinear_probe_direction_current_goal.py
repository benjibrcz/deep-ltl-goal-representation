#!/usr/bin/env python3
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
WORLD_IDX  = 32
MAX_STEPS  = 400
LAYER_NAME = 'mlp.0'
TOP_K      = 10
MLP_HIDDEN = 64
MLP_EPOCHS = 30
MLP_LR     = 1e-3
ABLATE_AFTER = 50

ZONE_NAMES = [c.color for c in FlatWorld.CIRCLES]
ZONE_CENTERS = [c.center for c in FlatWorld.CIRCLES]
N_ZONES = len(ZONE_NAMES)

def fit_mlp_probe(X, Y, hidden=64, epochs=30, lr=1e-3):
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim)
    )
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = mlp(X_t)
        loss = ((pred - Y_t) ** 2).mean()
        loss.backward()
        optimizer.step()
    return mlp

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

# 3. Fit MLP direction probe for each zone
mlp_us = []
topk_idxs = []
for zi in range(N_ZONES):
    mlp = fit_mlp_probe(X, U[zi], hidden=MLP_HIDDEN, epochs=MLP_EPOCHS, lr=MLP_LR)
    mlp_us.append(mlp)
    # Use input layer weights for ablation
    W = mlp[0].weight.detach().cpu().numpy()  # shape (hidden, in_dim)
    weight_norms = np.linalg.norm(W, axis=0)
    topk_idx = np.argsort(np.abs(weight_norms))[-TOP_K:][::-1]
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
        if goal_zi is not None and self.step >= ABLATE_AFTER:
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
# Plot all zones in their actual color
for zi, zc in enumerate(ZONE_CENTERS):
    ax.scatter([zc[0]], [zc[1]], color=ZONE_NAMES[zi], s=40, label='Zone' if zi == 0 else None)
# Plot a star at the current subgoal at each step
for step in range(len(ablated_pos)):
    goal_zi = get_current_goal_zone(agent)
    if goal_zi is not None:
        ax.scatter([ZONE_CENTERS[goal_zi][0]], [ZONE_CENTERS[goal_zi][1]], color=ZONE_NAMES[goal_zi], s=120, marker='*', edgecolor='black', linewidths=1.5, zorder=5, label='Current subgoal' if step == 0 else None)
# Add a vertical line at the ablation start step (projected onto trajectory)
if ABLATE_AFTER < len(ablated_pos):
    ax.scatter([ablated_pos[ABLATE_AFTER, 0]], [ablated_pos[ABLATE_AFTER, 1]], color='red', s=80, marker='|', label='Ablation start', zorder=6)
ax.set_title(f'Trajectory: original vs ablated (top {TOP_K} MLP dir dims for current goal zeroed after {ABLATE_AFTER})')
ax.set_xlabel('x')
ax.set_ylabel('y')
handles, labels = ax.get_legend_handles_labels()
# Remove duplicate labels
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('ablation_trajectory_nonlinear_direction_current_goal.png', dpi=150)
plt.close(fig) 