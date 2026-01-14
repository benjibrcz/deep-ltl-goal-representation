import sys, random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

SRC = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC))

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search.exhaustive_search import ExhaustiveSearch
from utils.model_store.model_store import ModelStore
from visualize.zones import draw_trajectories

env_name = 'PointLtl2-v0'
exp = 'big_test'
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

sampler = FixedSampler.partial('FG green')
deterministic = True

env = make_env(env_name, sampler, render_mode=None)
config = model_configs[env_name]
model_store = ModelStore(env_name, exp, seed)
model_store.load_vocab()
training_status = model_store.load_training_status(map_location='cpu')
model = build_model(env, training_status, config)

props = set(env.get_propositions())
search = ExhaustiveSearch(model, props, num_loops=2)
agent = Agent(model, search=search, propositions=props, verbose=False)

num_episodes = 5

trajectories = []
zone_poss = []

pbar = trange(num_episodes)
for i in pbar:
    env.load_world_info(f'eval_datasets/PointLtl2-v0/worlds/world_info_{i}.pkl')
    out = env.reset()
    obs = out[0] if isinstance(out, (tuple, list)) else out
    agent.reset()
    done = False

    zone_poss.append(env.zone_positions)
    agent_traj = []

    while not done:
        action = agent.get_action(obs, {}, deterministic=deterministic)
        action = action.flatten()
        step_out = env.step(action)
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            obs, reward, term, trunc, info = step_out
            done = bool(term or trunc)
        else:
            obs, reward, done, info = step_out
        pos = getattr(env, 'agent_pos', None)
        if pos is not None:
            agent_traj.append(np.asarray(pos, dtype=float)[:2])
        if done:
            trajectories.append(agent_traj)

env.close()
cols = 4 if len(zone_poss) > 4 else len(zone_poss)
rows = 1 if len(zone_poss) <= 4 else 2
fig = draw_trajectories(zone_poss, trajectories, cols, rows)
out_path = Path(__file__).resolve().parents[2] / 'interpretability' / 'audit_plots' / 'zone_trajectories.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=140, bbox_inches='tight')
print(f"Saved {out_path}")
