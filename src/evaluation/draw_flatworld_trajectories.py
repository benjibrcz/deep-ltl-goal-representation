import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

from envs import make_env
from envs.flatworld import FlatWorld
from ltl import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore
from visualize.zones import  setup_axis
import seaborn as sns

sns.set_theme(style='whitegrid')

env_name = 'FlatWorld-v0'
exp = 'deepset_complex'
seed = 1

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

sampler = FixedSampler.partial('F (red & magenta) & F (blue & green)')  # FG blue & GF green & GF aqua
deterministic = False

env = make_env(env_name, sampler, render_mode=None)
config = model_configs[env_name]
model_store = ModelStore(env_name, exp, seed)
training_status = model_store.load_training_status(map_location='cpu')
print(training_status['curriculum_stage'])
model_store.load_vocab()
model = build_model(env, training_status, config)

props = set(env.get_propositions())
search = ExhaustiveSearch(model, props, num_loops=2)
agent = Agent(model, search=search, propositions=props, verbose=False)

num_episodes = 8
trajectories = []
steps = []
success = []

for i in trange(num_episodes):
    if i > 3:
        sampler = FixedSampler.partial('(!red U (green & blue & aqua)) & F (orange & (F (red & magenta)))')  # FG blue & GF green & GF aqua
        env = make_env(env_name, sampler, render_mode=None)
    traj = []
    obs, info = env.reset(), {}
    traj.append(env.agent_pos)
    agent.reset()
    done = False
    num_steps = 0

    while not done:
        action = agent.get_action(obs, info, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        num_steps += 1
        traj.append(env.agent_pos)
        if done:
            success.append("success" in info)
            break
    trajectories.append(traj)
    steps.append(num_steps)

env.close()
print('SR:', np.mean(success))
print('Steps:', np.mean(steps))


fig = plt.figure(figsize=(20, 10))
cols = 4 if len(trajectories) > 4 else len(trajectories)
rows = 1 if len(trajectories) <= 4 else 2
for i, traj in enumerate(trajectories):
    ax = fig.add_subplot(rows, cols, i + 1)
    # setup_axis(ax)
    FlatWorld.render(traj, ax=ax)

plt.tight_layout(pad=4)
plt.savefig('/home/mathias/tmp/traj2.pdf', bbox_inches='tight')
plt.show()
