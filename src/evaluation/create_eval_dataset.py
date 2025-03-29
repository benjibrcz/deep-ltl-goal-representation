import os
import random

import numpy as np
import torch
from tqdm import trange

from envs import make_env
from ltl import AvoidSampler
from ltl.samplers.reach_sampler import ReachSampler
from ltl.samplers.super_sampler import SuperSampler

env_name = 'PointLtl2-v0'
seed = 1
num_eval_episodes = 50
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

def main():
    avoid_sampler = AvoidSampler.partial(2, 1, ltl2action_format=True)
    reach_sampler = ReachSampler.partial(3, ltl2action_format=True)
    sampler = SuperSampler.partial(reach_sampler, avoid_sampler)
    env = make_env(env_name, sampler, render_mode=None, max_steps=1000)
    env.reset(seed=seed)
    path = 'eval_datasets/zones'
    os.makedirs(f'{path}/worlds', exist_ok=True)

    for i in trange(num_eval_episodes):
        obs = env.reset()
        env.save_world_info(f'{path}/worlds/world_info_{i}.pkl')
        formula = obs['goal']
        with open(f'{path}/tasks.txt', 'a+') as f:
            f.write(formula)
            f.write('\n')
        ltl2action_formula = obs['ltl2action_goal']
        with open(f'{path}/tasks_ltl2action.txt', 'a+') as f:
            f.write(str(ltl2action_formula))
            f.write('\n')


if __name__ == '__main__':
    main()
