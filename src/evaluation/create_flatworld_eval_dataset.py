import os
import random

import numpy as np
import torch
from tqdm import trange

from envs import make_env
from ltl import AvoidSampler
from ltl.samplers.flatworld_avoid_sampler import FlatWorldAvoidSampler
from ltl.samplers.flatworld_reach_sampler import FlatWorldReachSampler
from ltl.samplers.reach_sampler import ReachSampler
from ltl.samplers.super_sampler import SuperSampler

seed = 1
num_eval_episodes = 50
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

def main():
    avoid_sampler = FlatWorldAvoidSampler.partial(2, ltl2action_format=True)
    reach_sampler = FlatWorldReachSampler.partial(3, ltl2action_format=True)
    sampler = SuperSampler.partial(reach_sampler, avoid_sampler)

    props = ['red', 'magenta', 'blue', 'green', 'aqua', 'yellow', 'orange', ('red', 'magenta'), ('blue', 'green'),
             ('green', 'aqua'), ('blue', 'aqua'), ('blue', 'green', 'aqua')]
    sample = sampler(props)

    path = f'eval_datasets/FlatWorld-v0'
    os.makedirs(path, exist_ok=True)

    for i in trange(num_eval_episodes):
        formula, ltl2action_formula = sample()
        with open(f'{path}/tasks.txt', 'a+') as f:
            f.write(formula)
            f.write('\n')
        with open(f'{path}/tasks_ltl2action.txt', 'a+') as f:
            f.write(str(ltl2action_formula))
            f.write('\n')


if __name__ == '__main__':
    main()
