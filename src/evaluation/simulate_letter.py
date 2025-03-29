import random

import numpy as np
import torch
from tqdm import trange

from envs import make_env
from ltl import PartiallyOrderedSampler
from ltl.samplers.avoid_multiple_sampler import AvoidMultipleSampler
from ltl.samplers.avoid_sampler import AvoidSampler
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import BFS, DijkstraSearch, ExhaustiveSearch
from utils.model_store import ModelStore

env_name = 'LetterEnv-v0'
exp = 'deepltl'
seed = 1
render_modes = [None, 'human', 'path']
render = render_modes[1]
render_on_fail = False

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

# sampler = RandomSequenceSampler.partial(length=2, unique=True)
# sampler = FixedSequenceSampler.partial([('b', 'a')])
# sampler = PartiallyOrderedSampler.partial(depth=15, num_conjuncts=1, disjunct_prob=0.25, as_list=False)
# sampler = PartiallyOrderedSampler.partial(depth=3, num_conjuncts=2, as_list=False, disjunct_prob=0)
# sampler = AvoidSampler.partial(depth=2, num_conjuncts=1)
# sampler = AvoidSampler.partial(depth=2, num_conjuncts=1)
# sampler = AvoidSampler.partial(depth=6, num_conjuncts=1)
# sampler = AvoidMultipleSampler.partial(depth=1, num_avoid=2)
# sampler = FixedSampler.partial('GF k & GF e & G (h => F f)')
# sampler = FixedSampler.partial('F (a & (!d U c))')
sampler = FixedSampler.partial('F ((a | b) & (GF k & GF e))')

deterministic = False

env = make_env(env_name, sampler, max_steps=75, render_mode=render)
config = model_configs[env_name]
model_store = ModelStore(env_name, exp, seed)
model_store.load_vocab()
training_status = model_store.load_training_status(map_location='cpu')
model = build_model(env, training_status, config)

props = set(env.get_propositions())
# search = BFS(model)
search = ExhaustiveSearch(model, props, num_loops=2)
agent = Agent(model, search=search, propositions=props, verbose=render is not None)

num_episodes = 500

num_successes = 0
num_violations = 0
rets = []
success_mask = []
steps = []
props = []

env.reset(seed=seed)

finish = False
pbar = range(num_episodes) if render else trange(num_episodes)
for i in pbar:
    actions = []
    obs = env.reset()
    agent.reset()
    info = {'ldba_state_changed': True}
    if render:
        print(obs['goal'])
    done = False
    num_steps = 0
    while not done:
        action = agent.get_action(obs, info, deterministic=deterministic)
        action = action.flatten()[0]
        actions.append(action)
        obs, reward, done, info = env.step(action)
        if len(info['propositions']) > 0:
            props.append(list(info['propositions'])[0])
        if render == 'human':
            print(reward)
            finish = env.wait_for_input()
            if finish:
                break
        num_steps += 1
        if done:
            if 'success' in info:
                num_successes += 1
                final_reward = 1
                steps.append(num_steps)  # only count steps for successful episodes
            elif 'violation' in info:
                num_violations += 1
                final_reward = -1
            else:
                final_reward = 0
            rets.append(final_reward * 0.94 ** (num_steps - 1))
            success_mask.append('success' in info)

            if render == 'path':
                if render_on_fail and final_reward != -1:
                    continue
                print(final_reward)
                env.render_path(actions)
                print(props)
                props = []
                finish = env.wait_for_input()
            elif not render:
                pbar.set_postfix({
                    's': num_successes / (i + 1),
                    'v': num_violations / (i + 1),
                    'ADR(t)': np.mean(rets),
                    'ADR(s)': np.mean(np.array(rets)[success_mask]),
                    'AS': np.mean(steps)
                })
    if finish:
        break

env.close()
print(f'Success rate: {num_successes / num_episodes:.3f}')
print(f'Violation rate: {num_violations / num_episodes:.3f}')
print(f'Num total: {num_episodes}')
print(f'ADR (total): {np.mean(rets):.3f}')
print(f'ADR (successful): {np.mean(np.array(rets)[success_mask]):.3f}')
print(f'AS: {np.mean(steps):.3f}')

print(
    f'{num_successes / num_episodes:.3f},{num_violations / num_episodes:.3f},{np.mean(rets):.3f},{np.mean(np.array(rets)[success_mask]):.3f},{np.mean(steps):.3f}')
