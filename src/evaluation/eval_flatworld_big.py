import argparse
import os

import pandas as pd
from tqdm import tqdm

from config import model_configs
from envs import make_env
from evaluation.simulate import simulate
import multiprocessing as mp

from ltl import FixedSampler
from model.model import build_model
from utils.model_store import ModelStore


def main():
    num_procs = 8
    envs = ['FlatWorld-big-8-v0', 'FlatWorld-big-12-v0', 'FlatWorld-big-16-v0', 'FlatWorld-big-20-v0']
    seeds = [4, 5]
    exp = 'first'
    gamma = 0.98
    deterministic = False
    num_episodes = 500
    #
    # for e in envs:
    #     for s in seeds:
    #         model_store = ModelStore(e, exp, s, None)
    #         training_status = model_store.load_training_status(map_location='cpu')
    #         model_store.load_vocab()
    #         env = make_env(e, FixedSampler.partial('F green'), render_mode=None)
    #         config = model_configs[e]
    #         model = build_model(env, training_status, config)

    with mp.Pool(num_procs) as pool:
        args = [[env, gamma, exp, seed, num_episodes, '', True, False, deterministic, False] for env in envs for seed in
                seeds]
        for result in tqdm(pool.imap_unordered(eval_task, args), total=len(args)):
            (sr, mean_steps, ret), seed, env = result
            num_colors = env.split('-')[-2]
            with open(f'eval_results/FlatWorld-big/results_new.csv', 'a+') as f:
                f.write(f'{num_colors},DeepLTL,{seed},{sr},{mean_steps},{ret}\n')


def eval_task(simulate_args):
    return simulate(*simulate_args), simulate_args[3], simulate_args[0]


if __name__ == '__main__':
    main()
