import os

import pandas as pd
from tqdm import tqdm

from evaluation.simulate import simulate

import argparse
import multiprocessing as mp

env_to_tasks = {
    'PointLtl2-v0': [
        'GF blue & GF green',
        'GF blue & GF green & GF yellow & G !magenta',
        'FG blue',
        'FG blue & F (yellow & F green)',
    ],
    'LetterEnv-v0': [
        'GF (e & (!a U f))',
        'GF a & GF b & GF c & GF d & G (!e & !f)',
    ],
    'FlatWorld-v0': [
        'GF (blue & green) & GF (red & magenta)',
        'GF (aqua & blue) & GF red & GF yellow & G !green'
    ]
}


def main(env, exp):
    num_episodes = 500
    tasks = env_to_tasks[env]
    gamma = {
        'PointLtl2-v0': 0.998,
        'LetterEnv-v0': 0.94,
        'FlatWorld-v0': 0.98
    }[env]
    seeds = range(1, 6)
    results = []
    if os.path.exists(f'results_infinite/{env}.csv'):
        df = pd.read_csv(f'results_infinite/{env}.csv')
        results = df.values.tolist()
    num_procs = 8
    with mp.Pool(num_procs) as pool:
        for task in tasks:
            print(f'Running task: {task}')
            # deterministic = task.startswith('GF')  # TODO: we evaluate deterministic policies for some tasks
            deterministic = False
            args = [[env, gamma, exp, seed, num_episodes, task, False, False, deterministic] for seed in seeds]
            for result in tqdm(pool.imap_unordered(eval_task, args), total=len(seeds)):
                accepting_visits, seed = result
                results.append(['DeepLTL', task, seed, accepting_visits])
                df = pd.DataFrame(results, columns=['method', 'task', 'seed', 'accepting_visits'])
                os.makedirs('results_infinite', exist_ok=True)
                df.to_csv(f'results_infinite/{env}.csv', index=False)


def eval_task(simulate_args):
    return simulate(*simulate_args), simulate_args[3]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['PointLtl2-v0', 'LetterEnv-v0', 'FlatWorld-v0'],
                        default='PointLtl2-v0')
    parser.add_argument('--exp', type=str, default='deepset')
    args = parser.parse_args()
    main(args.env, args.exp)
