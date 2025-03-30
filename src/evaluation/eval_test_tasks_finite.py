import argparse
import os
import time

import pandas as pd
from colorama import Fore
from tqdm import tqdm

from evaluation.simulate import simulate
import multiprocessing as mp


env_to_tasks = {
    'PointLtl2-v0': [
        'F (green & (!blue U yellow)) & F magenta',
        '(F blue) & (!blue U (green & F yellow))',
        'F (blue | green) & F yellow & F magenta',
        '!(magenta | yellow) U (blue & F green)',
        '!green U ((blue | magenta) & (!green U yellow))',
        '((green | blue) => (!yellow U magenta)) U yellow'
    ],
    'LetterEnv-v0': [
        'F (a & (!b U c)) & F d',
        '(F d) & (!f U (d & F b))',
        '(F ((a | c | j) & F b)) & (F (c & F d)) & F k',
        '!a U (b & (!c U (d & (!e U f))))',
        '((a | b | c | d) => F (e & (F (f & F g)))) U (h & F i)'
    ],
    'FlatWorld-v0': [
        'F ((red & magenta) & F ((blue & green) & F yellow))',
        'F (orange & (!red U magenta))',
        '(!red U (green & blue & aqua)) & F (orange & (F (red & magenta)))',
        '(((!yellow & !orange) U (green & blue)) & (!green U magenta))',
        '(blue => F magenta) U (yellow | ((green & blue) & F orange))'
    ]
}


def main(env, exp, deterministic):
    num_episodes = 500
    tasks = env_to_tasks[env]
    gamma = {
        'PointLtl2-v0': 0.998,
        'LetterEnv-v0': 0.94,
        'FlatWorld-v0': 0.98
    }[env]
    seeds = range(1, 6)
    results = []
    if os.path.exists(f'results/{env}.csv'):
        df = pd.read_csv(f'results/{env}.csv')
        results = df.values.tolist()
    num_procs = 8
    with mp.Pool(num_procs) as pool:
        for task in tasks:
            print(f'Running task: {task}')
            args = [[env, gamma, exp, seed, num_episodes, task, True, False, deterministic] for seed in seeds]
            for result in tqdm(pool.imap_unordered(eval_task, args), total=len(seeds)):
                (sr, mean_steps), seed = result
                results.append(['DeepLTL', task, seed, sr, mean_steps])
                df = pd.DataFrame(results, columns=['method', 'task', 'seed', 'success_rate', 'mean_steps'])
                os.makedirs('results', exist_ok=True)
                df.to_csv(f'results/{env}.csv', index=False)


def eval_task(simulate_args):
    return simulate(*simulate_args), simulate_args[3]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['PointLtl2-v0', 'LetterEnv-v0', 'FlatWorld-v0'], default='FlatWorld-v0')
    parser.add_argument('--exp', type=str, default='deepset_complex')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.deterministic and args.env != 'PointLtl2-v0':
        print(f'{Fore.RED}NOTE: we only recommend using the --deterministic flag for PointLtl2-v0{Fore.RESET}')
        time.sleep(2)
    main(args.env, args.exp, args.deterministic)
