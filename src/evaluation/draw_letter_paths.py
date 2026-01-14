import sys, random
from pathlib import Path
import argparse

import numpy as np
import torch
from tqdm import trange

SRC = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC))

from envs.env_utils import make_env, get_env_attr
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search.exhaustive_search import ExhaustiveSearch
from utils.model_store.model_store import ModelStore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='LetterEnv-v0')
    p.add_argument('--exp', type=str, default='test')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--formula', type=str, required=True, help='LTL formula to follow')
    p.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--out', type=str, default=str(Path(__file__).resolve().parents[2] / 'interpretability' / 'audit_plots' / 'letter_path.png'))
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    sampler = FixedSampler.partial(args.formula)
    # Use render_mode='path' so we can draw the complete path overlay at the end
    env = make_env(args.env, sampler, render_mode='path')
    cfg = model_configs[args.env]
    store = ModelStore(args.env, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    model = build_model(env, status, cfg)

    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)

    actions = []
    out = env.reset(seed=args.seed)
    obs = out[0] if isinstance(out, (tuple, list)) else out
    agent.reset()
    done = False
    while not done:
        action = agent.get_action(obs, {}, deterministic=args.deterministic)
        action = int(action.flatten()[0])
        actions.append(action)
        step_out = env.step(action)
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            obs, reward, term, trunc, info = step_out
            done = bool(term or trunc)
        else:
            obs, reward, done, info = step_out

    print(f'actions: {actions}')
    # Draw the whole path overlay in a window
    try:
        env.render_path(actions)
    except Exception:
        pass

    # Keep the window open until the user presses a key (or Enter fallback)
    try:
        wait_fn = get_env_attr(env, 'wait_for_input')
        wait_fn()
    except Exception:
        try:
            input("Press Enter to close...")
        except KeyboardInterrupt:
            pass

    # Save a static image if the env supports RGB rendering of the overlay
    # The LetterEnv renderer updates the pygame window; we could optionally dump the screen to file.
    # For now, we focus on visual inspection via the window and printed actions.

    env.close()


if __name__ == '__main__':
    main()


