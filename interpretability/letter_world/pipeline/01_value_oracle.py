import argparse
from pathlib import Path
import random

import numpy as np
import torch

import sys
SRC = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC))

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from config import model_configs
from utils.model_store import ModelStore
from preprocessing.preprocessing import preprocess_obss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='LetterEnv-v0')
    p.add_argument('--exp', type=str, default='test')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--formula', type=str, required=True)
    p.add_argument('--state_seed', type=int, default=1234, help='seed to reset env state for evaluation')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Build env with fixed formula; SequenceWrapper/LDBAWrapper provide obs.seq, ldba state, epsilon mask
    sampler = FixedSampler.partial(args.formula)
    env = make_env(args.env, sampler, render_mode=None)
    cfg = model_configs[args.env]
    store = ModelStore(args.env, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    model = build_model(env, status, cfg).eval()

    # Sample a state by resetting with a fixed seed (simple proxy); grab obs
    out = env.reset(seed=args.state_seed)
    obs = out[0] if isinstance(out, (tuple, list)) else out
    dl = preprocess_obss([obs], set(env.get_propositions()))
    with torch.no_grad():
        _, value = model(dl)
    print(float(value.squeeze(0)))


if __name__ == '__main__':
    main()


