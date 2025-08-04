#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0
FORMULA = "GF blue & GF green"
MAX_STEPS = 700
STEERING_STRENGTHS = [0.0, 0.5, 1.0, 2.0, 5.0]  # 0.0 = no steering
STEER_LAYER = 'set_network'

class SetNetworkSteerer:
    def __init__(self, model, steering_direction, steering_strength=1.0):
        self.model = model
        self.steering_direction = torch.tensor(steering_direction, dtype=torch.float32)
        self.steering_strength = steering_strength
        self.original_output = None

    def hook_fn(self, module, input, output):
        if self.steering_strength == 0.0:
            return output
        self.original_output = output.clone()
        steering_adjustment = self.steering_direction * self.steering_strength
        return output + steering_adjustment


def get_layer_and_hook(model, layer_name, hook_fn):
    if layer_name == 'set_network':
        if hasattr(model.ltl_net, 'set_network'):
            handle = model.ltl_net.set_network.register_forward_hook(hook_fn)
            return handle
    return None


def run_steered_rollout(model, env, sampler_fn, steering_direction, steering_strength, world_idx=0):
    print(f"Running steered rollout with strength {steering_strength}...")
    ret = env.reset(seed=SEED + world_idx)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    agent.reset()
    steerer = SetNetworkSteerer(model, steering_direction, steering_strength)
    handle = None
    if steering_strength > 0.0:
        handle = get_layer_and_hook(model, STEER_LAYER, steerer.hook_fn)
    agent_positions = []
    for step in range(MAX_STEPS):
        action = agent.get_action(obs, info, deterministic=True).flatten()
        pos = None
        if isinstance(obs, dict) and 'features' in obs:
            pos = obs['features'][:2]
        agent_positions.append(pos)
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
            term, trunc = done, done
        if done:
            break
    if handle:
        handle.remove()
    return agent_positions


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    sampler_fn = FixedSampler.partial(FORMULA)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, 0)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()

    # Use a random direction for demonstration (replace with probe direction if available)
    dummy_direction = np.ones(model.ltl_net.set_network.out_features)

    for strength in STEERING_STRENGTHS:
        print(f"\n--- Steering strength: {strength} ---")
        env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        positions = run_steered_rollout(model, env, sampler_fn, dummy_direction, strength)
        env.close()
        # You can add visualization or save positions here
        print(f"Trajectory length: {len(positions)}")

if __name__ == '__main__':
    main() 