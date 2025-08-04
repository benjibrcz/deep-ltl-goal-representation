#!/usr/bin/env python3
import random
import numpy as np
import torch
from tqdm import trange

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from envs.flatworld    import FlatWorld
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

ENV       = "PointLtl2-v0"
EXP       = "big_test"
SEED      = 1
MAX_STEPS = 1000
FORMULA   = "GF blue & GF green"

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    sampler_fn = FixedSampler.partial(FORMULA)

    # Load model and agent
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store     = ModelStore(ENV, EXP, 0)
    store.load_vocab()
    status    = store.load_training_status(map_location="cpu")
    cfg       = model_configs[ENV]
    model     = build_model(build_env, status, cfg).eval()
    search    = ExhaustiveSearch(model, set(props), num_loops=2)
    agent     = Agent(model, search=search, propositions=set(props), verbose=True)
    build_env.close()

    # Rollout
    rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    ret = rollout_env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()

    print("\nStarting rollout...")
    for step in trange(MAX_STEPS, desc="Rolling out"): 
        action = agent.get_action(obs, info, deterministic=True).flatten()
        agent_pos = None
        if isinstance(obs, dict) and 'features' in obs:
            features = obs['features']
            if isinstance(features, np.ndarray) or (hasattr(features, '__getitem__') and len(features) >= 2):
                agent_pos = features[:2]
        current_goal = None
        seq = getattr(agent, "sequence", None)
        if seq is not None and hasattr(seq, '__len__') and len(seq) > 0:
            goal_set = seq[0][0]
            if goal_set is not None and hasattr(goal_set, '__len__') and len(goal_set) == 1 and hasattr(goal_set, '__iter__'):
                assignment = next(iter(goal_set))
                true_props = {p for p, v in assignment.assignment if v}
                if len(true_props) == 1:
                    current_goal = next(iter(true_props))
        print(f"Step {step:3d} | Agent pos: {agent_pos} | Current subgoal: {current_goal}")
        print(f"  obs: {obs}")
        print(f"  info: {info}")
        ret = rollout_env.step(action)
        if isinstance(ret, tuple) and len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        elif isinstance(ret, tuple) and len(ret) == 4:
            obs, rew, done, info = ret
            term, trunc = done, done
        else:
            raise RuntimeError(f"Unexpected step return: {ret}")
        if done:
            print("Episode finished.")
            break
    rollout_env.close()

if __name__ == '__main__':
    main() 