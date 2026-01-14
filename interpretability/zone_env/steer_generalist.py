#!/usr/bin/env python3
import os, sys, argparse, torch, numpy as np
from preprocessing.preprocessing import preprocess_obss
from envs import make_env
from ltl import FixedSampler
import torch
from model.agent import Agent
from sequence.search import ExhaustiveSearch
from visualize.zones import draw_trajectories
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1) point Python at src/
sys.path.insert(0, os.path.abspath(os.path.join(__file__,"..","src")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env

def parse_args():
    p = argparse.ArgumentParser(
        description="Activation-steering on your zero-shot DeepLTL agent")
    p.add_argument('--env',    default='PointLtl2-v0')
    p.add_argument('--exp',    default='big_test', 
                   help="folder under experiments/ppo/<env>/")
    p.add_argument('--seed',   type=int, default=0)
    p.add_argument('--layer',  default='net.fc2',
                   help="the module name to hook (e.g. net.fc2)")
    p.add_argument('--alpha',  type=float, default=1.0,
                   help="steering strength multiplier")
    p.add_argument('--eps',    type=int, default=50,
                   help="episodes per eval")
    return p.parse_args()

def load_general_model(env_name, exp, seed, formula):
    store = ModelStore(env_name, exp, seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[env_name]
    sampler= FixedSampler.partial(formula)
    dummy  = make_env(env_name, sampler)
    model  = build_model(dummy, status, cfg)
    model.eval()
    return model

def collect_acts(model, layer, env_name, formula, eps, seed=0):
    acts = []
    h = layer.register_forward_hook(lambda m, i, o: acts.append(o.detach().cpu().squeeze(0)))

    sampler = FixedSampler.partial(formula)
    env     = make_env(env_name, sampler)
    props   = set(env.get_propositions())
    search  = ExhaustiveSearch(model, props, num_loops=2)
    agent   = Agent(model, search=search, propositions=props, verbose=False)

    env.reset(seed=seed)
    torch.random.manual_seed(seed)

    for _ in tqdm(range(eps), desc=f"Collecting `{formula}`"):
        obs = env.reset()
        info = {}
        agent.reset()
        done = False
        while not done:
            try:
                action = agent.get_action(obs, info, deterministic=True).flatten()
            except Exception:
                action = env.action_space.sample()
            obs, _, done, info = env.step(action)

    h.remove()
    return torch.stack(acts).mean(0)


def eval_agent(model, layer, env_name, formula, eps, seed=0,
               steer=False, steer_vec=None, alpha=1.0):
    orig = layer.forward
    if steer:
        def patched(x, *a, **k):
            out = orig(x, *a, **k)
            return out + alpha * steer_vec.to(out.device)
        layer.forward = patched
    else:
        layer.forward = orig

    sampler = FixedSampler.partial(formula)
    env     = make_env(env_name, sampler)
    props   = set(env.get_propositions())
    search  = ExhaustiveSearch(model, props, num_loops=2)
    agent   = Agent(model, search=search, propositions=props, verbose=False)

    env.reset(seed=seed)
    torch.random.manual_seed(seed)

    successes = 0
    for _ in tqdm(range(eps), desc=f"Evaluating `{formula}` steer={steer}"):
        obs = env.reset()
        info = {}
        agent.reset()
        done = False
        while not done:
            try:
                action = agent.get_action(obs, info, deterministic=True).flatten()
            except Exception:
                action = env.action_space.sample()
            obs, _, done, info = env.step(action)
        if info.get("success", False):
            successes += 1

    layer.forward = orig
    return successes / eps


def collect_trajectories(model, layer, env_name, formula, eps, seed=0, steer=False, steer_vec=None, alpha=1.0):
    """
    Run `eps` episodes, return (zone_positions, list of agent-paths).
    If steer=True, patches `layer` with steer_vec.
    """
    # patch/unpatch
    orig = layer.forward
    if steer:
        def patched(x,*a,**k):
            out = orig(x,*a,**k)
            return out + alpha * steer_vec.to(out.device)
        layer.forward = patched
    else:
        layer.forward = orig

    sampler = FixedSampler.partial(formula)
    env     = make_env(env_name, sampler, render_mode=None)
    props   = set(env.get_propositions())
    search  = ExhaustiveSearch(model, props, num_loops=2)
    agent   = Agent(model, search=search, propositions=props, verbose=False)

    env.reset(seed=seed)
    torch.random.manual_seed(seed)

    zone_positions = []
    trajectories   = []
    for _ in range(eps):
        # if your env supports loading world info: env.load_world_info(...)
        zone_positions.append(env.zone_positions)
        obs = env.reset()
        agent.reset()
        done = False
        traj = []
        while not done:
            try:
                action = agent.get_action(obs, {}, deterministic=True).flatten()
            except Exception:
                action = env.action_space.sample()
            obs, _, done, info = env.step(action)
            traj.append(env.agent_pos[:2])
        trajectories.append(traj)

    layer.forward = orig
    return zone_positions, trajectories

def main():
    args = parse_args()

    # 2) load the single generalist model
    print("Loading generalist agent…")
    model = load_general_model(args.env, args.exp, args.seed, "F yellow")

    # 3) grab the layer object
    layer = dict(model.named_modules())[args.layer]

    # 4) collect the yellow-finding vector
    print(f"Collecting activations on F yellow over {args.eps} eps…")
    steer_vec = collect_acts(
        model, layer,
        args.env,
        "F yellow",
        args.eps,
        seed=args.seed
    )
    print("Steering vector norm:", steer_vec.norm().item())

    tests = [
        ("Find blue",               "F blue"),
        ("Find blue & avoid yellow","F blue & G !yellow")
    ]
    for label, formula in tests:
        print(f"\n== {label} ==")

        # baseline
        base = eval_agent(
            model, layer,
            args.env,
            formula,
            args.eps,
            seed=args.seed,
            steer=False
        )
        print(f"  Baseline success:         {base:.2%}")

        # with yellow‐steering
        steered = eval_agent(
            model, layer,
            args.env,
            formula,
            args.eps,
            seed=args.seed,
            steer=True,
            steer_vec=steer_vec,
            alpha=args.alpha
        )
        print(f"  With yellow‐steer vector: {steered:.2%}")

    # 7) sanity-check on find-yellow
    print("\nEvaluating on F yellow (find yellow):")
    findY = eval_agent(
        model, layer,
        args.env,
        "F yellow",
        args.eps,
        seed=args.seed,
        steer=False
    )
    print(f"  Baseline success: {findY:.2%}")

    findY_steered = eval_agent(
        model, layer,
        args.env,
        "F yellow",
        args.eps,
        seed=args.seed,
        steer=True,
        steer_vec=steer_vec,
        alpha=args.alpha
    )
    print(f"  With yellow-steering: {findY_steered:.2%}")

    # 8) visualize a few rollouts side by side
    VIS = min(4, args.eps)  # how many to show
    print(f"\nRendering {VIS} baseline vs steered trajectories...")
    # baseline
    zones_b, trajs_b = collect_trajectories(
        model, layer, args.env, "G !yellow", VIS,
        seed=args.seed, steer=False
    )

    # now steered
    zones_s, trajs_s = collect_trajectories(
        model, layer, args.env, "G !yellow", VIS,
        seed=args.seed, steer=True, steer_vec=steer_vec, alpha=args.alpha
    )

    # plot as two rows of VIS columns
    cols = VIS
    rows = 2
    fig = plt.figure(figsize=(4*cols, 4*rows))
    # top row: baseline
    for i in range(VIS):
        ax = fig.add_subplot(rows, cols, i+1)
        draw_trajectories([zones_b[i]], [trajs_b[i]], num_cols=1, num_rows=1)
        ax.set_title(f"Base #{i+1}")
    # bottom row: steered
    for i in range(VIS):
        ax = fig.add_subplot(rows, cols, cols + i+1)
        draw_trajectories([zones_s[i]], [trajs_s[i]], num_cols=1, num_rows=1)
        ax.set_title(f"Steered #{i+1}")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
