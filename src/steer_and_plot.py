#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from model.agent       import Agent
from sequence.search   import ExhaustiveSearch
from visualize.zones   import draw_trajectories

def parse_args():
    p = argparse.ArgumentParser(description="Steer & Plot zero-shot DeepLTL agent")
    p.add_argument('--env',            default='PointLtl2-v0')
    p.add_argument('--exp',            default='big_test')
    p.add_argument('--seed',           type=int, default=0)
    p.add_argument('--layer',          required=True,
                   help="Module to hook (e.g. actor.enc.3)")
    p.add_argument('--eps',            type=int, default=50,
                   help="Episodes for steering vector & each eval")
    p.add_argument('--alpha',          type=float, default=1.0,
                   help="Steering strength multiplier")
    p.add_argument('--steer-formula',  default='F yellow',
                   help="LTL formula to collect steering vector from")
    p.add_argument('--test',           action='append', required=True,
                   help="Test spec as label:formula, e.g. 'FindBlue:F blue'. Repeatable.")
    p.add_argument('--monitor-prop',
                   type=str,
                   default=None,
                   help="Name of an atomic proposition to watch (e.g. 'blue')")
    p.add_argument('--plot-k',         type=int, default=1,
                       help="How many trajectories to plot per cell")
    p.add_argument('--v-steer',        type=str, default=None,
                   help="Path to a precomputed steering vector (e.g. v_steer.pt)")
    return p.parse_args()

def load_model(env, exp, seed, formula):
    store = ModelStore(env, exp, seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[env]
    sampler= FixedSampler.partial(formula)
    dummy  = make_env(env, sampler)
    model  = build_model(dummy, status, cfg)
    model.eval()
    return model

def collect_acts(model, layer, env_name, formula, eps, seed):
    acts = []

    def grab_rnn(module, inp, out):
        # out is (seq_output, h_n); we want the final hidden state h_n of shape [1, hidden_dim]
        if isinstance(out, tuple):
            h_n = out[1]                # GRU/LSTM cell hidden state
        else:
            h_n = out                    # fallback for non‐RNN layers
        # squeeze batch dim if present
        acts.append(h_n.detach().cpu().squeeze(0))

    hook = layer.register_forward_hook(grab_rnn)
    #hook = layer.register_forward_hook(lambda m, i, o: acts.append(o.detach().cpu().squeeze(0)))

    sampler = FixedSampler.partial(formula)
    env0    = make_env(env_name, sampler)
    props   = set(env0.get_propositions())
    search  = ExhaustiveSearch(model, props, num_loops=2)
    agent   = Agent(model, search=search, propositions=props, verbose=False)

    env0.reset(seed=seed)
    torch.random.manual_seed(seed)

    for _ in tqdm(range(eps), desc=f"Collect `{formula}`"):
        obs = env0.reset()
        info = {}
        agent.reset()
        done = False
        while not done:
            try:
                a = agent.get_action(obs, info, deterministic=True).flatten()
            except:
                a = env0.action_space.sample()
            obs, _, done, info = env0.step(a)

    hook.remove()
    return torch.stack(acts).mean(0)

def eval_agent(model, layer, env_name, formula, eps, seed,
               steer, steer_vec, alpha, monitor_prop=None):
    """
    Returns (success_rate, violation_rate, monitor_rate) over eps episodes.
    """
    # 1) Patch/unpatch the steering vector
    orig = layer.forward
    if steer:
        def fwd(x,*a,**k):
            return orig(x,*a,**k) + alpha * steer_vec.to(x.device)
        layer.forward = fwd
    else:
        layer.forward = orig

    # 2) Build env, get propositions
    sampler     = FixedSampler.partial(formula)
    env0        = make_env(env_name, sampler)
    props_list  = env0.get_propositions()     # e.g. ['blue','green','yellow']
    props_set   = set(props_list)

    # 3) Pre-compute the index of monitor_prop in props_list (if any)
    if monitor_prop is not None:
        try:
            prop_idx = props_list.index(monitor_prop)
        except ValueError:
            raise ValueError(f"Unknown proposition {monitor_prop!r}; choices: {props_list}")

    # 4) Build the search+agent
    search  = ExhaustiveSearch(model, props_set, num_loops=2)
    agent   = Agent(model, search=search, propositions=props_set, verbose=False)

    # 5) Seed
    env0.reset(seed=seed)
    torch.random.manual_seed(seed)

    succ = vio = mon = 0
    for _ in tqdm(range(eps), desc=f"Eval `{formula}` steer={steer}"):
        obs = env0.reset()
        info = {}
        agent.reset()
        done = False

        seen = False
        while not done:
            try:
                a = agent.get_action(obs, info, deterministic=True).flatten()
            except:
                a = env0.action_space.sample()
            obs, _, done, info = env0.step(a)

            # 6) Track monitor_prop in any format
            if monitor_prop is not None and not seen:
                pr = obs.get('propositions', None)
                if isinstance(pr, dict):
                    seen = bool(pr.get(monitor_prop, False))
                elif isinstance(pr, set):
                    seen = (monitor_prop in pr)
                elif pr is not None:
                    # assume list/array aligned to props_list
                    seen = bool(pr[prop_idx])

        # 7) Episode‐level tally
        if info.get("success", False):
            succ += 1
        if info.get("violation", False):
            vio += 1
        if monitor_prop is not None and seen:
            mon += 1

    # 8) Restore and return
    layer.forward = orig
    return succ/eps, vio/eps, (mon/eps if monitor_prop is not None else None)



def collect_trajectory(model, layer, env_name, formula, seed, steer, steer_vec, alpha):
    orig = layer.forward
    if steer:
        def fwd(x,*a,**k):
            return orig(x,*a,**k) + alpha * steer_vec.to(x.device)
        layer.forward = fwd
    else:
        layer.forward = orig

    sampler = FixedSampler.partial(formula)
    env0    = make_env(env_name, sampler)
    props   = set(env0.get_propositions())
    search  = ExhaustiveSearch(model, props, num_loops=2)
    agent   = Agent(model, search=search, propositions=props, verbose=False)

    env0.reset(seed=seed)
    torch.random.manual_seed(seed)

    zone_pos = env0.zone_positions
    obs = env0.reset(seed=seed)
    agent.reset()
    done = False
    path = []
    while not done:
        try:
            a = agent.get_action(obs, {}, deterministic=True).flatten()
        except:
            a = env0.action_space.sample()
        obs, _, done, info = env0.step(a)
        path.append(env0.agent_pos[:2])

    layer.forward = orig
    return zone_pos, path

def fig_to_array(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    rgb = buf[..., :3]
    plt.close(fig)
    return rgb

def main():
    args = parse_args()

    print("Loading model …")
    model = load_model(args.env, args.exp, args.seed, args.steer_formula)
    layer = dict(model.named_modules())[args.layer]

    # 1) Steering vector
    if args.v_steer:
        steer_vec = torch.load(args.v_steer)
    else:
        steer_vec = collect_acts(
            model, layer, args.env,
            args.steer_formula, args.eps, args.seed
        )
    print("Steering vector norm:", steer_vec.norm().item())

    # 2) Parse test specs: label:formula
    tests = []
    for spec in args.test:
        if ':' not in spec:
            raise ValueError("Each --test must be label:formula")
        lbl, frm = spec.split(':', 1)
        tests.append((lbl, frm))

    # 3) Evaluate each test
    # Prepare lists for all three metrics
    base_succ = []; base_vio = []; base_mon = []
    steer_succ = []; steer_vio = []; steer_mon = []
    for label, formula in tests:
        print(f"\n=== {label} ===")
        b_s, b_v, b_m = eval_agent(
        model, layer, args.env, formula,
        args.eps, args.seed,
        steer=False, steer_vec=None, alpha=args.alpha,
        monitor_prop=args.monitor_prop
        )
        s_s, s_v, s_m = eval_agent(
            model, layer, args.env, formula,
            args.eps, args.seed,
            steer=True,  steer_vec=steer_vec, alpha=args.alpha,
            monitor_prop=args.monitor_prop
        )
        line = f"Baseline → Success: {b_s:.2%}, Violation: {b_v:.2%}"
        if b_m is not None:
            line += f"  ⍟ {b_m:.2%}"
        print(line)

        line = f"Steered  → Sucess: {s_s:.2%}, Violation: {s_v:.2%}"
        if s_m is not None:
            line += f"  ⍟ {s_m:.2%}"
        print(line)

        # Append to your lists
        base_succ.append(b_s)
        base_vio.append(b_v)
        base_mon.append(b_m if b_m is not None else 0.0)

        steer_succ.append(s_s)
        steer_vio.append(s_v)
        steer_mon.append(s_m if s_m is not None else 0.0)

    # 4) Render one trajectory per test × (baseline, steered)
    K = args.plot_k
    rows, cols = len(tests), 2
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))

    for i, (label, formula) in enumerate(tests):
        # collect K baseline rollouts (each with a different seed offset)
        zb_list, tb_list = [], []
        for j in range(K):
            z, t = collect_trajectory(
                model, layer, args.env, formula,
                seed=args.seed + j,
                steer=False, steer_vec=None, alpha=args.alpha
            )
            zb_list.append(z)
            tb_list.append(t)

        # collect K steered rollouts
        zs_list, ts_list = [], []
        for j in range(K):
            z, t = collect_trajectory(
                model, layer, args.env, formula,
                seed=args.seed + j,
                steer=True, steer_vec=steer_vec, alpha=args.alpha
            )
            zs_list.append(z)
            ts_list.append(t)

        # Plot baseline cell
        ax0 = axes[i][0] if rows>1 else axes[0]
        f0  = draw_trajectories(zb_list, tb_list, num_cols=K, num_rows=1)
        im0 = fig_to_array(f0)
        ax0.imshow(im0); ax0.axis('off')
        # Baseline
        title0 = f"{label} (Baseline)\n" \
                f"Success: {base_succ[i]:.1%}, Violation: {base_vio[i]:.1%}"
        if args.monitor_prop:
            title0 += f", {args.monitor_prop.capitalize()} found: {base_mon[i]:.1%}"
        ax0.set_title(title0)

        # Plot steered cell
        ax1 = axes[i][1] if rows>1 else axes[1]
        f1  = draw_trajectories(zs_list, ts_list, num_cols=K, num_rows=1)
        im1 = fig_to_array(f1)
        ax1.imshow(im1); ax1.axis('off')
        title1 = f"{label} (Steered)\n" \
         f"Success: {steer_succ[i]:.1%}, Violation: {steer_vio[i]:.1%}"
        if args.monitor_prop:
            title1 += f", {args.monitor_prop.capitalize()} found: {steer_mon[i]:.1%}"
        ax1.set_title(title1)

    plt.tight_layout()
    plt.savefig("steering_summary.png", dpi=150)
    plt.show()

    print("Saved steering_summary.png")

if __name__ == '__main__':
    main()
