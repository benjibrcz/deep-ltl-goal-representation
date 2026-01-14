#!/usr/bin/env python3
"""
Intervention: goal-switch memory with clean isolation (single env per trial).

Conditions:
- baseline: reset agent at switch (new plan, new hidden)
- no_reset: carry hidden + planner across switch
- flush_planner: keep hidden, clear planner cache (agent.sequence = None)
- clear_hidden_only: reset hidden but KEEP stale planner cache
- delay_k: run k steps with carryover (hidden+plan), then reset and continue

Success = observe proposition 't' within per_link_steps.

Example:
  python interpretability/working_scripts/intervene_goal_memory.py \
    --env_id PointLtl2-v0 --exp big_test --seed 0 --deterministic \
    --colors "green,blue,yellow,magenta" \
    --num_trials 200 --pre_steps 200 --per_link_steps 200 \
    --delays "0,1,2" \
    --out_csv interpretability/working_scripts/intervene_goal_memory.csv
"""
import argparse, sys, re, gc, os
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from collections import defaultdict, Counter

# ---- repo imports ----
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC))
from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent

try:
    from gymnasium import spaces as gspaces
except Exception:
    from gym import spaces as gspaces  # type: ignore

torch.set_grad_enabled(False)

# ---------- helpers ----------
def goal_for_search(phi: str) -> str:
    m = re.match(r"^\s*F\s+([A-Za-z]+)\s*$", phi)
    return f"FG {m.group(1)}" if m else phi

def reset_unpack(env, **kwargs):
    out = env.reset(**kwargs)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}

def step_unpack(ret):
    if len(ret) == 5:
        obs, rew, terminated, truncated, info = ret
        return obs, rew, bool(terminated or truncated), info
    elif len(ret) == 4:
        obs, rew, done, info = ret
        return obs, rew, bool(done), info
    raise RuntimeError(f"Unexpected env.step return of length {len(ret)}")

def coerce_action(act, action_space):
    if isinstance(action_space, gspaces.Box):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        need = int(np.prod(action_space.shape))
        if a.size == 1 and need > 1: a = np.repeat(a, need)
        if a.size != need: raise ValueError(f"Action size {a.size} != {need}")
        a = np.clip(a, action_space.low, action_space.high)
        return a.reshape(action_space.shape)
    elif isinstance(action_space, gspaces.Discrete):
        a = int(np.asarray(act).ravel()[0]) if isinstance(act,(np.ndarray,list,tuple)) else int(act)
        if not (0 <= a < action_space.n): raise ValueError("Discrete out of range")
        return a
    elif isinstance(action_space, gspaces.MultiDiscrete):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        if a.size != action_space.nvec.size: raise ValueError("MultiDiscrete size mismatch")
        return a
    elif isinstance(action_space, gspaces.MultiBinary):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        need = int(np.prod(action_space.shape))
        if a.size != need: raise ValueError("MultiBinary size mismatch")
        return a.reshape(action_space.shape)
    return act

def saw_prop(info, name: str) -> bool:
    if "propositions" in info:
        return any(str(p) == name for p in info["propositions"])
    return False

def run_until_hit(env, agent, obs, info, target_prop, max_steps, deterministic):
    """Run up to max_steps from the *current* (obs,info); return success bool."""
    for _ in range(max_steps):
        with torch.no_grad():
            act = agent.get_action(obs, info, deterministic=deterministic)
        act = coerce_action(act, env.action_space)
        obs, rew, done, info = step_unpack(env.step(act))
        if saw_prop(info, target_prop):
            return True
        if done:
            return False
    return False

def replay_prephase(env, agent, s_color, steps, deterministic, seed):
    """Reset env to seed, set goal s, agent.reset(), run 'steps' with agent; return (obs,info,done)."""
    obs, info = reset_unpack(env, seed=seed)
    if hasattr(env, "set_goal"):
        env.set_goal(f"FG {s_color}")
    agent.reset()
    done = False
    for _ in range(steps):
        with torch.no_grad():
            act = agent.get_action(obs, info, deterministic=deterministic)
        act = coerce_action(act, env.action_space)
        obs, rew, done, info = step_unpack(env.step(act))
        if done:
            break
    return obs, info, done

def refresh_after_goal_change(env, obs, info):
    """Do a single no-op step to refresh obs/info after env.set_goal(...)."""
    if hasattr(env, "action_space"):
        if hasattr(env.action_space, "shape"):
            a0 = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        else:
            a0 = 0
        a0 = coerce_action(a0, env.action_space)
    else:
        a0 = 0
    obs2, _, done2, info2 = step_unpack(env.step(a0))
    return obs2, info2, done2

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default="PointLtl2-v0")
    ap.add_argument("--exp", default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_loops", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--colors", type=str, default="green,blue,yellow,magenta")
    ap.add_argument("--num_trials", type=int, default=200)
    ap.add_argument("--pre_steps", type=int, default=200, help="budget to reach s before switching")
    ap.add_argument("--per_link_steps", type=int, default=200, help="budget to reach t after switching")
    ap.add_argument("--delays", type=str, default="0,1,2,5")
    ap.add_argument("--no_auto_flush", action="store_true",
                    help="Disable Agent's auto-flush-on-goal-change for this run.")
    ap.add_argument("--out_csv", type=str, default="",
                    help="If set, write per-trial results to this CSV (append if exists).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Build env + agent
    first_goal = f"FG {args.colors.split(',')[0].strip()}"
    dummy = make_env(args.env_id, FixedSampler.partial(goal_for_search(first_goal)), sequence=False)
    cfg   = model_configs[args.env_id]
    store = ModelStore(args.env_id, args.exp, args.seed); store.load_vocab()
    status= store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    props = set(dummy.get_propositions())
    dummy.close()

    planner = ExhaustiveSearch(model, props, num_loops=args.num_loops)
    agent   = Agent(model, planner, propositions=props)
    agent.auto_flush_on_goal = (not args.no_auto_flush)

    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    delays = [int(x.strip()) for x in args.delays.split(",") if x.strip()]

    results = { "baseline": [], "no_reset": [], "flush_planner": [], "clear_hidden_only": [] }
    for d in delays:
        results[f"delay_{d}"] = []
    by_pair = defaultdict(lambda: Counter())
    log_rows = []

    print("\nIntervening on goal-switch memory (isolated, single-env per trial)...\n")
    for trial in range(args.num_trials):
        s = colors[rng.integers(len(colors))]
        t = rng.choice([c for c in colors if c != s])
        world_seed = int(args.seed + 101*trial)

        env = make_env(args.env_id, FixedSampler.partial(goal_for_search(f"FG {s}")), sequence=False)
        try:
            # Determine pre_used steps to a consistent switch state
            obs, info = reset_unpack(env, seed=world_seed)
            if hasattr(env, "set_goal"):
                env.set_goal(f"FG {s}")
            agent.reset()
            pre_used = 0
            while pre_used < args.pre_steps:
                with torch.no_grad():
                    act = agent.get_action(obs, info, deterministic=args.deterministic)
                act = coerce_action(act, env.action_space)
                obs, rew, done, info = step_unpack(env.step(act))
                pre_used += 1
                if saw_prop(info, s) or done:
                    break

            # Helper to record a result row
            def record(condition, ok):
                results[condition].append(int(ok))
                by_pair[(s,t)][condition] += int(ok)
                log_rows.append({
                    "trial": trial, "s": s, "t": t, "condition": condition,
                    "success": int(ok), "pre_used": pre_used,
                    "world_seed": world_seed, "per_link_steps": args.per_link_steps,
                    "auto_flush_on_goal": int(agent.auto_flush_on_goal)
                })

            # baseline: reset all
            obs_c, info_c, done_c = replay_prephase(env, agent, s, pre_used, args.deterministic, seed=world_seed)
            if hasattr(env, "set_goal"): env.set_goal(f"FG {t}")
            obs_c, info_c, done_c = refresh_after_goal_change(env, obs_c, info_c)
            agent.reset()
            ok = 0 if done_c else run_until_hit(env, agent, obs_c, info_c, t, args.per_link_steps, args.deterministic)
            record("baseline", ok)

            # no_reset: carry hidden + planner
            obs_c, info_c, done_c = replay_prephase(env, agent, s, pre_used, args.deterministic, seed=world_seed)
            if hasattr(env, "set_goal"): env.set_goal(f"FG {t}")
            obs_c, info_c, done_c = refresh_after_goal_change(env, obs_c, info_c)
            ok = 0 if done_c else run_until_hit(env, agent, obs_c, info_c, t, args.per_link_steps, args.deterministic)
            record("no_reset", ok)

            # flush_planner: keep hidden, clear plan
            obs_c, info_c, done_c = replay_prephase(env, agent, s, pre_used, args.deterministic, seed=world_seed)
            if hasattr(env, "set_goal"): env.set_goal(f"FG {t}")
            obs_c, info_c, done_c = refresh_after_goal_change(env, obs_c, info_c)
            if hasattr(agent, "sequence"):
                agent.sequence = None
            ok = 0 if done_c else run_until_hit(env, agent, obs_c, info_c, t, args.per_link_steps, args.deterministic)
            record("flush_planner", ok)

            # clear_hidden_only: reset hidden but KEEP stale plan
            obs_c, info_c, done_c = replay_prephase(env, agent, s, pre_used, args.deterministic, seed=world_seed)
            if hasattr(env, "set_goal"): env.set_goal(f"FG {t}")
            obs_c, info_c, done_c = refresh_after_goal_change(env, obs_c, info_c)
            if not done_c:
                saved_plan = getattr(agent, "sequence", None)
                saved_toggle = getattr(agent, "auto_flush_on_goal", False)
                # prevent auto-flush from deleting our stale plan
                agent.auto_flush_on_goal = False
                agent.reset()                      # clears hidden and (likely) sequence
                agent.sequence = saved_plan        # restore stale plan
                ok = run_until_hit(env, agent, obs_c, info_c, t, args.per_link_steps, args.deterministic)
                agent.auto_flush_on_goal = saved_toggle
            else:
                ok = False
            record("clear_hidden_only", ok)

            # delayed reset(s): run d steps with carryover, then reset and continue
            for d in delays:
                obs_c, info_c, done_c = replay_prephase(env, agent, s, pre_used, args.deterministic, seed=world_seed)
                if hasattr(env, "set_goal"): env.set_goal(f"FG {t}")
                obs_c, info_c, done_c = refresh_after_goal_change(env, obs_c, info_c)
                if done_c:
                    record(f"delay_{d}", False)
                    continue
                success_early = False
                steps = 0
                while steps < max(0, d):
                    with torch.no_grad():
                        act = agent.get_action(obs_c, info_c, deterministic=args.deterministic)
                    act = coerce_action(act, env.action_space)
                    obs_c, rew, done_c, info_c = step_unpack(env.step(act))
                    steps += 1
                    if saw_prop(info_c, t):
                        success_early = True
                        break
                    if done_c:
                        break
                if success_early:
                    record(f"delay_{d}", True)
                elif done_c:
                    record(f"delay_{d}", False)
                else:
                    agent.reset()
                    rem = max(0, args.per_link_steps - d)
                    ok = run_until_hit(env, agent, obs_c, info_c, t, rem, args.deterministic)
                    record(f"delay_{d}", ok)

        finally:
            env.close()
            del env
            if (trial + 1) % max(1, args.num_trials // 10) == 0:
                print(f"  progress {trial+1}/{args.num_trials}")
                gc.collect()

    def smoothed_rate(xs):
        xs = np.asarray(xs, dtype=float)
        return (xs.sum() + 0.5) / (len(xs) + 1.0)

    print("\nResults (smoothed success rates):")
    base = smoothed_rate(results["baseline"])
    print(f"  baseline (reset at switch):   {base:.3f}  (N={len(results['baseline'])})")
    nr   = smoothed_rate(results["no_reset"])
    print(f"  no_reset (carry hidden+planner): {nr:.3f}  (Δ={nr-base:+.3f})")
    fp   = smoothed_rate(results["flush_planner"])
    print(f"  flush_planner (keep hidden, clear plan): {fp:.3f}  (Δ={fp-base:+.3f})")
    ch   = smoothed_rate(results["clear_hidden_only"])
    print(f"  clear_hidden_only (reset hidden, keep plan): {ch:.3f}  (Δ={ch-base:+.3f})")
    for d in delays:
        r = smoothed_rate(results[f"delay_{d}"])
        print(f"  delay_{d} (reset after {d}):   {r:.3f}  (Δ={r-base:+.3f})")

    print("\nPer-pair baseline vs no_reset (top 10 pairs):")
    pairs_sorted = sorted(by_pair.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:10]
    for (s,t), cnts in pairs_sorted:
        print(f"  {s:>7s} → {t:<7s} | baseline {cnts['baseline']}  no_reset {cnts['no_reset']}  flush_plan {cnts['flush_planner']}  clear_hidden {cnts['clear_hidden_only']}")

    # Write CSV if requested
    if args.out_csv:
        df = pd.DataFrame(log_rows)
        header = (not os.path.exists(args.out_csv))
        df.to_csv(args.out_csv, mode="a" if not header else "w", header=header, index=False)
        print(f"\nAppended results → {args.out_csv}")

if __name__ == "__main__":
    main()
