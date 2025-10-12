#!/usr/bin/env python3
"""
Run the trained DeepLTL agent with the goal/sequence pathway nulled (or scrambled),
to reveal "default" policy fields / latent drives.

Examples:
  python interpretability/working_scripts/run_goal_free.py --mode zero
  python interpretability/working_scripts/run_goal_free.py --mode noise --noise-std 0.5
  python interpretability/working_scripts/run_goal_free.py --spec "FG blue"
  python interpretability/working_scripts/run_goal_free.py --episodes 10 --max-steps 700
"""

import argparse, os, random, sys, csv
from pathlib import Path
from typing import List

import numpy as np
import gymnasium as gym
import torch

# ─────────────────── repo imports ────────────────────
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC))
from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent
# ──────────────────────────────────────────────────────

def register_goal_killer_hooks(model: torch.nn.Module,
                               pattern: str = r"(^ltl_net(\.|$))|(\bltl\b)|(\bsequence\b)|(\bgoal\b)",
                               mode: str = "zero",
                               noise_std: float = 0.5):
    """
    Attach forward hooks to submodules whose *names* match `pattern`.
    Replace their outputs with zeros or Gaussian noise. Returns (handles, stats).
    NOTE: We only match on module *names* to avoid catching class 'Sequential'.
    """
    import re
    compiled = re.compile(pattern, re.IGNORECASE)
    handles = []
    stats = {"n_calls": 0, "modules": []}

    def _make_replacement(t: torch.Tensor):
        if mode == "zero":
            return torch.zeros_like(t)
        elif mode == "noise":
            return torch.randn_like(t) * noise_std
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def hook_fn(module, args, out):
        with torch.no_grad():
            stats["n_calls"] += 1
            if isinstance(out, torch.Tensor):
                return _make_replacement(out)
            elif isinstance(out, (tuple, list)):
                new_out = []
                for item in out:
                    if isinstance(item, torch.Tensor):
                        new_out.append(_make_replacement(item))
                    else:
                        new_out.append(item)
                return type(out)(new_out)
            else:
                return out

    for name, sub in model.named_modules():
        # Only check the *name*, not the class, to avoid matching 'Sequential'
        if compiled.search(name):
            h = sub.register_forward_hook(hook_fn)
            handles.append(h)
            stats["modules"].append(f"{name} ({sub.__class__.__name__})")

    return handles, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="PointLtl2-v0")
    ap.add_argument("--exp", type=str, default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--num-loops", type=int, default=2, help="planner loops for ExhaustiveSearch")
    ap.add_argument("--spec", type=str, default="FG blue", help="env sampler spec (still required by env)")
    ap.add_argument("--sequence", action="store_true", help="use env with sequence rendering if desired")
    ap.add_argument("--mode", choices=["zero","noise"], default="zero", help="zero out or scramble the goal pathway")
    ap.add_argument("--noise-std", type=float, default=0.5)
    ap.add_argument("--hook-pattern", type=str,
                    default=r"(^ltl_net(\.|$))|(\bltl\b)|(\bsequence\b)|(\bgoal\b)",
                    help="regex over module *names* to treat as goal/sequence pathway")
    ap.add_argument("--out", type=str, default="interpretability/working_scripts/rollouts_goal_free.csv")
    args = ap.parse_args()

    ENV, EXP, SEED = args.env_id, args.exp, args.seed
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # ── build dummy env and model (repo-native) ───────
    dummy_env = make_env(ENV, FixedSampler.partial(args.spec), sequence=args.sequence)
    cfg       = model_configs[ENV]
    store     = ModelStore(ENV, EXP, SEED); store.load_vocab()
    status    = store.load_training_status(map_location="cpu")
    model     = build_model(dummy_env, status, cfg).eval()
    torch.set_grad_enabled(False)

    # ── patch goal/sequence pathway ───────────────────
    handles, stats = register_goal_killer_hooks(model,
                                                pattern=args.hook_pattern,
                                                mode=args.mode,
                                                noise_std=args.noise_std)

    print("[goal-free] Hooked modules:")
    if stats["modules"]:
        for m in stats["modules"]:
            print("  -", m)
    else:
        print("  (none matched; try a broader --hook-pattern like 'ltl|goal|sequence|sigma|spec')")

    # ── rollout logger ────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("w", newline="")
    writer = csv.DictWriter(f, fieldnames=[
        "episode","t","reward","done",
        "action_0","action_1",
        "info_agent_x","info_agent_y",
        "chosen_zone_id","current_colour"
    ])
    writer.writeheader()

    # ── run episodes ─────────────────────────────────
    for ep in range(args.episodes):
        env   = make_env(ENV, FixedSampler.partial(args.spec), sequence=args.sequence)
        props = set(env.get_propositions())
        planner = ExhaustiveSearch(model, props, num_loops=args.num_loops)
        agent   = Agent(model, planner, propositions=props)

        # handle reset for both gym and gymnasium
        reset_out = env.reset(seed=SEED + 100 * ep)
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) and len(reset_out) >= 1 else reset_out
        agent.reset()

        done = False
        t = 0
        while not done and t < args.max_steps:
            with torch.no_grad():
                act = agent.get_action(obs, {}, deterministic=True)

            # Prepare action for env (discrete vs continuous)
            a_arr = np.asarray(act).flatten()
            if isinstance(env.action_space, gym.spaces.Discrete):
                step_action = int(a_arr[0])
            else:
                step_action = a_arr

            # Handle 4-tuple (gym) and 5-tuple (gymnasium) step APIs
            step_out = env.step(step_action)
            if isinstance(step_out, (tuple, list)):
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                elif len(step_out) == 4:
                    obs, reward, done_flag, info = step_out
                    done = bool(done_flag)
                else:
                    raise ValueError(f"Unexpected env.step() return length: {len(step_out)}")
            else:
                raise ValueError("env.step() did not return a tuple/list")

            # Capture agent position from observation features (root cause: env info often omits it)
            ax, ay = 0.0, 0.0
            try:
                if isinstance(obs, dict) and 'features' in obs:
                    feats = np.asarray(obs['features']).ravel()
                    if feats.size >= 2:
                        fx, fy = float(feats[0]), float(feats[1])
                        if not (np.isnan(fx) or np.isinf(fx)):
                            ax = fx
                        if not (np.isnan(fy) or np.isinf(fy)):
                            ay = fy
                elif hasattr(env, 'agent_pos'):
                    pos = np.asarray(env.agent_pos)[:2]
                    if pos.size >= 2:
                        px, py = float(pos[0]), float(pos[1])
                        if not (np.isnan(px) or np.isinf(px)):
                            ax = px
                        if not (np.isnan(py) or np.isinf(py)):
                            ay = py
            except Exception:
                pass

            # Derive current colour from Agent's planned sequence if available
            current_colour = ""
            try:
                if getattr(agent, 'sequence', None):
                    current_goal_assignment_set, _ = agent.sequence[0]
                    if isinstance(current_goal_assignment_set, (set, frozenset)):
                        for frozen_assignment in current_goal_assignment_set:
                            if hasattr(frozen_assignment, 'assignment'):
                                props_true = {p for p, v in frozen_assignment.assignment if v}
                                if len(props_true) == 1:
                                    current_colour = next(iter(props_true))
                                    break
            except Exception:
                pass

            # Map colour -> zone id if FlatWorld definitions are available
            chosen_zone_id = ""
            if current_colour:
                try:
                    from envs.flatworld import FlatWorld
                    for i, c in enumerate(FlatWorld.CIRCLES):
                        if getattr(c, 'color', None) == current_colour:
                            chosen_zone_id = i
                            break
                except Exception:
                    pass

            a0 = float(a_arr[0]) if a_arr.size >= 1 else 0.0
            a1 = float(a_arr[1]) if a_arr.size >= 2 else 0.0
            # sanitize actions
            if np.isnan(a0) or np.isinf(a0):
                a0 = 0.0
            if np.isnan(a1) or np.isinf(a1):
                a1 = 0.0

            r_val = float(reward)
            if np.isnan(r_val) or np.isinf(r_val):
                r_val = 0.0

            writer.writerow({
                "episode": ep, "t": t, "reward": r_val, "done": int(done),
                "action_0": a0,
                "action_1": a1,
                "info_agent_x": ax, "info_agent_y": ay,
                "chosen_zone_id": chosen_zone_id,
                "current_colour": current_colour,
            })
            t += 1

        env.close()
        print(f"[ep {ep}] steps={t} last_reward≈{float(reward):.3f}")

    f.close()
    for h in handles: h.remove()
    print(f"[goal-free] Hook calls during run: {stats['n_calls']} across {len(stats['modules'])} modules")
    print(f"[goal-free] Wrote CSV: {out_path.resolve()}")

if __name__ == "__main__":
    main()
