#!/usr/bin/env python3
"""
Log rollouts to a single NPZ with world/episode grouping and default DeepLTL hooks,
compatible with probe_forward_look.py.

Usage:
  python interpretability/working_scripts/log_rollouts_npz.py \
    --env_id PointLtl2-v0 --exp back_on_cpu \
    --n_worlds 10 --rollouts_per_world 10 --max_steps 500 \
    --use_repo_env --specs "FG blue, FG green, FG yellow, FG magenta" \
    --out_npz interpretability/working_scripts/test_new_fields.npz
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

try:
    import torch
except Exception:
    torch = None  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Log rollouts to NPZ for probing (DeepLTL-compatible)")
    ap.add_argument("--env_id", type=str, default="PointLtl2-v0")
    ap.add_argument("--n_worlds", type=int, default=5)
    ap.add_argument("--rollouts_per_world", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out_npz", type=Path, required=True)

    ap.add_argument("--use_repo_env", action="store_true")
    ap.add_argument("--exp", type=str, default="back_on_cpu")
    ap.add_argument("--spec", type=str, default="FG blue")
    ap.add_argument("--sequence", action="store_true")
    ap.add_argument("--specs", type=str, default="")
    ap.add_argument("--lookahead_steps", nargs="+", type=int, default=[0, 1, 3, 5, 10], 
                    help="Steps ahead to predict (e.g., 0 1 3 5 10). Use 0 for immediate next step.")
    return ap.parse_args()


def flatten(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, dict):
        if "features" in x:
            return np.asarray(x["features"]).reshape(-1)
        return None
    a = np.asarray(x)
    return a.reshape(-1) if a.ndim > 1 else a


def make_env(env_id: str, seed: int, use_repo_env: bool, spec: str, sequence: bool):
    if use_repo_env:
        try:
            SRC = Path(__file__).resolve().parents[2] / "src"
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))
            from envs.env_utils import make_env as repo_make_env  # type: ignore
            from ltl.samplers.fixed_sampler import FixedSampler  # type: ignore
            env = repo_make_env(env_id, FixedSampler.partial(spec), sequence=sequence)
            reset_out = env.reset(seed=seed)
            obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
            return env, obs, {}
        except Exception:
            pass
    env = gym.make(env_id)
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        try:
            env.seed(seed)
        except Exception:
            pass
        obs = env.reset(); info = {}
    return env, obs, info


def main():
    args = parse_args()
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)

    spec_list = [s.strip() for s in args.specs.split(",") if s.strip()] if args.specs else [args.spec]

    # Build DeepLTL agent and register default hooks (if repo env available)
    agent = None
    activations: Dict[str, List[np.ndarray]] = {}
    if args.use_repo_env:
        SRC = Path(__file__).resolve().parents[2] / "src"
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        from utils.model_store.model_store import ModelStore  # type: ignore
        from config import model_configs  # type: ignore
        from model.model import build_model  # type: ignore
        from sequence.search.exhaustive_search import ExhaustiveSearch  # type: ignore
        from model.agent import Agent  # type: ignore
        from envs.env_utils import make_env as repo_make_env  # type: ignore
        from ltl.samplers.fixed_sampler import FixedSampler  # type: ignore

        dummy_env = repo_make_env(args.env_id, FixedSampler.partial(spec_list[0]), sequence=args.sequence)
        store = ModelStore(args.env_id, args.exp, args.seed); store.load_vocab()
        status = store.load_training_status(map_location="cpu")
        model = build_model(dummy_env, status, model_configs[args.env_id]).eval()
        props = set(dummy_env.get_propositions())
        planner = ExhaustiveSearch(model, props, num_loops=2)
        agent = Agent(model, planner, propositions=props)

        def reg_hook(mod, key):
            activations.setdefault(key, [])
            def _cb(module, inp, out):
                y = out
                try:
                    if isinstance(out, (list, tuple)) and len(out) >= 2:
                        y = out[1][-1]
                    y = y.detach().cpu().numpy()
                except Exception:
                    y = np.asarray(y)
                activations[key].append(y.reshape(-1))
            mod.register_forward_hook(_cb)

        # Best-effort defaults
        try:
            reg_hook(getattr(model.env_net, 'mlp')[1], 'hook_env_mlp1')
        except Exception:
            pass
        try:
            reg_hook(getattr(model.ltl_net, 'rnn'), 'hook_ltl_rnn_h')
        except Exception:
            pass
        try:
            reg_hook(getattr(model.actor, 'enc'), 'hook_actor_h5')
        except Exception:
            pass
        try:
            reg_hook(getattr(model.critic, '0'), 'hook_critic_mlp0')
        except Exception:
            pass

    # Build buffer keys dynamically based on lookahead_steps
    base_keys = ["obs", "action", "policy_mu", "ap", "world", "episode_id", "t", "spec"]
    lookahead_keys = []
    for k in args.lookahead_steps:
        lookahead_keys.extend([
            f"next{k}_obs", f"next{k}_ap", f"next{k}_agent_pos", 
            f"next{k}_reward", f"next{k}_done", f"next{k}_wall_lidar", f"next{k}_zone_lidar"
        ])
    
    all_keys = base_keys + lookahead_keys
    buf: Dict[str, List[np.ndarray]] = {k: [] for k in all_keys}

    def append(key: str, val: Any):
        if val is None:
            return
        # Special handling for sets (convert to counts or binary flags)
        if isinstance(val, set):
            val = len(val)  # Convert set to count
        # Flatten observation and sensor arrays
        should_flatten = (key in ("obs", "action", "policy_mu") or 
                         key.endswith("_obs") or key.endswith("_wall_lidar") or key.endswith("_zone_lidar"))
        arr = flatten(val) if should_flatten else val
        buf[key].append(arr)

    # Collect trajectories first, then compute lookaheads
    trajectories = []
    total_steps = 0

    for w in range(args.n_worlds):
        world_seed = args.seed + 1000 * w
        for ep in range(args.rollouts_per_world):
            spec_ep = spec_list[((w * args.rollouts_per_world) + ep) % len(spec_list)]
            episode_id = f"w{w}_ep{ep}"
            env, obs, info = make_env(args.env_id, world_seed + ep, args.use_repo_env, spec_ep, args.sequence)
            if agent is not None:
                agent.reset()
            
            # Collect full trajectory for this episode
            traj = {
                "obs": [], "action": [], "policy_mu": [], "ap": [],
                "reward": [], "done": [], "agent_pos": [], "wall_lidar": [], "zone_lidar": [],
                "world": w, "episode_id": episode_id, "spec": spec_ep
            }
            
            for t in range(args.max_steps):
                # Action
                if agent is not None:
                    action = agent.get_action(obs, {}, deterministic=args.deterministic)
                    mu = agent.get_action(obs, {}, deterministic=True)
                    # Normalize action shape/type to env action space
                    try:
                        a_arr = np.asarray(action).reshape(-1)
                        if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
                            action = int(a_arr[0])
                        else:
                            if hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
                                dim = int(np.prod(env.action_space.shape))
                                a_arr = a_arr[:dim]
                                if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
                                    low = np.asarray(env.action_space.low).reshape(-1)[:dim]
                                    high = np.asarray(env.action_space.high).reshape(-1)[:dim]
                                    a_arr = np.clip(a_arr, low, high)
                                action = a_arr.astype(np.float32)
                            else:
                                action = a_arr.astype(np.float32)
                    except Exception:
                        pass
                else:
                    action = env.action_space.sample()
                    mu = None

                # Store current step
                traj["obs"].append(obs)
                traj["action"].append(action)
                traj["policy_mu"].append(mu)
                traj["ap"].append(info.get("propositions") if isinstance(info, dict) else None)

                # Step environment
                step_out = env.step(action)
                if isinstance(step_out, (list, tuple)):
                    if len(step_out) == 5:
                        obs_next, reward, terminated, truncated, info2 = step_out
                    elif len(step_out) == 4:
                        obs_next, reward, done, info2 = step_out
                        terminated, truncated = bool(done), False
                    else:
                        obs_next = step_out[0]; reward=0.0; terminated=False; truncated=False; info2={}
                else:
                    obs_next = step_out; reward=0.0; terminated=False; truncated=False; info2={}

                # Store outcome
                traj["reward"].append(reward)
                traj["done"].append(terminated or truncated)
                
                # Extract agent position
                if isinstance(obs_next, dict) and "features" in obs_next:
                    features = np.asarray(obs_next["features"])
                    traj["agent_pos"].append(features[:2] if len(features) >= 2 else None)
                elif hasattr(env, "agent_pos"):
                    try:
                        traj["agent_pos"].append(np.asarray(env.agent_pos))
                    except Exception:
                        traj["agent_pos"].append(None)
                else:
                    traj["agent_pos"].append(None)
                
                # Extract sensor data
                if isinstance(obs_next, dict):
                    traj["wall_lidar"].append(obs_next.get("wall_lidar"))
                    traj["zone_lidar"].append(obs_next.get("zone_lidar"))
                elif isinstance(obs_next, np.ndarray) and len(obs_next) >= 20:
                    traj["wall_lidar"].append(obs_next[2:10])  # 8 wall sensors
                    traj["zone_lidar"].append(obs_next[10:18])  # 8 zone sensors
                else:
                    traj["wall_lidar"].append(None)
                    traj["zone_lidar"].append(None)

                obs = obs_next
                info = info2
                if terminated or truncated:
                    break
            
            trajectories.append(traj)
            try:
                env.close()
            except Exception:
                pass

    # Now process trajectories to create lookahead targets
    print(f"Processing {len(trajectories)} trajectories for lookahead targets...")
    
    for traj in trajectories:
        traj_len = len(traj["obs"])
        for i in range(traj_len):
            # Base data
            append("obs", traj["obs"][i])
            append("action", traj["action"][i])
            append("policy_mu", traj["policy_mu"][i])
            append("ap", traj["ap"][i])
            append("world", traj["world"])
            append("episode_id", traj["episode_id"])
            append("t", i)
            append("spec", traj["spec"])
            
            # Lookahead targets - always append for consistency, use None for unavailable
            for k in args.lookahead_steps:
                # nextK means: what happens K+1 steps ahead from current observation
                # next0 = immediate next step (obs[i+1])
                # next1 = 1 step after that (obs[i+2])
                # nextK = K steps after immediate next (obs[i+K+1])
                future_idx = i + k + 1
                
                if future_idx < traj_len:
                    append(f"next{k}_obs", traj["obs"][future_idx])
                    append(f"next{k}_ap", traj["ap"][future_idx])
                    append(f"next{k}_agent_pos", traj["agent_pos"][future_idx-1] if future_idx-1 < len(traj["agent_pos"]) else None)
                    
                    if k == 0:
                        # For next0, use immediate reward and done from current step
                        append(f"next{k}_reward", traj["reward"][i] if i < len(traj["reward"]) else 0.0)
                        append(f"next{k}_done", traj["done"][i] if i < len(traj["done"]) else False)
                    else:
                        # For nextK (K>0), use cumulative reward and any done in horizon
                        append(f"next{k}_reward", sum(traj["reward"][i:future_idx]) if future_idx <= traj_len else 0.0)
                        append(f"next{k}_done", any(traj["done"][i:future_idx]) if future_idx <= traj_len else False)
                    
                    append(f"next{k}_wall_lidar", traj["wall_lidar"][future_idx-1] if future_idx-1 < len(traj["wall_lidar"]) else None)
                    append(f"next{k}_zone_lidar", traj["zone_lidar"][future_idx-1] if future_idx-1 < len(traj["zone_lidar"]) else None)
                else:
                    # Beyond trajectory end - use last available values or defaults
                    last_obs = traj["obs"][-1] if traj["obs"] else None
                    last_ap = traj["ap"][-1] if traj["ap"] else None
                    last_pos = traj["agent_pos"][-1] if traj["agent_pos"] else None
                    last_wall = traj["wall_lidar"][-1] if traj["wall_lidar"] else None
                    last_zone = traj["zone_lidar"][-1] if traj["zone_lidar"] else None
                    
                    append(f"next{k}_obs", last_obs)
                    append(f"next{k}_ap", last_ap) 
                    append(f"next{k}_agent_pos", last_pos)
                    append(f"next{k}_reward", 0.0)  # No more reward after episode end
                    append(f"next{k}_done", True)   # Episode is done
                    append(f"next{k}_wall_lidar", last_wall)
                    append(f"next{k}_zone_lidar", last_zone)
            
            total_steps += 1

    # Build output
    out: Dict[str, np.ndarray] = {}
    for k, vals in buf.items():
        if not vals:
            continue
        try:
            out[k] = np.stack(vals)
        except Exception:
            out[k] = np.array(vals, dtype=object)

    # Hooks
    for k, acts in activations.items():
        if acts:
            try:
                out[k] = np.stack(acts)
            except Exception:
                out[k] = np.array(acts, dtype=object)

    out["meta_json"] = np.array([json.dumps({
        "env_id": args.env_id,
        "exp": args.exp,
        "n_worlds": args.n_worlds,
        "rollouts_per_world": args.rollouts_per_world,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "specs": spec_list,
        "total_steps": total_steps,
    })])

    np.savez_compressed(args.out_npz, **out)
    print(f"Saved NPZ to {args.out_npz} with {total_steps} steps across {args.n_worlds*args.rollouts_per_world} episodes and {args.n_worlds} worlds.")


if __name__ == "__main__":
    main()
