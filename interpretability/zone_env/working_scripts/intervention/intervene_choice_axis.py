#!/usr/bin/env python3
"""
Causal intervention on 'choice axis' to test where the model commits to a zone.

Pipeline:
1) Collect ambiguous two-zone events (first t=0 snapshot only), label which zone is pursued
   via a short lookahead, and capture *pre-activations* for target modules.
2) Fit a 'choice axis' per module = mean(x|y=1) - mean(x|y=0) in z-scored space, then
   transform back to raw space.
3) Re-run fresh worlds and, at the first ambiguous event, inject ±ε * axis on that module's
   *input* (via forward_pre_hook) exactly once; record whether the pursued zone flips.

Targets (editable): actor.enc (pre), critic value head's INPUT (penultimate).
"""

import argparse, sys, math, random, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
import torch
import gymnasium as gym

# ───── repo imports ─────
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC))
from visualize.zones import draw_trajectories
from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent
# ────────────────────────

COLOURS = ["blue", "green", "yellow", "magenta"]

def get_zone_table(env) -> List[Dict]:
    """Robustly extract zone info: [{id, colour, center:(x,y)}]."""
    # unwrap wrappers
    def unwrap(e, depth=5):
        u = e
        for _ in range(depth):
            if hasattr(u, "unwrapped") and u.unwrapped is not None:
                u = u.unwrapped
            elif hasattr(u, "env") and u.env is not None:
                u = u.env
            else:
                break
        return u

    e = unwrap(env)

    # 1) Direct API
    if hasattr(e, "get_zone_info"):
        try:
            out = []
            for z in e.get_zone_info():
                cid = z.get("id", z.get("zone_id"))
                col = (z.get("colour") or z.get("color") or "").lower()
                ctr = z.get("center") or z.get("centre") or z.get("pos")
                if cid is None or not col or ctr is None:
                    continue
                ctr = np.asarray(ctr, dtype=float).ravel()
                if ctr.size >= 2:
                    out.append({"id": cid, "colour": col, "center": (float(ctr[0]), float(ctr[1]))})
            if out:
                return out
        except Exception:
            pass

    # 2) zone_positions dict
    if hasattr(e, "zone_positions") and isinstance(getattr(e, "zone_positions"), dict):
        out = []
        n = 0
        for key, val in e.zone_positions.items():
            try:
                name = str(key).lower()
                col = name.split("_zone", 1)[0] if "_zone" in name else name.split("_")[0]
                ctr = np.asarray(val, dtype=float).ravel()
                if ctr.size >= 2 and col:
                    out.append({"id": n, "colour": col, "center": (float(ctr[0]), float(ctr[1]))})
                    n += 1
            except Exception:
                continue
        if out:
            return out

    # 3) world.zones list
    if hasattr(e, "world") and hasattr(e.world, "zones"):
        out = []
        for idx, z in enumerate(e.world.zones):
            try:
                col = getattr(z, "colour", getattr(z, "color", "")).lower()
                cen = getattr(z, "center", None)
                if cen is None:
                    cx = getattr(z, "cx", getattr(z, "x", None))
                    cy = getattr(z, "cy", getattr(z, "y", None))
                    if cx is not None and cy is not None:
                        cen = (float(cx), float(cy))
                if not col or cen is None:
                    continue
                cen = np.asarray(cen, dtype=float).ravel()
                if cen.size >= 2:
                    out.append({"id": getattr(z, "id", idx), "colour": col, "center": (float(cen[0]), float(cen[1]))})
            except Exception:
                continue
        if out:
            return out

    # 4) common containers
    for attr in ("zones", "zone_centers", "objects", "scene", "task", "world"):
        try:
            c = getattr(e, attr)
        except Exception:
            continue
        if isinstance(c, dict):
            out = []
            n = 0
            for k, v in c.items():
                try:
                    col = str(k).lower()
                    if isinstance(v, dict) and "center" in v:
                        cen = np.asarray(v["center"], dtype=float).ravel()
                    else:
                        cen = np.asarray(v, dtype=float).ravel()
                    if cen.size >= 2 and col:
                        out.append({"id": n, "colour": col, "center": (float(cen[0]), float(cen[1]))})
                        n += 1
                except Exception:
                    continue
            if out:
                return out

    raise RuntimeError("Could not obtain zone info")

def parse_xy_from_obs_or_env(obs, env) -> Optional[np.ndarray]:
    """Return agent (x,y) in environment coordinates.
    Prefer env.agent_pos to avoid mis-indexing observation vectors.
    """
    try:
        if hasattr(env, "agent_pos") and env.agent_pos is not None:
            ap = np.asarray(env.agent_pos, dtype=float).ravel()
            if ap.size >= 2:
                return np.array([float(ap[0]), float(ap[1])], dtype=np.float32)
    except Exception:
        pass
    # Fallbacks deliberately conservative; avoid using generic feature vectors
    if isinstance(obs, dict):
        for k in ("agent_pos", "position", "pos"):
            if k in obs:
                try:
                    v = np.asarray(obs[k], dtype=float).ravel()
                    if v.size >= 2:
                        return np.array([float(v[0]), float(v[1])], dtype=np.float32)
                except Exception:
                    continue
    return None

def reset_unpack(env, **kwargs):
    out = env.reset(**kwargs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out

def is_ambiguous_two_zone(p_t: np.ndarray, zone_xy: Dict[int, np.ndarray], cand_ids: List[int], dist_tol: float) -> bool:
    d = [np.linalg.norm(p_t - zone_xy[z]) for z in cand_ids]
    return len(cand_ids) == 2 and abs(d[0] - d[1]) <= dist_tol

def pursued_label_by_lookahead(env, agent, obs, zone_xy, cand_ids, steps: int = 10) -> int:
    """Return which candidate is pursued in the next few steps (argmin future min distance)."""
    traj = []
    o = obs
    for _ in range(steps):
        with torch.no_grad():
            a = agent.get_action(o, {}, deterministic=True)
        a_arr = np.asarray(a).flatten()
        step_action = int(a_arr[0]) if isinstance(env.action_space, gym.spaces.Discrete) else a_arr
        step_out = env.step(step_action)
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            o, _, term, trunc, _ = step_out
            done = bool(term or trunc)
        else:
            o, _, done, _ = step_out
        p = parse_xy_from_obs_or_env(o, env)
        if p is not None: traj.append(p)
        if done: break
    def min_future_dist(zid):
        if not traj: return 1e9
        z = zone_xy[zid]
        return min(float(np.linalg.norm(pp - z)) for pp in traj)
    z_sorted = sorted(cand_ids)
    return 1 if min_future_dist(z_sorted[0]) < min_future_dist(z_sorted[1]) else 0  # 1 -> first id

# ----- Adaptive ambiguity and lookahead helpers -----
def rel_ambiguity(d1: float, d2: float) -> float:
    s = d1 + d2 + 1e-8
    return abs(d1 - d2) / s

def est_speed(recent_positions: List[np.ndarray]) -> float:
    if len(recent_positions) < 2:
        return 0.0
    ds = [float(np.linalg.norm(recent_positions[i+1]-recent_positions[i])) for i in range(len(recent_positions)-1)]
    if not ds:
        return 0.0
    return float(np.median(ds))

def auto_lookahead(dmin: float, vhat: float, alpha: float = 1.3, kmin: int = 8, kmax: int = 40) -> int:
    if vhat <= 1e-6:
        return kmin
    k = int(np.ceil(alpha * dmin / vhat))
    return int(np.clip(k, kmin, kmax))

def label_pursuit_by_first_hit_or_min_future(traj: List[np.ndarray], cand_ids: List[int], zone_xy: Dict[int, np.ndarray], enter_tol: float = 1.0) -> int:
    for p in traj:
        for j, zid in enumerate(sorted(cand_ids)):
            if float(np.linalg.norm(p - zone_xy[zid])) <= enter_tol:
                return 1 if j == 0 else 0
    def min_future_dist(zid: int) -> float:
        if not traj:
            return 1e9
        z = zone_xy[zid]
        return min(float(np.linalg.norm(pp - z)) for pp in traj)
    z_sorted = sorted(cand_ids)
    return 1 if min_future_dist(z_sorted[0]) < min_future_dist(z_sorted[1]) else 0

def plot_event(fname: str, zone_positions: Dict[str, Tuple[float, float]], p0: np.ndarray, traj: List[np.ndarray], rho: float, K: int, label: int):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    path = [np.asarray(p0, dtype=float)[:2]] + ([np.asarray(t, dtype=float)[:2] for t in traj] if traj else [])
    fig = draw_trajectories([zone_positions], [path], num_cols=1, num_rows=1)
    fig.suptitle(f"rho={rho:.3f}  K={K}  label={'A' if label==1 else 'B'}", fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)

# --- Short-trajectory helpers for audit triplets ---
def step_env_compat(env, action):
    out = env.step(action)
    if isinstance(out, (tuple, list)) and len(out) == 5:
        obs, _rew, term, trunc, _info = out
        return obs, bool(term or trunc)
    elif isinstance(out, (tuple, list)) and len(out) == 4:
        obs, _rew, done, _info = out
        return obs, bool(done)
    raise RuntimeError("Unexpected env.step return")

def record_short_traj(env, agent, obs, K: int) -> List[np.ndarray]:
    traj: List[np.ndarray] = []
    done = False
    for _ in range(K):
        a = agent.get_action(obs, {}, deterministic=True)
        a = int(np.asarray(a).flatten()[0]) if isinstance(env.action_space, gym.spaces.Discrete) else np.asarray(a).flatten()
        obs, done = step_env_compat(env, a)
        p = parse_xy_from_obs_or_env(obs, env)
        if p is not None:
            traj.append(p.copy())
        if done:
            break
    return traj

def save_zone_traj_png(out_path: str, zone_positions, traj: List[np.ndarray], title: str = ""):
    fig = draw_trajectories([zone_positions], [traj], num_cols=1, num_rows=1)
    if title:
        fig.suptitle(title, fontsize=11)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

def visualize_intervention_triplet(env_id, colour, seed, model, args, first_module_for_injection, axis_vec, logit_shift: float = 1.0, rho_gate: float = 0.12, K: int = 20):
    # Baseline to first ambiguous
    env0 = make_env(env_id, FixedSampler.partial(f"FG {colour}"), sequence=args.sequence)
    props0 = set(env0.get_propositions())
    agent0 = Agent(model, ExhaustiveSearch(model, props0, num_loops=args.num_loops), propositions=props0)
    obs0 = reset_unpack(env0, seed=seed); agent0.reset()
    ztab0 = get_zone_table(env0)
    zone_xy0 = {z["id"]: np.array(z["center"], dtype=np.float32) for z in ztab0}
    zone_positions0 = getattr(env0, "zone_positions", [(z["center"][0], z["center"][1]) for z in ztab0])
    found = False
    while True:
        p = parse_xy_from_obs_or_env(obs0, env0)
        cands = [z["id"] for z in ztab0 if z["colour"] == colour]
        if p is not None and len(cands) == 2:
            d1 = float(np.linalg.norm(p - zone_xy0[cands[0]])); d2 = float(np.linalg.norm(p - zone_xy0[cands[1]]))
            rho = abs(d1 - d2) / (d1 + d2 + 1e-8)
            if rho <= rho_gate:
                traj_base = record_short_traj(env0, agent0, obs0, K)
                save_zone_traj_png(os.path.join(args.audit_dir, f"traj_base_seed{seed}_{colour}.png"), zone_positions0, traj_base, f"baseline  rho={rho:.3f}  K={K}")
                found = True
                break
        a = agent0.get_action(obs0, {}, deterministic=True)
        a = int(np.asarray(a).flatten()[0]) if isinstance(env0.action_space, gym.spaces.Discrete) else np.asarray(a).flatten()
        obs0, done0 = step_env_compat(env0, a)
        if done0: break
    env0.close()
    if not found:
        print(f"[viz] no ambiguous moment for seed {seed} / {colour}")
        return

    class _Injector:
        def __init__(self, axis, shift, sign=+1, steps=3):
            self.axis = axis.astype(np.float32); self.shift = float(shift); self.sign = int(sign); self.steps = 0
        def enable(self, steps=3): self.steps = int(steps)
        def hook(self, _m, inputs):
            if self.steps <= 0: return inputs
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.shape[-1] != self.axis.shape[0]:
                self.steps = 0; return inputs
            dx = self.sign * self.shift * torch.from_numpy(self.axis).to(x.device, x.dtype)
            self.steps -= 1
            return (x + dx,)

    # +eps
    envp = make_env(env_id, FixedSampler.partial(f"FG {colour}"), sequence=args.sequence)
    props_p = set(envp.get_propositions()); agentp = Agent(model, ExhaustiveSearch(model, props_p, num_loops=args.num_loops), propositions=props_p)
    obsp = reset_unpack(envp, seed=seed); agentp.reset()
    injp = _Injector(axis_vec, logit_shift, +1, 3)
    hp = first_module_for_injection.register_forward_pre_hook(injp.hook)
    while True:
        p = parse_xy_from_obs_or_env(obsp, envp)
        cands = [z["id"] for z in ztab0 if z["colour"] == colour]
        if p is not None and len(cands) == 2:
            d1 = float(np.linalg.norm(p - zone_xy0[cands[0]])); d2 = float(np.linalg.norm(p - zone_xy0[cands[1]]))
            rho = abs(d1 - d2) / (d1 + d2 + 1e-8)
            if rho <= rho_gate:
                injp.enable(steps=3)
                trajp = record_short_traj(envp, agentp, obsp, K)
                save_zone_traj_png(os.path.join(args.audit_dir, f"traj_pluseps_seed{seed}_{colour}.png"), zone_positions0, trajp, f"+eps  rho={rho:.3f}  K={K}")
                break
        a = agentp.get_action(obsp, {}, deterministic=True)
        a = int(np.asarray(a).flatten()[0]) if isinstance(envp.action_space, gym.spaces.Discrete) else np.asarray(a).flatten()
        obsp, donep = step_env_compat(envp, a)
        if donep: break
    hp.remove(); envp.close()

    # -eps
    envn = make_env(env_id, FixedSampler.partial(f"FG {colour}"), sequence=args.sequence)
    props_n = set(envn.get_propositions()); agentn = Agent(model, ExhaustiveSearch(model, props_n, num_loops=args.num_loops), propositions=props_n)
    obsn = reset_unpack(envn, seed=seed); agentn.reset()
    injn = _Injector(axis_vec, logit_shift, -1, 3)
    hn = first_module_for_injection.register_forward_pre_hook(injn.hook)
    while True:
        p = parse_xy_from_obs_or_env(obsn, envn)
        cands = [z["id"] for z in ztab0 if z["colour"] == colour]
        if p is not None and len(cands) == 2:
            d1 = float(np.linalg.norm(p - zone_xy0[cands[0]])); d2 = float(np.linalg.norm(p - zone_xy0[cands[1]]))
            rho = abs(d1 - d2) / (d1 + d2 + 1e-8)
            if rho <= rho_gate:
                injn.enable(steps=3)
                trajn = record_short_traj(envn, agentn, obsn, K)
                save_zone_traj_png(os.path.join(args.audit_dir, f"traj_minuseps_seed{seed}_{colour}.png"), zone_positions0, trajn, f"-eps  rho={rho:.3f}  K={K}")
                break
        a = agentn.get_action(obsn, {}, deterministic=True)
        a = int(np.asarray(a).flatten()[0]) if isinstance(envn.action_space, gym.spaces.Discrete) else np.asarray(a).flatten()
        obsn, donen = step_env_compat(envn, a)
        if donen: break
    hn.remove(); envn.close()

def resolve_critic_value_head(model):
    """Find critic value Linear; prefer out_features==1. Return (module, name)."""
    value_mod, value_name = None, None
    for name, mod in model.named_modules():
        if name.startswith("critic") and isinstance(mod, torch.nn.Linear) and getattr(mod, "out_features", None) == 1:
            value_mod, value_name = mod, name
    if value_mod is None:
        # fallback to last Linear under critic
        linear_under_critic = [(n, m) for (n, m) in model.named_modules() if n.startswith("critic") and isinstance(m, torch.nn.Linear)]
        if linear_under_critic:
            value_name, value_mod = linear_under_critic[-1]
    return value_mod, value_name

class PreActRecorder:
    """Capture the INPUT to a module via forward_pre_hook."""
    def __init__(self):
        self.buf: List[np.ndarray] = []
    def hook(self, _m, inputs):
        x = inputs[0]
        if isinstance(x, torch.Tensor):
            self.buf.append(x.detach().cpu().numpy().squeeze())
        else:
            self.buf.append(np.array([]))

class Injector:
    """Add ±eps * axis to the INPUT of a module, on the first call after `enable_once()`."""
    def __init__(self, axis: np.ndarray, eps: float, sign: int = +1):
        self.axis = axis.astype(np.float32)
        self.eps  = float(eps)
        self.sign = int(sign)
        self.enabled = False
        self.used = False
        self.valid = bool(np.isfinite(self.axis).all() and (np.linalg.norm(self.axis) > 1e-8))
    def enable_once(self):
        self.enabled = True; self.used = False
    def hook(self, _m, inputs):
        if not self.enabled or self.used or not self.valid:
            return inputs
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return inputs
        if x.shape[-1] != self.axis.shape[0]:
            # dimension mismatch -> skip injection
            self.used = True
            return inputs
        dx = self.sign * self.eps * torch.from_numpy(self.axis).to(x.device, x.dtype)
        self.used = True
        return (x + dx,)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="PointLtl2-v0")
    ap.add_argument("--exp", type=str, default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--num-loops", type=int, default=2)
    ap.add_argument("--colours", type=str, default="blue,green,yellow,magenta")
    ap.add_argument("--sequence", action="store_true")
    ap.add_argument("--dist-tol", type=float, default=0.8, help="ambiguity |d1-d2| threshold")
    ap.add_argument("--lookahead", type=int, default=10, help="steps to label pursued zone")
    ap.add_argument("--epsilons", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--targets", type=str, default="actor.enc,critic.penult",
                    help="comma-separated: actor.enc, critic.penult")
    ap.add_argument("--eval-worlds", type=int, default=80, help="worlds per (module, ε, sign)")
    ap.add_argument("--auto-params", action="store_true", help="auto-select ambiguity threshold and lookahead")
    ap.add_argument("--audit", type=int, default=0, help="if >0, save this many sampled event plots")
    ap.add_argument("--audit-dir", type=str, default="interpretability/audit_plots")
    args = ap.parse_args()

    COLOUR_LIST = [c.strip().lower() for c in args.colours.split(",") if c.strip()]
    EPS_LIST = [float(x) for x in args.epsilons.split(",") if x.strip()]
    TARGETS  = [t.strip() for t in args.targets.split(",") if t.strip()]
    BASE_SEED = args.seed

    rng = np.random.default_rng(BASE_SEED)
    torch.manual_seed(BASE_SEED); np.random.seed(BASE_SEED); random.seed(BASE_SEED)
    torch.set_grad_enabled(False)

    # ── model
    dummy_env = make_env(args.env_id, FixedSampler.partial(f"FG {COLOUR_LIST[0]}"), sequence=args.sequence)
    cfg       = model_configs[args.env_id]
    store     = ModelStore(args.env_id, args.exp, BASE_SEED); store.load_vocab()
    status    = store.load_training_status(map_location="cpu")
    model     = build_model(dummy_env, status, cfg).eval()

    # ── pre-activation recorders for the training (axis fitting) phase
    recorders: Dict[str, PreActRecorder] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    # actor.enc
    if "actor.enc" in TARGETS:
        mod_actor = dict(model.named_modules()).get("actor.enc", None)
        if mod_actor is None:
            print("[warn] target actor.enc not found; removing"); TARGETS = [t for t in TARGETS if t != "actor.enc"]
        else:
            rec = PreActRecorder(); handles.append(mod_actor.register_forward_pre_hook(rec.hook))
            recorders["actor.enc"] = rec

    # critic.penult = INPUT to critic's final value layer
    if "critic.penult" in TARGETS:
        crit_value_mod, crit_value_name = resolve_critic_value_head(model)
        if crit_value_mod is None:
            print("[warn] critic value head not found; removing critic.penult target")
            TARGETS = [t for t in TARGETS if t != "critic.penult"]
        else:
            rec = PreActRecorder(); handles.append(crit_value_mod.register_forward_pre_hook(rec.hook))
            recorders["critic.penult"] = rec

    # ── Phase 1: collect ambiguous events → (X_pre, y) per target
    X_by_tgt: Dict[str, List[np.ndarray]] = {t: [] for t in TARGETS}
    y_choice: List[int] = []
    groups:   List[int] = []

    ep_count = 0
    saved_audits = 0  # deterministic cap for audit plot count
    for ep in (trange(args.episodes, desc="collect", leave=True)):
        colour = COLOUR_LIST[ep % len(COLOUR_LIST)]
        spec   = f"FG {colour}"
        world_seed = BASE_SEED + 777 * ep

        env   = make_env(args.env_id, FixedSampler.partial(spec), sequence=args.sequence)
        props = set(env.get_propositions())
        planner = ExhaustiveSearch(model, props, num_loops=args.num_loops)
        agent   = Agent(model, planner, propositions=props)

        reset_out = env.reset(seed=world_seed)
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        agent.reset()
        ztab = get_zone_table(env)
        zone_xy = {z["id"]: np.array(z["center"], dtype=np.float32) for z in ztab}

        step = 0
        took_snapshot = False
        recent_positions: List[np.ndarray] = []
        while step < args.max_steps:
            with torch.no_grad():
                a = agent.get_action(obs, {}, deterministic=True)

            a_arr = np.asarray(a).flatten()
            act = int(a_arr[0]) if isinstance(env.action_space, gym.spaces.Discrete) else a_arr
            step_out = env.step(act)
            if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                obs, reward, term, trunc, info = step_out
                done = bool(term or trunc)
            else:
                obs, reward, done, info = step_out

            p_t = parse_xy_from_obs_or_env(obs, env)
            if p_t is None:
                step += 1
                if done: break
                continue
            recent_positions.append(p_t)

            cands = [z["id"] for z in ztab if z["colour"] == colour]
            if not took_snapshot and len(cands) == 2:
                # relative ambiguity gate
                dists = [float(np.linalg.norm(p_t - zone_xy[z])) for z in cands]
                d1, d2 = dists[0], dists[1]
                rho = rel_ambiguity(d1, d2)
                if args.auto_params:
                    use_event = (rho <= 0.12)
                else:
                    tau = args.dist_tol if args.dist_tol <= 1.0 else (args.dist_tol / (d1 + d2 + 1e-8))
                    use_event = (rho <= tau)

                if use_event:
                    # get pre-activations (t0)
                    got_any = False
                    for tname, rec in recorders.items():
                        if rec.buf:
                            x = np.array(rec.buf[-1]).ravel().astype(np.float32)
                            X_by_tgt[tname].append(x); got_any = True
                    if got_any:
                        # adaptive lookahead
                        vhat = est_speed(recent_positions[-8:])
                        K = auto_lookahead(min(d1,d2), vhat) if args.auto_params else int(args.lookahead)
                        # roll K steps to build a small trajectory
                        traj = []
                        oK = obs
                        k = 0
                        while k < K:
                            with torch.no_grad():
                                aK = agent.get_action(oK, {}, deterministic=True)
                            a_arrK = np.asarray(aK).flatten()
                            actK = int(a_arrK[0]) if isinstance(env.action_space, gym.spaces.Discrete) else a_arrK
                            step_outK = env.step(actK)
                            if isinstance(step_outK, (tuple, list)) and len(step_outK) == 5:
                                oK, _, termK, truncK, _ = step_outK
                                doneK = bool(termK or truncK)
                            else:
                                oK, _, doneK, _ = step_outK
                            pK = parse_xy_from_obs_or_env(oK, env)
                            if pK is not None: traj.append(pK)
                            k += 1
                            if doneK: break

                        label = label_pursuit_by_first_hit_or_min_future(traj, cands, zone_xy, enter_tol=1.0)
                        y_choice.append(label)
                        groups.append(world_seed)
                        if args.audit:
                            if saved_audits < int(args.audit):
                                fname = os.path.join(args.audit_dir, f"audit_ep{ep}_seed{world_seed}.png")
                                plot_event(fname, getattr(env, "zone_positions", {}), p_t, traj, rho, K, label)
                                saved_audits += 1
                        took_snapshot = True

            step += 1
            if done: break
        env.close()
        ep_count += 1

    for h in handles: h.remove()

    # materialize per-target arrays and fit axes
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    axes_raw: Dict[str, np.ndarray] = {}
    stats: Dict[str, Dict[str, float]] = {}

    y = np.asarray(y_choice, dtype=np.int64)
    g = np.asarray(groups)
    if len(y) < 80:
        print(f"[warn] only {len(y)} ambiguous samples; consider more --episodes")

    for tname in TARGETS:
        X_list = X_by_tgt[tname]
        if not X_list:
            print(f"[axis] {tname}: no samples")
            continue
        X = np.vstack(X_list).astype(np.float32)

        # world-split to build axis on train
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=BASE_SEED+1)
        tr, te = next(gss.split(X, y, groups=g))

        sc = StandardScaler(with_mean=True, with_std=True).fit(X[tr])
        Xz_tr = sc.transform(X[tr])
        y_tr = y[tr]
        if (y_tr==0).sum() == 0 or (y_tr==1).sum() == 0:
            print(f"[axis] {tname}: insufficient class balance in train fold; skipping")
            continue
        m0 = Xz_tr[y_tr==0].mean(axis=0)
        m1 = Xz_tr[y_tr==1].mean(axis=0)
        axis_z = (m1 - m0)
        norm = np.linalg.norm(axis_z)
        if not np.isfinite(norm) or norm <= 1e-8:
            print(f"[axis] {tname}: degenerate axis; skipping")
            continue
        axis_z /= norm

        # back to raw space: raw = z / std
        std = sc.scale_
        with np.errstate(divide='ignore', invalid='ignore'):
            axis_raw = np.where(std > 0, axis_z / std, 0.0).astype(np.float32)
        if not np.isfinite(axis_raw).all() or np.linalg.norm(axis_raw) <= 1e-8:
            print(f"[axis] {tname}: non-finite/zero axis; skipping")
            continue
        axes_raw[tname] = axis_raw
        stats[tname] = {"dim": float(X.shape[1]), "train_n": float(len(tr)), "test_n": float(len(te))}

    # Optional: visualize a few baseline / +eps / -eps triplets for quick audit
    try:
        if args.audit and "actor.enc" in axes_raw and axes_raw["actor.enc"].size > 0:
            mod_inject = dict(model.named_modules()).get("actor.enc", None)
            if mod_inject is not None:
                seeds_demo = [BASE_SEED + 1337, BASE_SEED + 2027]
                colour_demo = COLOUR_LIST[0] if COLOUR_LIST else "green"
                for sd in seeds_demo:
                    visualize_intervention_triplet(
                        env_id=args.env_id,
                        colour=colour_demo,
                        seed=sd,
                        model=model,
                        args=args,
                        first_module_for_injection=mod_inject,
                        axis_vec=axes_raw["actor.enc"],
                        logit_shift=1.5,
                        rho_gate=0.12,
                        K=20,
                    )
    except Exception as e:
        print(f"[audit-triplet] skipped due to: {e}")

    # ── Phase 2: Intervene and measure flips
    def eval_module(tname: str, axis: np.ndarray, eps: float, sign: int) -> Tuple[int,int]:
        """Return (#flips, #total) over args.eval_worlds."""
        flips, total = 0, 0

        # prepare injector and register the pre-hook on the right module
        handle_inj = None
        try:
            if tname == "actor.enc":
                mod = dict(model.named_modules())["actor.enc"]
            elif tname == "critic.penult":
                crit_value_mod, _ = resolve_critic_value_head(model)
                if crit_value_mod is None: return 0,0
                mod = crit_value_mod
            else:
                return 0, 0

            injector = Injector(axis, eps, sign)
            handle_inj = mod.register_forward_pre_hook(injector.hook)

            for rep in range(args.eval_worlds):
                colour = COLOURS[rep % len(COLOURS)]
                spec = f"FG {colour}"
                seed0 = BASE_SEED + 9000 + rep

                # baseline world
                env0 = make_env(args.env_id, FixedSampler.partial(spec), sequence=args.sequence)
                props0 = set(env0.get_propositions())
                agent0 = Agent(model, ExhaustiveSearch(model, props0, num_loops=args.num_loops), propositions=props0)
                o0 = reset_unpack(env0, seed=seed0); agent0.reset()
                ztab0 = get_zone_table(env0)
                zone_xy0 = {z["id"]: np.array(z["center"], dtype=np.float32) for z in ztab0}
                base_label = None
                done = False
                while not done:
                    p = parse_xy_from_obs_or_env(o0, env0)
                    cands = [z["id"] for z in ztab0 if z["colour"] == colour]
                    if base_label is None and p is not None and is_ambiguous_two_zone(p, zone_xy0, cands, args.dist_tol):
                        base_label = pursued_label_by_lookahead(env0, agent0, o0, zone_xy0, cands, steps=args.lookahead)
                        break
                    a0 = agent0.get_action(o0, {}, deterministic=True)
                    a_arr0 = np.asarray(a0).flatten()
                    step_action0 = int(a_arr0[0]) if isinstance(env0.action_space, gym.spaces.Discrete) else a_arr0
                    s0 = env0.step(step_action0)
                    o0 = s0[0]; done = s0[2] if len(s0)==4 else (s0[2] or s0[3])
                env0.close()

                if base_label is None:
                    continue  # no ambiguous event in this world

                # intervened world (same seed)
                env1 = make_env(args.env_id, FixedSampler.partial(spec), sequence=args.sequence)
                props1 = set(env1.get_propositions())
                agent1 = Agent(model, ExhaustiveSearch(model, props1, num_loops=args.num_loops), propositions=props1)
                o1 = reset_unpack(env1, seed=seed0); agent1.reset()
                ztab1 = get_zone_table(env1)
                zone_xy1 = {z["id"]: np.array(z["center"], dtype=np.float32) for z in ztab1}

                # we enable the injector *once*, at the first ambiguous moment
                done = False
                alt_label = None
                while not done:
                    p = parse_xy_from_obs_or_env(o1, env1)
                    cands = [z["id"] for z in ztab1 if z["colour"] == colour]
                    if alt_label is None and p is not None and is_ambiguous_two_zone(p, zone_xy1, cands, args.dist_tol):
                        injector.enable_once()  # next module call will get nudged once
                        alt_label = pursued_label_by_lookahead(env1, agent1, o1, zone_xy1, cands, steps=args.lookahead)
                        break
                    a1 = agent1.get_action(o1, {}, deterministic=True)
                    a_arr1 = np.asarray(a1).flatten()
                    step_action1 = int(a_arr1[0]) if isinstance(env1.action_space, gym.spaces.Discrete) else a_arr1
                    s1 = env1.step(step_action1)
                    o1 = s1[0]; done = s1[2] if len(s1)==4 else (s1[2] or s1[3])
                env1.close()

                if alt_label is not None:
                    flips += int(alt_label != base_label)
                    total += 1

        finally:
            if handle_inj is not None:
                handle_inj.remove()
        return flips, total

    # evaluate each target × ε × sign
    for tname, axis in axes_raw.items():
        if axis.size == 0:
            print(f"[intervene] {tname}: empty axis")
            continue
        print(f"\n[intervene] target={tname}  dim={axis.shape[0]}  eps={EPS_LIST}")
        for eps in EPS_LIST:
            flips_p, tot_p = eval_module(tname, axis, eps, +1)
            flips_n, tot_n = eval_module(tname, axis, eps, -1)
            rate_p = (flips_p / max(1, tot_p)) * 100.0
            rate_n = (flips_n / max(1, tot_n)) * 100.0
            print(f"  eps={eps:.2f}  +ε flips {flips_p}/{tot_p} ({rate_p:.1f}%)  "
                  f"-ε flips {flips_n}/{tot_n} ({rate_n:.1f}%)")

if __name__ == "__main__":
    main()
