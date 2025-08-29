#!/usr/bin/env python3
"""
DeepLTL — Clean linear-probe suite for the Actor Fusion layer.

What it does
------------
1) Builds env+model from the DeepLTL repo (PointLtl2-v0), loads weights (EXP/SEED).
2) At each agent step, captures:
   • X_in  : actor fusion *input* embedding (model.compute_embedding)
   • X_out : policy head parameters (logits OR mean/log_std), i.e., what the actor will use
3) Builds target labels Y from stable sources (obs['features'], env.agent_pos, simple heuristics).
4) Trains simple linear probes (Ridge or LogisticRegression) with a *world-held-out* split.
5) Prints a compact summary table, optionally saves CSV.

Notes
-----
• Only depends on fields that exist in ZoneEnv (80-D features + agent_pos). Keep it robust.
• If you want extra targets, add them in TARGETS dict below.
"""

import os, sys, argparse, random, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from tqdm import trange

# ─── DeepLTL imports (repo layout) ────────────────────────────────────────────
# Ensure we import from src/ like the training/eval scripts
THIS = os.path.dirname(__file__)
SRC  = os.path.abspath(os.path.join(THIS, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent




COLOURS = ["blue", "green", "yellow", "magenta"]
COLOUR2IDX = {c:i for i,c in enumerate(COLOURS)}

class HeadTap:
    """
    Registers pre-hooks on *all* Linear layers. For each forward pass,
    we record the input to each Linear in call order. When the final
    output layer (out_features in out_dim_candidates) is hit, we can
    pull the last three 64-D inputs immediately before it.

    This gives us:
      - L1 pre-hook input  -> fusion (96-D)          [we already have X_in]
      - L2 pre-hook input  -> H1 (64-D, after ReLU1)
      - L3 pre-hook input  -> H2 (64-D, after ReLU2)
      - Lout pre-hook input-> H3 (64-D, after ReLU3)
    """
    def __init__(self, model, hidden_dim=64, out_dim_candidates=(2, 4), verbose=False):
        self.model = model
        self.hidden_dim = hidden_dim
        self.out_dim_candidates = set(out_dim_candidates)
        self.verbose = verbose

        self._handles = []
        self._step_calls = []   # list of dicts with: name, mod, in_feat, out_feat, x (np array)
        self._final_idx = None  # index in _step_calls where final out layer was called (this step)

        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                # register a pre-hook to see the *input* to this Linear
                def make_pre(name_, mod_):
                    def _pre(_m, inp):
                        x = inp[0]
                        if not torch.is_tensor(x):
                            return
                        x_np = x.detach().cpu().ravel().numpy()
                        self._step_calls.append(dict(
                            name=name_,
                            mod=mod_,
                            in_feat=mod_.in_features,
                            out_feat=mod_.out_features,
                            x=x_np
                        ))
                        if mod_.out_features in self.out_dim_candidates:
                            # assume this is the policy output layer
                            self._final_idx = len(self._step_calls) - 1
                    return _pre
                self._handles.append(m.register_forward_pre_hook(make_pre(name, m)))

    def begin_step(self):
        self._step_calls.clear()
        self._final_idx = None

    def end_step(self):
        """
        Return dict with possible keys: h1, h2, h3 (each 64-D np.array).
        These are chosen as the *last three 64-D Linear inputs* before the final output layer.
        """
        out = {}
        if self._final_idx is None:
            return out  # no forward seen
        # Work backwards from the final output layer, pick last three 64-D inputs
        i = self._final_idx - 1
        hits = []
        while i >= 0 and len(hits) < 3:
            rec = self._step_calls[i]
            if rec["in_feat"] == self.hidden_dim:
                hits.append(rec["x"])
            i -= 1
        hits = hits[::-1]  # chronological order: H1, H2, H3
        if len(hits) >= 1: out["h1"] = hits[0]
        if len(hits) >= 2: out["h2"] = hits[1]
        if len(hits) >= 3: out["h3"] = hits[2]
        return out

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def fvec(obs):  # 80-D features as np.float32
    if isinstance(obs, dict) and "features" in obs:
        return np.asarray(obs["features"], dtype=np.float32)
    return np.asarray(obs, dtype=np.float32)

def wall_lidar(obs):
    return fvec(obs)[3:19]

def zone_lidar(obs):
    return fvec(obs)[19:35]

def gyro(obs):
    return fvec(obs)[38:41]

def vel(obs):
    return fvec(obs)[35:38]

def contacts(obs):
    # fallback safe slice; skip if too short
    fv = fvec(obs)
    return fv[41:45] if fv.shape[0] >= 45 else np.zeros(4, np.float32)

# --- new “indirect” transforms ---------------------------------------------

def t_front_clearance(_env, obs, *_):
    rays = wall_lidar(obs)
    c = len(rays)//2
    k = 3
    return np.array([float(rays[max(0,c-k):c+k+1].mean())], dtype=np.float32)

def t_min_wall_dist(_env, obs, *_):
    return np.array([wall_lidar(obs).min()], dtype=np.float32)

def t_wall_sector_argmin(_env, obs, *_):
    return np.array([int(np.argmin(wall_lidar(obs)))], dtype=int)

def t_free_space_ahead(_env, obs, *_):
    rays = wall_lidar(obs)
    k = 3                    # forward window size
    center = len(rays)//2    # assume front-ish beams are around center
    val = rays[max(0, center-k):center+k+1].mean()
    return np.array([1 if val > 0.25 else 0], dtype=int)

def t_nearest_zone_id(_env, obs, *_):
    return np.array([int(np.argmax(zone_lidar(obs)))], dtype=int)

def t_nearest_zone_id_geom(env, _obs, *_):
    if not hasattr(env, "zone_positions"):
        return np.array([0], dtype=int)
    apos = np.asarray(env.agent_pos[:2], dtype=np.float32)
    zs = sorted(env.zone_positions.items())  # stable order
    dists = [np.linalg.norm(np.asarray(p[:2], dtype=np.float32) - apos) for _, p in zs]
    return np.array([int(np.argmin(dists))], dtype=int)

def t_nearest_zone_dir_cls(_env, obs, *_):
    return np.array([int(np.argmax(zone_lidar(obs)) % 16)], dtype=int)

def t_in_zone_flags(env, obs, *_):
    # prefer propositions if present; else threshold zone lidar
    props = obs.get("propositions", []) if isinstance(obs, dict) else []
    if props:
        flags = [1 if c in props else 0 for c in COLOURS]
    else:
        zl = zone_lidar(obs)
        # 4 colours × 4 beams each (if ordering matches); fallback: max-pooling into 4 bins
        bins = np.array([zl[i::4].max() for i in range(4)])
        flags = [1 if x > 0.4 else 0 for x in bins]
    return np.asarray(flags, dtype=int)

def t_contacts(env, obs, *_):
    return contacts(obs).astype(np.float32)

def t_goal_colour_id(env, obs, pre, post, *, rollout_goal_colour=None, **_):
    cid = COLOUR2IDX.get(rollout_goal_colour or "", 0)
    return np.array([cid], dtype=int)

def t_bearing_to_goal_cls(env, obs, pre, post, *, rollout_goal_colour=None, **_):
    # Preferred: exact bearing to color center, if available
    if hasattr(env, "zone_positions") and rollout_goal_colour in getattr(env, "zone_positions", {}):
        zpos = np.asarray(env.zone_positions[rollout_goal_colour][:2], dtype=np.float32)
        apos = np.asarray(env.agent_pos[:2], dtype=np.float32)
        d = zpos - apos
        ang = np.arctan2(d[1], d[0])  # [-pi, pi)
        cls = int(np.floor((ang + np.pi) / (2*np.pi/16))) % 16
        return np.array([cls], dtype=int)
    # Fallback: use zone-lidar argmax direction (color-agnostic)
    rays = zone_lidar(obs)
    cls = int(np.argmax(rays)) % 16
    return np.array([cls], dtype=int)

def t_delta_along_mu(env, obs, pre, post, *, out_params=None, **_):
    if pre is None or post is None or out_params is None or len(out_params) < 2:
        return np.array([0.0], dtype=np.float32)
    d = (post["pos"] - pre["pos"]).astype(np.float32)           # world Δ
    mu = np.asarray(out_params[:2], dtype=np.float32)
    n = np.linalg.norm(mu) + 1e-8
    mu_hat = mu / n
    # alignment of executed Δ with intended μ (treat μ as body “forward”)
    return np.array([float(np.dot(d, mu_hat))], dtype=np.float32)

def t_delta_perp_mu(env, obs, pre, post, *, out_params=None, **_):
    if pre is None or post is None or out_params is None or len(out_params) < 2:
        return np.array([0.0], dtype=np.float32)
    d = (post["pos"] - pre["pos"]).astype(np.float32)
    mu = np.asarray(out_params[:2], dtype=np.float32)
    n = np.linalg.norm(mu) + 1e-8
    # 2D perpendicular to μ̂
    mu_hat = mu / n
    perp = np.array([-mu_hat[1], mu_hat[0]], dtype=np.float32)
    return np.array([float(np.dot(d, perp))], dtype=np.float32)


def t_progress_sign_k3(env, obs, pre, post, *, ring=None, k=3, rollout_goal_colour=None, zl_max_ring=None, **_):
    # Preferred: true progress toward goal center if available
    if ring is not None and len(ring) >= k+1 and hasattr(env, "zone_positions") and rollout_goal_colour in getattr(env, "zone_positions", {}):
        zpos = np.asarray(env.zone_positions[rollout_goal_colour][:2], dtype=np.float32)
        d_then = np.linalg.norm(ring[-(k+1)] - zpos)
        d_now  = np.linalg.norm(ring[-1]       - zpos)
        return np.array([1 if d_now < d_then else 0], dtype=int)
    # Fallback: progress if the strongest zone signal increased
    if zl_max_ring is not None and len(zl_max_ring) >= k+1:
        return np.array([1 if zl_max_ring[-1] > zl_max_ring[-(k+1)] else 0], dtype=int)
    return np.array([0], dtype=int)


def t_next_wall_lidar(env, obs, pre, post, *, next_obs=None, **_):
    if next_obs is None:
        return np.zeros(16, dtype=np.float32)
    return wall_lidar(next_obs).astype(np.float32)

# Angle & magnitude bins from policy μ (linear classifiers do great on these)
def t_policy_angle_cls8(env, obs, pre, post, *, out_params=None, **_):
    if out_params is None or len(out_params) < 2:
        return np.array([0], dtype=int)
    mx, my = float(out_params[0]), float(out_params[1])
    ang = np.arctan2(my, mx)
    cls = int(np.floor((ang + np.pi) / (2*np.pi/8))) % 8
    return np.array([cls], dtype=int)

def t_policy_speed_bin(env, obs, pre, post, *, out_params=None, **_):
    if out_params is None or len(out_params) < 2:
        return np.array([0], dtype=int)
    s = np.hypot(float(out_params[0]), float(out_params[1]))
    # 4 bins: stop/slow/med/fast (tune thresholds if needed)
    edges = [0.05, 0.2, 0.5]
    cls = sum(s >= e for e in edges)  # 0..3
    return np.array([cls], dtype=int)

def t_policy_turn_sign(env, obs, pre, post, *, out_params=None, **_):
    # 1 if left turn (my>0), 0 otherwise — tweak for your convention
    if out_params is None or len(out_params) < 2:
        return np.array([0], dtype=int)
    return np.array([1 if float(out_params[1]) > 0 else 0], dtype=int)

def t_policy_stop_go(env, obs, pre, post, *, out_params=None, **_):
    if out_params is None or len(out_params) < 2:
        return np.array([0], dtype=int)
    s = np.hypot(float(out_params[0]), float(out_params[1]))
    return np.array([1 if s > 0.1 else 0], dtype=int)




# ───────────── Config ─────────────────────────────────────────────────────────

@dataclass
class RunCfg:
    env: str = "PointLtl2-v0"
    exp: str = "big_test"
    seed: int = 0
    n_worlds: int = 10
    n_rollout: int = 10
    max_step: int = 500
    deterministic: bool = True
    goals: Tuple[str, ...] = tuple(f"FG {c}" for c in ["blue", "green", "yellow", "magenta"])
    test_frac_worlds: float = 0.2          # fraction of worlds to hold out if --held_out_worlds not provided
    held_out_worlds: Optional[List[int]] = None
    results_csv: Optional[str] = None      # path to write CSV summary


# ───────────── Utilities ──────────────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def group_split_by_world(world_ids: np.ndarray, held_out: Optional[List[int]], frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (train_mask, test_mask), holding out entire worlds."""
    uniq = np.unique(world_ids).tolist()
    if held_out is None:
        rng = np.random.default_rng(seed)
        k = max(1, int(round(len(uniq) * frac)))
        held = set(rng.choice(uniq, size=k, replace=False).tolist())
    else:
        held = set(held_out)
    test_mask = np.isin(world_ids, list(held))
    return ~test_mask, test_mask

def as_features(obs) -> np.ndarray:
    """Return the 80-D feature vector, whether obs is a dict or ndarray."""
    if isinstance(obs, dict) and "features" in obs:
        return np.asarray(obs["features"], dtype=np.float32)
    arr = np.asarray(obs, dtype=np.float32)
    return arr

def action_to_index_or_vec(a):
    """Normalize an action to an int index or 1D vector; return np.ndarray."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(a, np.ndarray):
        a = a.flatten()
        if a.size == 1:
            return np.array([int(a.item())], dtype=int)
        return a.astype(np.float32)
    if isinstance(a, (int, np.integer)):
        return np.array([int(a)], dtype=int)
    if isinstance(a, (float, np.floating)):
        return np.array([int(a)], dtype=int)
    # Fallback
    return np.array([0], dtype=int)

def extract_policy_params(dist_like) -> np.ndarray:
    """
    Convert the model's policy output to a fixed numeric vector:
    • Categorical: logits
    • Normal/DiagNormal: [mean, log_std]
    Robust across common PyTorch dist wrappers.
    """
    try:
        # Common: (dist, ...) tuple
        if isinstance(dist_like, tuple):
            dist_like = dist_like[0]
        # Categorical-style
        if hasattr(dist_like, "logits"):
            return dist_like.logits.detach().cpu().ravel().numpy()
        # Torch Normal or Independent(Normal)
        if hasattr(dist_like, "loc") and hasattr(dist_like, "scale"):
            mu = dist_like.loc
            log_std = torch.log(dist_like.scale + 1e-12)
            return torch.cat([mu, log_std], dim=-1).detach().cpu().ravel().numpy()
        # Some wrappers use .dist
        if hasattr(dist_like, "dist"):
            d = dist_like.dist
            if hasattr(d, "loc") and hasattr(d, "scale"):
                mu = d.loc
                log_std = torch.log(d.scale + 1e-12)
                return torch.cat([mu, log_std], dim=-1).detach().cpu().ravel().numpy()
        # Fallback: raw tensor
        if torch.is_tensor(dist_like):
            return dist_like.detach().cpu().ravel().numpy()
    except Exception:
        pass
    # Last resort: zeros
    return np.zeros(4, dtype=np.float32)


# ───────────── Targets (stable, minimal dependencies) ─────────────────────────

# Slices for the 80-D features (ZoneEnv):
# acc(0:3), wall lidar(3:19), zone lidar(19:35), vel(35:38), gyro(38:41), contact(41:45), rest(45:80)
def t_agent_sensors(env, obs, _pre=None, _post=None):
    f = as_features(obs)
    acc = np.clip(f[0:3], -10.0, 10.0)
    gyro = f[38:41]
    vel = f[35:38]
    return np.concatenate([acc, gyro, vel]).astype(np.float32)

def t_wall_lidar(env, obs, _pre=None, _post=None):
    return as_features(obs)[3:19].astype(np.float32)

def t_zone_lidar(env, obs, _pre=None, _post=None):
    return as_features(obs)[19:35].astype(np.float32)

def t_agent_pos(env, _obs, _pre=None, _post=None):
    return np.asarray(env.agent_pos[:2], dtype=np.float32)

def t_delta_xy(env, _obs, pre, post):
    # executed displacement (post - pre)
    if pre is None or post is None:
        return np.zeros(2, dtype=np.float32)
    return (post["pos"] - pre["pos"]).astype(np.float32)

def t_vz(env, obs, _pre=None, _post=None):
    # body-frame forward speed (signed)
    return np.array([as_features(obs)[37]], dtype=np.float32)

def t_wz(env, obs, _pre=None, _post=None):
    # yaw rate from gyro z
    f = as_features(obs)
    wz = f[40] if f.shape[0] > 40 else 0.0
    return np.array([wz], dtype=np.float32)

def t_wz_sign(env, obs, _pre=None, _post=None):
    return np.array([1 if t_wz(env, obs)[0] > 0 else 0], dtype=int)

# K-step pose deltas are assembled after collection; placeholders here.
def t_pose_k_placeholder(_env, _obs, _pre=None, _post=None):
    return np.zeros(2, dtype=np.float32)


# ---- Output-native labels (head info) ----
def _split_out_params(out_params):
    if out_params is None:
        return None, None
    arr = np.asarray(out_params, dtype=np.float32).ravel()
    if arr.size >= 4:
        mu = arr[:2]; log_std = arr[-2:]
    elif arr.size >= 2:
        mu = arr[:2]; log_std = None
    else:
        mu = None; log_std = None
    return mu, log_std

def t_policy_mu(_env, _obs, _pre=None, _post=None, *, out_params=None, **_):
    mu, _ = _split_out_params(out_params)
    return np.zeros(2, np.float32) if mu is None else mu.astype(np.float32)

def t_policy_log_std(_env, _obs, _pre=None, _post=None, *, out_params=None, **_):
    _, log_std = _split_out_params(out_params)
    return np.zeros(2, np.float32) if log_std is None else log_std.astype(np.float32)

def t_policy_entropy(_env, _obs, _pre=None, _post=None, *, out_params=None, **_):
    _, log_std = _split_out_params(out_params)
    if log_std is None:
        return np.array([0.0], dtype=np.float32)
    const = 0.5 * np.log(2*np.pi*np.e)
    ent = float(np.sum(log_std + const))
    return np.array([ent], dtype=np.float32)

def t_policy_speed_mag(_env, _obs, _pre=None, _post=None, *, out_params=None, **_):
    mu, _ = _split_out_params(out_params)
    if mu is None:
        return np.array([0.0], dtype=np.float32)
    return np.array([np.hypot(mu[0], mu[1])], dtype=np.float32)

def t_policy_angle_cls8(_env, _obs, _pre=None, _post=None, *, out_params=None, **_):
    mu, _ = _split_out_params(out_params)
    if mu is None:
        return np.array([0], dtype=int)
    ang = np.arctan2(float(mu[1]), float(mu[0]))
    cls = int(np.floor((ang + np.pi) / (2*np.pi/8))) % 8
    return np.array([cls], dtype=int)

def t_policy_turn_sign(_env, _obs, _pre=None, _post=None, *, out_params=None, **_):
    mu, _ = _split_out_params(out_params)
    if mu is None:
        return np.array([0], dtype=int)
    return np.array([1 if float(mu[1]) > 0 else 0], dtype=int)

def t_policy_stop_go(_env, _obs, _pre=None, _post=None, *, out_params=None, **_):
    mu, _ = _split_out_params(out_params)
    if mu is None:
        return np.array([0], dtype=int)
    s = float(np.hypot(mu[0], mu[1]))
    return np.array([1 if s > 0.1 else 0], dtype=int)

# ---- Action-mediated short-horizon deltas ----

def t_d_front_clearance_sign(env, obs, pre, post, *, next_obs=None, **_):
    if next_obs is None:
        return np.array([0], dtype=int)
    rays_now  = wall_lidar(obs);  c, k = len(rays_now)//2, 3
    rays_next = wall_lidar(next_obs)
    fc_now  = float(rays_now[max(0,c-k):c+k+1].mean())
    fc_next = float(rays_next[max(0,c-k):c+k+1].mean())
    return np.array([1 if fc_next > fc_now else 0], dtype=int)

def t_d_bearing_to_goal_sign(env, obs, pre, post, *, next_obs=None, rollout_goal_colour=None, **_):
    # Use colour center if available; else fall back to zone-lidar argmax bearing proxy at t and t+1
    if next_obs is None:
        return np.array([0], dtype=int)
    if hasattr(env, "zone_positions") and rollout_goal_colour in getattr(env, "zone_positions", {}):
        z = np.asarray(env.zone_positions[rollout_goal_colour][:2], np.float32)
        a_now  = np.asarray(env.agent_pos[:2], np.float32)
        # "next obs" agent_pos isn't provided by env; approximate with post["pos"]
        a_next = np.asarray(post["pos"], np.float32) if post is not None else a_now
        ang_now  = np.arctan2(*(z - a_now)[[1,0]])
        ang_next = np.arctan2(*(z - a_next)[[1,0]])
        # compare absolute bearing magnitudes
        b_now  = np.abs(ang_now)
        b_next = np.abs(ang_next)
        return np.array([1 if b_next < b_now else 0], dtype=int)
    # Fallback proxy from beams
    z_now  = zone_lidar(obs);      b_now  = int(np.argmax(z_now))  % 16
    z_next = zone_lidar(next_obs); b_next = int(np.argmax(z_next)) % 16
    # "Improved" if wrapped distance to target bin decreases
    wrap = lambda x: (x + 8) % 16 - 8
    diff = abs(wrap(b_next)) - abs(wrap(b_now))
    return np.array([1 if diff < 0 else 0], dtype=int)

def t_avoidance_vs_goal(env, obs, pre, post, *, out_params=None, next_obs=None, rollout_goal_colour=None, **_):
    """
    Heuristic tag:
      1 = avoidance (turn/slow because wall ahead),
      0 = goal pursuit (move toward goal when clear).
    """
    # front-clearance now
    rays = wall_lidar(obs); c, k = len(rays)//2, 3
    fc = float(rays[max(0,c-k):c+k+1].mean())
    # action mean
    if out_params is None or len(out_params) < 2:
        return np.array([0], dtype=int)
    mu = np.asarray(out_params[:2], dtype=np.float32)
    speed = float(np.linalg.norm(mu))
    turning = abs(float(mu[0])) + abs(float(mu[1])) > 1e-6  # generic

    # bearing proxy
    if hasattr(env, "zone_positions") and rollout_goal_colour in getattr(env, "zone_positions", {}):
        z = np.asarray(env.zone_positions[rollout_goal_colour][:2], np.float32)
        a = np.asarray(env.agent_pos[:2], np.float32)
        to_goal = z - a
        align = float(np.dot(mu, to_goal))  # >0 if roughly toward goal
    else:
        zl = zone_lidar(obs); to_goal = np.zeros(2, np.float32)
        align = 0.0

    # Rules: wall close + turning/slow → avoidance; clear + aligned & moving → goal
    if fc < 0.25 and (turning or speed < 0.1):
        return np.array([1], dtype=int)
    if fc > 0.4 and align > 0 and speed > 0.1:
        return np.array([0], dtype=int)
    # default to "avoidance" near walls, else "goal"
    return np.array([1 if fc < 0.3 else 0], dtype=int)

def t_policy_alignment(_env, _obs, _pre=None, _post=None, *, out_params=None, **_):
    mu, _ = _split_out_params(out_params)
    if mu is None or (mu[0] == 0 and mu[1] == 0):
        return np.array([0.0], dtype=np.float32)
    # Use forward window used by front_clearance as a simple "free" direction proxy (unit forward)
    u = np.array([1.0, 0.0], dtype=np.float32)  # forward
    v = mu.astype(np.float32)
    dot = float(np.dot(u, v) / (np.linalg.norm(u) * (np.linalg.norm(v) + 1e-8)))
    return np.array([dot], dtype=np.float32)


# --- TARGETS and their functions -----------------------------------
TARGETS: Dict[str, Callable] = {
    # Sensors / geometry
    "agent_sensors": t_agent_sensors,   # 9D
    "wall_lidar":    t_wall_lidar,      # 16D
    "zone_lidar":    t_zone_lidar,      # 16D
    "agent_pos":     t_agent_pos,       # 2D

    # Dynamics-ish
    "delta_xy":      t_delta_xy,        # 2D (executed)
    "vz":            t_vz,              # 1D
    "wz":            t_wz,              # 1D
    "wz_sign":       t_wz_sign,         # 1D (classification)

    # Filled later from buffers
    "pose_k5":       t_pose_k_placeholder,
    "pose_k10":      t_pose_k_placeholder,
}

# --- extend TARGETS & add CATEGORY labels -----------------------------------
TARGETS.update({
    # direct inputs
    "contacts":             t_contacts,

    # indirect (trivial from inputs)
    "min_wall_dist":        t_min_wall_dist,
    "wall_sector_argmin":   t_wall_sector_argmin,
    "free_space_ahead":     t_free_space_ahead,
    "nearest_zone_id":      t_nearest_zone_id_geom,
    "nearest_zone_dir_cls": t_nearest_zone_dir_cls,
    "in_zone_flags":        t_in_zone_flags,

    # easy task-relevant
    "goal_colour_id":       t_goal_colour_id,
    "bearing_to_goal_cls":  t_bearing_to_goal_cls,
    "progress_sign_k3":     t_progress_sign_k3,
    "next_wall_lidar":      t_next_wall_lidar,
    "front_clearance":      t_front_clearance,
})

TARGETS.update({
    "policy_mu":             t_policy_mu,           # 2D
    "policy_speed_mag":      t_policy_speed_mag,    # 1D
    "policy_log_std":        t_policy_log_std,      # 2D
    "policy_entropy":        t_policy_entropy,      # 1D
    "d_front_clearance_sign": t_d_front_clearance_sign, # 1D (cls)
    "d_bearing_to_goal_sign": t_d_bearing_to_goal_sign, # 1D (cls)
    "avoidance_vs_goal":      t_avoidance_vs_goal,      # 1D (cls)
})

TARGETS.update({
    "policy_mu":           t_policy_mu,
    "policy_log_std":      t_policy_log_std,
    "policy_entropy":      t_policy_entropy,
    "policy_speed_mag":    t_policy_speed_mag,
    "policy_angle_cls8":   t_policy_angle_cls8,
    "policy_turn_sign":    t_policy_turn_sign,
    "policy_stop_go":      t_policy_stop_go,
})

TARGET_CATEGORY = {
    # direct inputs (come from obs/features)
    "wall_lidar": "direct", "zone_lidar": "direct", "agent_sensors": "direct",
    "wz": "direct", "vz": "direct", "contacts": "direct",

    # indirect (trivial from inputs via simple transforms)
    "min_wall_dist": "indirect", "wall_sector_argmin": "indirect",
    "free_space_ahead": "indirect", "nearest_zone_id": "indirect",
    "nearest_zone_dir_cls": "indirect", "in_zone_flags": "indirect",
    "front_clearance": "indirect",

    # trivial (literally what the head outputs)
    "policy_mu": "trivial",
    "policy_log_std": "trivial",
    "policy_entropy": "trivial",

    # easy (can be inferred from outputs with simple readouts)
    "goal_colour_id": "easy", "bearing_to_goal_cls": "easy",
    "progress_sign_k3": "easy", "next_wall_lidar": "easy",
    "collision_imminence": "easy", "wz_sign": "easy",
    "policy_angle_cls8": "easy",
    "policy_speed_bin": "easy",
    "policy_turn_sign": "easy",
    "policy_stop_go": "easy",

    # hard
    "agent_pos": "hard", "pose_k5": "hard", "pose_k10": "hard", "delta_xy": "hard",
}

TARGET_CATEGORY.update({
    "policy_speed_mag": "trivial",
    "d_front_clearance_sign": "indirect",
    "d_bearing_to_goal_sign": "indirect",
    "avoidance_vs_goal": "indirect",
})

TARGETS.update({
    "delta_along_mu": t_delta_along_mu,   # 1D (R²)
    "delta_perp_mu":  t_delta_perp_mu,    # 1D (R²)
})
TARGET_CATEGORY.update({
    "delta_along_mu": "easy",
    "delta_perp_mu":  "easy",
})

TARGETS["policy_alignment"] = t_policy_alignment
TARGET_CATEGORY["policy_alignment"] = "easy"

# For OUT-only quadratic features (to handle circular/curved boundaries from μ)
OUT_QUADRATIC_NAMES = {"policy_speed_mag", "policy_speed_bin", "policy_stop_go"}


def is_classification(y: np.ndarray, name: str) -> bool:
    if name in ("wz_sign",):
        return True
    return (np.issubdtype(y.dtype, np.integer) and np.unique(y).size <= 32)


# ───────────── Probing ────────────────────────────────────────────────────────
def fit_probe(X_train, X_test, y_train, y_test, name: str, use_quadratic: bool = False,
              probe_type: str = "linear",
              mlp_hidden: Tuple[int, ...] = (64,),
              mlp_alpha: float = 1e-4,
              mlp_max_iter: int = 200,
              seed: int = 0):
    """
    Train a probe and return (score, metric_name).
    - For regression: R²
    - For classification: Accuracy
    - Multi-dim regression: mean R² across dims
    """
    is_clf = is_classification(y_train, name)

    # --- MULTI-DIM REGRESSION (same behavior; MLPRegressor supports multi-output) ---
    if y_train.ndim == 2 and y_train.shape[1] > 1 and not is_clf:
        scores = []
        for i in range(y_train.shape[1]):
            yt_tr = y_train[:, i]; yt_te = y_test[:, i]
            if float(np.std(yt_tr)) < 1e-8 or float(np.std(yt_te)) < 1e-8:
                scores.append(0.0); continue

            if probe_type == "mlp":
                reg = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=mlp_hidden,
                        activation="relu",
                        alpha=mlp_alpha,
                        random_state=seed,
                        max_iter=mlp_max_iter,
                        early_stopping=True,
                        n_iter_no_change=10,
                        verbose=False
                    )
                )
            else:
                # linear (optionally quadratic)
                if use_quadratic:
                    reg = make_pipeline(
                        StandardScaler(),
                        PolynomialFeatures(degree=2, include_bias=False),
                        Ridge(alpha=10.0),
                    )
                else:
                    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))

            reg.fit(X_train, yt_tr)
            y_pred = reg.predict(X_test)
            ss_res = np.sum((yt_te - y_pred) ** 2)
            ss_tot = np.sum((yt_te - yt_te.mean()) ** 2)
            scores.append(1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0)

        return float(np.mean(scores)), "R²"

    # --- 1-D TARGETS ---
    ytr = y_train.ravel()
    yte = y_test.ravel()

    if is_clf:
        classes = np.unique(ytr)
        if classes.size < 2:
            maj = classes[0]
            acc = float(np.mean(yte == maj)) if yte.size else float("nan")
            return acc, "Acc*"

        if probe_type == "mlp":
            clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=mlp_hidden,
                    activation="relu",
                    alpha=mlp_alpha,
                    random_state=seed,
                    max_iter=mlp_max_iter,
                    early_stopping=True,
                    n_iter_no_change=10,
                    verbose=False
                )
            )
        else:
            # linear logistic
            if use_quadratic:
                clf = make_pipeline(
                    StandardScaler(),
                    PolynomialFeatures(degree=2, include_bias=False),
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                )
            else:
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, class_weight="balanced")
                )

        clf.fit(X_train, ytr)
        y_hat = clf.predict(X_test)
        return accuracy_score(yte, y_hat), "Acc"

    # --- 1-D REGRESSION ---
    if float(np.std(ytr)) < 1e-8 or float(np.std(yte)) < 1e-8:
        return 0.0, "R²"

    if probe_type == "mlp":
        reg = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=mlp_hidden,
                activation="relu",
                alpha=mlp_alpha,
                random_state=seed,
                max_iter=mlp_max_iter,
                early_stopping=True,
                n_iter_no_change=10,
                verbose=False
            )
        )
    else:
        if use_quadratic:
            reg = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=2, include_bias=False),
                Ridge(alpha=10.0),
            )
        else:
            reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    reg.fit(X_train, ytr)
    y_pred = reg.predict(X_test)
    return r2_score(yte, y_pred), "R²"





# ───────────── Data collection (hooks) ────────────────────────────────────────

def collect_dataset(cfg: RunCfg):
    """Run rollouts, capture X_in, X_out, targets, and k-step pose deltas."""
    # Build a dummy env to create model and read spaces
    dummy = make_env(cfg.env, FixedSampler.partial("FG blue"), sequence=False)
    store = ModelStore(cfg.env, cfg.exp, cfg.seed); store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    model  = build_model(dummy, status, model_configs[cfg.env]).eval()
    dummy.close()

    # State for hooks
    collect_now = {"flag": False}
    current_embedding = {"x": None}
    X_in, X_out = [], []
    A = []
    world_ids = []
    Y_buffers = {k: [] for k in TARGETS.keys()}  # 1-step targets captured at the same step

    H1, H2, H3 = [], [], []
    tap = HeadTap(model, hidden_dim=64, out_dim_candidates=(2, 4), verbose=False)


    # For k-step deltas
    pos_buf: List[np.ndarray] = []
    emb_buf: List[np.ndarray] = []
    X_k5: List[np.ndarray] = []; Y_k5: List[np.ndarray] = []; wid_k5: List[int] = []
    X_k10: List[np.ndarray] = []; Y_k10: List[np.ndarray] = []; wid_k10: List[int] = []

    # Hook model to capture actor fusion input
    orig_compute_embedding = model.compute_embedding
    def hooked_compute_embedding(obs):
        emb = orig_compute_embedding(obs)
        if collect_now["flag"]:
            current_embedding["x"] = emb.detach().cpu().ravel().numpy()
        return emb
    model.compute_embedding = hooked_compute_embedding

    # Hook forward to capture policy params and gather labels
    def forward_hook(_m, _inp, out):
        if not collect_now["flag"]:
            return
        collect_now["flag"] = False

        # back-fill previous sample's next_wall_lidar with the *current* obs
        if Y_buffers["next_wall_lidar"]:
            Y_buffers["next_wall_lidar"][-1] = wall_lidar(_state["obs"]).astype(np.float32)

        # X_in / X_out / wid
        if current_embedding["x"] is not None:
            X_in.append(current_embedding["x"])
        X_out.append(extract_policy_params(out))
        world_ids.append(_state["wid"])
        
        # output native params
        out_params = extract_policy_params(out)
        X_out.append(out_params)

        # NEW: reserve a slot for the action aligned with this sample; we'll fill it after get_action
        A.append(None)

        acts = tap.end_step()
        H1.append(acts.get("h1", np.zeros(64, dtype=np.float32)))
        H2.append(acts.get("h2", np.zeros(64, dtype=np.float32)))
        H3.append(acts.get("h3", np.zeros(64, dtype=np.float32)))


        # index of the row we just appended
        _last_row = len(X_in) - 1
        _state["last_row_for_delta"] = _last_row
        _state["obs_pre_for_delta"]  = _state["obs"]  # the obs used for this row


        # labels for *current* step
        env, obs, pre_s, post_s = _state["env"], _state["obs"], _state["pre"], _state["post"]
        extras = dict(
            rollout_goal_colour=_state["rollout_goal_colour"],
            ring=_state["pos_ring"],
            next_obs=None,  # we back-fill next_wall_lidar one step later
            zl_max_ring=_state["zl_max_ring"],   # NEW
            out_params=out_params,         # <— NEW: policy head params for label fns

        )
        for name, fn in TARGETS.items():
            try:
                Y_buffers[name].append(fn(env, obs, pre_s, post_s, **extras))
            except TypeError:
                Y_buffers[name].append(fn(env, obs, pre_s, post_s))

    model.register_forward_hook(forward_hook)

    # Rollouts
    pbar = trange(cfg.n_worlds * cfg.n_rollout, desc="Collecting", unit="rollout")
    rng_w = np.random.default_rng(cfg.seed)
    _state = {"env": None, "obs": None, "pre": None, "post": None, "wid": -1,
          "rollout_goal_colour": None,
          "pos_ring": [],
          "zl_max_ring": [],
          "next_obs": None}  # used to back-fill next_wall_lidar

    for wid in range(cfg.n_worlds):
        for rid in range(cfg.n_rollout):
            goal = cfg.goals[(wid * cfg.n_rollout + rid) % len(cfg.goals)]
            env  = make_env(cfg.env, FixedSampler.partial(goal), sequence=False)
            props = set(env.get_propositions())
            planner = ExhaustiveSearch(model, props, num_loops=2)
            agent   = Agent(model, planner, propositions=props, verbose=False)

            rollout_goal = cfg.goals[(wid*cfg.n_rollout + rid) % len(cfg.goals)]
            _state["rollout_goal_colour"] = rollout_goal.split()[-1]
            # _state["pos_ring"].append(pre["pos"].copy())

            obs = env.reset(seed=cfg.seed + 100*wid + rid)
            done = False; step = 0

            # reset per-rollout buffers
            pos_buf.clear(); emb_buf.clear()
            _state["pos_ring"] = []

            while not done and step < cfg.max_step:
                # pre-step state
                pre = {"pos": np.asarray(env.agent_pos[:2], dtype=np.float32)}
                _state["pos_ring"].append(pre["pos"].copy())
                _state.update(env=env, obs=obs, pre=pre, wid=wid)

                zl_max = float(zone_lidar(obs).max())
                _state["zl_max_ring"].append(zl_max)

                # set flag to collect this step
                collect_now["flag"] = True

                # store for k-step deltas (pose)
                pos_buf.append(pre["pos"].copy())
                if current_embedding["x"] is not None:
                    emb_buf.append(current_embedding["x"])

                tap.begin_step()

                with torch.no_grad():
                    act = agent.get_action(obs, {}, deterministic=cfg.deterministic)
                act_norm = action_to_index_or_vec(act)

                # NEW: fill the action slot created by the forward_hook
                act_vec = action_to_index_or_vec(act).astype(np.float32)
                if A and A[-1] is None:
                    A[-1] = act_vec
                else:
                    # safety fallback (shouldn't happen with the gating flag)
                    A.append(act_vec)

                # env step
                is_int = np.issubdtype(act_norm.dtype, np.integer) if isinstance(act_norm, np.ndarray) else False
                obs, _, done, _ = env.step(int(act_norm.item()) if is_int and act_norm.size == 1 else act_norm)

                # Backfill delta labels for the row created just before this step
                row = _state.get("last_row_for_delta", None)
                obs_pre = _state.get("obs_pre_for_delta", None)  # obs at time t

                # post-step state
                post = {"pos": np.asarray(env.agent_pos[:2], dtype=np.float32)}
                _state["post"] = post
                # update _state so the next hook sees the new obs
                _state["obs"]  = obs
                _state["post"] = {"pos": np.asarray(env.agent_pos[:2], dtype=np.float32)}
                _state["next_obs"] = obs  # cache for next_wall_lidar at *previous* step

                if row is not None and obs_pre is not None and row < len(Y_buffers["d_front_clearance_sign"]):
                    # compute deltas using obs_pre (t) and obs (t+1)
                    y_fc = t_d_front_clearance_sign(env, obs_pre, pre, post, next_obs=obs)
                    y_bg = t_d_bearing_to_goal_sign(env, obs_pre, pre, post,
                                                    next_obs=obs, rollout_goal_colour=_state["rollout_goal_colour"])
                    # overwrite placeholders at the correct row
                    Y_buffers["d_front_clearance_sign"][row] = y_fc
                    Y_buffers["d_bearing_to_goal_sign"][row] = y_bg

                step += 1

            # make k=5 / k=10 relative pose labels
            if len(pos_buf) > 5 and len(emb_buf) >= len(pos_buf)-5:
                pose_now = np.asarray(pos_buf[:-5]); pose_fut = np.asarray(pos_buf[5:])
                emb_now  = np.asarray(emb_buf[:pose_now.shape[0]])
                X_k5.extend(emb_now); Y_k5.extend((pose_fut - pose_now)); wid_k5.extend([wid]*emb_now.shape[0])

            if len(pos_buf) > 10 and len(emb_buf) >= len(pos_buf)-10:
                pose_now = np.asarray(pos_buf[:-10]); pose_fut = np.asarray(pos_buf[10:])
                emb_now  = np.asarray(emb_buf[:pose_now.shape[0]])
                X_k10.extend(emb_now); Y_k10.extend((pose_fut - pose_now)); wid_k10.extend([wid]*emb_now.shape[0])

            if len(Y_buffers["next_wall_lidar"]) > 0:
                Y_buffers["next_wall_lidar"][-1] = np.zeros(16, dtype=np.float32)


            env.close()
            pbar.update(1)

    pbar.close()
    tap.close()

    # Stack arrays
    X_in  = np.asarray(X_in);   X_out = np.asarray(X_out); world_ids = np.asarray(world_ids)
    Ys = {k: np.asarray(v) for k, v in Y_buffers.items()}

    # NEW: stack actions (pad with zeros if any None sneaks in)
    act_dim = max((len(a) for a in A if a is not None), default=1)
    A_np = np.zeros((len(A), act_dim), dtype=np.float32)
    for i, a in enumerate(A):
        if a is None:  # very first sample could be None; keep zeros
            continue
        A_np[i, :len(a)] = a

    # Fill k-step targets (replace placeholders)
    if len(X_k5):
        Ys["pose_k5"]  = np.asarray(Y_k5);  X_k5 = np.asarray(X_k5);  wid_k5 = np.asarray(wid_k5)
    else:
        Ys["pose_k5"]  = np.zeros((0,2), dtype=np.float32); X_k5 = np.zeros((0, X_in.shape[1]), dtype=np.float32); wid_k5 = np.zeros((0,), dtype=int)

    if len(X_k10):
        Ys["pose_k10"] = np.asarray(Y_k10); X_k10 = np.asarray(X_k10); wid_k10 = np.asarray(wid_k10)
    else:
        Ys["pose_k10"] = np.zeros((0,2), dtype=np.float32); X_k10 = np.zeros((0, X_in.shape[1]), dtype=np.float32); wid_k10 = np.zeros((0,), dtype=int)

    return dict(
        model=model,
        X_in=X_in, X_out=X_out, world_ids=world_ids,
        Ys=Ys,
        X_k5=X_k5, wid_k5=wid_k5,
        X_k10=X_k10, wid_k10=wid_k10,
        A=A_np,  # <— NEW
        H1=np.asarray(H1, dtype=np.float32),
        H2=np.asarray(H2, dtype=np.float32),
        H3=np.asarray(H3, dtype=np.float32),
    )

def _align_xout_to_actor(X_out, X_in, Ys):
    # already aligned?
    if len(X_out) == len(X_in):
        return X_out[:len(X_in)]
    # try to pick correct half using the fact that policy_mu labels came from the same call
    if "policy_mu" in Ys and len(Ys["policy_mu"]) >= 100:
        Ymu = np.asarray(Ys["policy_mu"], dtype=np.float32)
        halves = []
        if len(X_out) >= 2*len(X_in):
            halves = [X_out[::2], X_out[1::2]]
        else:
            # fallback: just clamp
            return X_out[:len(X_in)]
        # compare MSE on μ (first 2 dims) for the two halves
        mses = []
        for H in halves:
            n = min(len(H), len(Ymu))
            mses.append(float(np.mean((H[:n, :2] - Ymu[:n])**2)))
        best = halves[int(mses[1] < mses[0])]
        return best[:len(X_in)]
    # ultimate fallback
    return X_out[:len(X_in)]

def print_out_correlations(X_out, Ys):
        wanted = []
        if "front_clearance" in Ys and len(Ys["front_clearance"]) == len(X_out):
            wanted.append(("front_clearance", Ys["front_clearance"].ravel()))
        if "free_space_ahead" in Ys and len(Ys["free_space_ahead"]) == len(X_out):
            wanted.append(("free_space_ahead", Ys["free_space_ahead"].ravel().astype(np.float32)))
        if not wanted:
            return
        colnames = ["mu_x","mu_y","logstd_x","logstd_y"][:X_out.shape[1]]
        print("\n🔎 Correlation (OUT columns vs targets)")
        for tname, y in wanted:
            y = np.asarray(y, dtype=np.float32)
            if np.std(y) < 1e-8:
                print(f"  {tname:>18}: constant target, skip")
                continue
            vals = []
            for j in range(X_out.shape[1]):
                x = X_out[:, j].astype(np.float32)
                r = np.corrcoef(x, y)[0,1] if np.std(x) > 1e-8 else 0.0
                vals.append((colnames[j] if j < len(colnames) else f"col{j}", r))
            vals = sorted(vals, key=lambda p: -abs(p[1]))[:4]
            pretty = ", ".join([f"{n}={r:+.3f}" for n,r in vals])
            print(f"  {tname:>18}: {pretty}")

# ───────────── Main ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PointLtl2-v0")
    ap.add_argument("--exp", default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_worlds", type=int, default=10)
    ap.add_argument("--n_rollout", type=int, default=10)
    ap.add_argument("--max_step", type=int, default=500)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--test_frac_worlds", type=float, default=0.2)
    ap.add_argument("--held_out_worlds", type=str, default="", help="comma-separated world ids")
    ap.add_argument("--targets", type=str, default="agent_sensors,wall_lidar,zone_lidar,agent_pos,delta_xy,vz,wz,wz_sign,pose_k5,pose_k10")
    ap.add_argument("--results_csv", type=str, default="")
    ap.add_argument("--probe_type", type=str, choices=["linear", "mlp"], default="linear",
                    help="Which probe to use: linear (Ridge/LogReg) or mlp (MLPRegressor/MLPClassifier)")
    ap.add_argument("--mlp_hidden", type=str, default="64",
                    help='Comma-separated hidden sizes for MLP, e.g. "128,64"')
    ap.add_argument("--mlp_alpha", type=float, default=1e-4,
                    help="L2 regularization for MLP")
    ap.add_argument("--mlp_max_iter", type=int, default=200,
                    help="Max iterations for MLP with early stopping")

    args = ap.parse_args()

    cfg = RunCfg(
        env=args.env, exp=args.exp, seed=args.seed,
        n_worlds=args.n_worlds, n_rollout=args.n_rollout, max_step=args.max_step,
        deterministic=args.deterministic,
        test_frac_worlds=args.test_frac_worlds,
        held_out_worlds=[int(x) for x in args.held_out_worlds.split(",")] if args.held_out_worlds else None,
        results_csv=args.results_csv or None,
    )
    set_all_seeds(cfg.seed)

    probe_type = args.probe_type
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(",") if x.strip())
    mlp_alpha = args.mlp_alpha
    mlp_max_iter = args.mlp_max_iter


    data = collect_dataset(cfg)

    X_in, X_out, world_ids = data["X_in"], data["X_out"], data["world_ids"]
    Ys = data["Ys"]
    A_all = data["A"]
    H1 = data.get("H1", np.zeros((0, X_in.shape[1])))
    H2 = data.get("H2", np.zeros((0, X_in.shape[1])))
    H3 = data.get("H3", np.zeros((0, X_in.shape[1])))

    # Align policy head params to rows
    X_out = _align_xout_to_actor(X_out, X_in, Ys)
    print(f"  ↳ Aligned policy OUTPUT params: {X_out.shape}")  # should now match X_in rows

    if "policy_mu" in Ys and len(Ys["policy_mu"]) > 0:
        n = min(len(X_out), len(Ys["policy_mu"]))
        mse = np.mean((X_out[:n, :2] - Ys["policy_mu"][:n])**2)
        print(f"  Sanity policy_mu alignment MSE: {mse:.6f}")

    # Report collection stats
    print(f"\n📊 Collected: {len(X_in)} samples")
    print(f"  Actor fusion INPUT shape : {X_in.shape}")
    print(f"  Policy OUTPUT params     : {X_out.shape}")
    print(f"  Unique worlds            : {sorted(set(world_ids.tolist()))}")

    speed_edges = None
    if X_out.shape[1] >= 2:
        speed_all = np.hypot(X_out[:, 0], X_out[:, 1])
        speed_edges = np.quantile(speed_all, [0.25, 0.5, 0.75]).tolist()

    print_out_correlations(X_out, Ys)

    # Parse target list
    want_targets = [t.strip() for t in args.targets.split(",") if t.strip() in TARGETS]
    if not want_targets:
        print("No valid targets specified; exiting.")
        return

    # Also wire K-step data (different id arrays)
    X_k5, wid_k5 = data["X_k5"], data["wid_k5"]
    X_k10, wid_k10 = data["X_k10"], data["wid_k10"]

    # Sources we’ll probe
    sources = [
        ("IN",  X_in),
        ("H1",  H1),
        ("H2",  H2),
        ("H3",  H3),
        ("OUT", X_out),
    ]

    def score_source(label, Xsrc, Y_base, ids_base, name):
        """
        Align lengths, optionally augment with action (for next_wall_lidar),
        split by held-out worlds, and fit the probe.
        """
        # Skip OUT for the k-step pose targets (no meaningful OUT there)
        if name in ("pose_k5", "pose_k10") and label == "OUT":
            return float("nan"), "N/A"

        n_local = min(len(Xsrc), len(Y_base), len(ids_base))
        if n_local == 0:
            return float("nan"), "N/A"

        Xs = Xsrc[:n_local]
        Ys_loc = Y_base[:n_local]
        ids_loc = ids_base[:n_local]

        # Augment with action for predictive next_wall_lidar target
        if name == "next_wall_lidar":
            A_use = A_all[:n_local]
            if len(A_use) == n_local:
                Xs = np.hstack([Xs, A_use])

        tr_m, te_m = group_split_by_world(ids_loc, cfg.held_out_worlds, cfg.test_frac_worlds, cfg.seed)
        use_quad = (label == "OUT" and name in OUT_QUADRATIC_NAMES)
        return fit_probe(
            Xs[tr_m], Xs[te_m], Ys_loc[tr_m], Ys_loc[te_m],
            name,
            use_quadratic=use_quad,
            probe_type=probe_type,
            mlp_hidden=mlp_hidden,
            mlp_alpha=mlp_alpha,
            mlp_max_iter=mlp_max_iter,
            seed=cfg.seed,
        )
    print("\n🎯 Probing targets")
    print("="*80)
    results = []
    for name in want_targets:
        # Choose base Y/ids (IN’s indexing) for the target
        if name == "pose_k5":
            X_base = X_k5
            Y_base = Ys["pose_k5"]
            ids_base = wid_k5
        elif name == "pose_k10":
            X_base = X_k10
            Y_base = Ys["pose_k10"]
            ids_base = wid_k10
        else:
            X_base = X_in
            Y_base = Ys[name]
            ids_base = world_ids

        # Align base length
        n = min(len(X_base), len(Y_base), len(ids_base))
        if n == 0:
            print(f"  {name:<18} — no samples, skipping.")
            continue

        # Special case: recompute balanced bins for policy_speed_bin labels from OUT's μ
        if name == "policy_speed_bin" and speed_edges is not None and X_out.shape[0] >= n:
            mu = X_out[:n, :2]
            s = np.hypot(mu[:, 0], mu[:, 1])
            y_bin = np.array([sum(si >= e for e in speed_edges) for si in s], dtype=int)
            Y_base = y_bin[:, None]

        if name in ("goal_colour_id", "bearing_to_goal_cls", "progress_sign_k3"):
            uniq, cnt = np.unique(Y_base[:n], return_counts=True)
            print(f"  class balance: {dict(zip(uniq.tolist(), cnt.tolist()))}")

        # Score each source
        scores = {}
        metric = "R²"
        for label, Xsrc in sources:
            sc, met = score_source(label, Xsrc, Y_base[:n], ids_base[:n], name)
            scores[label] = sc
            if met != "N/A":
                metric = met

        # μ vs logσ ablation for the OUT source on clearance-like targets
        if name in ("free_space_ahead", "front_clearance"):
            n_al = min(len(X_out), len(Y_base))
            if n_al > 0:
                Xo_al = X_out[:n_al]
                tr_m, te_m = group_split_by_world(ids_base[:n_al], cfg.held_out_worlds, cfg.test_frac_worlds, cfg.seed)
                mu_tr, mu_te = Xo_al[tr_m, :2],   Xo_al[te_m, :2]
                ls_tr, ls_te = Xo_al[tr_m, -2:], Xo_al[te_m, -2:]
                Ytr, Yte = Y_base[:n_al][tr_m],   Y_base[:n_al][te_m]
                mu_score, _ = fit_probe(mu_tr, mu_te, Ytr, Yte, name,
                        use_quadratic=False,
                        probe_type=probe_type,
                        mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha,
                        mlp_max_iter=mlp_max_iter, seed=cfg.seed)
                ls_score, _ = fit_probe(ls_tr, ls_te, Ytr, Yte, name,
                        use_quadratic=False,
                        probe_type=probe_type,
                        mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha,
                        mlp_max_iter=mlp_max_iter, seed=cfg.seed)

                print(f"    OUT ablation — μ only: {mu_score:.3f}, logσ only: {ls_score:.3f}")

        # Compact per-target print
        print(f"  {name:<18}  IN:{metric}={scores.get('IN', np.nan):.3f}  "
              f"H1:{scores.get('H1', np.nan):.3f}  H2:{scores.get('H2', np.nan):.3f}  H3:{scores.get('H3', np.nan):.3f}  "
              f"OUT:{(scores.get('OUT', np.nan) if np.isfinite(scores.get('OUT', np.nan)) else float('nan')):.3f}")

        # Stash results with *both* key styles (new: in/out, legacy: input/output)
        res_row = dict(
            target=name,
            metric=metric,
            shape=int(Y_base[:n].shape[1]) if Y_base[:n].ndim > 1 else 1,
            in_=scores.get("IN", np.nan),     # temp key to avoid Python keyword
            h1=scores.get("H1", np.nan),
            h2=scores.get("H2", np.nan),
            h3=scores.get("H3", np.nan),
            out_=scores.get("OUT", np.nan),   # temp key
        )
        # normalize keys
        res_row["in"] = res_row.pop("in_")
        res_row["out"] = res_row.pop("out_")
        # legacy compatibility
        res_row["input"] = res_row["in"]
        res_row["output"] = res_row["out"]
        results.append(res_row)

    # ───────── Layered printing helpers ─────────
    def print_layered_table(results):
        print("\n📋 LAYERED SUMMARY (per target)")
        print("="*88)
        header = f"{'Target':<22}{'Dim':>4} {'Metric':>6} | {'IN':>6} {'H1':>6} {'H2':>6} {'H3':>6} {'OUT':>6}"
        print(header)
        print("-"*len(header))

        def fmt(x):
            try:
                x = float(x)
            except Exception:
                return "  nan"
            return "  nan" if not np.isfinite(x) else f"{x:6.3f}"

        for r in results:
            print(f"{r['target']:<22} {r['shape']:>3} {r['metric']:>5} |"
                  f"{fmt(r.get('in',  np.nan))}"
                  f"{fmt(r.get('h1',  np.nan))}"
                  f"{fmt(r.get('h2',  np.nan))}"
                  f"{fmt(r.get('h3',  np.nan))}"
                  f"{fmt(r.get('out', np.nan))}")

    def print_out_minus_in(results, top_k=12):
        print("\n🏁 OUT − IN leaderboard (top gains)")
        delta_rows = []
        for r in results:
            inn_val = r.get("in", np.nan)
            out_val = r.get("out", np.nan)
            try:
                inn = float(inn_val)
            except Exception:
                inn = np.nan
            try:
                out = float(out_val)
            except Exception:
                out = np.nan
            if np.isfinite(inn) and np.isfinite(out) and r.get("metric", "") != "N/A":
                delta_rows.append((out - inn, r["target"], inn, out, r["metric"]))
        delta_rows.sort(reverse=True)
        for d, name, s_in, s_out, m in delta_rows[:top_k]:
            print(f"  {name:<22} Δ={d:+.3f}   IN={s_in:.3f}   OUT={s_out:.3f}   ({m})")

    def print_category_medians(results):
        by_cat = {}
        for r in results:
            cat = TARGET_CATEGORY.get(r["target"], "uncat")
            by_cat.setdefault(cat, {k: [] for k in ("in","h1","h2","h3","out")})
            for k in ("in","h1","h2","h3","out"):
                v = r.get(k)
                try:
                    v = float(v)
                except Exception:
                    v = np.nan
                if np.isfinite(v):
                    by_cat[cat][k].append(v)

        print("\n📚 CATEGORY MEDIANS (per layer)")
        for cat, d in by_cat.items():
            med = {k: (np.median(v) if v else float("nan")) for k, v in d.items()}
            print(f"{cat:>10}: IN={med['in']:.3f}  H1={med['h1']:.3f}  H2={med['h2']:.3f}  H3={med['h3']:.3f}  OUT={med['out']:.3f}")

    # ───────── Use the helpers ─────────
    print_layered_table(results)
    print_out_minus_in(results, top_k=12)
    print_category_medians(results)

    # Optional: write CSV with all columns
    if cfg.results_csv:
        import csv, os
        cols = ["target","shape","metric","in","h1","h2","h3","out"]
        os.makedirs(os.path.dirname(cfg.results_csv), exist_ok=True)
        with open(cfg.results_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in results:
                row = {k: r.get(k, "") for k in cols}
                w.writerow(row)
        print(f"\n💾 Saved CSV → {cfg.results_csv}")


if __name__ == "__main__":
    main()

