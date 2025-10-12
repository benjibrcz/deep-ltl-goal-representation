#!/usr/bin/env python3
"""
Probe forward-looking structure in DeepLTL activations.

Goals (v0 - keep it simple):
  1) Predict *next observation* features (e.g., next_wall_lidar, next_zone_lidar, etc.)
     from network activations and (optionally) the action / policy mean.
  2) Compare to a sensors-only baseline to see if hidden state adds predictive power.
  3) Evaluate across multiple tapped layers (hooks) with a consistent train/test split.

Design choices:
  - Input format: NPZ produced by your rollout logger/prober. (Parquet support can be added later.)
  - Targets: By default, auto-detects keys starting with "next_" (e.g., next_wall_lidar).
  - Actions: If --include_action, will try keys in priority order: policy_mu, action.
  - Baselines: (A) sensors-only (obs vector or concatenated lidar-like fields); (B) hook-only; (C) hook+action.
  - Models: Ridge regression for continuous targets; LogisticRegression for single-label classification; multi-output handled via MultiOutputRegressor.
  - Split: GroupShuffleSplit if a grouping key exists (world_id, world, group, seed_world_color, etc.); otherwise random split.

Planned extensions (future PRs):
  - Multi-horizon k>1: predict o_{t+k} via per-episode shifting
  - Egocentric motion labels (Δ_forward, Δ_lateral, Δθ) if pose is available
  - Critic/sequence-specific targets (e.g., hitting-time proxies)

Usage (example):
  python probe_forward_look.py \
    --npz interpretability/working_scripts/test_new_fields.npz \
    --hooks hook_env_mlp1 hook_ltl_rnn_h hook_actor_h5 hook_critic_mlp0 \
    --include_action \
    --out_dir interpretability/working_scripts/probe_forward_look \
    --seed 0

"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, log_loss


# -----------------------------
# Helpers for data discovery
# -----------------------------

LIKELY_GROUP_KEYS = [
    "group",
    "world_id",
    "world",
    "seed_world_color",
    "world_color",
    "episode_id",
    "traj_id",
]

SENSOR_FIELD_HINTS = [
    "obs",  # full observation vector if present
    "wall_lidar",
    "zone_lidar",
    "agent_sensors",
    "sensor",
]

ACTION_KEY_CANDIDATES = [
    "policy_mu",
    "action",
]


def load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    # Normalize object arrays (rare) into numeric if possible
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and v.dtype == object:
            try:
                out[k] = np.stack(v)
            except Exception:
                pass
    return out


def pick_action_array(data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    for key in ACTION_KEY_CANDIDATES:
        if key in data:
            arr = data[key]
            if arr.ndim == 1:
                arr = arr[:, None]
            return arr
    return None


def pick_group_array(data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    for key in LIKELY_GROUP_KEYS:
        if key in data:
            g = data[key]
            if g.ndim > 1:
                g = g.reshape(len(g), -1)
                # if multi-col group, hash to a single id
                g = g.astype("U").astype(object)
                g = np.array(["|".join(map(str, row)) for row in g])
            return g
    return None


def discover_targets(data: Dict[str, np.ndarray], explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return [t for t in explicit if t in data]
    # Auto: anything that starts with next_ and is array-like
    cands = []
    for k, v in data.items():
        if not isinstance(v, np.ndarray):
            continue
        if k.startswith("next_") and v.ndim >= 1 and v.shape[0] == guess_length(data):
            cands.append(k)
    return sorted(cands)


def discover_sensor_matrix(data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Try to build a sensors-only matrix.
       Priority: a single 'obs' vector; else concat likely lidar/sensor fields if aligned.
    """
    n = guess_length(data)
    # 1) obs
    if "obs" in data and isinstance(data["obs"], np.ndarray) and data["obs"].ndim >= 1 and len(data["obs"]) == n:
        obs = data["obs"]
        obs = obs if obs.ndim == 2 else obs.reshape(n, -1)
        return obs
    # 2) concat lidar-like fields
    parts = []
    used_keys = []
    for hint in SENSOR_FIELD_HINTS[1:]:
        for k in list(data.keys()):
            if hint in k and isinstance(data[k], np.ndarray) and len(data[k]) == n:
                arr = data[k]
                arr = arr if arr.ndim == 2 else arr.reshape(n, -1)
                parts.append(arr)
                used_keys.append(k)
    if parts:
        try:
            return np.concatenate(parts, axis=1)
        except Exception:
            pass
    return None


def build_geometry_only_matrix(data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Use current-step goal geometry only: vec_to_goal_t if available.
    Returns [N,2] matrix or None.
    """
    try:
        if "vec_to_goal_t" in data:
            v = np.asarray(data["vec_to_goal_t"])  # [N, 2]
            if v.ndim == 1:
                v = v[:, None]
            if v.ndim >= 2 and v.shape[0] == guess_length(data):
                return v[:, :2]
    except Exception:
        pass
    return None


def guess_length(data: Dict[str, np.ndarray]) -> int:
    # Pick the most common array length, preferring non-hook keys
    non_hook_lengths = []
    hook_lengths = []
    
    for k, v in data.items():
        if not isinstance(v, np.ndarray) or v.ndim < 1:
            continue
        if k.startswith('hook_'):
            hook_lengths.append(len(v))
        else:
            non_hook_lengths.append(len(v))
    
    # Prefer non-hook lengths for base dataset size
    if non_hook_lengths:
        from collections import Counter
        return int(Counter(non_hook_lengths).most_common(1)[0][0])
    elif hook_lengths:
        from collections import Counter
        return int(Counter(hook_lengths).most_common(1)[0][0])
    else:
        return 0


# -----------------------------
# Derived motion targets
# -----------------------------

def _parse_pos_dims_arg(pos_dims_arg: str) -> Tuple[int, int]:
    try:
        parts = [int(x.strip()) for x in pos_dims_arg.split(",") if x.strip()]
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 0, 1


def add_motion_targets(
    data: Dict[str, np.ndarray],
    pos_dims: Tuple[int, int] = (0, 1),
) -> Dict[str, np.ndarray]:
    """Compute and add simple motion targets derived from obs/next_obs.

    Adds keys (if computable and well-formed):
      - next_delta_xy: next_obs[:, pos] - obs[:, pos]  (shape [N,2])
      - next_speed: ||next_delta_xy||_2                 (shape [N,])
      - next_direction_xy: unit(next_delta_xy)          (shape [N,2])
      - next_dist_to_goal: ||vec_to_next_pos||_2        (shape [N,])  if vec_to_next_pos exists
      - next_unit_vec_to_goal: unit(vec_to_next_pos)    (shape [N,2])  if vec_to_next_pos exists
    """
    try:
        if "obs" in data and "next_obs" in data:
            obs = np.asarray(data["obs"])  # [N, D]
            nxt = np.asarray(data["next_obs"])  # [N, D]
            if obs.ndim >= 2 and nxt.ndim >= 2 and len(obs) == len(nxt):
                pd0, pd1 = pos_dims
                # guard indices
                if pd0 < obs.shape[1] and pd1 < obs.shape[1] and pd0 >= 0 and pd1 >= 0:
                    p_obs = obs[:, [pd0, pd1]].astype(np.float32)
                    p_nxt = nxt[:, [pd0, pd1]].astype(np.float32)
                    dxy = (p_nxt - p_obs).astype(np.float32)
                    data["next_delta_xy"] = dxy

                    spd = np.linalg.norm(dxy, axis=1).astype(np.float32)
                    data["next_speed"] = spd

                    eps = 1e-8
                    denom = np.maximum(spd, eps)[:, None]
                    dir_xy = (dxy / denom).astype(np.float32)
                    data["next_direction_xy"] = dir_xy
        # Goal-related (from NPZ built by log_rollouts_full.py)
        if "vec_to_next_pos" in data:
            v = np.asarray(data["vec_to_next_pos"], dtype=np.float32)
            if v.ndim == 1:
                v = v[:, None]
            if v.ndim == 2 and v.shape[1] >= 2:
                v2 = v[:, :2].astype(np.float32)
                dist = np.linalg.norm(v2, axis=1).astype(np.float32)
                data["next_dist_to_goal"] = dist

                eps = 1e-8
                denom = np.maximum(dist, eps)[:, None]
                unit = (v2 / denom).astype(np.float32)
                data["next_unit_vec_to_goal"] = unit
    except Exception:
        # Best-effort; do not crash caller
        pass
    return data


def stack_features(arrs: List[np.ndarray]) -> np.ndarray:
    parts = []
    for a in arrs:
        if a is None:
            continue
        if a.ndim == 1:
            a = a[:, None]
        parts.append(a)
    if not parts:
        raise ValueError("No feature arrays to stack.")
    X = np.concatenate(parts, axis=1)
    return X


def is_classification_target(name: str, y: np.ndarray) -> bool:
    # Heuristics: integer dtype or *_cls / *_id naming
    if y.dtype.kind in {"i", "u"}:
        return True
    lname = name.lower()
    if lname.endswith("_cls") or lname.endswith("_id"):
        # also guard against vector-valued one-hot style
        if y.ndim == 1:
            return True
    return False


# -----------------------------
# Train/eval
# -----------------------------

def train_eval_regression(Xtr, ytr, Xte, yte) -> Dict[str, float]:
    try:
        # For very high-dimensional targets, use dimension-wise regression
        if ytr.shape[1] > 50:  # High-dimensional target like next0_obs
            print(f"[INFO] High-dimensional target ({ytr.shape[1]}D), using dimension-wise regression")
            
            # Train separate Ridge models for each output dimension
            r2_dims = []
            mse_dims = []
            
            for i in range(min(ytr.shape[1], 10)):  # Sample first 10 dimensions
                try:
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge(alpha=10.0))
                    ])
                    model.fit(Xtr, ytr[:, i])
                    yhat_i = model.predict(Xte)
                    
                    r2_i = r2_score(yte[:, i], yhat_i)
                    mse_i = mean_squared_error(yte[:, i], yhat_i)
                    
                    if not (np.isnan(r2_i) or np.isinf(r2_i)):
                        r2_dims.append(r2_i)
                        mse_dims.append(mse_i)
                        
                except Exception:
                    continue
            
            if r2_dims:
                r2 = float(np.mean(r2_dims))
                mse = float(np.mean(mse_dims))
            else:
                r2, mse = -np.inf, np.inf
                
        else:
            # Low-dimensional target: use standard multi-output regression
            alpha = 10.0 if ytr.shape[1] > 5 else 1.0
            
            model = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", MultiOutputRegressor(Ridge(alpha=alpha)))
            ])
            model.fit(Xtr, ytr)
            yhat = model.predict(Xte)
            
            # Check for numerical issues
            if np.any(np.isnan(yhat)) or np.any(np.isinf(yhat)):
                raise ValueError("Prediction contains NaN or Inf values")
            
            # Calculate r2 manually for better numerical stability
            if ytr.shape[1] == 1:
                r2 = float(r2_score(yte, yhat))
            else:
                r2_per_dim = []
                for i in range(ytr.shape[1]):
                    try:
                        r2_dim = r2_score(yte[:, i], yhat[:, i])
                        if not (np.isnan(r2_dim) or np.isinf(r2_dim)):
                            r2_per_dim.append(r2_dim)
                    except Exception:
                        continue
                
                r2 = float(np.mean(r2_per_dim)) if r2_per_dim else -np.inf
            
            mse = float(mean_squared_error(yte, yhat))
        
        return {"r2": r2, "mse": mse}
        
    except Exception as e:
        return {"r2": -np.inf, "mse": np.inf, "error": str(e)}


def train_eval_classification(Xtr, ytr, Xte, yte) -> Dict[str, float]:
    # Single-label multi-class or binary
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logreg", LogisticRegression(max_iter=200, solver="saga", n_jobs=None, multi_class="auto"))
    ])
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    metrics = {
        "acc": float(accuracy_score(yte, ypred)),
        "f1_macro": float(f1_score(yte, ypred, average="macro")),
    }
    # Try to report log loss if predict_proba available
    try:
        proba = clf.predict_proba(Xte)
        # predict_proba returns list for multi-class OvR; handle binary/multiclass
        if isinstance(proba, list):
            # pick the column corresponding to class labels in estimator; approximate
            # Fallback: skip log loss if ambiguous
            pass
        else:
            metrics["log_loss"] = float(log_loss(yte, proba))
    except Exception:
        pass
    return metrics


def compute_velocity_persistence_baseline(
    target_name: str,
    y: np.ndarray,
    data: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Optional[Dict[str, float]]:
    """
    Compute velocity/direction persistence baseline using previous-step motion within each trajectory.
    """
    try:
        # Prefer 'traj_id' grouping from NPZ; fall back to episode_id
        group_key = None
        for k in ("traj_id", "episode_id"):
            if k in data:
                group_key = k
                break
        if group_key is None:
            return None

        # Need obs positions to compute previous motion
        if "obs" not in data:
            return None
        obs = np.asarray(data["obs"])  # [N, D]
        if obs.ndim < 2 or obs.shape[1] < 2:
            return None
        pos = obs[:, :2].astype(np.float32)

        groups = {}
        gids = np.asarray(data[group_key])
        for i, g in enumerate(gids):
            groups.setdefault(g, []).append(i)

        naive_predictions = np.full_like(y, np.nan, dtype=np.float32)

        for g, indices in groups.items():
            indices = sorted(indices)
            if len(indices) < 2:
                continue
            p = pos[indices]
            prev_vel = (p[1:] - p[:-1]).astype(np.float32)  # motion from t-1->t

            # Assign to indices[1:]
            if y.ndim == 2 and y.shape[1] >= 2 and ("delta" in target_name or "velocity" in target_name):
                for j, idx in enumerate(indices[1:]):
                    if j < len(prev_vel):
                        naive_predictions[idx] = prev_vel[j]
            elif ("direction" in target_name) and y.ndim == 2 and y.shape[1] >= 2:
                for j, idx in enumerate(indices[1:]):
                    if j < len(prev_vel):
                        vel = prev_vel[j]
                        sp = float(np.linalg.norm(vel))
                        if sp > 1e-6:
                            naive_predictions[idx] = (vel / sp).astype(np.float32)
                        else:
                            naive_predictions[idx] = np.array([0.0, 0.0], dtype=np.float32)
            elif ("speed" in target_name) and y.ndim == 1:
                for j, idx in enumerate(indices[1:]):
                    if j < len(prev_vel):
                        naive_predictions[idx] = float(np.linalg.norm(prev_vel[j]))
        
        # Evaluate on test set only
        test_mask = np.isin(np.arange(len(y)), test_idx)
        if y.ndim == 1:
            valid_mask = np.isfinite(naive_predictions) & test_mask
        else:
            valid_mask = ~np.isnan(naive_predictions).any(axis=1) & test_mask
        
        if valid_mask.sum() < 10:  # Need enough valid predictions
            return None
        
        y_true = y[valid_mask]
        y_pred = naive_predictions[valid_mask]
        
        # Ensure 2D for regression metrics
        if y_true.ndim == 1:
            y_true = y_true[:, None]
            y_pred = y_pred[:, None]
        
        r2 = float(r2_score(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        
        return {"r2": r2, "mse": mse}
        
    except Exception as e:
        print(f"[WARN] Velocity persistence baseline failed for {target_name}: {e}")
        return None


def fit_one_target(
    target_name: str,
    y: np.ndarray,
    X_baseline: Optional[np.ndarray],
    X_hook: Optional[np.ndarray],
    X_action: Optional[np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    data: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict[str, float]]:
    results = {}

    def _fit(Xname: str, X: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        if X is None:
            return None
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        # Filter out rows with NaN/Inf in either X or y per split
        def _finite_mask(A: np.ndarray) -> np.ndarray:
            if A.ndim == 1:
                return np.isfinite(A)
            return np.isfinite(A).all(axis=1)

        mtr = _finite_mask(Xtr) & _finite_mask(ytr)
        mte = _finite_mask(Xte) & _finite_mask(yte)

        # Require minimum samples to train and evaluate
        if mtr.sum() < 50 or mte.sum() < 50:
            return None

        Xtr, ytr = Xtr[mtr], ytr[mtr]
        Xte, yte = Xte[mte], yte[mte]
        if y.ndim == 1 and is_classification_target(target_name, y):
            return train_eval_classification(Xtr, ytr, Xte, yte)
        else:
            # regression for vector or float targets
            ytr_ = ytr
            yte_ = yte
            # ensure 2D for regression metric if vector-valued
            if ytr_.ndim == 1:
                ytr_ = ytr_[:, None]
                yte_ = yte_[:, None]
            return train_eval_regression(Xtr, ytr_, Xte, yte_)

    # Velocity persistence baseline for motion-like targets
    if data is not None and ("velocity" in target_name or "direction" in target_name or "delta" in target_name or "speed" in target_name):
        velocity_persistence_result = compute_velocity_persistence_baseline(
            target_name, y, data, train_idx, test_idx
        )
        if velocity_persistence_result is not None:
            results["velocity_persistence"] = velocity_persistence_result

    # Baseline: sensors only
    if X_baseline is not None:
        results["baseline_sensors"] = _fit("baseline_sensors", X_baseline)

    # Baseline: geometry-only (no internals)
    if data is not None and "vec_to_goal_t" in data:
        try:
            X_geom_local = build_geometry_only_matrix(data)
        except Exception:
            X_geom_local = None
        if X_geom_local is not None:
            results["baseline_geometry_only"] = _fit("baseline_geometry_only", X_geom_local)

    # Hook only
    if X_hook is not None:
        results["hook_only"] = _fit("hook_only", X_hook)

    # Hook + action
    if X_hook is not None and X_action is not None:
        results["hook_plus_action"] = _fit("hook_plus_action", np.concatenate([X_hook, X_action], axis=1))

    # Sensors + action (control)
    if X_baseline is not None and X_action is not None:
        results["sensors_plus_action"] = _fit("sensors_plus_action", np.concatenate([X_baseline, X_action], axis=1))

    return results


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Probe forward-looking structure from DeepLTL activations (v0)")
    ap.add_argument("--npz", type=Path, required=True, help="Path to NPZ with rollout fields + hooks")
    ap.add_argument("--hooks", nargs="+", required=True, help="Keys to use as activation hooks")
    ap.add_argument("--targets", nargs="*", default=None, help="Target keys to predict. Default: auto-detect next_* keys.")
    ap.add_argument("--include_action", action="store_true", help="Append action/policy_mu to features (if available)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test fraction if no group key is present")
    ap.add_argument("--world_level_split", action="store_true", help="Split by worlds instead of episodes (stronger generalization test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--add_motion_targets", action="store_true", help="Derive next_delta_xy/next_speed/next_direction_xy/etc. from obs/next_obs")
    ap.add_argument("--pos_dims", type=str, default="0,1", help="Comma-separated indices of x,y in obs and next_obs (default '0,1')")
    ap.add_argument("--geometry_only_baseline", action="store_true", help="Include baseline using only vec_to_goal_t (no internals)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.npz)

    # Optionally add motion-related derived targets from obs/next_obs and vec_to_next_pos
    if args.add_motion_targets:
        pos_dims = _parse_pos_dims_arg(args.pos_dims)
        data = add_motion_targets(data, pos_dims)
    n = guess_length(data)
    if n == 0:
        raise RuntimeError("Could not infer dataset length from NPZ.")

    # Build sensors-only baseline
    X_sensors = discover_sensor_matrix(data)
    # Build geometry-only baseline
    X_geom = build_geometry_only_matrix(data) if args.geometry_only_baseline else None

    # Build action features (optional)
    X_action = pick_action_array(data) if args.include_action else None

    # Check hooks and align lengths
    hooks_present = []
    for h in args.hooks:
        if h not in data:
            print(f"[WARN] Hook '{h}' not found in NPZ keys; skipping.")
            continue
        arr = data[h]
        if not isinstance(arr, np.ndarray):
            print(f"[WARN] Hook '{h}' is not an ndarray; skipping.")
            continue
        
        # Handle length mismatch: hooks might be sampled at higher frequency
        if len(arr) != n:
            if len(arr) > n and len(arr) % n == 0:
                # Downsample by taking every k-th sample
                k = len(arr) // n
                print(f"[INFO] Hook '{h}' has {len(arr)} samples vs base {n}; downsampling by factor {k}")
                arr = arr[::k]
                data[h] = arr  # Update in place
            elif len(arr) > n:
                # Truncate to base length
                print(f"[INFO] Hook '{h}' has {len(arr)} samples vs base {n}; truncating")
                arr = arr[:n]
                data[h] = arr  # Update in place
            else:
                print(f"[WARN] Hook '{h}' has fewer samples ({len(arr)}) than base ({n}); skipping.")
                continue
        
        hooks_present.append(h)
    if not hooks_present:
        raise RuntimeError("No valid hooks found.")

    # Targets
    target_names = discover_targets(data, args.targets)
    if not target_names:
        raise RuntimeError("No targets discovered. Pass --targets explicitly or ensure next_* keys exist.")

    # Train/test split
    if args.world_level_split and "world" in data:
        # World-level split: entire worlds in train or test, no overlap
        worlds = data["world"]
        unique_worlds = np.unique(worlds)
        
        # Deterministic world split based on seed
        np.random.seed(args.seed)
        shuffled_worlds = np.random.permutation(unique_worlds)
        
        n_test_worlds = max(1, int(len(unique_worlds) * args.test_size))
        test_worlds = shuffled_worlds[:n_test_worlds]
        train_worlds = shuffled_worlds[n_test_worlds:]
        
        train_idx = np.where(np.isin(worlds, train_worlds))[0]
        test_idx = np.where(np.isin(worlds, test_worlds))[0]
        
        split_info = {
            "type": "world_level", 
            "train_worlds": train_worlds.tolist(),
            "test_worlds": test_worlds.tolist(),
            "test_size": len(test_idx) / n
        }
        print(f"World-level split: {len(train_worlds)} train worlds, {len(test_worlds)} test worlds")
        print(f"Train worlds: {train_worlds}")
        print(f"Test worlds: {test_worlds}")
        
    else:
        # Original episode-level or random split
        g = pick_group_array(data)
        if g is not None:
            splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
            (train_idx, test_idx) = next(splitter.split(np.arange(n), groups=g))
            split_info = {"type": "group", "group_key": True, "test_size": args.test_size}
        else:
            train_idx, test_idx = train_test_split(np.arange(n), test_size=args.test_size, random_state=args.seed, shuffle=True)
            split_info = {"type": "random", "test_size": args.test_size}

    results = {
        "meta": {
            "npz": str(args.npz),
            "n": int(n),
            "hooks": hooks_present,
            "targets": target_names,
            "include_action": bool(args.include_action),
            "split": split_info,
            "seed": int(args.seed),
        },
        "per_hook": {}
    }

    # Evaluate per hook and per target
    for h in hooks_present:
        H = data[h]
        H = H if H.ndim == 2 else H.reshape(n, -1)
        hook_results = {}
        for tgt in target_names:
            y = data[tgt]
            if y.ndim == 1:
                y_arr = y
            else:
                # Ensure 2D for regression; classification handler deals with 1D
                y_arr = y

            try:
                r = fit_one_target(
                    target_name=tgt,
                    y=y_arr,
                    X_baseline=X_sensors,
                    X_hook=H,
                    X_action=X_action,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    data=data,
                )
            except Exception as e:
                r = {"error": str(e)}
            hook_results[tgt] = r
        results["per_hook"][h] = hook_results

    # Save JSON + a compact Markdown summary
    json_path = args.out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {json_path}")

    # Markdown summary (per-hook per-target key metrics)
    md_lines = ["# Forward-look Probe (v0)\n"]
    md_lines.append(f"NPZ: `{args.npz}`  ")
    md_lines.append(f"Hooks: {', '.join(hooks_present)}  ")
    md_lines.append(f"Targets: {', '.join(target_names)}  ")
    md_lines.append(f"Include action: {args.include_action}  ")
    md_lines.append("")

    def fmt_metrics(m: Optional[Dict[str, float]]) -> str:
        if m is None:
            return "—"
        return ", ".join([f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, (int, float))])

    for h in hooks_present:
        md_lines.append(f"\n## Hook: `{h}`\n")
        
        # Check if any target has velocity persistence baseline
        has_velocity_persistence = any(
            "velocity_persistence" in results["per_hook"][h].get(tgt, {}) 
            for tgt in target_names
        )
        
        cols = ["target"]
        if has_velocity_persistence:
            cols.append("velocity_persistence")
        cols.append("baseline")
        if any(results["per_hook"][h].get(tgt, {}).get("baseline_geometry_only") for tgt in target_names):
            cols.append("geom_only")
        cols.extend(["sensors+action","hook","hook+action"])
        table_header = "| " + " | ".join(cols) + " |\n" + "|" + "---|"*len(cols)
        
        md_lines.append(table_header)
        for tgt in target_names:
            r = results["per_hook"][h][tgt]
            if "error" in r:
                if has_velocity_persistence:
                    md_lines.append(f"| {tgt} | — | ERROR: {r['error']} |  |  |  |")
                else:
                    md_lines.append(f"| {tgt} | ERROR: {r['error']} |  |  |  |")
                continue
            
            row = [tgt]
            if has_velocity_persistence:
                row.append(fmt_metrics(r.get("velocity_persistence")))
            row.append(fmt_metrics(r.get("baseline_sensors")))
            if "baseline_geometry_only" in r:
                row.append(fmt_metrics(r.get("baseline_geometry_only")))
            row.extend([
                fmt_metrics(r.get("sensors_plus_action")),
                fmt_metrics(r.get("hook_only")),
                fmt_metrics(r.get("hook_plus_action")),
            ])
            md_lines.append("| " + " | ".join(row) + " |")

    md_path = args.out_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved markdown summary to {md_path}")


if __name__ == "__main__":
    main()
