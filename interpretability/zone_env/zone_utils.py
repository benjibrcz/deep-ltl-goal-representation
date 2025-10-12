# interpretability/working_scripts/zone_utils.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ---------- small helpers ----------

def color_from_goal_text(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    # e.g. "FG blue" -> "blue"
    if " " in s:
        return s.split()[-1]
    return s

def ensure_numeric_xy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pos_x"] = pd.to_numeric(out["pos_x"], errors="coerce")
    out["pos_y"] = pd.to_numeric(out["pos_y"], errors="coerce")
    return out.dropna(subset=["pos_x", "pos_y"])

def stable_sort_centers(centers_xy: np.ndarray) -> List[Tuple[float, float]]:
    # deterministic ID assignment: sort by x then y
    cs = [(float(x), float(y)) for x, y in centers_xy]
    cs.sort(key=lambda p: (round(p[0], 4), round(p[1], 4)))
    return cs

def nearest_center_id(xy: Tuple[float, float], centers: List[Tuple[float, float]]) -> int:
    arr = np.asarray(centers, dtype=float)
    d = np.square(arr - np.asarray(xy, dtype=float)).sum(axis=1)
    return int(np.argmin(d))


# ---------- center inference from rollouts ----------

def infer_centers_from_df_for_color(
    df_seed: pd.DataFrame,
    color: str,
    min_points: int = 2,
    prefer_success: bool = True,
    random_state: int = 0,
    max_points_for_kmeans: int = 4000,
) -> List[Tuple[float, float]]:
    """
    Infer two zone centers for a given (seed,color) from rollout data.
    Order of attempts:
      1) SUCCESS rows for that color (very precise but sparse)
      2) ANY rows in this seed whose `propositions` mention the color (not gated by `color==...`)
      3) Fallback: last 20% of each (chain,link) segment while the current goal color==target

    Returns [] if we can't get >=2 usable points.
    """
    color = str(color).lower()

    # --- 1) STRICT: success rows while target color is active ---
    if {"color", "success_step"}.issubset(df_seed.columns):
        cand_goal = df_seed[df_seed["color"].astype(str).str.lower() == color]
        suc = ensure_numeric_xy(cand_goal[cand_goal["success_step"] == True])
        if len(suc) >= 2:
            pts = suc[["pos_x", "pos_y"]].to_numpy()
            if len(pts) > max_points_for_kmeans:
                pts = pts[:max_points_for_kmeans]
            km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
            centers = km.fit(pts).cluster_centers_
            return stable_sort_centers(centers)

    # --- 2) PERMISSIVE: any propositions row that mentions the color (across the whole seed) ---
    if "propositions" in df_seed.columns:
        mask = df_seed["propositions"].astype(str).str.lower().str.contains(color, na=False)
        prop_all = ensure_numeric_xy(df_seed[mask])
        if len(prop_all) >= 2:
            pts = prop_all[["pos_x", "pos_y"]].to_numpy()
            if len(pts) > max_points_for_kmeans:
                # subsample for stability/speed
                rng = np.random.default_rng(random_state)
                idx = rng.choice(len(pts), size=max_points_for_kmeans, replace=False)
                pts = pts[idx]
            km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
            centers = km.fit(pts).cluster_centers_
            return stable_sort_centers(centers)

    # --- 3) FALLBACK: last-20% of segments while goal color is active ---
    if {"chain_id", "link_idx", "step_idx", "color"}.issubset(df_seed.columns):
        cand_goal = df_seed[df_seed["color"].astype(str).str.lower() == color]
        pts_list = []
        for (_, _), seg in cand_goal.groupby(["chain_id", "link_idx"]):
            seg = seg.sort_values("step_idx")
            n = max(1, int(0.2 * len(seg)))
            tail = ensure_numeric_xy(seg.tail(n))
            if not tail.empty:
                pts_list.append(tail[["pos_x", "pos_y"]].mean().to_numpy())
        if len(pts_list) >= 2:
            pts = np.vstack(pts_list)
            km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
            centers = km.fit(pts).cluster_centers_
            return stable_sort_centers(centers)

    return []



def build_centers_map_from_rollouts(
    df: pd.DataFrame,
    colors: List[str],
    min_points: int = 12,
    random_state: int = 0,
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """
    Returns {seed_str: {color: [{"id":0,"x":...,"y":...}, {"id":1,...}]}}
    Only includes (seed,color) where 2 centers are found.
    """
    # normalize columns
    df = df.copy()
    df["color"] = df["color"].astype(str)
    df["world_seed"] = df["world_seed"].astype(int)
    seeds = np.sort(df["world_seed"].unique())

    result = {}
    for ws in seeds:
        df_seed = df[df["world_seed"] == ws]
        out_seed = {}
        for c in colors:
            centers = infer_centers_from_df_for_color(
                df_seed, c, min_points=min_points, random_state=random_state
            )
            if len(centers) == 2:
                out_seed[c] = [{"id": i, "x": cx, "y": cy} for i, (cx, cy) in enumerate(centers)]
        if out_seed:
            result[str(ws)] = out_seed
    return result


def save_centers_json(centers_map: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(centers_map, f, indent=2)
