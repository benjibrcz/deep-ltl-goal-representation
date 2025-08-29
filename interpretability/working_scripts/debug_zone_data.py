#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Optional import: tail/success/proposition-based center inference you already wrote
try:
    from zone_utils import infer_centers_from_df_for_color
except Exception:
    infer_centers_from_df_for_color = None


REQUIRED = {"world_seed", "chain_id", "link_idx", "step_idx", "color", "pos_x", "pos_y"}

def color_norm(s: Any) -> Optional[str]:
    if s is None: return None
    s = str(s).strip().lower()
    if " " in s:
        s = s.split()[-1]
    return s

def extract_xy(item):
    if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 2:
        return [float(item[0]), float(item[1])]
    if isinstance(item, dict):
        for key in ("xy", "center", "coords", "c"):
            v = item.get(key, None)
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 2:
                return [float(v[0]), float(v[1])]
        if "x" in item and "y" in item:
            return [float(item["x"]), float(item["y"])]
    return None

def normalize_two_centers(obj) -> Optional[np.ndarray]:
    if obj is None:
        return None
    if isinstance(obj, dict) and "centers" in obj:
        obj = obj["centers"]
    if isinstance(obj, dict) and not isinstance(next(iter(obj.values()), None), (str, int, float)):
        try:
            keys = sorted(obj.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
        except Exception:
            keys = list(obj.keys())
        lst = [obj[k] for k in keys]
    else:
        lst = obj if isinstance(obj, list) else [obj]
    coords = []
    for it in lst:
        xy = extract_xy(it)
        if xy is not None:
            coords.append(xy)
    if len(coords) >= 2:
        arr = np.asarray(coords[:2], dtype=float)
        if arr.shape == (2,2):
            return arr
    return None

def kmeans_two_centers_all_points(df_seed: pd.DataFrame, color: str, random_state: int = 0, max_points: int = 5000):
    cand = df_seed[df_seed["color"] == color][["pos_x","pos_y"]].dropna()
    if len(cand) < 2:
        return None
    pts = cand.to_numpy()
    if len(pts) > max_points:
        rng = np.random.default_rng(random_state)
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    km = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit(pts)
    return km.cluster_centers_.astype(float)

def first_non_null_shape(col: pd.Series) -> Optional[int]:
    for v in col:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            a = np.asarray(v, dtype=float).ravel()
            return int(a.shape[0])
        # other types -> skip
    return None

def check_monotonic_steps(df: pd.DataFrame) -> Tuple[int, int]:
    bad, total = 0, 0
    for (_, _, _), seg in df.groupby(["world_seed","chain_id","link_idx"]):
        total += 1
        s = seg["step_idx"].to_numpy()
        if not np.all(np.diff(s) >= 0):
            bad += 1
    return bad, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--centers_json", required=False)
    ap.add_argument("--max_pairs", type=int, default=10, help="show this many sample (seed,color) rows")
    ap.add_argument("--random_state", type=int, default=0)
    args = ap.parse_args()

    print(f"[load] parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"[rows] {len(df):,}")
    print(f"[cols] {list(df.columns)}")

    missing = sorted(list(REQUIRED - set(df.columns)))
    if missing:
        print(f"[ERROR] missing required columns: {missing}")
    else:
        print("[ok] all required columns present")

    # Coerce types & count NaNs/infs
    for c in ["world_seed","chain_id","link_idx","step_idx","pos_x","pos_y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Replace infs in positions
    if {"pos_x","pos_y"}.issubset(df.columns):
        before = len(df)
        df[["pos_x","pos_y"]] = df[["pos_x","pos_y"]].replace([np.inf, -np.inf], np.nan)
        dropped = df[["pos_x","pos_y"]].isna().any(axis=1).sum()
        print(f"[pos] rows with invalid pos_x/pos_y: {dropped:,}")

    # Color normalization preview
    if "color" in df.columns:
        print("\n[color] top raw values:")
        print(df["color"].astype(str).value_counts().head(10))
        df["color"] = df["color"].map(color_norm)
        print("\n[color] normalized top values:")
        print(df["color"].value_counts().head(10))

    # Seeds, coverage
    if "world_seed" in df.columns:
        df["world_seed"] = df["world_seed"].astype("Int64")
        null_ws = df["world_seed"].isna().sum()
        print(f"\n[world_seed] null count: {null_ws}")
        df = df.dropna(subset=["world_seed"]).copy()
        df["world_seed"] = df["world_seed"].astype(int)
        print(f"[world_seed] unique seeds: {df['world_seed'].nunique()}")

    # Per (seed,color) counts
    if REQUIRED.issubset(df.columns):
        grp = (df.groupby(["world_seed","color"])
                 .agg(n_rows=("pos_x","count"),
                      n_links=("link_idx","nunique"),
                      steps_median=("step_idx","median"),
                      steps_min=("step_idx","min"),
                      steps_max=("step_idx","max"))
                 .reset_index()
                 .sort_values("n_rows", ascending=False))
        print("\n[(seed,color) head]")
        print(grp.head(args.max_pairs).to_string(index=False))
        out_csv = Path(args.parquet).with_suffix(".seed_color_counts.csv")
        grp.to_csv(out_csv, index=False)
        print(f"[save] wrote per-(seed,color) counts to {out_csv}")

        # Link length distribution per link
        seg_sizes = df.groupby(["world_seed","chain_id","link_idx"]).size().to_numpy()
        if len(seg_sizes) > 0:
            q = np.quantile(seg_sizes, [0,0.25,0.5,0.75,0.9,0.99,1.0])
            print("\n[link lengths] count:", len(seg_sizes), "quantiles:", dict(zip(["min","q25","median","q75","q90","q99","max"], map(int, q))))
        bad, total = check_monotonic_steps(df)
        print(f"[step_idx] non-monotonic segments: {bad}/{total}")

    # Vector columns shapes
    for col in ["h","obs_features"]:
        if col in df.columns:
            dim = first_non_null_shape(df[col])
            nulls = df[col].isna().sum()
            print(f"\n[{col}] first non-null dim: {dim}, nulls: {nulls}")
        else:
            print(f"\n[{col}] column missing")

    # Centers JSON analysis
    centers = None
    if args.centers_json:
        path = Path(args.centers_json)
        if path.exists():
            with open(path, "r") as f:
                centers = json.load(f)
            print(f"\n[centers] loaded {path}")
            seeds_cjson = list(centers.keys())
            print(f"[centers] seeds in JSON: {len(seeds_cjson)} (example: {seeds_cjson[:5]})")
            c_pairs = []
            good_pairs = 0
            for s, cmap in centers.items():
                if isinstance(cmap, dict):
                    for c, entry in cmap.items():
                        twoc = normalize_two_centers(entry)
                        if twoc is not None and twoc.shape == (2,2):
                            good_pairs += 1
                        c_pairs.append((int(s), color_norm(c)))
            print(f"[centers] total pairs in JSON: {len(c_pairs)} ; with 2 centers: {good_pairs}")
            df_pairs = set(df[["world_seed","color"]].dropna().apply(lambda r: (int(r["world_seed"]), str(r["color"])), axis=1).unique()) \
                       if {"world_seed","color"}.issubset(df.columns) else set()
            centers_pairs = set(c_pairs)
            overlap = df_pairs & centers_pairs
            missing = sorted(list(df_pairs - centers_pairs))[:10]
            print(f"[centers] overlap with df pairs: {len(overlap)} ; example missing pairs: {missing}")
        else:
            print(f"\n[centers] file not found: {path}")

    # Try inference feasibility for a few pairs
    if infer_centers_from_df_for_color is not None:
        print("\n[infer] sampling up to 8 (seed,color) to check inference")
        sampled = (df.groupby(["world_seed","color"]).size()
                    .reset_index(name="n").sort_values("n", ascending=False).head(8))
        for _, row in sampled.iterrows():
            ws, c = int(row["world_seed"]), str(row["color"])
            df_seed = df[df["world_seed"] == ws]
            # Keep it permissive: min_points=2
            twoc_list = infer_centers_from_df_for_color(df_seed, c, min_points=2, random_state=args.random_state)
            ok = (len(twoc_list) == 2)
            print(f"  (seed={ws}, color={c}) rows={int(row['n'])} -> infer_tail: {ok}")
            if not ok:
                # Try all-points KMeans fallback
                twoc = kmeans_two_centers_all_points(df_seed, c, random_state=args.random_state)
                ok2 = isinstance(twoc, np.ndarray) and twoc.shape == (2,2)
                print(f"    fallback KMeans(all points): {ok2}")
    else:
        print("\n[infer] zone_utils.infer_centers_from_df_for_color not importable; skipping.")

    print("\n[done] debug complete.")

if __name__ == "__main__":
    main()
