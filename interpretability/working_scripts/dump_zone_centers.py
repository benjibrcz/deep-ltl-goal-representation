# interpretability/working_scripts/dump_zone_centers.py
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zone_utils import (
    build_centers_map_from_rollouts,
    save_centers_json,
)

def plot_seed(
    df_seed: pd.DataFrame,
    centers_seed: dict,
    colors: list[str],
    out_png: Path,
    max_points: int = 5000,
):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    df_seed = df_seed.copy()
    df_seed["pos_x"] = pd.to_numeric(df_seed["pos_x"], errors="coerce")
    df_seed["pos_y"] = pd.to_numeric(df_seed["pos_y"], errors="coerce")
    df_seed = df_seed.dropna(subset=["pos_x", "pos_y"])

    # subsample for speed/clarity
    if len(df_seed) > max_points:
        df_seed = df_seed.sample(max_points, random_state=0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_title(f"Seed {int(df_seed['world_seed'].iloc[0])} — samples & inferred centers")

    # scatter by goal color (the link color column)
    palette = dict(green="#2ca02c", blue="#1f77b4", yellow="#ffbf00", magenta="#d62728")
    for c in colors:
        pts = df_seed[df_seed["color"].str.lower() == c.lower()]
        if pts.empty:
            continue
        ax.scatter(pts["pos_x"], pts["pos_y"], s=6, alpha=0.25, label=c, c=palette.get(c, None))

    # centers
    for c in colors:
        centers = centers_seed.get(c, [])
        for z in centers:
            ax.scatter([z["x"]], [z["y"]], s=140, marker="x", linewidths=3, color=palette.get(c, "k"))
            ax.text(z["x"], z["y"], f"{c}[{z['id']}]", fontsize=9, color=palette.get(c, "k"))

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--colors", default="green,blue,yellow,magenta")
    ap.add_argument("--out_json", default="interpretability/working_scripts/zone_centers.json")
    ap.add_argument("--plot_dir", default="interpretability/working_scripts/zone_centers_plots")
    ap.add_argument("--min_points", type=int, default=12, help="min points to run KMeans")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    colors = [c.strip().lower() for c in args.colors.split(",") if c.strip()]

    print(f"[load] {args.parquet}")
    df = pd.read_parquet(args.parquet)
    if not {"world_seed", "color", "pos_x", "pos_y"}.issubset(df.columns):
        raise SystemExit("Parquet missing required columns: world_seed,color,pos_x,pos_y")

    # Normalize types
    df["color"] = df["color"].astype(str).str.lower()
    df["world_seed"] = df["world_seed"].astype(int)

    # Build centers map
    centers_map = build_centers_map_from_rollouts(
        df, colors=colors, min_points=args.min_points, random_state=args.seed
    )
    out_json = Path(args.out_json)
    save_centers_json(centers_map, out_json)
    print(f"[save] wrote centers → {out_json}  (seeds={len(centers_map)})")

    # Optional small visual sanity checks
    plot_root = Path(args.plot_dir)
    plot_root.mkdir(parents=True, exist_ok=True)
    seeds = sorted({int(s) for s in centers_map.keys()})
    print(f"[plots] generating up to 12 seed plots in {plot_root} …")
    for ws in seeds[:12]:
        df_seed = df[df["world_seed"] == ws]
        centers_seed = centers_map[str(ws)]
        out_png = plot_root / f"seed_{ws}.png"
        plot_seed(df_seed, centers_seed, colors, out_png)
        print(f"  • {out_png}")

if __name__ == "__main__":
    main()
