#!/usr/bin/env python3
"""
A/B comparison runner for DeepLTL probes:
- Calls probe_actor_fusion_clean.py twice (linear vs MLP)
- Merges results and computes deltas (MLP - Linear)
- Prints short analysis and saves combined CSV (and optional Markdown)

Usage (example):
  python interpretability/working_scripts/probe_compare_linear_vs_mlp.py \
    --env PointLtl2-v0 \
    --exp big_test \
    --seed 0 \
    --n_worlds 5 \
    --n_rollout 5 \
    --max_step 500 \
    --deterministic \
    --held_out_worlds "1,3" \
    --targets "min_wall_dist,nearest_zone_id,bearing_to_goal_cls,next_wall_lidar,d_front_clearance_sign,wz,agent_pos,delta_xy" \
    --mlp_hidden "128,64" \
    --mlp_alpha 3e-4 \
    --mlp_max_iter 400 \
    --out_dir interpretability/working_scripts/probe_ab \
    --make_markdown

Tip: keep --held_out_worlds fixed so splits are identical across runs.
"""

import argparse, os, shlex, subprocess, sys, textwrap, datetime, time
import pandas as pd
from pathlib import Path
import subprocess, shlex, sys


THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]  # repo root (adjust if needed)


def run(cmd: str, timeout_sec: int | None = None):
    print(f"\n$ {cmd}")
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    try:
        subprocess.run(
            shlex.split(cmd),
            check=True,
            timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"Process timed out after {timeout_sec}s: {cmd}", file=sys.stderr)
        raise SystemExit(124)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Command failed with code {e.returncode}: {cmd}")

def build_shared_args(args) -> str:
    parts = [
        f"--env {args.env}",
        f"--exp {args.exp}",
        f"--seed {args.seed}",
        f"--n_worlds {args.n_worlds}",
        f"--n_rollout {args.n_rollout}",
        f"--max_step {args.max_step}",
    ]
    if args.deterministic: parts.append("--deterministic")
    if args.held_out_worlds: parts.append(f'--held_out_worlds "{args.held_out_worlds}"')
    if args.targets: parts.append(f'--targets "{args.targets}"')
    return " ".join(parts)

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # shared DeepLTL rollout/probing args
    ap.add_argument("--env", default="PointLtl2-v0")
    ap.add_argument("--exp", default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_worlds", type=int, default=5)
    ap.add_argument("--n_rollout", type=int, default=5)
    ap.add_argument("--max_step", type=int, default=500)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--held_out_worlds", type=str, default="")
    ap.add_argument("--targets", type=str, default="")
    # MLP hyperparams
    ap.add_argument("--mlp_hidden", type=str, default="64,32")
    ap.add_argument("--mlp_alpha", type=float, default=1e-3)
    ap.add_argument("--mlp_max_iter", type=int, default=300)
    # output controls
    ap.add_argument("--out_dir", type=str, default="interpretability/working_scripts/probe_ab")
    ap.add_argument("--make_markdown", action="store_true", help="Also write a Markdown summary beside CSV")
    # execution controls
    ap.add_argument("--timeout_sec", type=int, default=0, help="Per-subprocess timeout; 0 disables")
    ap.add_argument("--fast", action="store_true", help="Clamp rollouts/steps to small debug values")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional fast-mode clamps
    if args.fast:
        args.n_worlds = min(args.n_worlds, 2)
        args.n_rollout = min(args.n_rollout, 2)
        args.max_step = min(args.max_step, 200)

    shared = build_shared_args(args)
    linear_csv = out_dir / "probe_linear.csv"
    mlp_csv    = out_dir / "probe_mlp.csv"

    # 1) Run linear
    cmd_lin = f"""
    python {ROOT/'interpretability/working_scripts/probe_actor_fusion_clean.py'} \
      {shared} \
      --probe_type linear \
      --results_csv {linear_csv}
    """.strip()
    run(cmd_lin, timeout_sec=args.timeout_sec)

    # 2) Run MLP
    cmd_mlp = f"""
    python {ROOT/'interpretability/working_scripts/probe_actor_fusion_clean.py'} \
      {shared} \
      --probe_type mlp --mlp_hidden "{args.mlp_hidden}" --mlp_alpha {args.mlp_alpha} --mlp_max_iter {args.mlp_max_iter} \
      --results_csv {mlp_csv}
    """.strip()
    run(cmd_mlp, timeout_sec=args.timeout_sec)

    # 3) Merge + compute deltas
    lin = pd.read_csv(linear_csv)
    mlp = pd.read_csv(mlp_csv)

    # Normalize column names just in case
    def norm(df):
        return df.rename(columns={c: c.strip() for c in df.columns})
    lin, mlp = norm(lin), norm(mlp)

    keys = ["target","shape","metric"]  # identity columns written by your script
    cols = ["in","h1","h2","h3","out"]

    both = lin.merge(mlp, on=keys, suffixes=("_lin","_mlp"), how="inner")
    if both.empty:
        print("No overlapping targets between runs. Check --targets values.", file=sys.stderr)
        sys.exit(1)

    for c in cols:
        both[f"Δ_{c}"] = both[f"{c}_mlp"] - both[f"{c}_lin"]

    # Reorder columns for readability
    side_by_side_cols = keys + \
        [f"{c}_lin" for c in cols] + [f"{c}_mlp" for c in cols] + [f"Δ_{c}" for c in cols]

    both = both[side_by_side_cols].sort_values(["target"])

    # 4) Save combined CSV
    combo_csv = out_dir / "probe_linear_vs_mlp.csv"
    both.to_csv(combo_csv, index=False)
    print(f"\n💾 Wrote combined CSV → {combo_csv}")

    # 5) Console summaries
    def fmt(x):
        try:
            return f"{float(x):.3f}"
        except Exception:
            return str(x)

    def print_table(df, title, sort_key):
        print("\n" + title)
        print("="*len(title))
        df2 = df.sort_values(sort_key, ascending=False)
        head_cols = ["target","metric","shape","in_lin","in_mlp","Δ_in","out_lin","out_mlp","Δ_out"]
        # fallbacks if any columns missing
        head_cols = [c for c in head_cols if c in df2.columns]
        print(df2[head_cols].to_string(index=False, formatters={c: fmt for c in head_cols}))

    # Meaningful changes (absolute delta ≥ 0.05 on IN or OUT)
    sig = both[(both["Δ_in"].abs() >= 0.05) | (both["Δ_out"].abs() >= 0.05)].copy()
    print_table(sig, "Meaningful changes (|Δ| ≥ 0.05 on IN or OUT)", "Δ_in")

    # Top IN gains
    top_in = both.copy()
    print_table(top_in, "Top IN gains (sorted by Δ_in)", "Δ_in")

    # Top OUT gains
    top_out = both.copy()
    print_table(top_out, "Top OUT gains (sorted by Δ_out)", "Δ_out")

    # 6) Optional Markdown report
    if args.make_markdown:
        md = []
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        md.append(f"# Probe A/B Report — Linear vs MLP\n")
        md.append(f"_Generated: {now}_\n")
        md.append("## Settings\n")
        md.append("```bash\n" + textwrap.dedent(f"""
        # Shared
        {shared}

        # Linear
        --probe_type linear

        # MLP
        --probe_type mlp --mlp_hidden "{args.mlp_hidden}" --mlp_alpha {args.mlp_alpha} --mlp_max_iter {args.mlp_max_iter}
        """).strip() + "\n```\n")
        md.append("## Combined table (CSV)\n")
        md.append(f"- `{combo_csv}`\n")
        def to_md_table(df, title, sort_key):
            if df.empty:
                return ""
            df2 = df.sort_values(sort_key, ascending=False).copy()
            cols_md = ["target","metric","shape","in_lin","in_mlp","Δ_in","out_lin","out_mlp","Δ_out"]
            cols_md = [c for c in cols_md if c in df2.columns]
            try:
                tbl = df2[cols_md].to_markdown(index=False)
            except Exception:
                # Fallback if optional 'tabulate' is not installed
                tbl = "```\n" + df2[cols_md].to_string(index=False) + "\n```"
            return "### " + title + "\n\n" + tbl + "\n"
        md.append(to_md_table(sig, "Meaningful changes (|Δ| ≥ 0.05 on IN or OUT)", "Δ_in"))
        md.append(to_md_table(top_in, "Top IN gains (MLP − Linear)", "Δ_in"))
        md.append(to_md_table(top_out, "Top OUT gains (MLP − Linear)", "Δ_out"))
        md_path = out_dir / "probe_linear_vs_mlp.md"
        with open(md_path, "w") as f:
            f.write("\n".join(md))
        print(f"📝 Wrote Markdown report → {md_path}")

if __name__ == "__main__":
    main()
