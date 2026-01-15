#!/usr/bin/env python3
"""
Generate Trajectory Plots for Report

Creates example trajectory visualizations for each paper capability test:
1. Optimality: F (blue & F green)
2. Safety: (F green | F yellow) & G !blue
3. Infinite Horizon: G F blue & G F green

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/generate_trajectory_plots.py
"""
import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(SRC))

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
    from gym import spaces as gspaces

torch.set_grad_enabled(False)

ZONE_RADIUS = 0.4

# Color mapping for zones
ZONE_COLORS = {
    'blue': '#3498db',
    'green': '#2ecc71',
    'yellow': '#f1c40f',
    'magenta': '#e91e63',
    'red': '#e74c3c',
    'cyan': '#00bcd4',
}


def coerce_action(act, action_space):
    if isinstance(action_space, gspaces.Box):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        need = int(np.prod(action_space.shape))
        if a.size == 1 and need > 1:
            a = np.repeat(a, need)
        a = np.clip(a, action_space.low, action_space.high)
        return a.reshape(action_space.shape)
    elif isinstance(action_space, gspaces.Discrete):
        return int(np.asarray(act).ravel()[0]) if isinstance(act, (np.ndarray, list, tuple)) else int(act)
    return act


def extract_zone_info(env) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]]]:
    agent_pos = None
    zone_positions = {}
    try:
        task = env.unwrapped.task
        agent_pos = task.agent.pos[:2].copy()
        for geom_name, geom in task._geoms.items():
            if hasattr(geom, 'color_name') and hasattr(geom, 'num'):
                color = geom.color_name
                if color not in zone_positions:
                    zone_positions[color] = []
                for i in range(geom.num):
                    try:
                        body_name = f'{color}_zone{i}'
                        pos = task.data.body(body_name).xpos[:2].copy()
                        zone_positions[color].append(pos)
                    except:
                        pass
    except:
        pass
    return agent_pos, zone_positions


def run_and_collect_trajectory(env_id, model, formula, seed, max_steps=300):
    """Run a scenario and collect trajectory data for plotting."""

    sampler_fn = FixedSampler.partial(formula)
    env = make_env(env_id, sampler_fn, sequence=False)
    props = set(env.get_propositions())

    planner = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, planner, propositions=props)

    reset_out = env.reset(seed=seed)
    obs, info = (reset_out, {}) if not isinstance(reset_out, tuple) else reset_out
    agent.reset()

    agent_start, zone_positions = extract_zone_info(env)

    if agent_start is None:
        env.close()
        return None

    trajectory = [agent_start.copy()]
    zone_visits = []

    for step in range(max_steps):
        with torch.no_grad():
            action = agent.get_action(obs, info, deterministic=True)
        action = coerce_action(action, env.action_space)

        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret

        try:
            agent_pos = env.unwrapped.task.agent.pos[:2].copy()
            trajectory.append(agent_pos.copy())

            # Track zone visits
            for color, positions in zone_positions.items():
                for pos in positions:
                    if np.linalg.norm(agent_pos - pos) < ZONE_RADIUS:
                        if not zone_visits or zone_visits[-1][0] != color or step - zone_visits[-1][1] > 5:
                            zone_visits.append((color, step))
        except:
            pass

        if done:
            break

    env.close()

    return {
        'trajectory': np.array(trajectory),
        'zone_positions': zone_positions,
        'agent_start': agent_start,
        'zone_visits': zone_visits,
        'formula': formula,
    }


def plot_trajectory(data, title, filename, highlight_colors=None, avoid_colors=None):
    """Plot a single trajectory with zones."""

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot zones
    for color, positions in data['zone_positions'].items():
        zone_color = ZONE_COLORS.get(color, 'gray')
        alpha = 0.3
        edgecolor = zone_color
        linewidth = 2

        # Highlight goal zones
        if highlight_colors and color in highlight_colors:
            alpha = 0.5
            linewidth = 3

        # Mark avoid zones differently
        if avoid_colors and color in avoid_colors:
            alpha = 0.4
            edgecolor = 'red'
            linewidth = 3

        for pos in positions:
            circle = patches.Circle(pos, ZONE_RADIUS,
                                   facecolor=zone_color,
                                   edgecolor=edgecolor,
                                   alpha=alpha,
                                   linewidth=linewidth)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], color[0].upper(),
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color='white' if color != 'yellow' else 'black')

    # Plot trajectory
    traj = data['trajectory']

    # Color trajectory by time
    n_points = len(traj)
    for i in range(n_points - 1):
        progress = i / max(n_points - 1, 1)
        color = plt.cm.plasma(progress)
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], color=color, linewidth=2, alpha=0.8)

    # Mark start and end
    ax.scatter(traj[0, 0], traj[0, 1], s=150, c='green', marker='o',
              zorder=5, edgecolors='white', linewidths=2, label='Start')
    ax.scatter(traj[-1, 0], traj[-1, 1], s=150, c='red', marker='s',
              zorder=5, edgecolors='white', linewidths=2, label='End')

    # Mark zone visits with numbers
    for i, (color, step) in enumerate(data['zone_visits']):
        # Find position at that step
        if step < len(traj):
            pos = traj[step]
            ax.annotate(str(i+1), pos, fontsize=12, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')

    # Add formula as subtitle
    ax.text(0.5, -0.08, f'Formula: {data["formula"]}',
           transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='interpretability/zone_env/reports/transition_function/figures')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.env_id} / {args.exp} / seed={args.seed}")
    dummy_env = make_env(args.env_id, FixedSampler.partial("F blue"), sequence=False)
    cfg = model_configs[args.env_id]
    store = ModelStore(args.env_id, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    model = build_model(dummy_env, status, cfg).eval()
    dummy_env.close()

    print("\nGenerating trajectory plots...")

    # 1. Optimality example
    print("\n1. Optimality test (F (blue & F green))...")
    for seed_offset in range(20):
        data = run_and_collect_trajectory(
            args.env_id, model, "F (blue & F green)",
            args.seed + seed_offset * 100, max_steps=200
        )
        if data and len(data['zone_visits']) >= 2:
            plot_trajectory(
                data,
                "Optimality Test: Visit Blue then Green",
                out_dir / "optimality_example.png",
                highlight_colors=['blue', 'green']
            )
            break

    # 2. Safety example
    print("\n2. Safety test ((F green | F yellow) & G !blue)...")
    for seed_offset in range(20):
        data = run_and_collect_trajectory(
            args.env_id, model, "(F green | F yellow) & G !blue",
            args.seed + 1000 + seed_offset * 100, max_steps=200
        )
        if data and len(data['zone_visits']) >= 1:
            plot_trajectory(
                data,
                "Safety Test: Reach Green or Yellow, Avoid Blue",
                out_dir / "safety_example.png",
                highlight_colors=['green', 'yellow'],
                avoid_colors=['blue']
            )
            break

    # 3. Infinite horizon example
    print("\n3. Infinite horizon test (G F blue & G F green)...")
    for seed_offset in range(20):
        data = run_and_collect_trajectory(
            args.env_id, model, "G F blue & G F green",
            args.seed + 2000 + seed_offset * 100, max_steps=300
        )
        if data and len(data['zone_visits']) >= 4:
            plot_trajectory(
                data,
                "Infinite Horizon: Keep Visiting Blue and Green",
                out_dir / "infinite_horizon_example.png",
                highlight_colors=['blue', 'green']
            )
            break

    # 4. Simple reach example (baseline)
    print("\n4. Simple reach (F blue)...")
    for seed_offset in range(20):
        data = run_and_collect_trajectory(
            args.env_id, model, "F blue",
            args.seed + 3000 + seed_offset * 100, max_steps=150
        )
        if data and len(data['zone_visits']) >= 1:
            plot_trajectory(
                data,
                "Simple Reach: Go to Blue",
                out_dir / "simple_reach_example.png",
                highlight_colors=['blue']
            )
            break

    # 5. Reach-avoid example
    print("\n5. Reach-avoid (!yellow U blue)...")
    for seed_offset in range(20):
        data = run_and_collect_trajectory(
            args.env_id, model, "!yellow U blue",
            args.seed + 4000 + seed_offset * 100, max_steps=200
        )
        if data:
            plot_trajectory(
                data,
                "Reach-Avoid: Reach Blue While Avoiding Yellow",
                out_dir / "reach_avoid_example.png",
                highlight_colors=['blue'],
                avoid_colors=['yellow']
            )
            break

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == '__main__':
    main()
