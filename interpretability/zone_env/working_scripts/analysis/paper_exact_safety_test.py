#!/usr/bin/env python3
"""
Exact Paper Safety Test: Replicating Figure 1c from DeepLTL paper

Uses the fixed environment configuration that matches the paper exactly:
- Green zone at (1.2, -1.9) - goal option 1
- Yellow zone at (1.1, 2.1) - goal option 2
- Blue zones (3) at (2, -1), (0.6, -1.05), (0.1, -2.3) - blocking path to green
- Agent starts at (-1.2, -0.6)

Task: (F green | F yellow) & G !blue
Expected: Agent should choose yellow (farther but unblocked) over green (closer but blocked)

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/paper_exact_safety_test.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
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

# Fixed zone positions from the paper (from ltl_fixed.py)
FIXED_CONFIG = {
    'agent_start': (-1.2, -0.6),
    'green': (1.2, -1.9),
    'yellow': (1.1, 2.1),
    'blue': [(2, -1), (0.6, -1.05), (0.1, -2.3)],
    'magenta': (1.8, 0.4),
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


def run_fixed_safety_test(env_id: str, model, seed: int, max_steps: int = 300):
    """Run the exact paper safety test with fixed zone positions."""

    formula = "(F green | F yellow) & G !blue"
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(env_id, sampler_fn, sequence=False)

    try:
        props = set(env.get_propositions())
        planner = ExhaustiveSearch(model, props, num_loops=2)
        agent = Agent(model, planner, propositions=props)

        reset_out = env.reset(seed=seed)
        obs, info = (reset_out, {}) if not isinstance(reset_out, tuple) else reset_out
        agent.reset()

        # Get actual positions from environment
        agent_start = None
        zone_positions = {}
        try:
            task = env.unwrapped.task
            agent_start = task.agent.pos[:2].copy()
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
        except Exception as e:
            print(f"Error extracting zone info: {e}")
            return None

        if agent_start is None:
            return None

        # Calculate distances
        green_pos = zone_positions.get('green', [None])[0]
        yellow_pos = zone_positions.get('yellow', [None])[0]
        blue_positions = zone_positions.get('blue', [])

        if green_pos is None or yellow_pos is None:
            return None

        dist_to_green = np.linalg.norm(agent_start - green_pos)
        dist_to_yellow = np.linalg.norm(agent_start - yellow_pos)

        print(f"\n  Agent start: ({agent_start[0]:.2f}, {agent_start[1]:.2f})")
        print(f"  Green zone:  ({green_pos[0]:.2f}, {green_pos[1]:.2f}) - distance: {dist_to_green:.2f}")
        print(f"  Yellow zone: ({yellow_pos[0]:.2f}, {yellow_pos[1]:.2f}) - distance: {dist_to_yellow:.2f}")
        print(f"  Blue zones:  {len(blue_positions)} blocking zones")
        print(f"  Closer goal: {'green' if dist_to_green < dist_to_yellow else 'yellow'}")

        # Run episode
        trajectory = [agent_start.copy()]
        reached_green = False
        reached_yellow = False
        touched_blue = False
        first_goal = None

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

                # Check zone contacts
                if np.linalg.norm(agent_pos - green_pos) < ZONE_RADIUS and not reached_green:
                    reached_green = True
                    if first_goal is None:
                        first_goal = 'green'

                if np.linalg.norm(agent_pos - yellow_pos) < ZONE_RADIUS and not reached_yellow:
                    reached_yellow = True
                    if first_goal is None:
                        first_goal = 'yellow'

                for blue in blue_positions:
                    if np.linalg.norm(agent_pos - blue) < ZONE_RADIUS:
                        touched_blue = True
            except:
                pass

            if done:
                break

        return {
            'seed': seed,
            'agent_start': agent_start.tolist(),
            'green_pos': green_pos.tolist(),
            'yellow_pos': yellow_pos.tolist(),
            'blue_positions': [b.tolist() for b in blue_positions],
            'dist_to_green': dist_to_green,
            'dist_to_yellow': dist_to_yellow,
            'trajectory': [t.tolist() for t in trajectory],
            'reached_green': reached_green,
            'reached_yellow': reached_yellow,
            'touched_blue': touched_blue,
            'first_goal': first_goal,
            'steps': len(trajectory),
        }

    finally:
        env.close()


def plot_result(result, filename):
    """Plot the trajectory on the fixed environment."""
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = {'blue': '#3498db', 'green': '#2ecc71', 'yellow': '#f1c40f', 'magenta': '#e91e63'}

    # Plot blue zones (avoid)
    for blue in result['blue_positions']:
        circle = patches.Circle(blue, ZONE_RADIUS,
                               facecolor=colors['blue'], alpha=0.5,
                               edgecolor='red', linewidth=3)
        ax.add_patch(circle)
        ax.text(blue[0], blue[1], 'B', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')

    # Plot green zone
    green = result['green_pos']
    circle = patches.Circle(green, ZONE_RADIUS,
                           facecolor=colors['green'], alpha=0.6,
                           edgecolor='darkgreen', linewidth=3)
    ax.add_patch(circle)
    ax.text(green[0], green[1], 'G\n(blocked)', ha='center', va='center',
           fontsize=10, fontweight='bold')

    # Plot yellow zone
    yellow = result['yellow_pos']
    circle = patches.Circle(yellow, ZONE_RADIUS,
                           facecolor=colors['yellow'], alpha=0.6,
                           edgecolor='orange', linewidth=3)
    ax.add_patch(circle)
    ax.text(yellow[0], yellow[1], 'Y\n(safe)', ha='center', va='center',
           fontsize=10, fontweight='bold')

    # Plot trajectory
    traj = np.array(result['trajectory'])
    n_points = len(traj)
    for i in range(n_points - 1):
        progress = i / max(n_points - 1, 1)
        color = plt.cm.plasma(progress)
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], color=color, linewidth=2.5, alpha=0.9)

    # Mark start
    ax.scatter(traj[0, 0], traj[0, 1], s=200, c='lime', marker='o',
              zorder=10, edgecolors='black', linewidths=2, label='Start')

    # Mark end
    ax.scatter(traj[-1, 0], traj[-1, 1], s=200, c='red', marker='s',
              zorder=10, edgecolors='black', linewidths=2, label='End')

    # Draw path lines from agent to goals
    ax.plot([traj[0, 0], green[0]], [traj[0, 1], green[1]],
           'g--', alpha=0.3, linewidth=2, label=f'To green: {result["dist_to_green"]:.1f}')
    ax.plot([traj[0, 0], yellow[0]], [traj[0, 1], yellow[1]],
           color='gold', linestyle='--', alpha=0.3, linewidth=2,
           label=f'To yellow: {result["dist_to_yellow"]:.1f}')

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    # Title with result
    title = f"Paper Safety Test: (F green | F yellow) & G !blue\n"
    if result['first_goal'] == 'yellow':
        title += "CHOSE YELLOW (safe, farther) - PLANNING EVIDENCE"
    elif result['first_goal'] == 'green':
        title += "CHOSE GREEN (blocked, closer) - NO PLANNING"
    else:
        title += f"Neither reached (touched blue: {result['touched_blue']})"

    if result['touched_blue']:
        title += "\n[SAFETY VIOLATION: touched blue]"

    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0.fixed')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--model_seed', type=int, default=0, help='Model checkpoint seed')
    ap.add_argument('--env_seed', type=int, default=0, help='Environment seed for runs')
    ap.add_argument('--n_runs', type=int, default=10)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/paper_exact_safety')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model (trained on PointLtl2-v0)
    print(f"Loading model: PointLtl2-v0 / {args.exp} / seed={args.model_seed}")
    dummy_env = make_env('PointLtl2-v0', FixedSampler.partial("F blue"), sequence=False)
    cfg = model_configs['PointLtl2-v0']
    store = ModelStore('PointLtl2-v0', args.exp, args.model_seed)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    model = build_model(dummy_env, status, cfg).eval()
    dummy_env.close()

    print(f"\nRunning {args.n_runs} episodes on fixed paper safety configuration...")
    print(f"Environment: {args.env_id}")
    print("="*60)

    results = []
    for run in range(args.n_runs):
        print(f"\n--- Run {run+1}/{args.n_runs} ---")
        result = run_fixed_safety_test(args.env_id, model, args.env_seed + run, args.max_steps)

        if result:
            results.append(result)
            print(f"\n  First goal reached: {result['first_goal']}")
            print(f"  Touched blue: {result['touched_blue']}")
            print(f"  Steps: {result['steps']}")

            # Save plot for first few runs
            if run < 3:
                plot_result(result, out_dir / f'run_{run+1}.png')

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    n_total = len(results)
    n_yellow = sum(1 for r in results if r['first_goal'] == 'yellow')
    n_green = sum(1 for r in results if r['first_goal'] == 'green')
    n_neither = sum(1 for r in results if r['first_goal'] is None)
    n_blue_touched = sum(1 for r in results if r['touched_blue'])

    print(f"\nTotal runs: {n_total}")
    print(f"Chose YELLOW (safe, farther):  {n_yellow}/{n_total} ({100*n_yellow/n_total:.1f}%)")
    print(f"Chose GREEN (blocked, closer): {n_green}/{n_total} ({100*n_green/n_total:.1f}%)")
    print(f"Neither reached:               {n_neither}/{n_total} ({100*n_neither/n_total:.1f}%)")
    print(f"Safety violations (touched blue): {n_blue_touched}/{n_total} ({100*n_blue_touched/n_total:.1f}%)")

    if n_yellow > n_green:
        print(f"\n>>> EVIDENCE OF PLANNING: Agent prefers safe path!")
    elif n_yellow < n_green:
        print(f"\n>>> NO PLANNING: Agent goes for closer goal despite blocking")
    else:
        print(f"\n>>> INCONCLUSIVE: 50/50 split")

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
