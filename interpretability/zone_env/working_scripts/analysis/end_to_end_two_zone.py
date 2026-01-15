#!/usr/bin/env python3
"""
End-to-end two-zone reach-avoid experiments for zone environment.

Analogous to letter_world two-letter e2e experiments:
- Create controlled scenarios with one "reach" zone and one "avoid" zone
- Run agent with goal "F reach" and track if avoid is touched
- Categorize outcomes: safe_success, risky_success, fail, neither
- Log features for analysis

Example:
    PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/end_to_end_two_zone.py \
        --n_scenarios 100 --max_steps 200 --out_dir interpretability/zone_env/results/two_zone_e2e
"""
import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from gymnasium import spaces as gspaces
except Exception:
    from gym import spaces as gspaces  # type: ignore

# Repo imports
SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(SRC))

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent

torch.set_grad_enabled(False)


def coerce_action(act, action_space):
    """Coerce action to match the action space format."""
    if isinstance(action_space, gspaces.Box):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        need = int(np.prod(action_space.shape))
        if a.size == 1 and need > 1:
            a = np.repeat(a, need)
        if a.size != need:
            raise ValueError(f"Action size {a.size} != {need}")
        a = np.clip(a, action_space.low, action_space.high)
        return a.reshape(action_space.shape)
    elif isinstance(action_space, gspaces.Discrete):
        a = int(np.asarray(act).ravel()[0]) if isinstance(act, (np.ndarray, list, tuple)) else int(act)
        if not (0 <= a < action_space.n):
            raise ValueError("Discrete out of range")
        return a
    elif isinstance(action_space, gspaces.MultiDiscrete):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        if a.size != action_space.nvec.size:
            raise ValueError("MultiDiscrete size mismatch")
        return a
    elif isinstance(action_space, gspaces.MultiBinary):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        need = int(np.prod(action_space.shape))
        if a.size != need:
            raise ValueError("MultiBinary size mismatch")
        return a.reshape(action_space.shape)
    return act


# Zone colors available in PointLtl2-v0
COLORS = ["blue", "green", "yellow", "magenta"]

# Color map for visualization
COLOR_MAP = {
    'blue': '#1f77b4',
    'green': '#2ca02c',
    'yellow': '#ffd700',
    'magenta': '#ff00ff',
}


def plot_trajectory(result: Dict, out_path: Path, zone_radius: float = 0.4):
    """Plot a trajectory with zone positions."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Extract trajectory
    traj = result['trajectory']
    xs = [t['pos_x'] for t in traj]
    ys = [t['pos_y'] for t in traj]

    # Plot zones
    zone_pos = result.get('zone_positions', {})
    for color, positions in zone_pos.items():
        for pos in positions:
            circle = patches.Circle(
                pos, zone_radius,
                facecolor=COLOR_MAP.get(color, 'gray'),
                edgecolor='black',
                alpha=0.4,
                linewidth=2,
            )
            ax.add_patch(circle)
            # Label zone
            label = 'REACH' if color == result['reach_color'] else ('AVOID' if color == result['avoid_color'] else '')
            if label:
                ax.annotate(label, xy=pos, ha='center', va='center', fontsize=8, fontweight='bold')

    # Plot trajectory
    ax.plot(xs, ys, 'k-', linewidth=1.5, alpha=0.7, label='Trajectory')
    ax.plot(xs[0], ys[0], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(xs[-1], ys[-1], 'rs', markersize=12, label='End', zorder=5)

    # Mark zone entries
    for i, t in enumerate(traj):
        if t['in_reach']:
            ax.plot(t['pos_x'], t['pos_y'], 'g^', markersize=8, alpha=0.8)
        if t['in_avoid']:
            ax.plot(t['pos_x'], t['pos_y'], 'rv', markersize=8, alpha=0.8)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    outcome = result['outcome']
    reach = result['reach_color']
    avoid = result['avoid_color']
    ax.set_title(f"Scenario {result['scenario_id']}: reach={reach}, avoid={avoid}\nOutcome: {outcome}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()


def create_two_zone_scenario(
    reach_color: str,
    avoid_color: str,
    reach_direction: str,
    avoid_direction: str,
    distance: float = 1.5,
    rng: random.Random = None,
) -> Dict:
    """
    Create a scenario definition with reach and avoid zones.

    Directions: 'up', 'down', 'left', 'right'
    Agent starts at origin (0, 0).
    """
    if rng is None:
        rng = random.Random()

    direction_vectors = {
        'up': (0, 1),
        'down': (0, -1),
        'left': (-1, 0),
        'right': (1, 0),
        'up_left': (-0.707, 0.707),
        'up_right': (0.707, 0.707),
        'down_left': (-0.707, -0.707),
        'down_right': (0.707, -0.707),
    }

    reach_vec = direction_vectors[reach_direction]
    avoid_vec = direction_vectors[avoid_direction]

    # Add small random jitter to avoid perfectly symmetric scenarios
    jitter = 0.2
    reach_pos = (
        reach_vec[0] * distance + rng.uniform(-jitter, jitter),
        reach_vec[1] * distance + rng.uniform(-jitter, jitter)
    )
    avoid_pos = (
        avoid_vec[0] * distance + rng.uniform(-jitter, jitter),
        avoid_vec[1] * distance + rng.uniform(-jitter, jitter)
    )

    return {
        'reach_color': reach_color,
        'avoid_color': avoid_color,
        'reach_pos': reach_pos,
        'avoid_pos': avoid_pos,
        'reach_direction': reach_direction,
        'avoid_direction': avoid_direction,
        'distance': distance,
    }


def generate_scenario_batch(
    n_scenarios: int,
    seed: int = 42,
    balanced_directions: bool = True,
) -> List[Dict]:
    """Generate a batch of two-zone scenarios."""
    rng = random.Random(seed)
    scenarios = []

    # Directions to use
    directions = ['up', 'down', 'left', 'right']

    # Color pairs (reach, avoid)
    color_pairs = [
        ('blue', 'green'),
        ('green', 'blue'),
        ('yellow', 'magenta'),
        ('magenta', 'yellow'),
        ('blue', 'yellow'),
        ('green', 'magenta'),
    ]

    if balanced_directions:
        # Generate balanced across direction pairs
        direction_pairs = []
        for r_dir in directions:
            for a_dir in directions:
                if r_dir != a_dir:  # Different directions
                    direction_pairs.append((r_dir, a_dir))

        # Repeat to get enough scenarios
        while len(scenarios) < n_scenarios:
            for r_dir, a_dir in direction_pairs:
                if len(scenarios) >= n_scenarios:
                    break
                reach_color, avoid_color = rng.choice(color_pairs)
                distance = rng.uniform(1.2, 2.0)
                scenario = create_two_zone_scenario(
                    reach_color, avoid_color, r_dir, a_dir, distance, rng
                )
                scenario['scenario_id'] = len(scenarios)
                scenarios.append(scenario)
    else:
        # Fully random
        for i in range(n_scenarios):
            reach_color, avoid_color = rng.choice(color_pairs)
            r_dir = rng.choice(directions)
            a_dir = rng.choice([d for d in directions if d != r_dir])
            distance = rng.uniform(1.2, 2.0)
            scenario = create_two_zone_scenario(
                reach_color, avoid_color, r_dir, a_dir, distance, rng
            )
            scenario['scenario_id'] = i
            scenarios.append(scenario)

    return scenarios


def run_scenario(
    scenario: Dict,
    model,
    env_id: str,
    max_steps: int = 200,
    deterministic: bool = True,
    capture_features: bool = True,
) -> Dict:
    """
    Run a single scenario and return results.

    Uses the standard PointLtl2-v0 environment with random zone placement,
    then analyzes which zones the agent reaches.
    """
    reach_color = scenario['reach_color']
    avoid_color = scenario['avoid_color']

    # Create environment with goal to reach the target color
    formula = f"F {reach_color}"
    sampler_fn = FixedSampler.partial(formula)

    env = make_env(env_id, sampler_fn, sequence=False)
    props = set(env.get_propositions())

    # Set up agent
    planner = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, planner, propositions=props)

    # Hook to capture hidden states
    hidden_states = []
    values = []
    action_probs = []

    def rnn_hook(mod, inp, out):
        try:
            h = out[1]
            if isinstance(h, (tuple, list)):
                h = h[-1]
            if isinstance(h, torch.Tensor):
                hidden_states.append(h.detach().cpu().numpy().squeeze().copy())
        except:
            pass

    if capture_features:
        handle = model.ltl_net.rnn.register_forward_hook(rnn_hook)

    # Reset environment with scenario-specific seed
    seed = scenario.get('scenario_id', 0) + 1000
    reset_out = env.reset(seed=seed)
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    agent.reset()

    # Try to extract zone positions from the environment
    zone_positions = {}
    try:
        task = env.unwrapped.task
        # Access the geoms through _geoms dict
        for geom_name, geom in task._geoms.items():
            if hasattr(geom, 'color_name') and hasattr(geom, 'num'):
                color = geom.color_name
                if color not in zone_positions:
                    zone_positions[color] = []
                # Get positions - body names are like 'blue_zone0', 'blue_zone1' etc
                for i in range(geom.num):
                    try:
                        body_name = f'{color}_zone{i}'
                        pos = task.data.body(body_name).xpos[:2].copy()
                        zone_positions[color].append(pos.tolist())
                    except:
                        pass
    except Exception as e:
        pass  # Zone positions will be empty if extraction fails

    # Get initial agent position
    try:
        initial_agent_pos = env.unwrapped.task.agent.pos[:2].copy()
    except:
        initial_agent_pos = np.array([0.0, 0.0])

    # Track trajectory
    trajectory = []
    reached_goal = False
    reached_avoid = False
    goal_step = -1
    avoid_step = -1

    for step in range(max_steps):
        # Get action
        with torch.no_grad():
            action = agent.get_action(obs, info, deterministic=deterministic)
        action = coerce_action(action, env.action_space)

        # Capture value estimate
        if capture_features:
            try:
                # Get value from critic
                obs_tensor = {k: torch.tensor(v).unsqueeze(0).float()
                             for k, v in obs.items() if isinstance(v, np.ndarray)}
                with torch.no_grad():
                    embed = model.compute_embedding(obs_tensor)
                    if hasattr(model, 'critic'):
                        v = model.critic(embed).item()
                        values.append(v)
            except:
                pass

        # Get agent position
        try:
            agent_pos = env.unwrapped.task.agent.pos[:2].copy()
        except:
            agent_pos = np.array([np.nan, np.nan])

        # Step environment
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret

        # Check propositions
        current_props = info.get('propositions', set())
        in_reach = reach_color in current_props
        in_avoid = avoid_color in current_props

        trajectory.append({
            'step': step,
            'pos_x': float(agent_pos[0]),
            'pos_y': float(agent_pos[1]),
            'action': action.tolist() if hasattr(action, 'tolist') else list(action),
            'in_reach': in_reach,
            'in_avoid': in_avoid,
        })

        if in_reach and not reached_goal:
            reached_goal = True
            goal_step = step

        if in_avoid and not reached_avoid:
            reached_avoid = True
            avoid_step = step

        if done:
            break

    if capture_features:
        handle.remove()

    env.close()

    # Categorize outcome
    if reached_goal and not reached_avoid:
        outcome = 'safe_success'
    elif reached_goal and reached_avoid:
        if goal_step <= avoid_step:
            outcome = 'safe_success'  # Reached goal first
        else:
            outcome = 'risky_success'  # Touched avoid before goal
    elif reached_avoid and not reached_goal:
        outcome = 'fail'
    else:
        outcome = 'neither'

    return {
        'scenario_id': scenario['scenario_id'],
        'reach_color': reach_color,
        'avoid_color': avoid_color,
        'reach_direction': scenario['reach_direction'],
        'avoid_direction': scenario['avoid_direction'],
        'outcome': outcome,
        'reached_goal': reached_goal,
        'reached_avoid': reached_avoid,
        'goal_step': goal_step,
        'avoid_step': avoid_step,
        'n_steps': len(trajectory),
        'trajectory': trajectory,
        'hidden_states': hidden_states if capture_features else [],
        'values': values if capture_features else [],
        'zone_positions': zone_positions,
        'initial_agent_pos': initial_agent_pos.tolist() if hasattr(initial_agent_pos, 'tolist') else list(initial_agent_pos),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', default='PointLtl2-v0')
    ap.add_argument('--exp', default='big_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_scenarios', type=int, default=100)
    ap.add_argument('--max_steps', type=int, default=200)
    ap.add_argument('--deterministic', action='store_true', default=True)
    ap.add_argument('--out_dir', default='interpretability/zone_env/results/two_zone_e2e')
    ap.add_argument('--capture_features', action='store_true', default=True)
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

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())}")

    # Generate scenarios
    print(f"\nGenerating {args.n_scenarios} scenarios...")
    scenarios = generate_scenario_batch(args.n_scenarios, seed=args.seed)

    # Save scenarios
    with open(out_dir / 'scenarios.json', 'w') as f:
        json.dump(scenarios, f, indent=2)

    # Run scenarios
    print(f"\nRunning {len(scenarios)} scenarios...")
    results = []
    for i, scenario in enumerate(scenarios):
        result = run_scenario(
            scenario, model, args.env_id,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            capture_features=args.capture_features,
        )
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(scenarios)} completed")

    # Save full results
    with open(out_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Create summary CSV
    summary_rows = []
    for r in results:
        summary_rows.append({
            'scenario_id': r['scenario_id'],
            'reach_color': r['reach_color'],
            'avoid_color': r['avoid_color'],
            'reach_direction': r['reach_direction'],
            'avoid_direction': r['avoid_direction'],
            'outcome': r['outcome'],
            'reached_goal': r['reached_goal'],
            'reached_avoid': r['reached_avoid'],
            'goal_step': r['goal_step'],
            'avoid_step': r['avoid_step'],
            'n_steps': r['n_steps'],
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / 'summary.csv', index=False)

    # Print summary statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    outcome_counts = df['outcome'].value_counts()
    total = len(df)

    print(f"\nOutcome distribution (n={total}):")
    for outcome in ['safe_success', 'risky_success', 'fail', 'neither']:
        count = outcome_counts.get(outcome, 0)
        pct = 100 * count / total
        print(f"  {outcome:15s}: {count:3d} ({pct:5.1f}%)")

    # Success rate by reach direction
    print("\nSafe success rate by reach direction:")
    for direction in df['reach_direction'].unique():
        subset = df[df['reach_direction'] == direction]
        safe = (subset['outcome'] == 'safe_success').sum()
        rate = 100 * safe / len(subset) if len(subset) > 0 else 0
        print(f"  {direction:10s}: {rate:5.1f}% ({safe}/{len(subset)})")

    # Success rate by avoid direction
    print("\nSafe success rate by avoid direction:")
    for direction in df['avoid_direction'].unique():
        subset = df[df['avoid_direction'] == direction]
        safe = (subset['outcome'] == 'safe_success').sum()
        rate = 100 * safe / len(subset) if len(subset) > 0 else 0
        print(f"  {direction:10s}: {rate:5.1f}% ({safe}/{len(subset)})")

    # Direction pair analysis
    print("\nSafe success rate by (reach, avoid) direction pair:")
    for r_dir in df['reach_direction'].unique():
        for a_dir in df['avoid_direction'].unique():
            if r_dir != a_dir:
                subset = df[(df['reach_direction'] == r_dir) & (df['avoid_direction'] == a_dir)]
                if len(subset) > 0:
                    safe = (subset['outcome'] == 'safe_success').sum()
                    rate = 100 * safe / len(subset)
                    print(f"  reach={r_dir:6s} avoid={a_dir:6s}: {rate:5.1f}% ({safe}/{len(subset)})")

    # Analyze spatial relationships
    print("\n" + "="*60)
    print("SPATIAL ANALYSIS")
    print("="*60)

    # Compute distances and analyze blocking
    spatial_data = []
    for r in results:
        agent_pos = np.array(r.get('initial_agent_pos', [0, 0]))
        zone_pos = r.get('zone_positions', {})
        reach_color = r['reach_color']
        avoid_color = r['avoid_color']

        # Find nearest zone of each color
        reach_zones = zone_pos.get(reach_color, [])
        avoid_zones = zone_pos.get(avoid_color, [])

        if reach_zones and avoid_zones:
            # Distance to nearest reach zone
            reach_dists = [np.linalg.norm(np.array(z) - agent_pos) for z in reach_zones]
            avoid_dists = [np.linalg.norm(np.array(z) - agent_pos) for z in avoid_zones]
            min_reach_dist = min(reach_dists) if reach_dists else float('inf')
            min_avoid_dist = min(avoid_dists) if avoid_dists else float('inf')

            # Check if avoid is closer than reach (blocking scenario)
            avoid_blocks = min_avoid_dist < min_reach_dist

            spatial_data.append({
                'scenario_id': r['scenario_id'],
                'outcome': r['outcome'],
                'min_reach_dist': min_reach_dist,
                'min_avoid_dist': min_avoid_dist,
                'avoid_blocks': avoid_blocks,
            })

    if spatial_data:
        spatial_df = pd.DataFrame(spatial_data)

        # Success rate when avoid blocks vs doesn't block
        blocking = spatial_df[spatial_df['avoid_blocks'] == True]
        non_blocking = spatial_df[spatial_df['avoid_blocks'] == False]

        if len(blocking) > 0:
            blocking_safe = (blocking['outcome'] == 'safe_success').sum()
            print(f"\nWhen avoid zone is CLOSER than reach zone (n={len(blocking)}):")
            print(f"  Safe success: {100*blocking_safe/len(blocking):.1f}%")

        if len(non_blocking) > 0:
            non_blocking_safe = (non_blocking['outcome'] == 'safe_success').sum()
            print(f"\nWhen reach zone is CLOSER than avoid zone (n={len(non_blocking)}):")
            print(f"  Safe success: {100*non_blocking_safe/len(non_blocking):.1f}%")

        spatial_df.to_csv(out_dir / 'spatial_analysis.csv', index=False)

    # Save example trajectory plots
    print("\n" + "="*60)
    print("SAVING EXAMPLE PLOTS")
    print("="*60)

    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Save a few examples from each outcome category
    for outcome in ['safe_success', 'risky_success', 'fail', 'neither']:
        outcome_results = [r for r in results if r['outcome'] == outcome]
        for i, r in enumerate(outcome_results[:3]):  # Save up to 3 per category
            plot_path = plots_dir / f"{outcome}_{r['scenario_id']}.png"
            try:
                plot_trajectory(r, plot_path)
                print(f"  Saved: {plot_path.name}")
            except Exception as e:
                print(f"  Failed to plot {r['scenario_id']}: {e}")

    print(f"\nResults saved to: {out_dir}")
    print("  - scenarios.json: scenario definitions")
    print("  - results.pkl: full results with trajectories and features")
    print("  - summary.csv: summary statistics")
    print("  - spatial_analysis.csv: spatial relationship analysis")
    print("  - plots/: trajectory visualizations")


if __name__ == '__main__':
    main()
