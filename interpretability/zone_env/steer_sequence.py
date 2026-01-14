#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent
from ltl.logic.assignment import Assignment, FrozenAssignment
from ltl.automata.ldba_sequence import LDBASequence
from visualize.zones import draw_trajectories, draw_zones, draw_path, draw_diamond, setup_axis
from matplotlib.axes import Axes

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0
FORMULA = "GF blue & GF green"
MAX_STEPS = 1000


def force_sequence_to_goal(agent, forced_goal):
    """Overwrite the agent's sequence so the next goal is always forced_goal."""
    seq = getattr(agent, "sequence", None)
    if seq and len(seq) > 0:
        # Overwrite the assignment in the first goal set
        goal_set, avoid_set = seq[0]
        assignment = next(iter(goal_set))
        # Create a new assignment dictionary with forced_goal set to True, others to False
        new_assignment_dict = {}
        for p, _ in assignment.assignment:
            new_assignment_dict[p] = (p == forced_goal)
        new_assignment = Assignment(new_assignment_dict).to_frozen()
        new_goal_set = frozenset([new_assignment])
        # Rebuild the sequence with the new first element
        new_seq_list = list(seq)
        new_seq_list[0] = (new_goal_set, avoid_set)
        agent.sequence = LDBASequence(new_seq_list)


def run_and_plot_rollout(forced_goal, rollout_idx):
    random.seed(SEED + rollout_idx)
    np.random.seed(SEED + rollout_idx)
    torch.manual_seed(SEED + rollout_idx)
    print(f"\n=== Sequence-Level Steering: Forcing Next Goal = {forced_goal} (Rollout {rollout_idx+1}) ===")
    
    sampler_fn = FixedSampler.partial(FORMULA)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()
    
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    
    ret = env.reset(seed=SEED + rollout_idx)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    goals = []
    trajectory = []
    goal_satisfaction_points = []
    
    for step in range(MAX_STEPS):
        force_sequence_to_goal(agent, forced_goal)
        
        # Get current goal
        seq = getattr(agent, "sequence", None)
        if seq and len(seq) > 0:
            goal_set = seq[0][0]
            assignment = next(iter(goal_set))
            true_props = {p for p, v in assignment.assignment if v}
            if len(true_props) == 1:
                prop = next(iter(true_props))
                goals.append(prop)
            else:
                goals.append('other')
        else:
            goals.append('none')
        
        # Record agent position
        trajectory.append(env.agent_pos[:2])
        
        action = agent.get_action(obs, info, deterministic=True).flatten()
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
        
        # Check if goal was satisfied
        if 'goal_satisfied' in info and info['goal_satisfied']:
            goal_satisfaction_points.append(env.agent_pos[:2])
        
        if done:
            break
    
    env.close()
    
    # Create visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    # Setup axis like in draw_zone_trajectories
    setup_axis(ax)
    
    # Draw zones
    zone_positions = env.zone_positions if hasattr(env, 'zone_positions') else {}
    draw_zones(ax, zone_positions)
    
    # Draw trajectory
    if len(trajectory) > 0:
        draw_path(ax, trajectory, color='green', linewidth=3)
        draw_diamond(ax, trajectory[0], color='orange')  # Start position
    
    # Mark goal satisfaction points
    for i, point in enumerate(goal_satisfaction_points):
        color = 'blue' if i % 2 == 0 else 'green'  # Alternate colors for different goals
        ax.scatter(point[0], point[1], c=color, s=100, marker='*', zorder=15, 
                  label=f'Goal {i+1}' if i < 2 else None)
    
    plt.title(f'Sequence Steering: {forced_goal.capitalize()} (Rollout {rollout_idx+1})')
    plt.legend()
    plt.tight_layout()
    
    fname = f'sequence_steering_{forced_goal}_{rollout_idx+1}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {fname}")
    plt.close()


def main():
    for forced_goal in ['blue', 'green']:
        for rollout_idx in range(2):
            run_and_plot_rollout(forced_goal, rollout_idx)
    print("\n=== Sequence-Level Steering Rollout Plots Complete ===")

if __name__ == '__main__':
    main() 