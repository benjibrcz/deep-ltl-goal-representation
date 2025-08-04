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
from visualize.zones import draw_trajectories, draw_zones, draw_path, draw_diamond, setup_axis

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0
FORMULA = "GF blue & GF green"
MAX_STEPS = 1000


class SteeredExhaustiveSearch(ExhaustiveSearch):
    """ExhaustiveSearch that biases value predictions toward certain goals."""
    
    def __init__(self, model, propositions, num_loops, value_threshold=0.4, 
                 steering_goal=None, steering_strength=1.0):
        super().__init__(model, propositions, num_loops, value_threshold)
        self.steering_goal = steering_goal  # 'blue' or 'green'
        self.steering_strength = steering_strength
    
    def get_value(self, seq, obs):
        """Override get_value to bias toward steering_goal."""
        original_value = super().get_value(seq, obs)
        
        if self.steering_goal is None:
            return original_value
        
        # Check if the first goal in the sequence matches our steering goal
        if len(seq) > 0:
            first_goal_set, _ = seq[0]
            if isinstance(first_goal_set, (set, frozenset)):
                for assignment in first_goal_set:
                    if hasattr(assignment, 'assignment'):
                        goal_props = {prop for prop, val in assignment.assignment if val}
                        if self.steering_goal in goal_props:
                            # Boost the value for sequences that start with our target goal
                            return original_value + self.steering_strength
        
        return original_value


def run_and_plot_steered_rollout(steering_goal, steering_strength, rollout_idx):
    random.seed(SEED + rollout_idx)
    np.random.seed(SEED + rollout_idx)
    torch.manual_seed(SEED + rollout_idx)
    print(f"\n=== Sequence Generation Steering: {steering_goal.capitalize()} (Strength: {steering_strength}, Rollout {rollout_idx+1}) ===")
    
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
    
    # Use our steered search instead of regular ExhaustiveSearch
    search = SteeredExhaustiveSearch(
        model, props, num_loops=2, 
        steering_goal=steering_goal, 
        steering_strength=steering_strength
    )
    agent = Agent(model, search=search, propositions=props, verbose=False)
    
    ret = env.reset(seed=SEED + rollout_idx)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    goals = []
    trajectory = []
    goal_satisfaction_points = []
    sequence_changes = []
    
    for step in range(MAX_STEPS):
        # Get current goal before action
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
        
        # Get action (this may trigger sequence regeneration)
        old_sequence = agent.sequence
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        # Check if sequence changed
        if agent.sequence != old_sequence:
            sequence_changes.append(step)
        
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
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    # Setup axis
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
        color = 'blue' if i % 2 == 0 else 'green'
        ax.scatter(point[0], point[1], c=color, s=100, marker='*', zorder=15, 
                  label=f'Goal {i+1}' if i < 2 else None)
    
    # Mark sequence change points
    for step in sequence_changes:
        if step < len(trajectory):
            ax.scatter(trajectory[step][0], trajectory[step][1], c='red', s=50, 
                      marker='s', zorder=10, alpha=0.7)
    
    plt.title(f'Sequence Generation Steering: {steering_goal.capitalize()} (Strength: {steering_strength}, Rollout {rollout_idx+1})')
    plt.legend()
    plt.tight_layout()
    
    fname = f'sequence_generation_steering_{steering_goal}_{steering_strength}_{rollout_idx+1}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {fname}")
    plt.close()
    
    # Print statistics
    goal_counts = {}
    for goal in goals:
        goal_counts[goal] = goal_counts.get(goal, 0) + 1
    
    print(f"Goal distribution: {goal_counts}")
    print(f"Sequence changes: {len(sequence_changes)} at steps {sequence_changes}")
    
    return goal_counts


def main():
    steering_strengths = [5.0, 10.0, 50.0, 100.0]  # Much stronger steering
    
    for steering_goal in ['blue', 'green']:
        for strength in steering_strengths:
            for rollout_idx in range(2):
                goal_counts = run_and_plot_steered_rollout(steering_goal, strength, rollout_idx)
    
    print("\n=== Sequence Generation Steering Complete ===")
    print("This approach steers the model's value predictions to prefer sequences")
    print("that start with the target goal, rather than editing the sequence directly.")
    print("Testing much stronger steering strengths to find threshold effects.")

if __name__ == '__main__':
    main() 