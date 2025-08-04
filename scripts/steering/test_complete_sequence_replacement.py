#!/usr/bin/env python3
import sys
sys.path.append('src')
sys.path.append('scripts/visualization')
import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from visualize.zones import draw_trajectories

from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from ltl import FixedSampler
from envs import make_env
from envs.flatworld import FlatWorld
from sequence.search import ExhaustiveSearch
from model.agent import Agent

ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 1
MAX_STEPS = 1000
WORLD_INDICES = list(range(20))  # Use 20 different world_info files for 20 rollouts

def test_complete_sequence_replacement():
    """Test complete sequence replacement with yellow and magenta"""
    print("=== Complete Sequence Replacement Test ===\n")
    
    # Set up
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Load model
    build_env = make_env(ENV, FixedSampler.partial("GF green & GF blue"), sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, 0)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()
    print("Model loaded successfully")
    
    # Test original sequence vs replaced sequence
    results = {}
    
    # Test 1: Original sequence (green & blue)
    print("--- Test 1: Original Sequence (Green & Blue) ---")
    physical_yellow_count, total_goals, goal_sequences, trajectories, zone_positions = run_goal_pursuit_with_original_sequence(
        model, "GF green & GF blue", num_rollouts=20, world_indices=WORLD_INDICES
    )
    
    yellow_pursuit_rate = physical_yellow_count / 20
    results['original'] = {
        'physical_yellow_count': physical_yellow_count,
        'yellow_pursuit_rate': yellow_pursuit_rate,
        'total_goals': total_goals,
        'goal_sequences': goal_sequences
    }
    
    print(f"Physical yellow achievement rate: {yellow_pursuit_rate:.1%}")
    print(f"Physical yellow achievements: {physical_yellow_count}/20")
    print(f"Rollouts with yellow pursuit: {physical_yellow_count}/20 ({yellow_pursuit_rate:.1%})")
    print()
    
    # Visualize subset of 4 zone maps
    cols = 2
    rows = 2
    fig = draw_trajectories(zone_positions[:4], trajectories[:4], cols, rows)
    plt.suptitle("Original Sequence (Green & Blue)", fontsize=16, y=0.95)
    plt.savefig("zone_trajectory_original_sequence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [INFO] Zone trajectory saved as zone_trajectory_original_sequence.png")
    
    # Test 2: Replaced sequence (yellow & magenta)
    print("--- Test 2: Replaced Sequence (Yellow & Magenta) ---")
    physical_yellow_count, total_goals, goal_sequences, trajectories, zone_positions = run_goal_pursuit_with_replaced_sequence(
        model, num_rollouts=20, world_indices=WORLD_INDICES
    )
    
    yellow_pursuit_rate = physical_yellow_count / 20
    results['replaced'] = {
        'physical_yellow_count': physical_yellow_count,
        'yellow_pursuit_rate': yellow_pursuit_rate,
        'total_goals': total_goals,
        'goal_sequences': goal_sequences
    }
    
    print(f"Physical yellow achievement rate: {yellow_pursuit_rate:.1%}")
    print(f"Physical yellow achievements: {physical_yellow_count}/20")
    print(f"Rollouts with yellow pursuit: {physical_yellow_count}/20 ({yellow_pursuit_rate:.1%})")
    print()
    
    # Visualize subset of 4 zone maps
    cols = 2
    rows = 2
    fig = draw_trajectories(zone_positions[:4], trajectories[:4], cols, rows)
    plt.suptitle("Replaced Sequence (Yellow & Magenta)", fontsize=16, y=0.95)
    plt.savefig("zone_trajectory_replaced_sequence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [INFO] Zone trajectory saved as zone_trajectory_replaced_sequence.png")
    
    # Print summary
    print("=== Complete Sequence Replacement Summary ===")
    print(f"Original sequence (green & blue): {results['original']['physical_yellow_count']}/20 ({results['original']['yellow_pursuit_rate']:.1%}) yellow pursuit rate")
    print(f"Replaced sequence (yellow & magenta): {results['replaced']['physical_yellow_count']}/20 ({results['replaced']['yellow_pursuit_rate']:.1%}) yellow pursuit rate")
    
    return results

def run_goal_pursuit_with_original_sequence(model, formula, num_rollouts=5, world_indices=None):
    """Run goal pursuit test with original sequence"""
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    search = ExhaustiveSearch(model, set(props), num_loops=2)
    agent = Agent(model, search=search, propositions=set(props), verbose=False)
    
    physical_yellow_count = 0
    total_goals = 0
    goal_sequences = []
    trajectories = []
    zone_positions = []
    
    for rollout in range(num_rollouts):
        # Use specified world index for each rollout
        world_idx = world_indices[rollout] if world_indices is not None else rollout
        world_info_path = f"eval_datasets/PointLtl2-v0/worlds/world_info_{world_idx}.pkl"
        if hasattr(env, 'load_world_info') and os.path.exists(world_info_path):
            env.load_world_info(world_info_path)
        ret = env.reset(seed=SEED + rollout)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        agent.reset()
        
        visited_goals = set()
        achieved_goals = []
        last_goal = None
        trajectory = []
        
        # Save zone positions for this rollout
        if hasattr(env, 'zone_positions'):
            zone_pos = env.zone_positions.copy()
            zone_positions.append(zone_pos)
        else:
            zone_pos = {}
            zone_positions.append(zone_pos)
        
        for step in range(MAX_STEPS):
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
            # Store agent position for trajectory
            if hasattr(env, 'agent_pos'):
                trajectory.append(env.agent_pos[:2].copy())
            elif hasattr(env.unwrapped, 'agent_pos'):
                trajectory.append(env.unwrapped.agent_pos[:2].copy())
            
            # Track current goal and achievements
            seq = getattr(agent, "sequence", None)
            if seq and len(seq) > 0:
                goal_set = seq[0][0]
                if len(goal_set) == 1:
                    try:
                        assignment = next(iter(goal_set))
                        true_props = {p for p, v in assignment.assignment if v}
                        if len(true_props) == 1:
                            prop = next(iter(true_props))
                            visited_goals.add(prop)
                            
                            # Track goal achievements (when goal changes)
                            if prop != last_goal:
                                achieved_goals.append(prop)
                                last_goal = prop
                                
                                # Debug: Print current sequence after each goal achievement
                                print(f"  [DEBUG] After achieving {prop}, current sequence: {seq}")
                                print(f"  [DEBUG] Sequence length: {len(seq)}")
                                if len(seq) > 0:
                                    print(f"  [DEBUG] Next subgoal: {seq[0]}")
                    except (StopIteration, AttributeError):
                        pass
            
            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret
            
            if done:
                break
        
        total_goals += len(visited_goals)
        goal_sequences.append(achieved_goals)
        trajectories.append(trajectory)

        # --- Physical yellow zone achievement check ---
        yellow_reached = False
        yellow_center = None
        yellow_radius = 0.3
        for k in zone_pos:
            if k.startswith('yellow'):
                yellow_center = np.array(zone_pos[k])
                break
        if yellow_center is not None:
            for t, pos in enumerate(trajectory):
                if np.linalg.norm(np.array(pos) - yellow_center) <= yellow_radius:
                    print(f"  [Rollout {rollout}] PHYSICALLY REACHED YELLOW at step {t}, pos {pos}, center {yellow_center}, radius {yellow_radius}")
                    yellow_reached = True
                    physical_yellow_count += 1
                    break
        if not yellow_reached:
            print(f"  [Rollout {rollout}] DID NOT PHYSICALLY REACH YELLOW zone.")
    
    env.close()
    
    return physical_yellow_count, total_goals, goal_sequences, trajectories, zone_positions

def run_goal_pursuit_with_replaced_sequence(model, num_rollouts=5, world_indices=None):
    """Run goal pursuit test with completely replaced sequence (yellow & magenta)"""
    sampler_fn = FixedSampler.partial("GF green & GF blue")  # Original formula, but we'll replace the sequence
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    search = ExhaustiveSearch(model, set(props), num_loops=2)
    agent = Agent(model, search=search, propositions=set(props), verbose=False)
    
    physical_yellow_count = 0
    total_goals = 0
    goal_sequences = []
    trajectories = []
    zone_positions = []
    
    for rollout in range(num_rollouts):
        # Use specified world index for each rollout
        world_idx = world_indices[rollout] if world_indices is not None else rollout
        world_info_path = f"eval_datasets/PointLtl2-v0/worlds/world_info_{world_idx}.pkl"
        if hasattr(env, 'load_world_info') and os.path.exists(world_info_path):
            env.load_world_info(world_info_path)
        ret = env.reset(seed=SEED + rollout)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        agent.reset()
        
        visited_goals = set()
        achieved_goals = []
        last_goal = None
        manipulation_applied = False
        trajectory = []
        
        # Save zone positions for this rollout
        if hasattr(env, 'zone_positions'):
            zone_pos = env.zone_positions.copy()
            zone_positions.append(zone_pos)
        else:
            zone_pos = {}
            zone_positions.append(zone_pos)
        
        for step in range(MAX_STEPS):
            # Apply sequence replacement during the rollout (like the working script)
            if step == 5 and not manipulation_applied:
                print(f"  [Rollout {rollout}] Replacing sequence with yellow and magenta")
                replace_sequence_with_yellow_magenta(agent)
                manipulation_applied = True
            
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
            # Store agent position for trajectory
            if hasattr(env, 'agent_pos'):
                trajectory.append(env.agent_pos[:2].copy())
            elif hasattr(env.unwrapped, 'agent_pos'):
                trajectory.append(env.unwrapped.agent_pos[:2].copy())
            
            # Track current goal and achievements
            seq = getattr(agent, "sequence", None)
            if seq and len(seq) > 0:
                goal_set = seq[0][0]
                if len(goal_set) == 1:
                    try:
                        assignment = next(iter(goal_set))
                        true_props = {p for p, v in assignment.assignment if v}
                        if len(true_props) == 1:
                            prop = next(iter(true_props))
                            visited_goals.add(prop)
                            
                            # Track goal achievements (when goal changes)
                            if prop != last_goal:
                                achieved_goals.append(prop)
                                last_goal = prop
                                
                                # Debug: Print current sequence after each goal achievement
                                print(f"  [DEBUG] After achieving {prop}, current sequence: {seq}")
                                print(f"  [DEBUG] Sequence length: {len(seq)}")
                                if len(seq) > 0:
                                    print(f"  [DEBUG] Next subgoal: {seq[0]}")
                    except (StopIteration, AttributeError):
                        pass
            
            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret
            
            if done:
                break
        
        total_goals += len(visited_goals)
        goal_sequences.append(achieved_goals)
        trajectories.append(trajectory)

        # --- Physical yellow zone achievement check ---
        yellow_reached = False
        yellow_center = None
        yellow_radius = 0.3
        for k in zone_pos:
            if k.startswith('yellow'):
                yellow_center = np.array(zone_pos[k])
                break
        if yellow_center is not None:
            for t, pos in enumerate(trajectory):
                if np.linalg.norm(np.array(pos) - yellow_center) <= yellow_radius:
                    print(f"  [Rollout {rollout}] PHYSICALLY REACHED YELLOW at step {t}, pos {pos}, center {yellow_center}, radius {yellow_radius}")
                    yellow_reached = True
                    physical_yellow_count += 1
                    break
        if not yellow_reached:
            print(f"  [Rollout {rollout}] DID NOT PHYSICALLY REACH YELLOW zone.")
    
    env.close()
    
    return physical_yellow_count, total_goals, goal_sequences, trajectories, zone_positions

def replace_sequence_with_yellow_magenta(agent):
    """Replace the agent's sequence with yellow and magenta instead of green and blue"""
    if not hasattr(agent, "sequence") or not agent.sequence:
        print(f"  [WARN] No sequence found to replace")
        return
    
    # Get the current sequence structure
    current_seq = agent.sequence
    print(f"  [DEBUG] Original sequence: {current_seq}")
    
    # Create new assignments for yellow and magenta
    from ltl.logic import Assignment
    from ltl.automata import LDBASequence
    
    props = set(['blue', 'green', 'yellow', 'magenta'])
    
    # Create yellow assignment
    yellow_assignment = Assignment.single_proposition('yellow', props)
    yellow_frozen = yellow_assignment.to_frozen()
    
    # Create magenta assignment
    magenta_assignment = Assignment.single_proposition('magenta', props)
    magenta_frozen = magenta_assignment.to_frozen()
    
    print(f"  [DEBUG] Yellow assignment: {yellow_frozen}")
    print(f"  [DEBUG] Magenta assignment: {magenta_frozen}")
    
    # Create new sequence with yellow and magenta
    new_reach_avoid_pairs = [
        ({yellow_frozen}, set()),  # First subgoal: yellow
        ({magenta_frozen}, set()),  # Second subgoal: magenta
    ]
    
    # Create a new LDBASequence
    new_sequence = LDBASequence(new_reach_avoid_pairs)
    agent.sequence = new_sequence
    print(f"  [DEBUG] Replaced sequence: {new_sequence}")

if __name__ == "__main__":
    test_complete_sequence_replacement() 