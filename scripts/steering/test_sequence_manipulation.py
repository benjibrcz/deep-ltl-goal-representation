#!/usr/bin/env python3
"""
Sequence Manipulation Test

This script tests if directly manipulating the agent's sequence attribute
can force it to pursue different goals (e.g., yellow instead of green).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'visualization'))
sys.path.append('src')

from visualize.zones import draw_trajectories

import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import to_rgba

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from envs.flatworld    import FlatWorld
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent
from preprocessing.vocab import VOCAB

ENV       = "PointLtl2-v0"
EXP       = "big_test"
SEED      = 1
MAX_STEPS = 1000
WORLD_INDICES = list(range(20))  # Use 20 different world_info files for 20 rollouts

def test_sequence_manipulation():
    """Test sequence manipulation at different positions"""
    print("=== Sequence Manipulation Test ===\n")
    
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
    
    # Test all four positions
    positions_to_test = [0, 1, 2, 3]  # 0-indexed positions
    results = {}
    
    for position in positions_to_test:
        print(f"--- Test: Forcing Yellow as {position + 1}st Subgoal via Sequence Manipulation ---")
        
        # Run 20 rollouts for this position
        physical_yellow_count, total_goals, goal_sequences, trajectories, zone_positions = run_goal_pursuit_with_sequence_manipulation(
            model, "GF green & GF blue", force_goal="yellow", force_position=position, num_rollouts=20, world_indices=WORLD_INDICES
        )
        
        # Calculate statistics based on physical yellow achievement
        yellow_pursuit_rate = physical_yellow_count / 20
        results[position] = {
            'physical_yellow_count': physical_yellow_count,
            'yellow_pursuit_rate': yellow_pursuit_rate,
            'total_goals': total_goals,
            'goal_sequences': goal_sequences
        }
        
        print(f"Physical yellow achievement rate: {yellow_pursuit_rate:.1%}")
        print(f"Physical yellow achievements: {physical_yellow_count}/20")
        print(f"Rollouts with yellow pursuit: {physical_yellow_count}/20 ({yellow_pursuit_rate:.1%})")
        print()
        
        # Debug: Print trajectory lengths before plotting
        for i, traj in enumerate(trajectories):
            if len(traj) == 0:
                print(f"[DEBUG] Trajectory {i} is empty! zone_positions: {zone_positions[i]}, goal_sequences: {goal_sequences[i]}")
            else:
                print(f"[DEBUG] Trajectory {i} length: {len(traj)}")
        
        # Debug: Print trajectory and zone_positions lengths and first few elements before plotting
        print(f"[DEBUG] Number of trajectories: {len(trajectories)}")
        print(f"[DEBUG] Number of zone_positions: {len(zone_positions)}")
        print(f"[DEBUG] First 4 trajectory lengths: {[len(traj) for traj in trajectories[:4]]}")
        print(f"[DEBUG] First 4 zone_positions: {zone_positions[:4]}")
        print(f"[DEBUG] Trajectories slice for plotting: {trajectories[:4]}")
        print(f"[DEBUG] Zone_positions slice for plotting: {zone_positions[:4]}")
        
        # Filter out empty trajectories before plotting
        filtered = [(traj, zone) for traj, zone in zip(trajectories, zone_positions) if len(traj) > 0]
        if len(filtered) < len(trajectories):
            empty_indices = [i for i, traj in enumerate(trajectories) if len(traj) == 0]
            print(f"[WARN] Empty trajectories at rollouts: {empty_indices}")
        if len(filtered) == 0:
            print("[ERROR] All trajectories are empty, skipping plot.")
        else:
            filtered_trajectories, filtered_zone_positions = zip(*filtered)
            # Visualize subset of 4 zone maps
            cols = 2
            rows = 2
            fig = draw_trajectories(filtered_zone_positions[:4], filtered_trajectories[:4], cols, rows)
            plt.suptitle(f"Forcing Yellow as {position + 1}st Subgoal", fontsize=16, y=0.95)
            plt.savefig(f"zone_trajectory_forced_yellow_position_{position}.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [INFO] Zone trajectory saved as zone_trajectory_forced_yellow_position_{position}.png")
    
    # Print summary
    print("=== Sequence Manipulation Summary ===")
    for position in positions_to_test:
        rate = results[position]['yellow_pursuit_rate']
        count = results[position]['physical_yellow_count']
        print(f"Position {position + 1}: {count}/20 ({rate:.1%}) yellow pursuit rate")
    
    return results

def run_goal_pursuit_test(model, formula, num_rollouts=5):
    """Run goal pursuit test and collect metrics"""
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    search = ExhaustiveSearch(model, set(props), num_loops=2)
    agent = Agent(model, search=search, propositions=set(props), verbose=False)
    
    completion_count = 0
    total_goals = 0
    goal_sequences = []
    
    for rollout in range(num_rollouts):
        ret = env.reset(seed=SEED + rollout)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        agent.reset()
        
        visited_goals = set()
        achieved_goals = []
        last_goal = None
        
        for step in range(MAX_STEPS):
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
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
        
        # Check if all goals were visited (completion)
        if len(visited_goals) >= 2:  # Need at least 2 goals for completion
            completion_count += 1
        
        total_goals += len(visited_goals)
        goal_sequences.append(achieved_goals)
    
    env.close()
    
    return {
        'completion_rate': completion_count / num_rollouts,
        'avg_goals': total_goals / num_rollouts,
        'goal_sequences': goal_sequences
    }

def run_goal_pursuit_with_sequence_manipulation(model, formula, force_goal="yellow", force_position=1, num_rollouts=5, world_indices=None):
    """Run goal pursuit test with sequence manipulation (clean, working logic)"""
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
        manipulation_applied = False

        if hasattr(env, 'zone_positions'):
            zone_pos = env.zone_positions.copy()
            zone_positions.append(zone_pos)
        else:
            zone_pos = {}
            zone_positions.append(zone_pos)

        for step in range(MAX_STEPS):
            # Manipulate sequence at step 5
            if step == 5 and not manipulation_applied:
                force_sequence_to_goal(agent, force_goal, force_position)
                manipulation_applied = True

            action = agent.get_action(obs, info, deterministic=True).flatten()
            # Use env.agent_pos for trajectory
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
                            if prop != last_goal:
                                achieved_goals.append(prop)
                                last_goal = prop
                    except (StopIteration, AttributeError):
                        pass

            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret
                term, trunc = done, done
            if done:
                break

        total_goals += len(visited_goals)
        goal_sequences.append(achieved_goals)
        trajectories.append(trajectory)

        # Physical yellow zone achievement check
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
                    yellow_reached = True
                    physical_yellow_count += 1
                    break

    env.close()
    return physical_yellow_count, total_goals, goal_sequences, trajectories, zone_positions

def force_sequence_to_goal(agent, target_goal, position=0):
    """Force the agent's sequence to pursue a specific goal at a specific position"""
    if not hasattr(agent, "sequence") or not agent.sequence:
        print(f"  [WARN] No sequence found to manipulate")
        return
    
    # Get the current sequence structure
    current_seq = agent.sequence
    print(f"  [DEBUG] Original sequence: {current_seq}")
    
    # Create a new sequence that forces the target goal
    from ltl.logic import Assignment
    from ltl.automata import LDBASequence
    
    # We need to know what propositions are available
    props = set(['blue', 'green', 'yellow', 'magenta'])  # All colors in vocab
    new_assignment = Assignment.single_proposition(target_goal, props)
    frozen_assignment = new_assignment.to_frozen()
    print(f"  [DEBUG] Injected assignment type: {type(frozen_assignment)}, value: {frozen_assignment}")
    
    # Create a new goal set with this assignment
    new_goal_set = {frozen_assignment}
    
    # Create a new sequence with the forced goal
    if len(current_seq) > 0:
        # Get the structure of the current sequence
        reach_avoid_pairs = list(current_seq)
        
        # Replace the goal at the specified position
        if len(reach_avoid_pairs) > position:
            # Keep the avoid part from the original, but change the reach part
            original_avoid = reach_avoid_pairs[position][1]
            new_reach_avoid = (new_goal_set, original_avoid)
            reach_avoid_pairs[position] = new_reach_avoid
            
            # Create a new LDBASequence
            new_sequence = LDBASequence(reach_avoid_pairs)
            agent.sequence = new_sequence
            print(f"  [DEBUG] Modified sequence: {new_sequence}")
        else:
            print(f"  [WARN] Position {position} out of range for sequence length {len(reach_avoid_pairs)}")
    else:
        print(f"  [WARN] No sequence to modify")

def test_dynamic_subgoal_swapping():
    """Test dynamic subgoal swapping at different positions during rollout"""
    print("=== Dynamic Subgoal Swapping Test ===\n")
    
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
    
    # Test different positions for dynamic swapping
    positions_to_test = [1, 2, 3]  # 0-indexed positions (2nd, 3rd, 4th subgoal)
    results = {}
    
    for position in positions_to_test:
        print(f"--- Test: Dynamically Swapping {position + 1}st Subgoal to Yellow ---")
        
        # Run 20 rollouts for this position
        physical_yellow_count, total_goals, goal_sequences, trajectories, zone_positions = run_goal_pursuit_with_dynamic_swapping(
            model, "GF green & GF blue", swap_position=position, num_rollouts=20, world_indices=WORLD_INDICES
        )
        
        # Calculate statistics based on physical yellow achievement
        yellow_pursuit_rate = physical_yellow_count / 20
        results[position] = {
            'physical_yellow_count': physical_yellow_count,
            'yellow_pursuit_rate': yellow_pursuit_rate,
            'total_goals': total_goals,
            'goal_sequences': goal_sequences
        }
        
        print(f"Physical yellow achievement rate: {yellow_pursuit_rate:.1%}")
        print(f"Physical yellow achievements: {physical_yellow_count}/20")
        print(f"Rollouts with yellow pursuit: {physical_yellow_count}/20 ({yellow_pursuit_rate:.1%})\n")
        
        # Visualize subset of 4 zone maps
        cols = 2
        rows = 2
        fig = draw_trajectories(zone_positions[:4], trajectories[:4], cols, rows)
        plt.suptitle(f"Dynamically Swapping {position + 1}st Subgoal to Yellow", fontsize=16, y=0.95)
        plt.savefig(f"zone_trajectory_dynamic_swap_position_{position}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [INFO] Zone trajectory saved as zone_trajectory_dynamic_swap_position_{position}.png")
    
    # Print summary
    print("=== Dynamic Subgoal Swapping Summary ===")
    for position, result in results.items():
        print(f"Position {position + 1} (2nd/3rd/4th subgoal): {result['physical_yellow_count']}/20 ({result['yellow_pursuit_rate']:.1%}) yellow pursuit rate")
    
    return results

def run_goal_pursuit_with_dynamic_swapping(model, formula, swap_position=1, num_rollouts=5, world_indices=None):
    """Run goal pursuit test with dynamic subgoal swapping"""
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    search = ExhaustiveSearch(model, set(props), num_loops=2)
    agent = Agent(model, search=search, propositions=set(props), verbose=False)
    
    physical_yellow_count = 0  # Track physical yellow achievements
    total_goals = 0
    goal_sequences = []
    trajectories = []  # Store agent trajectories
    zone_positions = []  # Store zone positions for each rollout
    
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
        swap_applied = False
        subgoal_count = 0  # Track which subgoal we're on
        trajectory = []
        
        # Save zone positions for this rollout
        if hasattr(env, 'zone_positions'):
            zone_pos = env.zone_positions.copy()
            zone_positions.append(zone_pos)
        else:
            zone_pos = {}
            zone_positions.append(zone_pos)
        
        for step in range(MAX_STEPS):
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
                                subgoal_count += 1
                                
                                # Apply dynamic swapping at the specified position
                                if subgoal_count == swap_position and not swap_applied:
                                    print(f"  [Rollout {rollout + 1}] Dynamically swapping {swap_position + 1}st subgoal to yellow")
                                    print(f"  [DEBUG] Original sequence: {seq}")
                                    
                                    # Create yellow assignment
                                    yellow_assignment = frozenset([
                                        ('green', False), ('yellow', True), ('blue', False), ('magenta', False)
                                    ])
                                    
                                    # Replace the current subgoal with yellow
                                    new_seq = list(seq)
                                    if len(new_seq) > 0:
                                        new_seq[0] = ({yellow_assignment}, set())
                                        agent.sequence = tuple(new_seq)
                                        swap_applied = True
                                        print(f"  [DEBUG] Swapped sequence: {agent.sequence}")
                                
                                # Debug: Print current sequence after each goal achievement
                                print(f"  [DEBUG] After achieving {prop}, current sequence: {seq}")
                                print(f"  [DEBUG] Sequence length: {len(seq)}")
                                if len(seq) > 0:
                                    print(f"  [DEBUG] Next subgoal: {seq[0]}")
                    except (StopIteration, AttributeError):
                        pass
            
            action = agent.get_action(obs, info, deterministic=True).flatten()
            try:
                obs_array = np.array(obs)
                if obs_array.ndim > 0 and obs_array.shape[0] >= 2:
                    trajectory.append(obs_array[:2].tolist())  # Store agent position as list
            except Exception as e:
                print(f"  [DEBUG] Skipping trajectory append due to obs shape: {obs}, error: {e}")
            
            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret
                term, trunc = done, done
            
            if done:
                break
        
        # Check if agent physically reached yellow zone
        yellow_reached = False
        if hasattr(env, 'zone_positions') and 'yellow' in env.zone_positions:
            yellow_center = env.zone_positions['yellow']
            yellow_radius = 0.3  # Yellow zone radius
            
            for step, pos in enumerate(trajectory):
                if len(pos) >= 2:
                    dist = np.linalg.norm(pos - yellow_center)
                    if dist <= yellow_radius:
                        print(f"  [Rollout {rollout + 1}] PHYSICALLY REACHED YELLOW at step {step}, pos {pos}, center {yellow_center}, radius {yellow_radius}")
                        yellow_reached = True
                        physical_yellow_count += 1
                        break
        
        if not yellow_reached:
            print(f"  [Rollout {rollout + 1}] DID NOT PHYSICALLY REACH YELLOW zone.")
        
        total_goals += len(achieved_goals)
        goal_sequences.append(achieved_goals)
        trajectories.append(trajectory)
    
    env.close()
    
    print(f"Physical yellow achievement rate: {physical_yellow_count/num_rollouts:.1%}")
    print(f"Physical yellow achievements: {physical_yellow_count}/{num_rollouts}")
    print(f"Rollouts with yellow pursuit: {physical_yellow_count}/{num_rollouts} ({physical_yellow_count/num_rollouts:.1%})\n")
    
    return physical_yellow_count, total_goals, goal_sequences, trajectories, zone_positions

if __name__ == "__main__":
    # Run the original sequence manipulation test
    test_sequence_manipulation()
    
    # Run the new dynamic subgoal swapping test
    test_dynamic_subgoal_swapping() 