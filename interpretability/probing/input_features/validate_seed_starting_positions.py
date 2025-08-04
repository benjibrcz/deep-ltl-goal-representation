#!/usr/bin/env python3
import os, sys
import numpy as np

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..", "src")))

from envs import make_env
from ltl import FixedSampler

# Test configurations - we'll test multiple environments
TEST_ENVS = ["PointLtl2-v0", "FlatWorld-v0", "LetterEnv-v0"]
WORLD_ID = 0
BASE_SEED = 42

def test_seed_variation(env_name):
    """Test if different seeds produce different starting positions."""
    print(f"Testing seed variation in {env_name}")
    
    formula = "FG blue"
    sampler = FixedSampler.partial(formula)
    
    # Create environment
    env = make_env(env_name, sampler, sequence=False, render_mode=None)
    
    # Load a world if world files exist for this environment
    world_dir = f"eval_datasets/{env_name}/worlds"
    world_file = f"{world_dir}/world_info_{WORLD_ID}.pkl"
    if os.path.exists(world_file):
        env.load_world_info(world_file)
        print(f"Loaded world {WORLD_ID} from {world_file}")
    else:
        print(f"No world file found, testing without fixed world layout")
    
    starting_positions = []
    test_seeds = [
        BASE_SEED + 0,
        BASE_SEED + 1000,
        BASE_SEED + 2000,
        BASE_SEED + 3000,
        BASE_SEED + 4000,
    ]
    
    print(f"Testing seeds: {test_seeds}")
    print(f"Testing in world {WORLD_ID}")
    print()
    
    for i, seed in enumerate(test_seeds):
        try:
            obs = env.reset(seed=seed)
            agent_pos = env.agent_pos[:2].copy()
            starting_positions.append(agent_pos)
            print(f"Seed {seed:6d}: Starting position = ({agent_pos[0]:.3f}, {agent_pos[1]:.3f})")
        except Exception as e:
            print(f"Seed {seed:6d}: Failed to reset - {e}")
            starting_positions.append(None)
    
    env.close()
    
    # Check for variation
    valid_positions = [pos for pos in starting_positions if pos is not None]
    if len(valid_positions) < 2:
        print("\nERROR: Not enough valid starting positions to compare!")
        return
    
    print(f"\nAnalysis of {len(valid_positions)} valid starting positions:")
    
    # Calculate pairwise distances
    distances = []
    for i in range(len(valid_positions)):
        for j in range(i+1, len(valid_positions)):
            dist = np.linalg.norm(valid_positions[i] - valid_positions[j])
            distances.append(dist)
    
    distances = np.array(distances)
    
    print(f"Pairwise distances: min={distances.min():.3f}, max={distances.max():.3f}, mean={distances.mean():.3f}")
    
    # Check if all positions are identical
    all_identical = all(np.allclose(pos, valid_positions[0], atol=1e-6) for pos in valid_positions[1:])
    
    if all_identical:
        print("❌ WARNING: All starting positions are identical! Seeds may not be affecting starting positions.")
    else:
        print("✅ SUCCESS: Different seeds produce different starting positions.")
        
        # Show position ranges
        positions_array = np.array(valid_positions)
        x_range = positions_array[:, 0].max() - positions_array[:, 0].min()
        y_range = positions_array[:, 1].max() - positions_array[:, 1].min()
        print(f"Position variation - X range: {x_range:.3f}, Y range: {y_range:.3f}")

def test_rollout_generation_pattern(env_name):
    """Test the specific seed pattern used in the probing script."""
    print(f"\nTesting rollout generation pattern from probe_comprehensive_generalization.py")
    print(f"Environment: {env_name}")
    
    formula = "FG blue"
    sampler = FixedSampler.partial(formula)
    env = make_env(env_name, sampler, sequence=False, render_mode=None)
    
    world_dir = f"eval_datasets/{env_name}/worlds"
    world_file = f"{world_dir}/world_info_{WORLD_ID}.pkl"
    if os.path.exists(world_file):
        env.load_world_info(world_file)
        print(f"Loaded world {WORLD_ID} from {world_file}")
    else:
        print(f"No world file found, testing without fixed world layout")
        # For environments without world files, we can still test seeding
    
    # Use the same seed generation pattern as the probe script
    SEED = 0  # Base seed from the script
    n_rollouts = 5
    max_attempts_per_rollout = 10
    
    print(f"Testing seed pattern: SEED + world_id * 1000 + rollout_idx * {max_attempts_per_rollout} + attempt")
    print(f"Base SEED = {SEED}, world_id = {WORLD_ID}")
    print()
    
    rollout_positions = []
    
    for rollout_idx in range(n_rollouts):
        for attempt in range(max_attempts_per_rollout):
            seed = SEED + WORLD_ID * 1000 + rollout_idx * max_attempts_per_rollout + attempt
            try:
                obs = env.reset(seed=seed)
                agent_pos = env.agent_pos[:2].copy()
                print(f"Rollout {rollout_idx}, Attempt {attempt}: Seed {seed:4d} -> Position ({agent_pos[0]:.3f}, {agent_pos[1]:.3f})")
                rollout_positions.append((rollout_idx, attempt, seed, agent_pos))
                break  # Success, move to next rollout
            except AssertionError as e:
                if "World has starting cost" in str(e):
                    continue  # Try next seed
                else:
                    print(f"Rollout {rollout_idx}, Attempt {attempt}: Seed {seed:4d} -> Unexpected error: {e}")
                    break
    
    env.close()
    
    # Analyze successful rollouts
    if len(rollout_positions) >= 2:
        print(f"\n✅ Successfully generated {len(rollout_positions)} rollouts")
        
        # Check if rollouts from different rollout_idx have different positions
        unique_rollout_indices = set(pos[0] for pos in rollout_positions)
        if len(unique_rollout_indices) >= 2:
            positions_by_rollout = {}
            for rollout_idx, attempt, seed, pos in rollout_positions:
                if rollout_idx not in positions_by_rollout:
                    positions_by_rollout[rollout_idx] = []
                positions_by_rollout[rollout_idx].append(pos)
            
            print("\nPosition comparison across different rollouts:")
            rollout_keys = sorted(positions_by_rollout.keys())
            for i in range(len(rollout_keys)):
                for j in range(i+1, len(rollout_keys)):
                    rollout_i, rollout_j = rollout_keys[i], rollout_keys[j]
                    pos_i = positions_by_rollout[rollout_i][0]  # Take first position
                    pos_j = positions_by_rollout[rollout_j][0]
                    distance = np.linalg.norm(pos_i - pos_j)
                    print(f"  Rollout {rollout_i} vs Rollout {rollout_j}: Distance = {distance:.3f}")
                    
            # Check if any positions are identical
            all_positions = [pos for positions in positions_by_rollout.values() for pos in positions]
            all_identical = all(np.allclose(pos, all_positions[0], atol=1e-6) for pos in all_positions[1:])
            
            if all_identical:
                print("❌ WARNING: All rollout positions are identical!")
            else:
                print("✅ SUCCESS: Different rollouts have different starting positions.")
        else:
            print("❌ Only one unique rollout generated - cannot test variation.")
    else:
        print("❌ ERROR: Could not generate sufficient rollouts for testing.")

if __name__ == "__main__":
    for env_name in TEST_ENVS:
        print("=" * 80)
        print(f"TESTING ENVIRONMENT: {env_name}")
        print("=" * 80)
        
        try:
            test_seed_variation(env_name)
            test_rollout_generation_pattern(env_name)
        except Exception as e:
            print(f"ERROR testing {env_name}: {e}")
        
        print("\n" + "="*80 + "\n") 