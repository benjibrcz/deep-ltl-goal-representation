#!/usr/bin/env python3
"""
Probe GRU internals for planning and goal-related features.

This script probes the LTL-Net GRU for planning-related information:
1. Current automaton state
2. Next primitive action  
3. Is current sub-goal reached?
4. Chosen target-zone ID
5. Direction to chosen zone
6. Time-to-sub-goal
7. Remaining LTL suffix
8. Success probability

Usage:
    python probe_gru_planning.py --target current_automaton_state
    python probe_gru_planning.py --all
"""

import argparse
import random
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config import model_configs
from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from preprocessing.preprocessing import preprocess_obss
from utils.model_store.model_store import ModelStore

# ── Constants ────────────────────────────────────────────────────────────────
ENV, EXP  = "PointLtl2-v0", "big_test"
SEED = 0
N_WORLDS = 10
N_ROLLOUT = 10  
MAX_STEP = 200

# Goals to vary across rollouts
COLOURS = ["blue", "green", "yellow", "magenta"]
COLOUR2IDX = {c: i for i, c in enumerate(COLOURS)}
GOALS = [f"FG {c}" for c in COLOURS]

# ── All available targets ───────────────────────────────────────────────────────
ALL_TARGETS = [
    # Planning/Goal-related targets (GRU internals)
    "current_automaton_state",
    "next_action", 
    "subgoal_reached_soon",
    "target_zone_id",
    "direction_to_zone", 
    "time_to_subgoal",
    "remaining_ltl_suffix",
    "success_probability"
]

# Global variables for hook data collection
collect_this_step = False
current_gru_input = None
current_gru_hidden = None
current_gru_output = None

def is_classification(y):
    """Determine if target is classification or regression."""
    return y.dtype == int or len(np.unique(y)) <= 10

def validate_target_data(Y, target_name):
    """Validate target data quality."""
    print(f"📊 {target_name} Data Quality:")
    print(f"  Samples: {len(Y)}")
    print(f"  Shape: {Y.shape}")
    print(f"  Range: [{Y.min():.3f}, {Y.max():.3f}]")
    print(f"  Mean: {Y.mean():.3f}")
    print(f"  Std: {Y.std():.3f}")
    
    # Check for low variance
    if Y.std() < 1e-6:
        print(f"  ⚠️  WARNING: Very low variance (std={Y.std():.6f})")
        return False
    
    # Target-specific validation
    if target_name == "current_automaton_state":
        unique_states = np.unique(Y)
        print(f"  Automaton states: {unique_states}")
        if len(unique_states) < 2:
            print(f"  ⚠️  WARNING: Only {len(unique_states)} unique automaton state(s)")
    elif target_name == "target_zone_id":
        unique_zones = np.unique(Y)
        print(f"  Target zones: {unique_zones}")
        if len(unique_zones) < 2:
            print(f"  ⚠️  WARNING: Only {len(unique_zones)} unique target zone(s)")
    elif target_name == "subgoal_reached_soon":
        unique_vals = np.unique(Y)
        print(f"  Subgoal reached values: {unique_vals}")
        if len(unique_vals) < 2:
            print(f"  ⚠️  WARNING: Only {len(unique_vals)} unique subgoal reached value(s)")
    
    return True

def get_automaton_state(env, obs):
    """Extract current automaton state from environment."""
    if hasattr(env, 'automaton') and hasattr(env.automaton, 'current_state'):
        return np.array([env.automaton.current_state])
    elif hasattr(env, '_ldba_sequence') and hasattr(env._ldba_sequence, 'current_state'):
        return np.array([env._ldba_sequence.current_state])
    else:
        # Fallback: try to infer from propositions
        props = obs.get('propositions', {})
        # This is a simplified heuristic - you may need to adapt based on your automaton structure
        if any(props.values()):
            return np.array([1])  # In a goal zone
        else:
            return np.array([0])  # Not in a goal zone

def get_next_action(model, obs, props):
    """Get the next action that will be executed."""
    try:
        # Preprocess observation
        preprocessed = preprocess_obss([obs], props)
        
        # Get action from model
        with torch.no_grad():
            dist, value = model(preprocessed)
            # Sample action from distribution
            if hasattr(dist, 'sample'):
                action = dist.sample()
            else:
                # For MixedDistribution, get the underlying distribution
                if hasattr(dist, 'dist'):
                    action = dist.dist.sample()
                else:
                    action = torch.zeros(1)  # Fallback
            
        # Convert to class index for discrete actions
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return np.array([int(action.item())])
            else:
                return np.array([np.argmax(action)])
        else:
            return np.array([int(action)])
    except Exception as e:
        print(f"Warning: Could not get next action: {e}")
        return np.array([0])

def check_subgoal_reached_soon(env, obs, horizon=3):
    """Check if current subgoal will be reached within horizon steps."""
    # This is a simplified heuristic - you may need to adapt based on your environment
    try:
        current_zone = None
        props = obs.get('propositions', {})
        
        # Find current zone from propositions
        for zone_color, in_zone in props.items():
            if in_zone:
                current_zone = zone_color
                break
        
        # Check if we're getting closer to a goal zone
        if hasattr(env, 'agent_pos') and hasattr(env, 'zones'):
            agent_pos = env.agent_pos[:2]
            min_distance = float('inf')
            
            for zone in env.zones:
                if hasattr(zone, 'pos'):
                    zone_pos = zone.pos[:2]
                    distance = np.linalg.norm(agent_pos - zone_pos)
                    min_distance = min(min_distance, distance)
            
            # Heuristic: if very close to a zone, subgoal will be reached soon
            return np.array([1 if min_distance < 0.5 else 0])
        
        # Fallback: if currently in a goal zone, subgoal is "reached"
        return np.array([1 if current_zone is not None else 0])
        
    except Exception as e:
        print(f"Warning: Could not check subgoal reached: {e}")
        return np.array([0])

def get_target_zone_id(env, obs):
    """Identify which specific zone the agent is heading towards."""
    try:
        if hasattr(env, 'agent_pos') and hasattr(env, 'zones'):
            agent_pos = env.agent_pos[:2]
            min_distance = float('inf')
            target_zone_idx = 0
            
            # Find closest zone as proxy for target
            for i, zone in enumerate(env.zones):
                if hasattr(zone, 'pos'):
                    zone_pos = zone.pos[:2]
                    distance = np.linalg.norm(agent_pos - zone_pos)
                    if distance < min_distance:
                        min_distance = distance
                        target_zone_idx = i
            
            return np.array([target_zone_idx])
        else:
            # Fallback: random zone
            return np.array([np.random.randint(0, 4)])
            
    except Exception as e:
        print(f"Warning: Could not get target zone ID: {e}")
        return np.array([0])

def get_direction_to_zone(env, obs):
    """Get unit vector direction to target zone."""
    try:
        target_zone_id = get_target_zone_id(env, obs)[0]
        
        if hasattr(env, 'agent_pos') and hasattr(env, 'zones') and target_zone_id < len(env.zones):
            agent_pos = env.agent_pos[:2]
            zone = env.zones[target_zone_id]
            
            if hasattr(zone, 'pos'):
                zone_pos = zone.pos[:2]
                direction = zone_pos - agent_pos
                
                # Normalize to unit vector
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction = direction / norm
                
                return direction
        
        # Fallback: random unit vector
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)])
        
    except Exception as e:
        print(f"Warning: Could not get direction to zone: {e}")
        return np.array([0.0, 0.0])

def get_time_to_subgoal(env, obs):
    """Estimate time steps until subgoal is reached."""
    try:
        # Simple heuristic based on distance to closest zone
        if hasattr(env, 'agent_pos') and hasattr(env, 'zones'):
            agent_pos = env.agent_pos[:2]
            min_distance = float('inf')
            
            for zone in env.zones:
                if hasattr(zone, 'pos'):
                    zone_pos = zone.pos[:2]
                    distance = np.linalg.norm(agent_pos - zone_pos)
                    min_distance = min(min_distance, distance)
            
            # Rough estimate: distance / typical_speed
            estimated_steps = min_distance / 0.1  # Assuming speed ~0.1 units per step
            return np.array([max(1, int(estimated_steps))])
        
        # Fallback: random estimate
        return np.array([np.random.randint(1, 20)])
        
    except Exception as e:
        print(f"Warning: Could not estimate time to subgoal: {e}")
        return np.array([10])

def get_remaining_ltl_suffix(env, obs):
    """Get remaining LTL suffix after current subgoal."""
    try:
        # This would require deeper integration with the LTL automaton
        # For now, use a simplified heuristic based on current goal
        if hasattr(env, 'current_goal_colour'):
            goal_color = env.current_goal_colour
            if goal_color in COLOUR2IDX:
                return np.array([COLOUR2IDX[goal_color]])
        
        # Fallback: infer from current LTL goal string
        if hasattr(env, '_ltl_goal'):
            goal_str = env._ltl_goal
            for color in COLOURS:
                if color in goal_str.lower():
                    return np.array([COLOUR2IDX[color]])
        
        # Final fallback
        return np.array([0])
        
    except Exception as e:
        print(f"Warning: Could not get remaining LTL suffix: {e}")
        return np.array([0])

def get_success_probability(model, obs, props):
    """Get success probability from critic value."""
    try:
        # Preprocess observation
        preprocessed = preprocess_obss([obs], props)
        
        # Get value estimate from critic
        with torch.no_grad():
            dist, value = model(preprocessed)
            
        # Convert value to probability-like score (sigmoid)
        prob = torch.sigmoid(value).item()
        return np.array([prob])
        
    except Exception as e:
        print(f"Warning: Could not get success probability: {e}")
        return np.array([0.5])

def get_planning_target(env, obs, model, props, name):
    """Extract planning/goal-related target features."""
    if name == "current_automaton_state":
        return get_automaton_state(env, obs)
    elif name == "next_action":
        return get_next_action(model, obs, props)
    elif name == "subgoal_reached_soon":
        return check_subgoal_reached_soon(env, obs)
    elif name == "target_zone_id":
        return get_target_zone_id(env, obs)
    elif name == "direction_to_zone":
        return get_direction_to_zone(env, obs)
    elif name == "time_to_subgoal":
        return get_time_to_subgoal(env, obs)
    elif name == "remaining_ltl_suffix":
        return get_remaining_ltl_suffix(env, obs)
    elif name == "success_probability":
        return get_success_probability(model, obs, props)
    else:
        raise ValueError(f"Unknown planning target: {name}")

def train_and_evaluate_probe(X_train, X_test, y_train, y_test, target_name):
    """Train and evaluate a probe for a specific target."""
    clf_task = is_classification(y_train)
    
    # For multi-dimensional targets, we need to handle each dimension separately
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        print(f"  DEBUG: Multi-dimensional target detected: {y_train.shape}")
        # Multi-dimensional regression - handle each dimension separately
        scores = []
        for i in range(y_train.shape[1]):
            y_train_dim = y_train[:, i]
            y_test_dim = y_test[:, i]
            
            pipe_dim = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            pipe_dim.fit(X_train, y_train_dim)
            y_pred_dim = pipe_dim.predict(X_test)
            
            # Calculate R² for this dimension
            ss_res = np.sum((y_test_dim - y_pred_dim) ** 2)
            ss_tot = np.sum((y_test_dim - np.mean(y_test_dim)) ** 2)
            if ss_tot > 0:
                score_dim = 1 - (ss_res / ss_tot)
            else:
                score_dim = 1.0
            scores.append(score_dim)
        
        # Return average R² across dimensions
        return np.mean(scores), "R²"
    
    # Create pipeline for classification or single-dimensional regression
    if clf_task:
        pipe = make_pipeline(StandardScaler(), 
                           LogisticRegression(max_iter=1000, class_weight="balanced"))
        y_train_flat = y_train.ravel()
        y_test_flat = y_test.ravel()
    else:
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        # Single-dimensional regression
        y_train_flat = y_train.ravel()
        y_test_flat = y_test.ravel()
    
    # Train and predict
    pipe.fit(X_train, y_train_flat)
    y_pred = pipe.predict(X_test)
    
    # Calculate score
    if clf_task:
        if target_name == "subgoal_reached_soon":
            # Use AUROC for binary classification
            if len(np.unique(y_test_flat)) > 1:
                score = roc_auc_score(y_test_flat, y_pred)
                metric = "AUROC"
            else:
                score = accuracy_score(y_test_flat, y_pred)
                metric = "accuracy"
        else:
            score = accuracy_score(y_test_flat, y_pred)
            metric = "accuracy"
    else:
        score = r2_score(y_test_flat, y_pred, multioutput="uniform_average")
        metric = "R²"
    
    return score, metric

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="specific target to decode")
    ap.add_argument("--all", action="store_true", help="probe all available targets")
    args = ap.parse_args()

    if not args.target and not args.all:
        ap.error("Please specify either --target or --all")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # Build dummy env + model
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
    cfg   = model_configs[ENV]
    store = ModelStore(ENV, EXP, SEED); store.load_vocab()
    status= store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    dummy.close()

    print(f"📊 Model has recurrent LTL-Net: {hasattr(model.ltl_net, 'rnn')}")
    if hasattr(model.ltl_net, 'rnn'):
        print(f"📊 GRU hidden size: {model.ltl_net.rnn.hidden_size}")
        print(f"📊 GRU num layers: {model.ltl_net.rnn.num_layers}")

    # ── Hook GRU internals ─────────────────────────────────────────────
    def gru_hook(module, input_tuple, output_tuple):
        global current_gru_input, current_gru_hidden, current_gru_output, collect_this_step
        if not collect_this_step:
            return
            
        # GRU returns (output, hidden)
        # input_tuple contains (packed_sequence,) 
        # output_tuple contains (packed_output, hidden_state)
        
        # Extract packed input sequence
        packed_input = input_tuple[0]
        unpacked_input, lengths = nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)
        
        # Store GRU internals (take last timestep for each sequence)
        batch_size = unpacked_input.size(0)
        gru_inputs = []
        for i in range(batch_size):
            seq_len = lengths[i].item()
            last_input = unpacked_input[i, seq_len-1, :]  # Last timestep
            gru_inputs.append(last_input.detach().cpu().numpy())
        
        current_gru_input = np.vstack(gru_inputs) if len(gru_inputs) > 1 else gru_inputs[0]
        
        # Extract hidden state (final layer, all sequences)
        hidden_state = output_tuple[1]  # Shape: (num_layers, batch, hidden_size)
        current_gru_hidden = hidden_state[-1].detach().cpu().numpy()  # Last layer
        
        # Extract output (would need unpacking for sequence output)
        current_gru_output = current_gru_hidden  # For now, use hidden as output

    # Register hook on GRU
    hook_handle = model.ltl_net.rnn.register_forward_hook(gru_hook)

    # ── Data collection ─────────────────────────────────────────────────
    buf_gru_input = []
    buf_gru_hidden = []  
    buf_gru_output = []
    buf_lbl_dict = {target: [] for target in ALL_TARGETS}
    buf_lbl_single = []
    world_ids = []
    goal_sequences = []

    global collect_this_step
    
    print("🔄 Collecting GRU data with varied goals...")
    
    # Collect data across multiple worlds and goals
    for wid in range(N_WORLDS):
        for rid in range(N_ROLLOUT):
            # Cycle through goals
            ltl_goal = GOALS[(wid * N_ROLLOUT + rid) % len(GOALS)]
            
            # Create environment with specific goal
            env = make_env(ENV, FixedSampler.partial(ltl_goal), sequence=False)
            
            obs = env.reset(seed=SEED + 100*wid + rid)
            done = False; step = 0
            
            while not done and step < MAX_STEP:
                # Set flag to collect GRU activations for this step
                collect_this_step = True
                
                # Get current propositions
                props = set()
                if hasattr(env, 'automaton_graph') and len(env.automaton_graph.nodes()) > 0:
                    props = list(env.automaton_graph.nodes())[0]
                elif hasattr(env, 'propositions'):
                    props = env.propositions
                else:
                    # Default propositions for zones environment
                    props = {'blue', 'green', 'yellow', 'magenta'}
                
                # Create dummy agent for getting actions/values (simplified approach)
                # We'll manually call model.forward instead of using Agent class
                
                # Collect labels for all targets
                if args.all:
                    for target_name in ALL_TARGETS:
                        try:
                            label = get_planning_target(env, obs, model, props, target_name)
                            buf_lbl_dict[target_name].append(label)
                        except Exception as e:
                            print(f"Warning: Error collecting {target_name}: {e}")
                            # Use default values for failed targets
                            if target_name in ["direction_to_zone"]:
                                buf_lbl_dict[target_name].append(np.array([0.0, 0.0]))
                            else:
                                buf_lbl_dict[target_name].append(np.array([0]))
                else:
                    # Single target
                    try:
                        label = get_planning_target(env, obs, model, props, args.target)
                        buf_lbl_single.append(label)
                    except Exception as e:
                        print(f"Warning: Error collecting {args.target}: {e}")
                        if args.target in ["direction_to_zone"]:
                            buf_lbl_single.append(np.array([0.0, 0.0]))
                        else:
                            buf_lbl_single.append(np.array([0]))

                # Trigger model forward pass to collect GRU activations
                try:
                    preprocessed = preprocess_obss([obs], props)
                    with torch.no_grad():
                        _ = model(preprocessed)
                    
                    # Store GRU activations if collected
                    if current_gru_input is not None:
                        buf_gru_input.append(current_gru_input)
                        buf_gru_hidden.append(current_gru_hidden)
                        buf_gru_output.append(current_gru_output)
                        world_ids.append(wid)
                    
                    # Reset collection flag and temporaries
                    collect_this_step = False
                    current_gru_input = None
                    current_gru_hidden = None 
                    current_gru_output = None
                    
                except Exception as e:
                    print(f"Warning: Error in model forward pass: {e}")
                    collect_this_step = False
                
                # Step environment with simple action
                try:
                    # Get action from model
                    preprocessed = preprocess_obss([obs], props)
                    with torch.no_grad():
                        dist, value = model(preprocessed)
                        # Sample action
                        if hasattr(dist, 'sample'):
                            action = dist.sample()
                        else:
                            if hasattr(dist, 'dist'):
                                action = dist.dist.sample()
                            else:
                                action = torch.zeros(1)
                    
                    action = action.cpu().numpy()
                    if action.size == 1:
                        action = int(action.item())
                    else:
                        action = action.flatten()
                    
                    obs, _, done, _ = env.step(action)
                except Exception as e:
                    print(f"Warning: Error stepping environment: {e}")
                    done = True
                
                step += 1

            env.close()
            
            # Record goal sequence
            if hasattr(env, 'goal_sequence'):
                goal_sequences.append(env.goal_sequence)

    # Remove hook
    hook_handle.remove()

    # Align data
    min_length = min(len(buf_gru_input), len(buf_gru_hidden), len(buf_gru_output), len(world_ids))
    if min_length == 0:
        print("❌ No data collected! Check GRU hook setup.")
        return

    X_gru_input = np.vstack(buf_gru_input[:min_length])
    X_gru_hidden = np.vstack(buf_gru_hidden[:min_length])
    X_gru_output = np.vstack(buf_gru_output[:min_length])
    world_ids = np.array(world_ids[:min_length])
    
    print(f"📊 Collected {len(X_gru_input)} samples")
    print(f"  GRU input shape: {X_gru_input.shape}")
    print(f"  GRU hidden shape: {X_gru_hidden.shape}")
    print(f"  GRU output shape: {X_gru_output.shape}")
    print(f"  World IDs shape: {world_ids.shape}")
    
    # Analyze goal variety
    unique_goals = set(tuple(seq) for seq in goal_sequences if seq is not None)
    print(f"  Unique goal sequences: {len(unique_goals)}")
    print(f"  Total goal sequences: {len(goal_sequences)}")

    # Split data using held-out worlds for better generalization
    held_out_worlds = [8, 9]  # Last 2 worlds for testing
    
    train_mask = ~np.isin(world_ids, held_out_worlds)
    test_mask = np.isin(world_ids, held_out_worlds)
    
    X_gru_input_train, X_gru_input_test = X_gru_input[train_mask], X_gru_input[test_mask]
    X_gru_hidden_train, X_gru_hidden_test = X_gru_hidden[train_mask], X_gru_hidden[test_mask]
    X_gru_output_train, X_gru_output_test = X_gru_output[train_mask], X_gru_output[test_mask]
    
    print(f"  Train samples: {np.sum(train_mask)}")
    print(f"  Test samples: {np.sum(test_mask)}")

    # Determine targets to probe
    targets_to_probe = ALL_TARGETS if args.all else [args.target]
    
    print(f"\n🎯 Probing {len(targets_to_probe)} target(s) on GRU internals...")
    print("=" * 80)
    
    results = []
    
    for target_name in targets_to_probe:
        print(f"\n🔍 Probing: {target_name}")
        print("-" * 40)
        
        # Get target data
        if args.all:
            # Use pre-collected labels
            Y = np.vstack(buf_lbl_dict[target_name][:min_length])
        else:
            # Use pre-collected single target labels
            Y = np.vstack(buf_lbl_single[:min_length])
        
        # Validate data quality
        if not validate_target_data(Y, target_name):
            print(f"  ❌ Skipping {target_name} due to poor data quality")
            continue
        
        # Split target data
        Y_train, Y_test = Y[train_mask], Y[test_mask]
        
        # Train and evaluate probes on different GRU representations
        try:
            # Probe GRU INPUT (pre-recurrence)
            input_score, input_metric = train_and_evaluate_probe(
                X_gru_input_train, X_gru_input_test, Y_train, Y_test, target_name
            )
            
            # Probe GRU HIDDEN STATE (post-recurrence)
            hidden_score, hidden_metric = train_and_evaluate_probe(
                X_gru_hidden_train, X_gru_hidden_test, Y_train, Y_test, target_name
            )
            
            # Probe GRU OUTPUT (same as hidden for now)
            output_score, output_metric = train_and_evaluate_probe(
                X_gru_output_train, X_gru_output_test, Y_train, Y_test, target_name
            )
            
            # Store results
            result = {
                'target': target_name,
                'input_score': input_score,
                'hidden_score': hidden_score,
                'output_score': output_score,
                'metric': input_metric,
                'shape': Y.shape[1] if len(Y.shape) > 1 else 1
            }
            results.append(result)
            
            # Print results
            print(f"  GRU INPUT      {input_metric}: {input_score:.3f}")
            print(f"  GRU HIDDEN     {hidden_metric}: {hidden_score:.3f}")
            print(f"  GRU OUTPUT     {output_metric}: {output_score:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error probing {target_name}: {e}")
            continue
    
    # Print summary table
    if results:
        print(f"\n📋 SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Target':<25} {'Shape':<6} {'INPUT':<10} {'HIDDEN':<10} {'OUTPUT':<10} {'Metric':<8}")
        print("-" * 80)
        for result in results:
            print(f"{result['target']:<25} {result['shape']:<6} "
                  f"{result['input_score']:<10.3f} {result['hidden_score']:<10.3f} "
                  f"{result['output_score']:<10.3f} {result['metric']:<8}")

if __name__ == "__main__":
    main()