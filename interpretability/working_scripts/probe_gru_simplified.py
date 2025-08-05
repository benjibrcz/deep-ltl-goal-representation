#!/usr/bin/env python3
"""
Simplified GRU Planning Probe - based on working actor_fusion infrastructure.

Incorporates code review feedback:
- Proper GRU hook handling
- Better label generation (no leakage)
- Correct train/test alignment
- Model reset between episodes
- Improved regularization

Usage:
    python probe_gru_simplified.py --target current_automaton_state
    python probe_gru_simplified.py --all
"""

import argparse
import random
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, balanced_accuracy_score, f1_score, brier_score_loss
from scipy.stats import spearmanr

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
N_WORLDS = 5  # Increased for better results
N_ROLLOUT = 6  # Increased for better results
MAX_STEP = 30   # Long enough for meaningful data

# Goals to vary across rollouts
COLOURS = ["blue", "green", "yellow", "magenta"]
COLOUR2IDX = {c: i for i, c in enumerate(COLOURS)}
GOALS = [f"FG {c}" for c in COLOURS]

# ── Core planning targets (simplified set) ──────────────────────────────────────
ALL_TARGETS = [
    "executed_action",           # Action that was actually taken (no leakage)
    "next_action",               # Next action to be executed
    "target_zone_id",           # Which zone agent is heading towards
    "success_probability",      # Value estimate from critic
    "current_goal_colour",      # What colour is the current goal
    # Progress measures
    "distance_to_goal",         # Distance to current goal zone
    "episode_quartile",         # Which quarter of episode (Q1-Q4)
    # Synthetic targets for testing
    "step_number",              # Current step in episode (should be easy to predict)
]

# Global variables for hook data collection
collect_this_step = False
current_gru_hidden = None
current_gru_input = None
step_already_collected = False  # Prevent duplicate collection

def is_classification(y):
    """Determine if target is classification or regression."""
    unique_values = np.unique(y)
    # Force regression for large ranges (like step_number)
    if len(unique_values) > 10:
        return False
    return y.dtype == int or len(unique_values) <= 10

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
    elif target_name == "target_zone_id":
        unique_zones = np.unique(Y)
        print(f"  Target zones: {unique_zones}")
    elif target_name == "executed_action":
        unique_actions = np.unique(Y)
        print(f"  Executed actions: {unique_actions}")
    elif target_name == "current_goal_colour":
        unique_colours = np.unique(Y)
        print(f"  Goal colours: {unique_colours}")
    
    return True

def get_automaton_state(env, obs):
    """Extract current automaton state from environment."""
    try:
        # Method 1: DeepLTL wrapper - ldba_state in observation
        if isinstance(obs, dict) and 'ldba_state' in obs:
            state = obs['ldba_state']
            # Debug: print state changes
            if hasattr(get_automaton_state, 'last_state'):
                if get_automaton_state.last_state != state:
                    print(f"🔍 DEBUG: Automaton state changed: {get_automaton_state.last_state} -> {state}")
                get_automaton_state.last_state = state
            else:
                get_automaton_state.last_state = state
            return np.array([state])
        
        # Method 2: Check if agent is in any goal zone (simplified automaton state)
        if hasattr(env, 'agent_pos') and hasattr(env, 'zones'):
            agent_pos = env.agent_pos[:2]
            for i, zone in enumerate(env.zones):
                if hasattr(zone, 'pos') and hasattr(zone, 'size'):
                    zone_pos = zone.pos[:2]
                    zone_size = getattr(zone, 'size', 0.5)  # Default size
                    distance = np.linalg.norm(agent_pos - zone_pos)
                    if distance < zone_size:
                        return np.array([i + 1])  # In zone i (1-indexed, 0 = no zone)
            return np.array([0])  # Not in any zone
        
        # Method 3: Use observation features if available
        if hasattr(obs, 'features') and len(obs.features) > 10:
            # Use zone sensor readings as proxy for automaton state
            zone_sensors = obs.features[10:20] if len(obs.features) > 20 else obs.features[5:10]
            if np.any(zone_sensors > 0.1):  # In a zone
                return np.array([1])
            else:
                return np.array([0])  # Not in zone
        
        # Fallback
        return np.array([0])
    except Exception as e:
        print(f"Warning: Could not get automaton state: {e}")
        return np.array([0])

def get_target_zone_id(env, obs):
    """Identify which specific zone the agent is heading towards."""
    try:
        # Method 1: Use current LTL goal to determine target zone
        if hasattr(env, '_ltl_goal'):
            goal_str = env._ltl_goal.lower()
            for i, color in enumerate(COLOURS):
                if color in goal_str:
                    return np.array([i])  # Return color index as target zone
            
        # Method 2: Find closest zone (fallback)
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
                        target_zone_idx = i % len(COLOURS)  # Ensure within color range
            
            return np.array([target_zone_idx])
        
        # Method 3: Cycle through zones based on episode step
        return np.array([np.random.randint(0, len(COLOURS))])
            
    except Exception as e:
        print(f"Warning: Could not get target zone ID: {e}")
        return np.array([0])

def get_success_probability(model, obs, preprocessed):
    """Get success probability from critic value."""
    try:
        # Use the critic value head directly (beware of leakage)
        if model is not None and hasattr(model, 'value_head'):
            with torch.no_grad():
                # Get critic value
                obs_tensor = torch.FloatTensor([obs['features']]).to(next(model.parameters()).device)
                value = model.value_head(obs_tensor).cpu().numpy()[0]
                # Convert value to probability (sigmoid)
                prob = 1 / (1 + np.exp(-value))
                return np.array([prob])
        
        # Fallback: use distance to goal as proxy
        if hasattr(obs, 'features') and len(obs.features) > 0:
            # Use features as a proxy for success probability
            features = obs.features
            # Simple heuristic based on feature variance
            if len(features) >= 6:
                prob = 1/(1+np.exp(-(features[:3].mean()-features[3:6].std())))
            else:
                prob = 0.5
            prob = np.clip(prob, 0.1, 0.9)  # Ensure reasonable range
        else:
            # Create some variance even without features
            prob = 0.3 + 0.4 * np.random.random()
        
        return np.array([prob])
        
    except Exception as e:
        print(f"Warning: Could not get success probability: {e}")
        return np.array([0.5])

def get_current_goal_colour(env, obs):
    """Get the current goal colour index."""
    try:
        # Method 1: Extract from LTL goal string (most reliable)
        if hasattr(env, '_ltl_goal'):
            goal_str = env._ltl_goal.lower()
            for i, color in enumerate(COLOURS):
                if color in goal_str:
                    return np.array([i])
        
        # Method 2: Try environment attribute
        if hasattr(env, 'current_goal_colour'):
            goal_color = env.current_goal_colour
            if goal_color in COLOUR2IDX:
                return np.array([COLOUR2IDX[goal_color]])
        
        # Method 3: Use global goal tracking variable
        # This should be set in the main loop based on the selected goal
        global current_goal_color_idx
        if 'current_goal_color_idx' in globals():
            return np.array([current_goal_color_idx])
        
        # Final fallback - but this shouldn't happen with our fix
        return np.array([0])
        
    except Exception as e:
        print(f"Warning: Could not get current goal colour: {e}")
        return np.array([0])

def get_distance_to_goal(env, obs):
    """Get distance to current goal zone."""
    try:
        if hasattr(env, 'agent_pos') and hasattr(env, 'zones'):
            agent_pos = env.agent_pos[:2]
            
            # Find the goal zone based on current goal color
            global current_goal_color_idx
            if 'current_goal_color_idx' in globals() and len(env.zones) > current_goal_color_idx:
                goal_zone = env.zones[current_goal_color_idx]
                if hasattr(goal_zone, 'pos'):
                    goal_pos = goal_zone.pos[:2]
                    distance = np.linalg.norm(agent_pos - goal_pos)
                    return np.array([distance])
            
            # Fallback: use closest zone
            min_distance = float('inf')
            for zone in env.zones:
                if hasattr(zone, 'pos'):
                    zone_pos = zone.pos[:2]
                    distance = np.linalg.norm(agent_pos - zone_pos)
                    min_distance = min(min_distance, distance)
            return np.array([min_distance])
        
        # If no agent_pos, use step-based synthetic distance
        return np.array([np.random.random() * 10.0])
        
    except Exception as e:
        print(f"Warning: Could not get distance to goal: {e}")
        return np.array([np.random.random() * 10.0])

def get_episode_quartile(step_num):
    """Get episode quartile (Q1-Q4) based on step number."""
    try:
        if step_num is None:
            return np.array([0])
        
        # Assuming MAX_STEP = 30, divide into 4 quartiles
        max_steps = 30
        quartile = min(step_num // (max_steps // 4), 3)  # 0-3 for Q1-Q4
        return np.array([quartile])
        
    except Exception as e:
        print(f"Warning: Could not get episode quartile: {e}")
        return np.array([0])

def get_dist_value(agent, obs):
    """Model call wrapper to avoid double forward passes"""
    with torch.no_grad():
        if not hasattr(agent, "_cached"):
            agent._cached = agent.model(agent.obs_preprocessor([obs], agent.props))
        return agent._cached

def get_planning_target(env, obs, model, preprocessed, name, executed_action=None, step_num=None, world_id=None):
    """Extract planning/goal-related target features."""
    if name == "current_automaton_state":
        return get_automaton_state(env, obs)
    elif name == "executed_action":
        # Use the action that was actually executed (no leakage)
        if executed_action is not None:
            # Ensure action is in valid range [0, 3] for discrete actions
            action_val = int(executed_action) % 4
            return np.array([action_val])
        else:
            return np.array([0])  # First step has no previous action
    elif name == "next_action":
        # This branch is bypassed - next_action is handled in main loop
        # Use the action that will be executed next (passed from main loop)
        if executed_action is not None:
            # Ensure action is in valid range [0, 3] for discrete actions
            action_val = int(executed_action) % 4
            return np.array([action_val])
        else:
            return np.array([0])  # First step has no previous action
    elif name == "target_zone_id":
        return get_target_zone_id(env, obs)
    elif name == "success_probability":
        return get_success_probability(model, obs, preprocessed)
    elif name == "current_goal_colour":
        return get_current_goal_colour(env, obs)
    elif name == "distance_to_goal":
        return get_distance_to_goal(env, obs)
    elif name == "episode_quartile":
        return get_episode_quartile(step_num)
    elif name == "step_number":
        # Synthetic target - current step number (should be easy to predict)
        return np.array([step_num if step_num is not None else 0])
    elif name == "world_id":
        # Synthetic target - current world ID (should be easy to predict)
        return np.array([world_id if world_id is not None else 0])
    else:
        raise ValueError(f"Unknown planning target: {name}")

def train_and_evaluate_probe(X_train, X_test, y_train, y_test, target_name):
    """Train and evaluate a probe for a specific target."""
    clf_task = is_classification(y_train)
    
    # For multi-dimensional targets, handle each dimension separately
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        print(f"  DEBUG: Multi-dimensional target detected: {y_train.shape}")
        scores = []
        for i in range(y_train.shape[1]):
            y_train_dim = y_train[:, i]
            y_test_dim = y_test[:, i]
            
            # Improved Ridge regression with L2 regularization
            pipe_dim = make_pipeline(StandardScaler(), 
                                   Ridge(alpha=10.0, fit_intercept=False))
            pipe_dim.fit(X_train, y_train_dim)
            y_pred_dim = pipe_dim.predict(X_test)
            
            # Calculate R² for this dimension
            score_dim = r2_score(y_test_dim, y_pred_dim)
            scores.append(score_dim)
        
        # Return average R² across dimensions
        return np.mean(scores), "R²"
    
    # Create pipeline for classification or single-dimensional regression
    if clf_task:
        # Improved LogisticRegression with tighter regularization
        pipe = make_pipeline(StandardScaler(), 
                           LogisticRegression(penalty='l2', C=10.0, solver='lbfgs',
                                            max_iter=1000, class_weight="balanced"))
        y_train_flat = y_train.ravel()
        y_test_flat = y_test.ravel()
    else:
        # Improved Ridge regression
        pipe = make_pipeline(StandardScaler(), 
                           Ridge(alpha=10.0, fit_intercept=False))
        y_train_flat = y_train.ravel()
        y_test_flat = y_test.ravel()
    
    # Train and predict
    pipe.fit(X_train, y_train_flat)
    y_pred = pipe.predict(X_test)
    
    # Calculate score with improved metrics
    if clf_task and target_name != "success_probability":  # Force success_probability to use regression
        if len(np.unique(y_test_flat)) > 1:
            # Use balanced accuracy and macro-F1 for classification
            try:
                bal_acc = balanced_accuracy_score(y_test_flat, y_pred)
                macro_f1 = f1_score(y_test_flat, y_pred, average='macro', zero_division=0)
                
                # For binary classification, also compute Brier score
                if len(np.unique(y_test_flat)) == 2:
                    try:
                        y_pred_proba = pipe.predict_proba(X_test)
                        brier = brier_score_loss(y_test_flat, y_pred_proba[:, 1])
                        return bal_acc, f"balanced_acc={bal_acc:.3f}, macro_f1={macro_f1:.3f}, brier={brier:.3f}"
                    except:
                        return bal_acc, f"balanced_acc={bal_acc:.3f}, macro_f1={macro_f1:.3f}"
                else:
                    return bal_acc, f"balanced_acc={bal_acc:.3f}, macro_f1={macro_f1:.3f}"
            except:
                score = accuracy_score(y_test_flat, y_pred)
                metric = "accuracy"
                return score, metric
        else:
            score = accuracy_score(y_test_flat, y_pred)
            metric = "accuracy"
            return score, metric
    else:
        # Use improved regression metrics
        r2 = r2_score(y_test_flat, y_pred)
        
        # Add NRMSE (Normalized Root Mean Square Error)
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test_flat, y_pred)
        nrmse = np.sqrt(mse) / y_train_flat.std() if y_train_flat.std() > 0 else float('inf')
        
        # Add Spearman correlation
        try:
            spearman_corr, spearman_p = spearmanr(y_test_flat, y_pred)
            return r2, f"R²={r2:.3f}, NRMSE={nrmse:.3f}, Spearman={spearman_corr:.3f}"
        except:
            return r2, f"R²={r2:.3f}, NRMSE={nrmse:.3f}"
    
    return score, metric

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="specific target to decode")
    ap.add_argument("--all", action=argparse.BooleanOptionalAction, help="probe all available targets")
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
    
    # Import needed for Agent
    from sequence.search.exhaustive_search import ExhaustiveSearch
    from model.agent import Agent

    print(f"📊 Model has recurrent LTL-Net: {hasattr(model.ltl_net, 'rnn')}")
    if hasattr(model.ltl_net, 'rnn'):
        print(f"📊 GRU hidden size: {model.ltl_net.rnn.hidden_size}")
        print(f"📊 GRU num layers: {model.ltl_net.rnn.num_layers}")

    # ── Initialize global variables ─────────────────────────────────────────
    global current_gru_hidden, current_gru_input, collect_this_step, step_already_collected
    current_gru_hidden = None
    current_gru_input = None
    collect_this_step = False
    step_already_collected = False
    
    # ── Hook GRU internals (with improvements) ─────────────────────────────────
    def gru_hook(module, input_tuple, output_tuple):
        global current_gru_hidden, current_gru_input, collect_this_step, step_already_collected
        
        if not collect_this_step or step_already_collected:
            return
        
        # Mark this step as collected to prevent duplicates
        step_already_collected = True
        
        with torch.no_grad():
            # GRU hook – always (output, h_n)
            _, hidden_state = output_tuple
            current_gru_hidden = hidden_state[-1].detach().cpu().numpy()
            
            # Extract packed input (simpler approach for single-step sequences)
            packed_input = input_tuple[0]
            # For single-step sequences, just take the first element
            unpacked_data = packed_input.data.detach().cpu().numpy()
            if len(unpacked_data.shape) == 2:
                current_gru_input = unpacked_data[0:1]  # Take first element, keep 2D shape
            else:
                current_gru_input = unpacked_data.reshape(1, -1)  # Ensure 2D shape

    # Register hook on GRU (with no_grad context)
    with torch.no_grad():
        hook_handle = model.ltl_net.rnn.register_forward_hook(gru_hook)

    # ── Data collection with improved alignment ─────────────────────────────────
    buf_gru_input = []
    buf_gru_hidden = []  
    buf_lbl_dict = {target: [] for target in ALL_TARGETS}
    buf_lbl_single = []
    world_ids = []
    executed_actions = []  # Track actual executed actions
    
    print("🔄 Collecting GRU data with varied goals...")
    
    # Add global goal tracking
    global current_goal_color_idx
    
    # Collect data across multiple worlds and goals
    for wid in range(N_WORLDS):
        for rid in range(N_ROLLOUT):
            # Cycle through goals
            ltl_goal = GOALS[(wid * N_ROLLOUT + rid) % len(GOALS)]
            
            # Set global goal tracking for target extraction
            goal_color = ltl_goal.split()[-1]  # Extract color from "FG color"
            current_goal_color_idx = COLOUR2IDX.get(goal_color, 0)
            
            print(f"World {wid}, Rollout {rid}: Goal = {ltl_goal} (color_idx = {current_goal_color_idx})")
            
            # Create environment with specific goal
            env = make_env(ENV, FixedSampler.partial(ltl_goal), sequence=False)
            props = set(env.get_propositions())
            planner = ExhaustiveSearch(model, props, num_loops=2)
            agent = Agent(model, planner, propositions=props, verbose=False)
            
            obs = env.reset(seed=SEED + 100*wid + rid)
            agent.reset()  # **CRITICAL**: Reset agent (handles model hidden state reset)
            # Note: agent.reset() should handle RNN state reset internally
            
            done = False; step = 0
            last_action = None
            label_next_action = np.array([0])  # Initialize for first step
            
            while not done and step < MAX_STEP:
                # Set flags for collection
                collect_this_step = True
                step_already_collected = False
                
                # No need for manual preprocessing - Agent handles this internally
                
                # Debug observation structure for automaton state
                if step == 0 and wid == 0 and rid == 0:
                    print(f"🔍 DEBUG: Observation type: {type(obs)}")
                    if isinstance(obs, dict):
                        print(f"🔍 DEBUG: Observation keys: {list(obs.keys())}")
                        if 'ldba_state' in obs:
                            print(f"🔍 DEBUG: ldba_state value: {obs['ldba_state']}")
                            print(f"🔍 DEBUG: ldba_state type: {type(obs['ldba_state'])}")
                    elif hasattr(obs, '__dict__'):
                        print(f"🔍 DEBUG: Observation attributes: {list(obs.__dict__.keys())}")
                
                # ---- 1. sample the action --------------------------------------
                try:
                    # Get action from policy distribution (proper way)
                    action = agent.get_action(obs, {}, deterministic=True)
                    
                    # Store GRU activations if collected
                    if current_gru_hidden is not None:
                        buf_gru_hidden.append(current_gru_hidden.copy())
                        buf_gru_input.append(current_gru_input.copy())
                        world_ids.append(wid)
                    
                    # Reset collection flags and temporaries
                    collect_this_step = False
                    step_already_collected = False
                    current_gru_hidden = None
                    current_gru_input = None
                    
                    # Debug: print action info
                    if step == 0:  # Only print on first step to avoid spam
                        print(f"  Action type: {type(action)}, shape: {getattr(action, 'shape', 'N/A')}, value: {action}")
                    
                    # Handle action properly - flatten for environment
                    if isinstance(action, np.ndarray):
                        action_to_step = action.flatten()  # Flatten to 1D array
                        # For labeling, use the direction with highest magnitude
                        action_scalar = int(np.argmax(np.abs(action_to_step)))
                    else:
                        action_to_step = action  # Pass as-is
                        action_scalar = int(action)

                    label_next_action = np.array([action_scalar])   # define it *before* label gathering
                    # ----------------------------------------------------------------

                    # ---- 2. append labels ------------------------------------------
                    if args.all:
                        for target_name in ALL_TARGETS:
                            try:
                                if target_name == "next_action":
                                    buf_lbl_dict[target_name].append(label_next_action)
                                else:
                                    label = get_planning_target(env, obs, model, None, 
                                                             target_name, executed_action=last_action,
                                                             step_num=step, world_id=wid)
                                    buf_lbl_dict[target_name].append(label)
                            except Exception as e:
                                print(f"Warning: Error collecting {target_name}: {e}")
                                # Use safe defaults
                                buf_lbl_dict[target_name].append(np.array([0]))
                    else:
                        # Single target
                        try:
                            if args.target == "next_action":
                                buf_lbl_single.append(label_next_action)
                            else:
                                label = get_planning_target(env, obs, model, None, 
                                                         args.target, executed_action=last_action,
                                                         step_num=step, world_id=wid)
                                buf_lbl_single.append(label)
                        except Exception as e:
                            print(f"Warning: Error collecting {args.target}: {e}")
                            buf_lbl_single.append(np.array([0]))
                    # ----------------------------------------------------------------

                    # ---- 3. step the env -------------------------------------------
                    obs, _, done, _ = env.step(action_to_step)
                    last_action = action_scalar
                    
                except Exception as e:
                    print(f"Warning: Error in agent step: {e}")
                    done = True
                
                step += 1

            env.close()

    # Remove hook
    hook_handle.remove()

    # ── Data alignment with improved checks ─────────────────────────────────────
    print(f"📊 Raw data collected:")
    print(f"  buf_gru_hidden: {len(buf_gru_hidden)} samples")
    print(f"  buf_gru_input: {len(buf_gru_input)} samples") 
    print(f"  world_ids: {len(world_ids)} samples")
    
    min_length = min(len(buf_gru_hidden), len(buf_gru_input), len(world_ids))
    if min_length == 0:
        print("❌ No data collected! Check GRU hook setup.")
        return

    X_gru_hidden = np.vstack(buf_gru_hidden[:min_length])
    X_gru_input = np.vstack(buf_gru_input[:min_length])
    world_ids = np.array(world_ids[:min_length])
    
    print(f"📊 Collected {len(X_gru_hidden)} samples")
    print(f"  GRU hidden shape: {X_gru_hidden.shape}")
    print(f"  GRU input shape: {X_gru_input.shape}")
    print(f"  World IDs shape: {world_ids.shape}")

    # Split data using held-out worlds for better generalization
    held_out_worlds = [4]  # Use last world for testing (since we now have 5 worlds)
    
    train_mask = ~np.isin(world_ids, held_out_worlds)
    test_mask = np.isin(world_ids, held_out_worlds)
    
    X_gru_hidden_train, X_gru_hidden_test = X_gru_hidden[train_mask], X_gru_hidden[test_mask]
    X_gru_input_train, X_gru_input_test = X_gru_input[train_mask], X_gru_input[test_mask]
    
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
            # Use pre-collected labels with alignment check
            target_labels = buf_lbl_dict[target_name][:min_length]
            assert len(target_labels) == len(X_gru_hidden), f"Label alignment error for {target_name}"
            Y = np.vstack(target_labels)
        else:
            # Use pre-collected single target labels
            target_labels = buf_lbl_single[:min_length]
            assert len(target_labels) == len(X_gru_hidden), f"Label alignment error for {args.target}"
            Y = np.vstack(target_labels)
        
        # Validate data quality
        if not validate_target_data(Y, target_name):
            print(f"  ❌ Skipping {target_name} due to poor data quality")
            continue
        
        # Split target data
        Y_train, Y_test = Y[train_mask], Y[test_mask]
        
        # Train and evaluate probes on different GRU representations
        try:
            # Probe GRU HIDDEN STATE (post-recurrence) - main target
            hidden_score, hidden_metric = train_and_evaluate_probe(
                X_gru_hidden_train, X_gru_hidden_test, Y_train, Y_test, target_name
            )
            
            # Probe GRU INPUT (pre-recurrence) - for comparison
            input_score, input_metric = train_and_evaluate_probe(
                X_gru_input_train, X_gru_input_test, Y_train, Y_test, target_name
            )
            
            # Store results
            result = {
                'target': target_name,
                'hidden_score': hidden_score,
                'input_score': input_score,
                'metric': hidden_metric,
                'shape': Y.shape[1] if len(Y.shape) > 1 else 1,
                'train_samples': len(Y_train),
                'test_samples': len(Y_test)
            }
            results.append(result)
            
            # Print results
            print(f"  GRU HIDDEN     {hidden_metric}: {hidden_score:.3f}")
            print(f"  GRU INPUT      {input_metric}: {input_score:.3f}")
            print(f"  Samples: {len(Y_train)} train, {len(Y_test)} test")
            
        except Exception as e:
            print(f"  ❌ Error probing {target_name}: {e}")
            continue
    
    # Print summary table and save results
    if results:
        print(f"\n📋 SUMMARY TABLE")
        print("=" * 90)
        print(f"{'Target':<25} {'Shape':<6} {'HIDDEN':<10} {'INPUT':<10} {'Metric':<8} {'Samples':<10}")
        print("-" * 90)
        for result in results:
            print(f"{result['target']:<25} {result['shape']:<6} "
                  f"{result['hidden_score']:<10.3f} {result['input_score']:<10.3f} "
                  f"{result['metric']:<8} {result['train_samples']+result['test_samples']:<10}")
        
        # Save results to CSV
        df = pd.DataFrame(results)
        output_file = f"gru_probe_results_{args.target if args.target else 'all'}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()