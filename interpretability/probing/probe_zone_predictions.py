#!/usr/bin/env python3
"""
Zone Prediction Probe

Test "will the agent visit zone X in the next k steps?" across different goal colors.
This should be much more balanced than grid squares since zones are larger targets
and the agent's goals are often zone-directed.
"""

import os
import sys
import random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from ltl import FixedSampler
from envs import make_env
from sequence.search import ExhaustiveSearch
from model.agent import Agent

ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0

def get_zone_info(env):
    """Extract zone color information from environment."""
    # Get zone colors from environment propositions
    if hasattr(env, 'get_propositions'):
        zone_colors = list(env.get_propositions())
        return sorted(zone_colors)
    
    # Fallback: check if wrapped environment has colors attribute
    if hasattr(env, 'colors'):
        return sorted(list(env.colors))
    
    return []

def get_current_zone(agent_pos, zone_colors, zone_size=0.4):
    """Determine which zone the agent is currently in."""
    # This is a simplified version - in reality we'd need the actual zone positions
    # For now, we'll use a heuristic based on the safety-gymnasium setup
    # The exact positions depend on the specific LTL level being used
    
    # For PointLtl2-v0 (which uses LtlLevel2), zones are placed randomly
    # We'll return None for now and rely on the environment's propositions
    return None

def collect_zone_data(goal_ltl, num_worlds=5, num_rollouts=5, max_steps=200, horizon=10, min_zone_visits=1):
    """Collect data for zone prediction across multiple trajectories."""
    print(f"Collecting data for goal: {goal_ltl}")
    
    # Load model
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial(goal_ltl), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Get zone information
    env = make_env(ENV, FixedSampler.partial(goal_ltl), sequence=False, render_mode=None)
    zone_colors = get_zone_info(env)
    print(f"Available zone colors: {zone_colors}")
    
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    all_samples = []
    
    # Collect data across multiple worlds and rollouts
    for world_id in range(num_worlds):
        for rollout_id in range(num_rollouts):
            seed = world_id * 1000 + rollout_id * 100
            
            done = False
            obs = env.reset(seed=seed)
            agent.reset()
            
            trajectory_data = []
            visited_zones = set()  # Track all zones visited during this rollout
            
            step_id = 0
            while step_id < max_steps:
                if done:
                    break
                
                # Get current propositions (which zones the agent is currently in)
                current_props = set(obs.get('propositions', []))
                visited_zones.update(current_props)  # Track zone visits
                
                # Get network input features
                obs_features = obs.get('features', np.zeros(80))
                
                # Create goal encoding
                goal_encoding = np.zeros(10)
                if 'blue' in goal_ltl.lower():
                    goal_encoding[0] = 1.0
                elif 'green' in goal_ltl.lower():
                    goal_encoding[1] = 1.0
                elif 'yellow' in goal_ltl.lower():
                    goal_encoding[2] = 1.0
                elif 'magenta' in goal_ltl.lower():
                    goal_encoding[3] = 1.0
                
                network_input = np.concatenate([obs_features, goal_encoding])
                
                trajectory_data.append({
                    'features': network_input,
                    'current_zones': current_props.copy(),
                    'world_id': world_id,
                    'rollout_id': rollout_id,
                    'step_id': step_id
                })
                
                action = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, info = env.step(action)
                
                # Also store propositions from info if available
                if isinstance(info, dict) and 'propositions' in info:
                    info_props = set(info['propositions']) if isinstance(info['propositions'], (list, set)) else set()
                    visited_zones.update(info_props)  # Track info zone visits too
                    # Update trajectory data with info propositions if they're more reliable
                    if info_props and len(trajectory_data) > 0:
                        trajectory_data[-1]['current_zones'] = info_props
                
                step_id += 1
                
                # Early stopping: if we have enough zone visits and enough steps for meaningful prediction
                if len(visited_zones) >= min_zone_visits and step_id >= horizon + 10:
                    print(f"  Early stop at step {step_id}: visited {visited_zones}")
                    break
            
            print(f"  Rollout {rollout_id}: {step_id} steps, visited zones: {visited_zones}")
            
            # Create samples with future zone visits
            if len(trajectory_data) >= horizon + 2:
                for i in range(len(trajectory_data) - horizon):
                    current_data = trajectory_data[i]
                    
                    # Look ahead k steps to see which zones are visited
                    future_zone_visits = set()
                    for j in range(1, horizon + 1):
                        if i + j < len(trajectory_data):
                            future_zones = trajectory_data[i + j]['current_zones']
                            future_zone_visits.update(future_zones)
                    
                    # Create binary labels for each zone color
                    zone_labels = {}
                    for color in zone_colors:
                        zone_labels[f'will_visit_{color}'] = 1 if color in future_zone_visits else 0
                    
                    all_samples.append({
                        'features': current_data['features'],
                        'zone_labels': zone_labels,
                        'current_zones': current_data['current_zones'],
                        'future_zones': future_zone_visits,
                        'world_id': current_data['world_id'],
                        'rollout_id': current_data['rollout_id'],
                        'step_id': current_data['step_id'],
                        'goal': goal_ltl
                    })
    
    env.close()
    print(f"Collected {len(all_samples)} samples for {goal_ltl}")
    return all_samples, zone_colors

def evaluate_zone_prediction(samples, zone_colors, target_zone, split_name="default"):
    """Evaluate zone prediction for a specific zone color."""
    if not samples:
        return {'status': 'no_data'}
    
    # Extract features and labels
    features = np.array([s['features'] for s in samples])
    labels = [s['zone_labels'][f'will_visit_{target_zone}'] for s in samples]
    
    positive_count = sum(labels)
    positive_ratio = positive_count / len(labels)
    
    print(f"\n🎯 ZONE PREDICTION: {target_zone.upper()} ({split_name})")
    print(f"Samples: {len(samples)}, Positive: {positive_count} ({positive_ratio:.3f})")
    
    # Skip if too few positive samples
    if positive_count < 5:
        print(f"❌ Too few positive samples ({positive_count})")
        return {'status': 'insufficient_data', 'positive_ratio': positive_ratio}
    
    # Apply PCA
    pca = PCA(n_components=min(10, features.shape[1]))
    features_pca = pca.fit_transform(features)
    
    # Train/test split (70/30)
    n_train = int(0.7 * len(samples))
    indices = list(range(len(samples)))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = features_pca[train_indices]
    X_test = features_pca[test_indices]
    y_train = [labels[i] for i in train_indices]
    y_test = [labels[i] for i in test_indices]
    
    train_positive = sum(y_train)
    test_positive = sum(y_test)
    
    print(f"Train: {len(X_train)} samples, {train_positive} positive ({train_positive/len(y_train):.3f})")
    print(f"Test: {len(X_test)} samples, {test_positive} positive")
    
    # Skip if no positive samples in train or test
    if train_positive == 0 or len(set(y_train)) == 1:
        print(f"❌ No positive samples in training set")
        return {'status': 'no_positive_train', 'positive_ratio': positive_ratio}
    
    if test_positive == 0 or len(set(y_test)) == 1:
        print(f"❌ No positive samples in test set")
        return {'status': 'no_positive_test', 'positive_ratio': positive_ratio}
    
    # Train balanced classifier
    try:
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight_dict)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division='warn')
        
        print(f"✅ Results: Acc={accuracy:.3f}, Bal-Acc={balanced_acc:.3f}, F1={f1:.3f}")
        
        return {
            'status': 'success',
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'positive_ratio': positive_ratio
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return {'status': 'training_failed', 'error': str(e), 'positive_ratio': positive_ratio}

def run_comprehensive_zone_evaluation():
    """Run zone prediction evaluation across multiple goals and generalization splits."""
    
    # Test goals
    goals = [
        "FG blue",
        "FG green", 
        "FG yellow",
        "FG magenta"
    ]
    
    # Collect data for each goal
    all_goal_data = {}
    all_zone_colors = set()
    
    for goal in goals:
        try:
            samples, zone_colors = collect_zone_data(goal, num_worlds=8, num_rollouts=8, max_steps=400, horizon=15, min_zone_visits=2)
            all_goal_data[goal] = samples
            all_zone_colors.update(zone_colors)
        except Exception as e:
            print(f"❌ Failed to collect data for {goal}: {e}")
            all_goal_data[goal] = []
    
    zone_colors = sorted(list(all_zone_colors))
    print(f"\n🎯 ALL ZONE COLORS FOUND: {zone_colors}")
    
    # Test 1: Within-goal prediction (train and test on same goal)
    print(f"\n" + "="*60)
    print(f"TEST 1: WITHIN-GOAL ZONE PREDICTION")
    print(f"="*60)
    
    within_goal_results = {}
    for goal in goals:
        if not all_goal_data[goal]:
            continue
            
        within_goal_results[goal] = {}
        print(f"\n--- Testing {goal} ---")
        
        for zone_color in zone_colors:
            result = evaluate_zone_prediction(
                all_goal_data[goal], 
                zone_colors, 
                zone_color, 
                f"within_{goal.replace(' ', '_')}"
            )
            within_goal_results[goal][zone_color] = result
    
    # Focus on within-goal performance only (cross-goal removed per user feedback)
    
    # Summary
    print(f"\n" + "="*60)
    print(f"SUMMARY")
    print(f"="*60)
    
    print(f"\n📊 WITHIN-GOAL PERFORMANCE:")
    for goal, results in within_goal_results.items():
        successful_zones = [zone for zone, result in results.items() if result.get('status') == 'success']
        if successful_zones:
            avg_f1 = np.mean([results[zone]['f1_score'] for zone in successful_zones])
            print(f"{goal}: {len(successful_zones)} zones, avg F1 = {avg_f1:.3f}")
        else:
            print(f"{goal}: No successful predictions")

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("="*60)
    print("ZONE PREDICTION PROBE")
    print("="*60)
    print("Testing 'will the agent visit zone X in the next k steps?'")
    print("across different goal colors with proper class balancing")
    print("="*60)
    
    run_comprehensive_zone_evaluation()

if __name__ == "__main__":
    main() 