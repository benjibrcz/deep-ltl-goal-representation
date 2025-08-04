#!/usr/bin/env python3
"""
Simple Navigation Success Probing

Tests if the complete network can predict navigation success - whether the agent
will be closer to the target zone in the next few steps.

Usage: python interpretability/probing/probe_simple_navigation_success.py
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from ltl import FixedSampler
from envs import make_env
from sequence.search import ExhaustiveSearch
from model.agent import Agent

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0

def collect_navigation_success_data(model, world_ids, goal="FG blue", n_rollouts_per_world=8, max_steps=50, k_steps=5):
    """Collect data for navigation success prediction."""
    print(f"Collecting navigation success data for goal: {goal}")
    
    all_features = []
    all_success_labels = []
    all_metadata = []
    
    env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    target_zone_name = goal.split()[-1]  # "FG blue" -> "blue"
    
    total_samples = 0
    successful_samples = 0
    
    for world_id in world_ids:
        print(f"Processing world {world_id}...")
        for rollout_id in trange(n_rollouts_per_world, desc=f"Rollouts for world {world_id}"):
            trajectory_data = []
            
            done = False
            obs = env.reset(seed=world_id + rollout_id * 1000)
            agent.reset()
            
            # Find target zone position
            target_zone_pos = None
            if hasattr(env, 'zone_positions') and env.zone_positions:
                for zone_name, zone_pos in env.zone_positions.items():
                    if target_zone_name.lower() in zone_name.lower():
                        target_zone_pos = zone_pos[:2]
                        break
            
            if target_zone_pos is None:
                continue
            
            for step_id in range(max_steps):
                if done:
                    break
                    
                current_pos = env.agent_pos[:2].copy()
                current_dist = np.linalg.norm(current_pos - target_zone_pos)
                
                # Get network representation
                obs_features = obs.get('features', np.zeros(80))
                goal_encoding = np.zeros(10)
                if 'blue' in goal.lower():
                    goal_encoding[0] = 1.0
                elif 'red' in goal.lower():
                    goal_encoding[1] = 1.0
                
                network_representation = np.concatenate([obs_features, goal_encoding])
                
                trajectory_data.append({
                    'network_representation': network_representation,
                    'position': current_pos,
                    'distance_to_target': current_dist
                })
                
                action = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, info = env.step(action)
            
            # Create success labels
            valid_samples = len(trajectory_data) - k_steps
            if valid_samples <= 0:
                continue
                
            for i in range(valid_samples):
                current_data = trajectory_data[i]
                future_data = trajectory_data[i + k_steps] if i + k_steps < len(trajectory_data) else trajectory_data[-1]
                
                current_dist = current_data['distance_to_target']
                future_dist = future_data['distance_to_target']
                
                # Success = getting closer to target
                is_successful = 1 if future_dist < current_dist - 0.1 else 0  # Small threshold to avoid noise
                
                all_features.append(current_data['network_representation'])
                all_success_labels.append(is_successful)
                all_metadata.append({
                    'world_id': world_id,
                    'rollout_id': rollout_id,
                    'step_id': i,
                    'current_dist': current_dist,
                    'future_dist': future_dist
                })
                
                total_samples += 1
                if is_successful:
                    successful_samples += 1
    
    env.close()
    
    print(f"Collected {total_samples} samples, {successful_samples} successful ({successful_samples/total_samples*100:.1f}%)")
    
    return {
        'features': np.array(all_features),
        'success_labels': np.array(all_success_labels),
        'metadata': all_metadata
    }

def evaluate_navigation_success_prediction(data):
    """Evaluate navigation success prediction."""
    features = data['features']
    success_labels = data['success_labels']
    
    # Apply PCA
    pca = PCA(n_components=0.95)
    features_pca = pca.fit_transform(features)
    print(f"PCA: {features.shape[1]} → {features_pca.shape[1]} dimensions")
    
    # Simple train/test split
    n_samples = len(features_pca)
    n_train = int(0.7 * n_samples)
    
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = features_pca[train_indices]
    X_test = features_pca[test_indices]
    y_train = success_labels[train_indices]
    y_test = success_labels[test_indices]
    
    # Check if we have both classes
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)
    
    print(f"Train classes: {unique_train}, Test classes: {unique_test}")
    
    if len(unique_train) < 2:
        print("❌ Not enough variety in training data")
        return None
    
    # Train classifier
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Navigation Success Prediction Accuracy: {accuracy:.3f}")
    
    return {
        'accuracy': accuracy,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

def main():
    parser = argparse.ArgumentParser(description='Simple Navigation Success Probing')
    parser.add_argument('--goal', default="FG blue", help='LTL goal to test')
    parser.add_argument('--k-steps', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--n-worlds', type=int, default=8, help='Number of worlds')
    parser.add_argument('--n-rollouts', type=int, default=8, help='Rollouts per world')
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps per rollout')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=== NAVIGATION SUCCESS PREDICTION PROBE ===")
    print(f"🎯 Can the network predict when it's making progress toward the goal?")
    print(f"Goal: {args.goal}")
    print(f"Prediction horizon: {args.k_steps} steps")
    
    # Load model
    print("\nLoading model...")
    store = ModelStore(ENV, EXP, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial(args.goal), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Collect data
    world_ids = list(range(args.n_worlds))
    data = collect_navigation_success_data(model, world_ids, args.goal, args.n_rollouts, args.max_steps, args.k_steps)
    
    # Evaluate
    print("\nEvaluating navigation success prediction...")
    results = evaluate_navigation_success_prediction(data)
    
    if results:
        print(f"\n=== RESULTS ===")
        print(f"🎯 Navigation Success Prediction: {results['accuracy']:.3f} accuracy")
        
        if results['accuracy'] > 0.6:
            print("✅ SUCCESS! Network can predict navigation progress!")
            print("   This supports the associative navigation hypothesis! 🧠")
        elif results['accuracy'] > 0.55:
            print("🤔 MODERATE success - some predictive capability")
        else:
            print("❌ Low prediction accuracy")

if __name__ == "__main__":
    main() 