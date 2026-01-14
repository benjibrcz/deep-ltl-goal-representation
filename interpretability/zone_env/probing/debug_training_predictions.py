#!/usr/bin/env python3
"""
Debug Training Predictions

Investigate why train predictions aren't close to 1.0 for positive samples.
This should reveal if there's a bug in our methodology.
"""

import os
import sys
import random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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

def position_to_grid_cell(pos, grid_size, map_bounds):
    x, y = pos
    x_min, y_min, x_max, y_max = map_bounds
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    grid_x = int(np.clip(x_norm * grid_size, 0, grid_size - 1))
    grid_y = int(np.clip(y_norm * grid_size, 0, grid_size - 1))
    return grid_x, grid_y

def debug_single_cell_training():
    """Debug training for a single cell to understand what's happening."""
    print("=== DEBUGGING SINGLE CELL TRAINING ===")
    
    # Load model
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Collect small dataset
    env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    grid_size = 5
    horizon = 5
    map_bounds = (-2, -2, 2, 2)
    
    samples = []
    
    # Collect data from 2 trajectories
    for traj_id in range(2):
        print(f"Collecting trajectory {traj_id}...")
        
        done = False
        obs = env.reset(seed=traj_id * 100)
        agent.reset()
        
        trajectory_data = []
        
        for step_id in range(30):
            if done:
                break
                
            current_pos = env.agent_pos[:2].copy()
            
            obs_features = obs.get('features', np.zeros(80))
            goal_encoding = np.zeros(10)
            goal_encoding[0] = 1.0  # blue
            raw_features = np.concatenate([obs_features, goal_encoding])
            
            trajectory_data.append({
                'position': current_pos,
                'features': raw_features
            })
            
            action = agent.get_action(obs, {}, deterministic=True).flatten()
            obs, _, done, info = env.step(action)
        
        # Create samples from this trajectory
        if len(trajectory_data) >= horizon + 2:
            for i in range(len(trajectory_data) - horizon):
                current_data = trajectory_data[i]
                
                # Get future positions
                future_positions = [trajectory_data[i+j]['position'] for j in range(1, horizon+1)]
                
                # Create grid visits set
                grid_visits = set()
                for future_pos in future_positions:
                    gx, gy = position_to_grid_cell(future_pos, grid_size, map_bounds)
                    grid_visits.add((gx, gy))
                
                samples.append({
                    'features': current_data['features'],
                    'grid_visits': grid_visits,
                    'trajectory_id': traj_id,
                    'step_id': i
                })
    
    env.close()
    print(f"Collected {len(samples)} samples")
    
    # Focus on one cell that has some positive samples
    target_cell = None
    for i in range(grid_size):
        for j in range(grid_size):
            labels = [1 if (i, j) in s['grid_visits'] else 0 for s in samples]
            positive_count = sum(labels)
            positive_ratio = positive_count / len(labels)
            
            if 0.1 <= positive_ratio <= 0.9 and positive_count >= 5:  # Good balance and enough samples
                target_cell = (i, j)
                print(f"Found target cell ({i},{j}) with {positive_count}/{len(labels)} positive samples ({positive_ratio:.3f})")
                break
        if target_cell:
            break
    
    if not target_cell:
        print("❌ No suitable target cell found!")
        return
    
    # Extract features and labels for target cell
    features = np.array([s['features'] for s in samples])
    labels = [1 if target_cell in s['grid_visits'] else 0 for s in samples]
    
    print(f"\n🔍 DEBUGGING CELL {target_cell}:")
    print(f"Features shape: {features.shape}")
    print(f"Labels: {sum(labels)} positive, {len(labels)-sum(labels)} negative")
    
    # Apply PCA
    pca = PCA(n_components=min(5, features.shape[1]))
    features_pca = pca.fit_transform(features)
    print(f"After PCA: {features_pca.shape}")
    
    # Split data
    n_train = int(0.7 * len(samples))
    indices = list(range(len(samples)))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = features_pca[train_indices]
    X_test = features_pca[test_indices]
    y_train = [labels[i] for i in train_indices]
    y_test = [labels[i] for i in test_indices]
    
    print(f"Train: {len(X_train)} samples, {sum(y_train)} positive")
    print(f"Test: {len(X_test)} samples, {sum(y_test)} positive")
    
    # Train classifier
    print(f"\n🧠 TRAINING CLASSIFIER:")
    
    # Try both balanced and unbalanced
    for name, class_weight in [("Unbalanced", None), ("Balanced", 'balanced')]:
        print(f"\n--- {name} Classifier ---")
        
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
        clf.fit(X_train, y_train)
        
        # Predictions and probabilities
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        train_proba = clf.predict_proba(X_train)[:, 1]
        test_proba = clf.predict_proba(X_test)[:, 1]
        
        # Training performance
        train_acc = accuracy_score(y_train, train_pred)
        train_bal_acc = balanced_accuracy_score(y_train, train_pred)
        
        print(f"Training accuracy: {train_acc:.3f}")
        print(f"Training balanced accuracy: {train_bal_acc:.3f}")
        
        # Test performance
        if len(set(y_test)) > 1:
            test_acc = accuracy_score(y_test, test_pred)
            test_bal_acc = balanced_accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, zero_division=0.0)
            print(f"Test accuracy: {test_acc:.3f}")
            print(f"Test balanced accuracy: {test_bal_acc:.3f}")
            print(f"Test F1: {test_f1:.3f}")
        
        # CRITICAL: Check predictions for positive training samples
        positive_train_indices = [i for i, label in enumerate(y_train) if label == 1]
        if len(positive_train_indices) > 0:
            positive_train_probas = [train_proba[i] for i in positive_train_indices]
            avg_positive_proba = np.mean(positive_train_probas)
            min_positive_proba = np.min(positive_train_probas)
            max_positive_proba = np.max(positive_train_probas)
            
            print(f"🚨 KEY DIAGNOSTIC - Probabilities for POSITIVE training samples:")
            print(f"   Average: {avg_positive_proba:.3f}")
            print(f"   Min: {min_positive_proba:.3f}")
            print(f"   Max: {max_positive_proba:.3f}")
            print(f"   Expected: Should be close to 1.0 if learning properly!")
            
            if avg_positive_proba < 0.7:
                print(f"   🚨 PROBLEM: Classifier can't even fit training data!")
                print(f"   🚨 This suggests features have no predictive power")
        
        # Check predictions for negative training samples
        negative_train_indices = [i for i, label in enumerate(y_train) if label == 0]
        if len(negative_train_indices) > 0:
            negative_train_probas = [train_proba[i] for i in negative_train_indices]
            avg_negative_proba = np.mean(negative_train_probas)
            
            print(f"Probabilities for NEGATIVE training samples:")
            print(f"   Average: {avg_negative_proba:.3f} (should be close to 0.0)")
        
        # Feature importance
        if hasattr(clf, 'coef_'):
            feature_weights = clf.coef_[0]
            print(f"Feature weights (first 5): {feature_weights[:5]}")
            print(f"Max absolute weight: {np.max(np.abs(feature_weights)):.6f}")
            
            if np.max(np.abs(feature_weights)) < 0.01:
                print(f"🚨 PROBLEM: Feature weights are tiny - features likely uninformative!")

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    debug_single_cell_training()

if __name__ == "__main__":
    main() 