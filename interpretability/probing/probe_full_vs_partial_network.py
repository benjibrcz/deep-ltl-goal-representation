#!/usr/bin/env python3
"""
Full vs Partial Network Grid Prediction Comparison

Compare spatial prediction performance between:
1. Raw network inputs (obs.features + goal encoding)
2. Full network computation (using actual agent actions)

This tests whether spatial planning emerges from the complete network processing.
"""

import os
import sys
import random
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score

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

def collect_both_representations(model, world_ids, goal="FG blue", grid_size=5, 
                                n_rollouts_per_world=6, max_steps=50, 
                                horizon=5, map_bounds=(-2, -2, 2, 2)):
    """Collect BOTH raw inputs and full network outputs for comparison."""
    print(f"🔍 COLLECTING BOTH REPRESENTATIONS")
    print(f"Goal: {goal}, Grid: {grid_size}x{grid_size}, Horizon: {horizon} steps")
    
    env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    samples = []
    
    for world_id in world_ids:
        print(f"Processing world {world_id}...")
        for rollout_id in trange(n_rollouts_per_world, desc=f"Rollouts"):
            trajectory_data = []
            
            done = False
            obs = env.reset(seed=world_id + rollout_id * 1000)
            agent.reset()
            
            for step_id in range(max_steps):
                if done:
                    break
                    
                current_pos = env.agent_pos[:2].copy()
                
                # 1. RAW INPUTS (what we were testing before)
                obs_features = obs.get('features', np.zeros(80))
                goal_encoding = np.zeros(10)
                if 'blue' in goal.lower():
                    goal_encoding[0] = 1.0
                elif 'red' in goal.lower():
                    goal_encoding[1] = 1.0
                elif 'green' in goal.lower():
                    goal_encoding[2] = 1.0
                
                raw_features = np.concatenate([obs_features, goal_encoding])
                
                # 2. FULL NETWORK COMPUTATION (agent's actual action)
                action = agent.get_action(obs, {}, deterministic=True).flatten()
                
                trajectory_data.append({
                    'position': current_pos,
                    'raw_features': raw_features,      # 90D: raw obs + goal encoding
                    'network_action': action,          # 2D: full network output
                    'step_id': step_id
                })
                
                obs, _, done, info = env.step(action)
            
            # Create samples for this horizon
            if len(trajectory_data) < horizon + 2:
                continue
                
            for i in range(len(trajectory_data) - horizon):
                current_data = trajectory_data[i]
                
                # Get future positions
                future_positions = []
                for j in range(1, horizon + 1):
                    if i + j < len(trajectory_data):
                        future_positions.append(trajectory_data[i + j]['position'])
                
                if len(future_positions) == 0:
                    continue
                
                # Create grid visits set
                grid_visits = set()
                for future_pos in future_positions:
                    gx, gy = position_to_grid_cell(future_pos, grid_size, map_bounds)
                    grid_visits.add((gx, gy))
                
                sample = {
                    'raw_features': current_data['raw_features'],
                    'network_action': current_data['network_action'],
                    'grid_visits': grid_visits,
                    'position': current_data['position']
                }
                samples.append(sample)
    
    env.close()
    print(f"Collected {len(samples)} samples")
    return samples

def test_spatial_prediction(samples, representation_type, grid_size, test_cells=None):
    """Test spatial prediction with specified representation."""
    print(f"\n🧠 TESTING {representation_type.upper()} REPRESENTATION")
    
    if test_cells is None:
        # Test a few representative cells
        test_cells = [(1, 1), (2, 2), (3, 1), (1, 3)]
    
    # Extract features based on representation type
    if representation_type == "raw_inputs":
        features = np.array([s['raw_features'] for s in samples])
        print(f"Raw features shape: {features.shape}")
    elif representation_type == "full_network":
        features = np.array([s['network_action'] for s in samples])
        print(f"Network action shape: {features.shape}")
    else:
        raise ValueError(f"Unknown representation: {representation_type}")
    
    # Apply PCA if needed
    if features.shape[1] > 10:
        pca = PCA(n_components=min(10, features.shape[1]))
        features_pca = pca.fit_transform(features)
        print(f"PCA: {features.shape[1]} → {features_pca.shape[1]} dimensions")
    else:
        features_pca = features
    
    # Train/test split
    n_train = int(0.7 * len(samples))
    indices = list(range(len(samples)))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = features_pca[train_indices]
    X_test = features_pca[test_indices]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Test prediction for each cell
    results = {}
    meaningful_predictions = 0
    
    for cell_x, cell_y in test_cells:
        # Create labels
        y_all = [1 if (cell_x, cell_y) in s['grid_visits'] else 0 for s in samples]
        y_train = [y_all[i] for i in train_indices]
        y_test = [y_all[i] for i in test_indices]
        
        positive_ratio = sum(y_all) / len(y_all)
        
        print(f"\n  Cell ({cell_x}, {cell_y}):")
        print(f"    Positive samples: {sum(y_all)}/{len(y_all)} ({positive_ratio*100:.1f}%)")
        
        # Check if we can train
        if len(set(y_train)) < 2 or positive_ratio < 0.02:
            print(f"    ❌ Insufficient data for training")
            results[(cell_x, cell_y)] = {
                'balanced_accuracy': 0.5,
                'status': 'insufficient_data',
                'positive_ratio': positive_ratio
            }
            continue
        
        meaningful_predictions += 1
        
        try:
            # Train balanced classifier
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            
            print(f"    ✅ Balanced Accuracy: {balanced_acc:.3f}")
            
            results[(cell_x, cell_y)] = {
                'balanced_accuracy': balanced_acc,
                'status': 'success',
                'positive_ratio': positive_ratio
            }
            
        except Exception as e:
            print(f"    ❌ Error: {str(e)[:50]}")
            results[(cell_x, cell_y)] = {
                'balanced_accuracy': 0.5,
                'status': 'error',
                'positive_ratio': positive_ratio
            }
    
    # Summary statistics
    successful_results = [r for r in results.values() if r['status'] == 'success']
    if successful_results:
        avg_accuracy = np.mean([r['balanced_accuracy'] for r in successful_results])
        max_accuracy = max([r['balanced_accuracy'] for r in successful_results])
        
        print(f"\n  📊 SUMMARY:")
        print(f"    Meaningful predictions: {meaningful_predictions}/{len(test_cells)}")
        print(f"    Average balanced accuracy: {avg_accuracy:.3f}")
        print(f"    Best balanced accuracy: {max_accuracy:.3f}")
        
        return {
            'representation': representation_type,
            'meaningful_predictions': meaningful_predictions,
            'avg_accuracy': avg_accuracy,
            'max_accuracy': max_accuracy,
            'individual_results': results
        }
    else:
        print(f"\n  📊 SUMMARY: No successful predictions")
        return {
            'representation': representation_type,
            'meaningful_predictions': 0,
            'avg_accuracy': 0.5,
            'max_accuracy': 0.5,
            'individual_results': results
        }

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("=== FULL vs PARTIAL NETWORK GRID PREDICTION COMPARISON ===")
    print("🔍 Testing whether spatial planning emerges from complete network processing")
    
    # Load model
    print("\nLoading model...")
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Collect data with both representations
    world_ids = list(range(3))  # Small sample for speed
    samples = collect_both_representations(
        model, world_ids, "FG blue", grid_size=5, 
        n_rollouts_per_world=6, max_steps=40, horizon=5
    )
    
    if len(samples) == 0:
        print("❌ No samples collected!")
        return
    
    # Test both representations
    raw_results = test_spatial_prediction(samples, "raw_inputs", 5)
    network_results = test_spatial_prediction(samples, "full_network", 5)
    
    # Compare results
    print(f"\n=== 🏆 COMPARISON RESULTS ===")
    print(f"Raw Inputs (obs.features + goal encoding):")
    print(f"  Meaningful predictions: {raw_results['meaningful_predictions']}")
    print(f"  Average accuracy: {raw_results['avg_accuracy']:.3f}")
    print(f"  Best accuracy: {raw_results['max_accuracy']:.3f}")
    
    print(f"\nFull Network (complete computation → actions):")
    print(f"  Meaningful predictions: {network_results['meaningful_predictions']}")
    print(f"  Average accuracy: {network_results['avg_accuracy']:.3f}")
    print(f"  Best accuracy: {network_results['max_accuracy']:.3f}")
    
    # Determine winner
    raw_score = raw_results['avg_accuracy']
    network_score = network_results['avg_accuracy']
    
    print(f"\n🎯 CONCLUSION:")
    if network_score > raw_score + 0.05:
        print(f"🚀 FULL NETWORK shows better spatial prediction!")
        print(f"   The complete processing pipeline adds spatial planning capability")
        print(f"   Improvement: {network_score - raw_score:.3f}")
    elif raw_score > network_score + 0.05:
        print(f"🤔 RAW INPUTS actually perform better")
        print(f"   Network processing may remove spatial information")
        print(f"   Difference: {raw_score - network_score:.3f}")
    else:
        print(f"❌ NO significant difference between representations")
        print(f"   Neither raw inputs nor full network show spatial planning")
        print(f"   Both perform near chance level (~0.5)")
    
    print(f"\n💡 INTERPRETATION:")
    if max(raw_score, network_score) > 0.6:
        print(f"Strong spatial prediction found - network has planning capability")
    elif max(raw_score, network_score) > 0.55:
        print(f"Weak spatial prediction - limited planning capability")
    else:
        print(f"No spatial prediction - confirms reactive navigation hypothesis")
        print(f"Spatial success through adaptive responses, not predictable planning")

if __name__ == "__main__":
    main() 