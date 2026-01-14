#!/usr/bin/env python3
"""
Diagnostic script to understand what should be perfectly decodable from inputs.
Tests fundamental assumptions about the observation space.
"""

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from ltl import FixedSampler
from envs import make_env
from sequence.search import ExhaustiveSearch
from model.agent import Agent

def test_perfect_decodability():
    """Test what should be perfectly decodable from observations."""
    
    print("=== DIAGNOSING OBSERVATION SPACE DECODABILITY ===")
    
    # Load model and create environment
    ENV = "PointLtl2-v0"
    EXP = "big_test"
    SEED = 0
    
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                 propositions=props, verbose=False)
    
    # Collect observations
    observations = []
    env.reset(seed=42)
    
    for step in range(100):
        obs = env.get_observation()
        features = obs['features']  # 80D vector
        observations.append(features)
        
        # Take random action
        action = env.action_space.sample()
        try:
            result = env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
                
            if done:
                env.reset()
        except:
            env.reset()
    
    observations = np.array(observations)
    print(f"Collected {len(observations)} observations of shape {observations.shape}")
    
    # Test 1: Perfect Self-Recovery (should be R² = 1.0)
    print("\n1. PERFECT SELF-RECOVERY TEST:")
    test_perfect_self_recovery(observations)
    
    # Test 2: Slice Recovery (should be R² = 1.0 for any slice)
    print("\n2. SLICE RECOVERY TEST:")
    test_slice_recovery(observations)
    
    # Test 3: Individual Element Recovery (should be R² = 1.0)
    print("\n3. INDIVIDUAL ELEMENT RECOVERY TEST:")
    test_element_recovery(observations)
    
    # Test 4: Temporal Consistency
    print("\n4. TEMPORAL CONSISTENCY TEST:")
    test_temporal_consistency(observations)
    
    env.close()

def test_perfect_self_recovery(observations):
    """Test if we can perfectly recover observations from themselves."""
    X = observations[:-1]  # Input
    y = observations[1:]   # Target (next observation)
    
    # Split data
    split = len(X) // 2
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    try:
        # Test recovering the SAME observation (should be perfect)
        X_same = observations[:80]
        y_same = observations[:80]  # Same as input
        
        probe = Ridge(alpha=0.01)
        probe.fit(X_same, y_same)
        y_pred = probe.predict(X_same)
        r2 = r2_score(y_same, y_pred)
        
        print(f"  Self-recovery R²: {r2:.6f} (should be ~1.0)")
        
        if r2 < 0.99:
            print(f"  🚨 ANOMALY: Self-recovery should be perfect!")
            
    except Exception as e:
        print(f"  ❌ Self-recovery failed: {e}")

def test_slice_recovery(observations):
    """Test if we can perfectly recover slices from full observations."""
    X = observations  # Full 80D observations
    
    # Test recovering different slices
    slices_to_test = [
        (0, 5, "First 5 elements"),
        (10, 20, "Elements 10-19"), 
        (16, 32, "Zone lidar approx (16-31)"),
        (32, 36, "Wall sensor guess (32-35)"),
        (52, 58, "Joint positions guess (52-57)"),
        (75, 80, "Last 5 elements")
    ]
    
    for start, end, name in slices_to_test:
        try:
            y = X[:, start:end]  # Target slice
            
            probe = Ridge(alpha=0.01)
            probe.fit(X, y)
            y_pred = probe.predict(X)
            r2 = r2_score(y, y_pred)
            
            print(f"  {name}: R² = {r2:.4f} (should be ~1.0)")
            
            if r2 < 0.99:
                print(f"    🤔 Not perfect - might not be directly encoded")
                
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")

def test_element_recovery(observations):
    """Test recovery of individual elements."""
    X = observations  # Full 80D observations
    
    # Test a few individual elements
    elements_to_test = [0, 10, 20, 30, 40, 50, 60, 70, 79]
    
    perfect_elements = []
    for element_idx in elements_to_test:
        try:
            y = X[:, element_idx]  # Single element
            
            probe = Ridge(alpha=0.01)
            probe.fit(X, y.reshape(-1, 1))
            y_pred = probe.predict(X).flatten()
            r2 = r2_score(y, y_pred)
            
            print(f"  Element {element_idx}: R² = {r2:.4f}")
            
            if r2 > 0.99:
                perfect_elements.append(element_idx)
                
        except Exception as e:
            print(f"  ❌ Element {element_idx} failed: {e}")
    
    print(f"  Perfect elements (R² > 0.99): {perfect_elements}")
    if len(perfect_elements) < len(elements_to_test):
        print(f"  🤔 Some elements not perfectly recoverable - observation preprocessing?")

def test_temporal_consistency(observations):
    """Test if observations change in expected ways over time."""
    print(f"  Observation mean: {observations.mean():.4f}")
    print(f"  Observation std: {observations.std():.4f}")
    print(f"  Min value: {observations.min():.4f}")
    print(f"  Max value: {observations.max():.4f}")
    
    # Check for constant elements
    constant_elements = []
    for i in range(observations.shape[1]):
        if observations[:, i].std() < 1e-6:
            constant_elements.append(i)
    
    print(f"  Constant elements: {constant_elements[:10]}...")  # Show first 10
    print(f"  Total constant elements: {len(constant_elements)}/80")
    
    # Check for binary elements  
    binary_elements = []
    for i in range(observations.shape[1]):
        unique_vals = np.unique(observations[:, i])
        if len(unique_vals) <= 2:
            binary_elements.append(i)
    
    print(f"  Binary elements: {binary_elements[:10]}...")  # Show first 10
    print(f"  Total binary elements: {len(binary_elements)}/80")

if __name__ == "__main__":
    test_perfect_decodability() 