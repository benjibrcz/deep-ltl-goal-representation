#!/usr/bin/env python3
"""
Comprehensive Agent Observation Space Audit
===========================================
This script systematically analyzes every component of the agent's observation
to understand exactly what information it receives about:
- World state (zones, walls, physics)
- Goals and objectives (LTL formulas, LDBA states)
- Internal state (velocity, orientation, etc.)
- Task context (propositions, automata states)
"""

import os, sys
sys.path.insert(0, "src")

import numpy as np
from ltl import FixedSampler
from envs import make_env
from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from sequence.search import ExhaustiveSearch
from model.agent import Agent

def analyze_observation_component(name, data, step_num=0):
    """Analyze a single component of the observation"""
    info = {
        'name': name,
        'type': type(data).__name__,
        'value': data
    }
    
    if hasattr(data, 'shape'):
        info['shape'] = data.shape
        info['dtype'] = data.dtype
        if data.size > 0:
            info['min'] = np.min(data)
            info['max'] = np.max(data)
            info['mean'] = np.mean(data)
            info['std'] = np.std(data)
            info['unique_values'] = len(np.unique(data))
            if info['unique_values'] <= 10:
                info['unique_list'] = list(np.unique(data))
    elif hasattr(data, '__len__') and not isinstance(data, str):
        info['length'] = len(data)
        if len(data) <= 10:
            info['contents'] = list(data)
    
    return info

def interpret_features_array(features):
    """Try to interpret the 80D features array based on patterns"""
    interpretations = {}
    
    # Zone lidars (likely first 64 dimensions: 4 colors × 16 bins)
    if len(features) >= 64:
        interpretations['blue_zones'] = {
            'indices': '0:16',
            'data': features[0:16],
            'description': 'Blue zone lidar (16 angular bins)',
            'active_bins': np.sum(features[0:16] != 0),
            'max_reading': np.max(features[0:16]),
            'interpretation': 'Distance to blue zones in each angular direction'
        }
        
        interpretations['green_zones'] = {
            'indices': '16:32', 
            'data': features[16:32],
            'description': 'Green zone lidar (16 angular bins)',
            'active_bins': np.sum(features[16:32] != 0),
            'max_reading': np.max(features[16:32])
        }
        
        interpretations['yellow_zones'] = {
            'indices': '32:48',
            'data': features[32:48], 
            'description': 'Yellow zone lidar (16 angular bins)',
            'active_bins': np.sum(features[32:48] != 0),
            'max_reading': np.max(features[32:48])
        }
        
        interpretations['magenta_zones'] = {
            'indices': '48:64',
            'data': features[48:64],
            'description': 'Magenta zone lidar (16 angular bins)', 
            'active_bins': np.sum(features[48:64] != 0),
            'max_reading': np.max(features[48:64])
        }
    
    # Wall sensors (likely next 16 dimensions)
    if len(features) >= 80:
        interpretations['wall_sensors'] = {
            'indices': '64:80',
            'data': features[64:80],
            'description': 'Wall proximity sensors (16 angular bins)',
            'active_bins': np.sum(features[64:80] != 0),
            'min_distance': np.min(features[64:80][features[64:80] > 0]) if np.any(features[64:80] > 0) else 'No walls detected',
            'interpretation': 'Distance to walls in each angular direction'
        }
    
    # Look for special patterns
    interpretations['special_patterns'] = {}
    
    # Constant values (might be gravity, bias terms, etc.)
    constants = []
    for i, val in enumerate(features):
        if abs(val - 9.81) < 0.01:  # Gravity
            constants.append(f"Position {i}: {val:.3f} (likely gravity)")
        elif abs(val) > 5:  # Large constant values
            constants.append(f"Position {i}: {val:.3f} (large constant)")
    
    if constants:
        interpretations['special_patterns']['constants'] = constants
    
    # Zero regions (inactive sensors)
    zero_regions = []
    in_zero_region = False
    region_start: int = 0
    
    for i, val in enumerate(features):
        if val == 0:
            if not in_zero_region:
                region_start = i
                in_zero_region = True
        else:
            if in_zero_region:
                region_end = i - 1
                if region_end > region_start + 2:  # Only report regions of 3+ zeros
                    zero_regions.append(f"{region_start}:{region_end+1} ({region_end-region_start+1} zeros)")
                in_zero_region = False
    
    if in_zero_region:
        region_end = len(features) - 1
        if region_end > region_start + 2:
            zero_regions.append(f"{region_start}:{region_end+1} ({region_end-region_start+1} zeros)")
    
    if zero_regions:
        interpretations['special_patterns']['zero_regions'] = zero_regions
    
    return interpretations

def audit_environment_setup():
    """Audit the environment setup and configuration"""
    ENV = "PointLtl2-v0"
    EXP = "big_test" 
    SEED = 0
    formula = "FG blue"
    sampler = FixedSampler.partial(formula)
    
    print("=" * 80)
    print("AGENT OBSERVATION SPACE AUDIT")
    print("=" * 80)
    print(f"Environment: {ENV}")
    print(f"Experiment: {EXP}")
    print(f"LTL Formula: {formula}")
    print(f"Seed: {SEED}")
    
    # Set up model and environment
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    
    env = make_env(ENV, sampler, sequence=False, render_mode=None)
    model = build_model(env, status, cfg).eval()
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    print(f"Environment propositions: {props}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return env, agent

def main():
    env, agent = audit_environment_setup()
    
    print("\n" + "=" * 80)
    print("OBSERVATION STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Collect observations from multiple steps to see variation
    observations = []
    
    for step in range(10):
        if step == 0:
            obs = env.reset(seed=42 + step)
            agent.reset()
        else:
            # Take action to generate new observation
            action = agent.get_action(obs, {}, deterministic=True).flatten()
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset(seed=42 + step)
                agent.reset()
        
        observations.append(obs.copy())
    
    # Analyze the structure of the first observation in detail
    print(f"\nStep 0 - Detailed Observation Analysis:")
    print("-" * 50)
    
    obs = observations[0]
    for key, value in obs.items():
        info = analyze_observation_component(key, value)
        print(f"\n[{key}]")
        print(f"  Type: {info['type']}")
        if 'shape' in info:
            print(f"  Shape: {info['shape']}")
            print(f"  Data type: {info['dtype']}")
            print(f"  Range: [{info['min']:.6f}, {info['max']:.6f}]")
            print(f"  Mean: {info['mean']:.6f}, Std: {info['std']:.6f}")
            print(f"  Unique values: {info['unique_values']}")
            if 'unique_list' in info:
                print(f"  Values: {info['unique_list']}")
        elif 'length' in info:
            print(f"  Length: {info['length']}")
            if 'contents' in info:
                print(f"  Contents: {info['contents']}")
        else:
            print(f"  Value: {info['value']}")
    
    # Deep dive into the features array
    if 'features' in obs:
        print(f"\n" + "=" * 80)
        print("FEATURES ARRAY DEEP DIVE")
        print("=" * 80)
        
        features = obs['features']
        interpretations = interpret_features_array(features)
        
        print(f"Features array shape: {features.shape}")
        print(f"Total dimensions: {len(features)}")
        print(f"Non-zero elements: {np.sum(features != 0)}")
        print(f"Data range: [{np.min(features):.6f}, {np.max(features):.6f}]")
        
        for category, details in interpretations.items():
            if category == 'special_patterns':
                print(f"\n[SPECIAL PATTERNS]")
                for pattern_type, pattern_list in details.items():
                    print(f"  {pattern_type}:")
                    for pattern in pattern_list:
                        print(f"    {pattern}")
            else:
                print(f"\n[{category.upper()}]")
                print(f"  Indices: {details['indices']}")
                print(f"  Description: {details['description']}")
                print(f"  Active bins: {details['active_bins']}/16")
                if 'max_reading' in details:
                    print(f"  Max reading: {details['max_reading']:.6f}")
                if 'min_distance' in details:
                    print(f"  Min distance: {details['min_distance']}")
                if 'interpretation' in details:
                    print(f"  Meaning: {details['interpretation']}")
                
                # Show first few values
                data = details['data']
                non_zero = data[data != 0]
                if len(non_zero) > 0:
                    print(f"  Non-zero values: {non_zero[:5]}{'...' if len(non_zero) > 5 else ''}")
                else:
                    print(f"  All values are zero")
    
    # Analyze variation across time steps
    print(f"\n" + "=" * 80)
    print("TEMPORAL VARIATION ANALYSIS")
    print("=" * 80)
    
    print("Analyzing how observations change over time steps...")
    
    # Track which components change
    changing_components = {}
    
    for key in observations[0].keys():
        values_over_time = []
        for obs in observations:
            if key == 'features':
                # For features, track summary statistics
                features = obs[key]
                values_over_time.append({
                    'mean': np.mean(features),
                    'std': np.std(features),
                    'non_zero': np.sum(features != 0),
                    'max': np.max(features),
                    'min': np.min(features)
                })
            elif hasattr(obs[key], 'shape'):
                values_over_time.append(np.mean(obs[key]) if obs[key].size > 0 else 0)
            else:
                values_over_time.append(str(obs[key]))
        
        # Check if this component varies
        if key == 'features':
            varies = len(set(v['non_zero'] for v in values_over_time)) > 1
            changing_components[key] = {
                'varies': varies,
                'summary': f"Non-zero elements range: {min(v['non_zero'] for v in values_over_time)}-{max(v['non_zero'] for v in values_over_time)}"
            }
        else:
            varies = len(set(values_over_time)) > 1
            changing_components[key] = {
                'varies': varies,
                'summary': f"Unique values: {len(set(values_over_time))}"
            }
    
    print("\nComponent variation over 10 time steps:")
    for key, info in changing_components.items():
        status = "DYNAMIC" if info['varies'] else "STATIC"
        print(f"  {key:20s}: {status:8s} - {info['summary']}")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("The agent receives the following information:")
    print("\n1. WORLD STATE:")
    print("   • Zone proximity sensors (4 colors × 16 directions = 64D)")
    print("   • Wall proximity sensors (16 directions = 16D)")
    print("   • Physics/IMU data (gravity, velocity, orientation)")
    
    print("\n2. TASK INFORMATION:")
    print(f"   • LTL goal formula: {obs['goal'] if 'goal' in obs else 'Unknown'}")
    print(f"   • Current LDBA automaton state: {obs['ldba_state'] if 'ldba_state' in obs else 'Unknown'}")
    print(f"   • Active propositions: {obs['propositions'] if 'propositions' in obs else 'Unknown'}")
    print("   • LDBA automaton object (for state transitions)")
    
    print("\n3. SENSOR MODALITIES:")
    print("   • Directional distance sensors (zones and walls)")
    print("   • Inertial measurement unit (IMU) data")
    print("   • Task-specific semantic information (LTL states)")
    
    print("\n4. INFORMATION ENCODING:")
    print("   • Total observation dimensionality: 80D + semantic components")
    print("   • Spatial sensors use 16-bin angular discretization")
    print("   • Some components are static (goal), others highly dynamic (sensors)")
    print("   • Rich directional information preserved (not just scalar distances)")
    
    env.close()
    print(f"\nAudit complete! 🎯")

if __name__ == "__main__":
    main() 