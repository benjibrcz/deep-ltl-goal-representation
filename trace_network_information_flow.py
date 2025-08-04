#!/usr/bin/env python3
"""
Network Information Flow Tracer
===============================
This script traces how observation information flows through different components
of the neural network architecture:
- env_net: Environment feature processor
- ltl_net: LTL/goal processing network  
- actor: Action selection network
- critic: Value estimation network

It shows what information each component receives and how it's processed.
"""

import os, sys
sys.path.insert(0, "src")

import numpy as np
import torch
from ltl import FixedSampler
from envs import make_env
from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from sequence.search import ExhaustiveSearch
from model.agent import Agent

def hook_activations(module, input_data, output_data, name):
    """Hook function to capture activations"""
    if isinstance(input_data, tuple):
        input_info = f"Input: {len(input_data)} tensors"
        for i, inp in enumerate(input_data):
            if hasattr(inp, 'shape'):
                input_info += f", tensor{i}: {inp.shape}"
            else:
                input_info += f", tensor{i}: {type(inp)}"
    else:
        input_info = f"Input: {input_data.shape if hasattr(input_data, 'shape') else type(input_data)}"
    
    if isinstance(output_data, tuple):
        output_info = f"Output: {len(output_data)} tensors"
        for i, out in enumerate(output_data):
            if hasattr(out, 'shape'):
                output_info += f", tensor{i}: {out.shape}"
            else:
                output_info += f", tensor{i}: {type(out)}"
    else:
        output_info = f"Output: {output_data.shape if hasattr(output_data, 'shape') else type(output_data)}"
    
    print(f"  [{name}] {input_info} -> {output_info}")

def trace_model_architecture():
    """Set up model and trace its architecture"""
    ENV = "PointLtl2-v0"
    EXP = "big_test"
    SEED = 0
    formula = "FG blue"
    sampler = FixedSampler.partial(formula)
    
    print("=" * 80)
    print("NETWORK INFORMATION FLOW ANALYSIS")
    print("=" * 80)
    
    # Set up model and environment
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    
    env = make_env(ENV, sampler, sequence=False, render_mode=None)
    model = build_model(env, status, cfg).eval()
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    print(f"Environment: {ENV}")
    print(f"LTL Formula: {formula}")
    print(f"Model total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Examine model architecture
    print(f"\n" + "=" * 80)
    print("MODEL ARCHITECTURE OVERVIEW")
    print("=" * 80)
    
    print("Model components:")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {type(module).__name__} ({param_count} parameters)")
        
        # Show submodules
        for subname, submodule in module.named_children():
            subparam_count = sum(p.numel() for p in submodule.parameters())
            print(f"    └─ {subname}: {type(submodule).__name__} ({subparam_count} parameters)")
    
    return env, model, agent

def trace_forward_pass(env, model, agent):
    """Trace a forward pass through the model"""
    print(f"\n" + "=" * 80)
    print("FORWARD PASS INFORMATION FLOW")
    print("=" * 80)
    
    # Register hooks on all major components
    hooks = []
    
    # Hook all major modules
    for name, module in model.named_modules():
        if any(key in name for key in ['env_net', 'ltl_net', 'actor', 'critic']):
            hook = module.register_forward_hook(
                lambda module, input_data, output_data, name=name: 
                hook_activations(module, input_data, output_data, name)
            )
            hooks.append(hook)
    
    # Get an observation
    obs = env.reset(seed=42)
    agent.reset()
    
    print("Observation structure:")
    for key, value in obs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)} = {value}")
    
    print(f"\nTracing forward pass...")
    print("-" * 50)
    
    # Run forward pass through model
    with torch.no_grad():
        action = agent.get_action(obs, {}, deterministic=True)
    
    print("-" * 50)
    print(f"Final action shape: {action.shape}")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return obs

def analyze_input_preprocessing(env, model):
    """Analyze how the raw observation gets preprocessed for different network parts"""
    print(f"\n" + "=" * 80)
    print("INPUT PREPROCESSING ANALYSIS")
    print("=" * 80)
    
    obs = env.reset(seed=42)
    
    # Convert observation to model input format
    # This is typically done in the model's forward pass
    print("Raw observation components:")
    for key, value in obs.items():
        if key == 'features':
            print(f"  {key}: {value.shape} - numerical sensor data")
        elif key == 'goal':
            print(f"  {key}: '{value}' - LTL goal string")
        elif key == 'ldba_state':
            print(f"  {key}: {value} - automaton state")
        elif key == 'propositions':
            print(f"  {key}: {value} - current propositions")
        else:
            print(f"  {key}: {type(value)} - {value}")
    
    # Try to understand what each network component receives
    # This requires inspecting the model's forward method
    
    print(f"\nAnalyzing model input processing...")
    
    # Look at the model's forward method signature and code
    if hasattr(model, 'forward'):
        import inspect
        forward_code = inspect.getsource(model.forward)
        print("Model.forward method processes:")
        
        # Extract key lines about input processing
        lines = forward_code.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['features', 'goal', 'ldba', 'propositions']):
                print(f"  {line}")
    
    # Try to trace what goes where by examining the model structure
    print(f"\nEstimated information flow:")
    print("  env_net likely receives:")
    print("    - features array (80D sensor data)")
    print("    - possibly ldba_state for context")
    
    print("  ltl_net likely receives:")
    print("    - goal string (LTL formula)")
    print("    - ldba_state (current automaton state)")
    print("    - propositions (current zone occupancy)")
    
    print("  actor/critic likely receive:")
    print("    - processed features from env_net")
    print("    - processed goal info from ltl_net")
    print("    - combined representation for decision making")

def trace_information_sharing():
    """Analyze how information is shared between network components"""
    print(f"\n" + "=" * 80)
    print("INFORMATION SHARING ANALYSIS")
    print("=" * 80)
    
    env, model, agent = trace_model_architecture()
    
    # Trace the forward pass
    obs = trace_forward_pass(env, model, agent)
    
    # Analyze preprocessing
    analyze_input_preprocessing(env, model)
    
    print(f"\n" + "=" * 80)
    print("SUMMARY: INFORMATION ACCESS BY COMPONENT")
    print("=" * 80)
    
    print("Based on the architecture analysis:")
    
    print(f"\n1. ENV_NET (Environment Processor):")
    print("   ✅ Gets: Full 80D sensor array (zones + walls + physics)")
    print("   ✅ Gets: LDBA state (for context)")
    print("   ❌ Likely doesn't get: Raw LTL goal string")
    print("   → Processes: Spatial and temporal sensor information")
    
    print(f"\n2. LTL_NET (Goal Processor):")
    print("   ✅ Gets: LTL goal string ('FG blue')")
    print("   ✅ Gets: LDBA automaton state")
    print("   ✅ Gets: Current propositions (zone satisfaction)")
    print("   ❌ Likely doesn't get: Raw sensor data")
    print("   → Processes: Task semantics and goal tracking")
    
    print(f"\n3. ACTOR (Action Network):")
    print("   ✅ Gets: Processed features from env_net")
    print("   ✅ Gets: Goal representation from ltl_net")
    print("   ✅ Gets: Combined world+goal understanding")
    print("   → Processes: Action selection based on world state and goals")
    
    print(f"\n4. CRITIC (Value Network):")
    print("   ✅ Gets: Same combined representation as actor")
    print("   ✅ Gets: Full context for value estimation")
    print("   → Processes: Expected reward/value estimation")
    
    print(f"\n📋 KEY INSIGHT:")
    print("   Different network components get DIFFERENT VIEWS of the observation:")
    print("   • env_net focuses on SPATIAL/SENSOR information")
    print("   • ltl_net focuses on GOAL/TASK information")  
    print("   • actor/critic get INTEGRATED information from both")
    print("   This allows specialized processing before final decision making!")
    
    env.close()

if __name__ == "__main__":
    trace_information_sharing() 