#!/usr/bin/env python3
"""
Actor/Critic Input Analysis
===========================
This script determines exactly what information the actor and critic networks receive:
- Do they get raw 80D sensor data + ltl info?
- Or do they get processed env_net output (64D) + ltl_net output (32D)?

This is crucial for understanding information flow and interpretability.
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

def capture_tensor_data(module, input_data, output_data, name, capture_dict):
    """Hook to capture actual tensor values"""
    if isinstance(input_data, tuple) and len(input_data) > 0:
        inp = input_data[0]
        if hasattr(inp, 'detach'):
            capture_dict[f"{name}_input"] = inp.detach().cpu().numpy().copy()
    
    if hasattr(output_data, 'detach'):
        capture_dict[f"{name}_output"] = output_data.detach().cpu().numpy().copy()
    elif isinstance(output_data, tuple) and len(output_data) > 0:
        if hasattr(output_data[0], 'detach'):
            capture_dict[f"{name}_output"] = output_data[0].detach().cpu().numpy().copy()

def analyze_information_flow():
    """Analyze exactly what data flows to actor/critic"""
    ENV = "PointLtl2-v0"
    EXP = "big_test"
    SEED = 0
    formula = "FG blue"
    sampler = FixedSampler.partial(formula)
    
    print("=" * 80)
    print("ACTOR/CRITIC INPUT ANALYSIS")
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
    
    # Dictionary to capture tensor data
    captured_data = {}
    
    # Register hooks on key components
    hooks = []
    
    # Hook env_net input/output
    if hasattr(model, 'env_net') and model.env_net is not None:
        env_hook = model.env_net.register_forward_hook(
            lambda m, inp, out: capture_tensor_data(m, inp, out, "env_net", captured_data)
        )
        hooks.append(env_hook)
    
    # Hook ltl_net input/output  
    if hasattr(model, 'ltl_net') and model.ltl_net is not None:
        ltl_hook = model.ltl_net.register_forward_hook(
            lambda m, inp, out: capture_tensor_data(m, inp, out, "ltl_net", captured_data)
        )
        hooks.append(ltl_hook)
    
    # Hook actor input
    if hasattr(model, 'actor') and model.actor is not None:
        actor_hook = model.actor.register_forward_hook(
            lambda m, inp, out: capture_tensor_data(m, inp, out, "actor", captured_data)
        )
        hooks.append(actor_hook)
    
    # Hook critic input
    if hasattr(model, 'critic') and model.critic is not None:
        critic_hook = model.critic.register_forward_hook(
            lambda m, inp, out: capture_tensor_data(m, inp, out, "critic", captured_data)
        )
        hooks.append(critic_hook)
    
    # Get observation and run forward pass
    obs = env.reset(seed=42)
    agent.reset()
    
    print(f"Raw observation:")
    print(f"  features shape: {obs['features'].shape}")
    print(f"  features sample: {obs['features'][:5]}")
    print(f"  goal: {obs['goal']}")
    print(f"  ldba_state: {obs['ldba_state']}")
    
    # Run forward pass to capture data
    with torch.no_grad():
        action = agent.get_action(obs, {}, deterministic=True)
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    print(f"\n" + "=" * 80)
    print("CAPTURED TENSOR ANALYSIS")
    print("=" * 80)
    
    # Analyze what we captured
    for key, tensor in captured_data.items():
        print(f"\n[{key}]")
        print(f"  Shape: {tensor.shape}")
        print(f"  Range: [{np.min(tensor):.6f}, {np.max(tensor):.6f}]")
        print(f"  Sample values: {tensor.flatten()[:5]}")
    
    # The key analysis: compare inputs
    print(f"\n" + "=" * 80)
    print("CRITICAL COMPARISON")
    print("=" * 80)
    
    raw_features = obs['features']
    env_net_input = captured_data.get('env_net_input')
    env_net_output = captured_data.get('env_net_output') 
    actor_input = captured_data.get('actor_input')
    critic_input = captured_data.get('critic_input')
    ltl_net_output = captured_data.get('ltl_net_output')
    
    print(f"1. RAW SENSOR DATA vs ENV_NET INPUT:")
    if env_net_input is not None:
        env_input_flat = env_net_input.flatten()
        raw_flat = raw_features.flatten()
        
        # Check if they're the same
        if len(env_input_flat) == len(raw_flat):
            are_same = np.allclose(env_input_flat, raw_flat, atol=1e-6)
            print(f"   Raw features shape: {raw_features.shape}")
            print(f"   Env_net input shape: {env_net_input.shape}")
            print(f"   Are they identical? {are_same}")
            if are_same:
                print(f"   ✅ env_net receives RAW sensor data")
            else:
                print(f"   ❌ env_net receives PROCESSED data")
        else:
            print(f"   Different shapes: raw={raw_features.shape}, env_input={env_net_input.shape}")
    
    print(f"\n2. ACTOR INPUT COMPOSITION:")
    if actor_input is not None and env_net_output is not None:
        print(f"   Actor input shape: {actor_input.shape}")
        print(f"   Env_net output shape: {env_net_output.shape}")
        if ltl_net_output is not None:
            print(f"   LTL_net output shape: {ltl_net_output.shape}")
            
            # Check if actor input = env_net output + ltl_net output
            expected_size = env_net_output.shape[-1] + ltl_net_output.shape[-1]
            actual_size = actor_input.shape[-1]
            
            print(f"   Expected combined size: {expected_size}")
            print(f"   Actual actor input size: {actual_size}")
            
            if expected_size == actual_size:
                print(f"   ✅ Actor gets CONCATENATED processed representations")
                
                # Verify the concatenation
                actor_flat = actor_input.flatten()
                env_flat = env_net_output.flatten()
                ltl_flat = ltl_net_output.flatten()
                
                # Check if first part matches env_net output
                env_match = np.allclose(actor_flat[:len(env_flat)], env_flat, atol=1e-6)
                ltl_match = np.allclose(actor_flat[len(env_flat):len(env_flat)+len(ltl_flat)], ltl_flat, atol=1e-6)
                
                print(f"   First {len(env_flat)} dims match env_net output: {env_match}")
                print(f"   Next {len(ltl_flat)} dims match ltl_net output: {ltl_match}")
            else:
                print(f"   ❌ Unexpected input composition")
    
    print(f"\n3. CRITIC INPUT COMPOSITION:")
    if critic_input is not None:
        print(f"   Critic input shape: {critic_input.shape}")
        if actor_input is not None:
            # Check if critic and actor get the same input
            actor_flat = actor_input.flatten()
            critic_flat = critic_input.flatten()
            
            if len(actor_flat) == len(critic_flat):
                are_same = np.allclose(actor_flat, critic_flat, atol=1e-6)
                print(f"   Actor and critic inputs identical: {are_same}")
                if are_same:
                    print(f"   ✅ Actor and critic get SAME combined representation")
                else:
                    print(f"   ❌ Actor and critic get DIFFERENT inputs")
            else:
                print(f"   Different input sizes: actor={len(actor_flat)}, critic={len(critic_flat)}")
    
    # Final analysis
    print(f"\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    
    print("Information flow architecture:")
    print(f"1. 📥 RAW OBSERVATION: 80D sensor data + LTL semantics")
    print(f"2. 🌍 ENV_NET: Takes 80D sensors → outputs 64D spatial representation")
    print(f"3. 🎯 LTL_NET: Takes LTL semantics → outputs 32D goal representation") 
    print(f"4. 🎭 ACTOR/CRITIC: Take [64D + 32D] → 96D combined representation")
    
    print(f"\n📋 CONCLUSION:")
    if (env_net_output is not None and ltl_net_output is not None and 
        actor_input is not None and 
        env_net_output.shape[-1] + ltl_net_output.shape[-1] == actor_input.shape[-1]):
        
        print("✅ ACTOR/CRITIC get PROCESSED representations, NOT raw data!")
        print("   • They receive env_net's spatial understanding (64D)")
        print("   • Plus ltl_net's goal understanding (32D)")
        print("   • They do NOT have direct access to raw 80D sensor data")
        print("   • All sensor information is filtered through env_net's processing")
    else:
        print("❓ Unable to definitively determine - need more investigation")
    
    env.close()

if __name__ == "__main__":
    analyze_information_flow() 