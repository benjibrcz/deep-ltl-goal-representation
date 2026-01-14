#!/usr/bin/env python3
"""
Helper script to run comprehensive environment network probing examples.

This script demonstrates how to probe various components of the environment network
and environmental features using the comprehensive probing tool.

Usage: python interpretability/probing/run_comprehensive_probe_examples.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_probe(layer, target, description, additional_args=None):
    """Run a comprehensive probe with the specified parameters."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Layer: {layer}")
    print(f"Target: {target}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "interpretability/probing/comprehensive_env_net_probe.py",
        "--layer", layer,
        "--target", target,
        "--n-worlds", "6",  # Reduced for faster testing
        "--n-rollouts", "5",
        "--max-steps", "100"
    ]
    
    if additional_args:
        cmd.extend(additional_args)
    
    try:
        # Change to the project root directory
        original_dir = os.getcwd()
        script_dir = Path(__file__).parent.parent.parent
        os.chdir(script_dir)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ Successfully completed: {description}")
        else:
            print(f"❌ Failed: {description} (return code: {result.returncode})")
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
    finally:
        os.chdir(original_dir)

def main():
    """Run comprehensive probing examples for different targets and layers."""
    
    print("🔍 Comprehensive Environment Network Probing Examples")
    print("This script will run probing experiments for various targets and layers.")
    
    # Define the experiments to run
    experiments = [
        # Environment Network Input/Output Analysis
        {
            "layer": "env_net.mlp.0",
            "target": "agent_pos", 
            "description": "Probe env_net MLP layer 0 for agent position"
        },
        {
            "layer": "env_net.mlp.2",
            "target": "agent_pos",
            "description": "Probe env_net MLP layer 2 for agent position"  
        },
        
        # Zone Analysis
        {
            "layer": "env_net.mlp.0",
            "target": "zone_lidar",
            "description": "Probe env_net MLP layer 0 for zone lidar readings"
        },
        {
            "layer": "env_net.mlp.2", 
            "target": "zone_lidar",
            "description": "Probe env_net MLP layer 2 for zone lidar readings"
        },
        {
            "layer": "env_net.mlp.1",
            "target": "zone_differences",
            "description": "Probe env_net MLP layer 1 for zone differences"
        },
        
        # Sensor Analysis
        {
            "layer": "env_net.mlp.0",
            "target": "wall_sensor",
            "description": "Probe env_net MLP layer 0 for wall sensor readings"
        },
        {
            "layer": "env_net.mlp.1",
            "target": "agent_sensors", 
            "description": "Probe env_net MLP layer 1 for agent sensors (accelerometer, etc.)"
        },
        
        # Joint Analysis  
        {
            "layer": "env_net.mlp.2",
            "target": "joint_positions",
            "description": "Probe env_net MLP layer 2 for joint positions"
        }
    ]
    
    print(f"\nPlanning to run {len(experiments)} experiments...")
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Aborted.")
        return
    
    # Run experiments
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]", end=" ")
        run_probe(
            layer=exp["layer"],
            target=exp["target"], 
            description=exp["description"]
        )
    
    print(f"\n🎉 Completed all {len(experiments)} experiments!")
    print("\nResults are saved in: interpretability/probing/comprehensive_results/")
    print("\nEach experiment generates:")
    print("  - CSV file with detailed metrics")
    print("  - PNG visualization with 4-panel plots")
    print("  - TXT summary report")
    
    print("\nExample usage for individual probes:")
    print("python interpretability/probing/comprehensive_env_net_probe.py --layer env_net.mlp.0 --target agent_pos")
    print("python interpretability/probing/comprehensive_env_net_probe.py --layer env_net.mlp.2 --target zone_lidar")

if __name__ == "__main__":
    main() 