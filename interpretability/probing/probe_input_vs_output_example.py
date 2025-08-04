#!/usr/bin/env python3
"""
Environment Network Input vs Output Probing Example

This script demonstrates how to probe the input layer vs output layer 
of the environment network to understand information processing.

Usage: python interpretability/probing/probe_input_vs_output_example.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_probe_comparison():
    """Run probes comparing input vs output layers of env_net."""
    
    print("🔬 Environment Network Input vs Output Analysis")
    print("=" * 60)
    
    # Define the comparison experiments
    experiments = [
        {
            "name": "Agent Position - INPUT Layer",
            "layer": "env_net.mlp.0",  # First layer (input side)
            "target": "agent_pos",
            "description": "How well does the INPUT layer encode agent position?"
        },
        {
            "name": "Agent Position - OUTPUT Layer", 
            "layer": "env_net.mlp.2",  # Last layer (output side)
            "target": "agent_pos",
            "description": "How well does the OUTPUT layer encode agent position?"
        },
        {
            "name": "Zone Lidar - INPUT Layer",
            "layer": "env_net.mlp.0",
            "target": "zone_lidar", 
            "description": "How well does the INPUT layer process zone lidar?"
        },
        {
            "name": "Zone Lidar - OUTPUT Layer",
            "layer": "env_net.mlp.2",
            "target": "zone_lidar",
            "description": "How well does the OUTPUT layer process zone lidar?"
        },
        {
            "name": "Wall Sensor - INPUT Layer",
            "layer": "env_net.mlp.0",
            "target": "wall_sensor",
            "description": "How well does the INPUT layer process wall sensors?"
        },
        {
            "name": "Wall Sensor - OUTPUT Layer", 
            "layer": "env_net.mlp.2",
            "target": "wall_sensor",
            "description": "How well does the OUTPUT layer process wall sensors?"
        }
    ]
    
    print(f"Planning to run {len(experiments)} experiments to compare input vs output processing...")
    print("\nExperiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")
    
    response = input(f"\nProceed with all {len(experiments)} experiments? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Aborted.")
        return
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running: {exp['name']}")
        print(f"Description: {exp['description']}")
        print(f"Layer: {exp['layer']}, Target: {exp['target']}")
        print("-" * 40)
        
        cmd = [
            sys.executable,
            "interpretability/probing/comprehensive_env_net_probe.py",
            "--layer", exp['layer'],
            "--target", exp['target'],
            "--n-worlds", "6",  # Smaller for faster testing
            "--n-rollouts", "5", 
            "--max-steps", "100"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: {exp['name']}")
                # Extract R² scores from output
                lines = result.stdout.split('\n')
                r2_scores = {}
                for line in lines:
                    if 'temporal:' in line.lower() and 'r²=' in line:
                        r2_scores['temporal'] = float(line.split('R²=')[1].split(',')[0])
                    elif 'spatial:' in line.lower() and 'r²=' in line:
                        r2_scores['spatial'] = float(line.split('R²=')[1].split(',')[0])
                    elif 'environmental:' in line.lower() and 'r²=' in line:
                        r2_scores['environmental'] = float(line.split('R²=')[1].split(',')[0])
                
                results.append({
                    'name': exp['name'],
                    'layer': exp['layer'],
                    'target': exp['target'],
                    'success': True,
                    'r2_scores': r2_scores
                })
                
            else:
                print(f"❌ FAILED: {exp['name']}")
                print("STDERR:", result.stderr)
                results.append({
                    'name': exp['name'],
                    'layer': exp['layer'], 
                    'target': exp['target'],
                    'success': False,
                    'error': result.stderr
                })
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {exp['name']}")
            results.append({
                'name': exp['name'],
                'layer': exp['layer'],
                'target': exp['target'], 
                'success': False,
                'error': 'Timeout'
            })
        except Exception as e:
            print(f"💥 ERROR: {exp['name']} - {e}")
            results.append({
                'name': exp['name'],
                'layer': exp['layer'],
                'target': exp['target'],
                'success': False,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 INPUT vs OUTPUT ANALYSIS SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) >= 2:
        # Group by target for comparison
        targets = set(r['target'] for r in successful_results)
        
        for target in targets:
            target_results = [r for r in successful_results if r['target'] == target]
            if len(target_results) >= 2:
                input_results = [r for r in target_results if 'mlp.0' in r['layer']]
                output_results = [r for r in target_results if 'mlp.2' in r['layer']]
                
                if input_results and output_results:
                    input_r = input_results[0]
                    output_r = output_results[0] 
                    
                    print(f"\n📊 {target.upper()} COMPARISON:")
                    print(f"INPUT Layer (env_net.mlp.0):")
                    for split, score in input_r['r2_scores'].items():
                        print(f"  {split}: R² = {score:.3f}")
                    
                    print(f"OUTPUT Layer (env_net.mlp.2):")
                    for split, score in output_r['r2_scores'].items():
                        print(f"  {split}: R² = {score:.3f}")
                    
                    # Calculate differences
                    print(f"DIFFERENCES (Output - Input):")
                    for split in ['temporal', 'spatial', 'environmental']:
                        if split in input_r['r2_scores'] and split in output_r['r2_scores']:
                            diff = output_r['r2_scores'][split] - input_r['r2_scores'][split]
                            direction = "📈" if diff > 0.05 else "📉" if diff < -0.05 else "➡️"
                            print(f"  {split}: {diff:+.3f} {direction}")
    
    print(f"\n📁 Results saved in: interpretability/probing/comprehensive_results/")
    print(f"✅ Successfully completed {len(successful_results)}/{len(experiments)} experiments")

def show_layer_info():
    """Show information about env_net layers."""
    print("🏗️ ENVIRONMENT NETWORK ARCHITECTURE")
    print("=" * 50)
    print("Raw Observations (80 dimensions)")
    print("    ↓")
    print("env_net.mlp.0 (Linear: 80→128)  ← 🎯 INPUT LAYER")
    print("    ↓")
    print("env_net.mlp.1 (Tanh activation)")
    print("    ↓") 
    print("env_net.mlp.2 (Linear: 128→64)  ← 🎯 OUTPUT LAYER")
    print("    ↓")
    print("env_net.mlp.3 (Tanh activation)")
    print("    ↓")
    print("Final embedding (64 dimensions)")
    print()
    print("🎯 KEY LAYERS TO PROBE:")
    print("• env_net.mlp.0 = INPUT side (processes raw observations)")
    print("• env_net.mlp.2 = OUTPUT side (final processed representation)")

def main():
    """Main function to run input vs output analysis."""
    
    show_layer_info()
    
    print("\n" + "="*60)
    print("Choose an option:")
    print("1. Run full input vs output comparison")
    print("2. Show manual command examples")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        run_probe_comparison()
    elif choice == '2':
        print("\n🔧 MANUAL COMMAND EXAMPLES:")
        print("=" * 40)
        print("# Probe INPUT layer for agent position:")
        print("python interpretability/probing/comprehensive_env_net_probe.py \\")
        print("    --layer env_net.mlp.0 \\")
        print("    --target agent_pos")
        print()
        print("# Probe OUTPUT layer for agent position:")
        print("python interpretability/probing/comprehensive_env_net_probe.py \\")
        print("    --layer env_net.mlp.2 \\")
        print("    --target agent_pos")
        print()
        print("# Probe INPUT layer for zone lidar:")
        print("python interpretability/probing/comprehensive_env_net_probe.py \\")
        print("    --layer env_net.mlp.0 \\")
        print("    --target zone_lidar")
        print()
        print("# Probe OUTPUT layer for zone lidar:")
        print("python interpretability/probing/comprehensive_env_net_probe.py \\")
        print("    --layer env_net.mlp.2 \\")
        print("    --target zone_lidar")
        print()
        print("💡 TIP: Compare R² scores between input and output layers!")
        print("   Higher scores = better information encoding")
    else:
        print("Goodbye!")

if __name__ == "__main__":
    main() 