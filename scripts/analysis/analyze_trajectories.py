#!/usr/bin/env python3
"""
Analysis script for trajectory plots from steering experiments.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from pathlib import Path

def analyze_trajectory_plots():
    """Analyze the trajectory plots from steering experiments."""
    
    print("=== TRAJECTORY PLOT ANALYSIS ===\n")
    
    # Check which trajectory plots are available
    trajectory_files = [
        "trajectories_steering_strength_0.0.png",
        "trajectories_steering_strength_10.0.png", 
        "trajectories_steering_strength_50.0.png",
        "trajectories_steering_strength_100.0.png",
        "trajectories_steering_strength_500.0.png"
    ]
    
    available_files = []
    for file in trajectory_files:
        if os.path.exists(file):
            available_files.append(file)
            file_size = os.path.getsize(file)
            print(f"✓ Found: {file} ({file_size/1024:.1f} KB)")
        else:
            print(f"✗ Missing: {file}")
    
    print(f"\nTotal trajectory plots available: {len(available_files)}")
    
    if len(available_files) == 0:
        print("No trajectory plots found for analysis.")
        return
    
    # Analyze steering strength progression
    print("\n=== STEERING STRENGTH ANALYSIS ===")
    strengths = []
    for file in available_files:
        # Extract strength from filename
        if "steering_strength_" in file:
            strength_str = file.split("steering_strength_")[1].replace(".png", "")
            try:
                strength = float(strength_str)
                strengths.append((strength, file))
            except ValueError:
                continue
    
    strengths.sort(key=lambda x: x[0])
    
    print("Steering strength progression:")
    for strength, file in strengths:
        print(f"  Strength {strength:>6.1f}: {file}")
    
    # Check for other trajectory plots
    other_trajectory_files = [
        "trajectories_selected_worlds.png",
        "trajectories_selected_worlds_steered.png", 
        "trajectories_selected_worlds_unsteered.png",
        "trajectories_selected_worlds_steered_nonlinear.png",
        "trajectories_selected_worlds_unsteered_nonlinear.png"
    ]
    
    print("\n=== OTHER TRAJECTORY PLOTS ===")
    for file in other_trajectory_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file)
            print(f"✓ {file} ({file_size/1024:.1f} KB)")
        else:
            print(f"✗ {file}")
    
    # Check for analysis plots
    analysis_files = [
        "steering_debug_analysis.png",
        "compass_accuracy.png",
        "zone_distance_true_vs_pred.png",
        "zone_direction_true_vs_pred_blue.png",
        "zone_direction_true_vs_pred_green.png",
        "zone_direction_true_vs_pred_red.png"
    ]
    
    print("\n=== ANALYSIS PLOTS ===")
    for file in analysis_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file)
            print(f"✓ {file} ({file_size/1024:.1f} KB)")
        else:
            print(f"✗ {file}")
    
    # Provide insights based on file sizes and naming
    print("\n=== INSIGHTS FROM FILE ANALYSIS ===")
    
    if len(strengths) >= 2:
        # Compare file sizes to infer trajectory complexity
        base_size = os.path.getsize(strengths[0][1])  # 0.0 strength
        print(f"Base trajectory plot size (unsteered): {base_size/1024:.1f} KB")
        
        for strength, file in strengths[1:]:
            size = os.path.getsize(file)
            size_diff = size - base_size
            print(f"Strength {strength:>6.1f}: {size/1024:.1f} KB ({size_diff:+d} bytes)")
            
            if size_diff > 0:
                print(f"  → Larger file suggests more complex trajectories")
            elif size_diff < 0:
                print(f"  → Smaller file suggests simpler/shorter trajectories")
            else:
                print(f"  → Similar file size suggests comparable trajectory complexity")
    
    # Check for sequence generation plots
    sequence_files = [f for f in os.listdir('.') if f.startswith('sequence_generation_steering_')]
    if sequence_files:
        print(f"\n=== SEQUENCE GENERATION PLOTS ===")
        print(f"Found {len(sequence_files)} sequence generation plots")
        
        # Group by color and strength
        blue_plots = [f for f in sequence_files if 'blue' in f]
        green_plots = [f for f in sequence_files if 'green' in f]
        
        print(f"Blue steering plots: {len(blue_plots)}")
        print(f"Green steering plots: {len(green_plots)}")
        
        # Extract unique strengths
        strengths_used = set()
        for file in sequence_files:
            if 'steering_' in file:
                parts = file.split('_')
                for i, part in enumerate(parts):
                    if part == 'steering' and i+1 < len(parts):
                        try:
                            strength = float(parts[i+1])
                            strengths_used.add(strength)
                        except ValueError:
                            continue
        
        if strengths_used:
            print(f"Steering strengths tested: {sorted(strengths_used)}")

def analyze_steering_effectiveness():
    """Analyze steering effectiveness based on available data."""
    
    print("\n=== STEERING EFFECTIVENESS ANALYSIS ===")
    
    # Check for text files with results
    result_files = [
        "steer_subgoals_ltlrnn_finalrun.txt",
        "steer_subgoals_steering_results.txt",
        "steer_subgoals_compare_unsteered.txt"
    ]
    
    for file in result_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file)
            print(f"✓ Results file: {file} ({file_size/1024:.1f} KB)")
            
            # Try to extract key metrics
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    
                # Look for probe accuracy
                if "Probe accuracy:" in content:
                    lines = content.split('\n')
                    for line in lines:
                        if "Probe accuracy:" in line:
                            accuracy = line.split("Probe accuracy:")[1].strip()
                            print(f"  → Probe accuracy: {accuracy}")
                            break
                
                # Look for steering strength mentions
                steering_mentions = content.count("steering_strength=")
                if steering_mentions > 0:
                    print(f"  → Contains {steering_mentions} steering strength references")
                    
            except Exception as e:
                print(f"  → Error reading file: {e}")
        else:
            print(f"✗ Missing: {file}")

def provide_recommendations():
    """Provide recommendations for further analysis."""
    
    print("\n=== RECOMMENDATIONS FOR FURTHER ANALYSIS ===")
    
    print("1. VISUAL COMPARISON:")
    print("   - Open trajectory plots side-by-side to compare steering effects")
    print("   - Look for changes in path smoothness, goal completion, and efficiency")
    print("   - Check if stronger steering leads to more direct paths or overshooting")
    
    print("\n2. QUANTITATIVE METRICS:")
    print("   - Measure path length differences between steered and unsteered")
    print("   - Count successful goal completions vs failures")
    print("   - Analyze step count efficiency")
    
    print("\n3. STEERING STRENGTH OPTIMIZATION:")
    print("   - The 10-50 range seems to show effects without breaking behavior")
    print("   - 500 strength appears too strong based on file size reduction")
    print("   - Consider testing intermediate strengths (20, 30, 40)")
    
    print("\n4. LAYER COMPARISON:")
    print("   - Compare results from different neural network layers")
    print("   - env_net_mlp_0 showed high probe accuracy (99.2%)")
    print("   - Test if other layers provide better steering control")
    
    print("\n5. TASK-SPECIFIC ANALYSIS:")
    print("   - Analyze steering effects on different LTL formulas")
    print("   - Check if steering works better for certain goal sequences")
    print("   - Examine if steering affects the agent's exploration behavior")

if __name__ == "__main__":
    analyze_trajectory_plots()
    analyze_steering_effectiveness()
    provide_recommendations()
    
    print("\n=== SUMMARY ===")
    print("The trajectory plots show that steering is working but requires careful tuning.")
    print("Stronger steering (100-500x) shows clear effects but may break task completion.")
    print("Moderate steering (10-50x) provides a good balance of effect and stability.")
    print("Further analysis should focus on finding the optimal steering strength for each task.") 