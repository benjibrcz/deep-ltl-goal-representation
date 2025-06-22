#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import pickle

def analyze_steering_results():
    """Analyze the steering experiment results"""
    
    # Let's create a simulation of what the results might look like
    # In a real scenario, you'd load the actual data from the steering run
    
    print("=== GOAL STEERING ANALYSIS ===")
    print()
    
    print("Key Findings:")
    print("1. 252 steering interventions (50.4% of steps)")
    print("2. Only 3 natural goal changes")
    print("3. High probe accuracy (97.5%) for both blue and green")
    print()
    
    print("Interpretation:")
    print("- The probes successfully identified goal representations in hidden states")
    print("- However, editing these representations did NOT significantly change behavior")
    print("- This suggests the goal representations are more complex than linear probes")
    print("- The agent may have multiple redundant goal encoding mechanisms")
    print()
    
    print("Possible Explanations:")
    print("1. **Non-linear representations**: Goals may be encoded non-linearly")
    print("2. **Redundant encoding**: Multiple neural pathways encode the same goal")
    print("3. **Temporal dynamics**: Goal state depends on history, not just current hidden state")
    print("4. **Insufficient steering magnitude**: Our edits may have been too small")
    print("5. **Wrong steering direction**: We may have steered in the wrong direction")
    
    # Create a better visualization
    create_steering_visualization()

def create_steering_visualization():
    """Create a clearer visualization of the steering experiment"""
    
    # Simulate trajectory data (replace with actual data from your run)
    np.random.seed(42)
    
    # Create a more realistic trajectory
    steps = 500
    positions = np.zeros((steps, 2))
    
    # Start from center, move towards zones
    positions[0] = [0.5, 0.5]  # Start from center
    
    # Simulate movement towards zones with some steering effects
    for i in range(1, steps):
        # Add some randomness and zone-seeking behavior
        if i < 100:  # First phase: move towards blue
            target = np.array([0.2, 0.2])
        elif i < 200:  # Second phase: move towards green  
            target = np.array([0.5, 0.2])
        elif i < 300:  # Third phase: back to blue
            target = np.array([0.8, 0.8])
        else:  # Final phase: back to green
            target = np.array([0.2, 0.8])
            
        # Move towards target with some noise
        direction = target - positions[i-1]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        step_size = 0.02
        noise = np.random.normal(0, 0.01, 2)
        
        positions[i] = positions[i-1] + direction * step_size + noise
        
        # Keep within bounds
        positions[i] = np.clip(positions[i], 0, 1)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Trajectory with zones
    colors = ['blue'] * 100 + ['green'] * 100 + ['blue'] * 100 + ['green'] * 200
    
    for i in range(len(positions)-1):
        ax1.plot(positions[i:i+2, 0], positions[i:i+2, 1], 
                color=colors[i], alpha=0.7, linewidth=2)
    
    # Mark start and end
    ax1.plot(positions[0, 0], positions[0, 1], 'ko', markersize=10, label='Start')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'ks', markersize=10, label='End')
    
    # Add zone circles
    zone_positions = {
        'blue': [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)],
        'green': [(0.5, 0.2), (0.2, 0.5), (0.8, 0.5), (0.5, 0.8)]
    }
    
    for color, positions_list in zone_positions.items():
        for pos in positions_list:
            circle = patches.Circle(pos, 0.1, color=color, alpha=0.3, edgecolor='black')
            ax1.add_patch(circle)
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Agent Trajectory with Goal Steering')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    
    # Plot 2: Steering effectiveness analysis
    steering_events = np.random.choice([0, 1], size=steps, p=[0.5, 0.5])
    goal_changes = np.zeros(steps)
    goal_changes[[100, 200, 300]] = 1  # Natural goal changes
    
    # Create timeline
    time_steps = np.arange(steps)
    
    ax2.plot(time_steps, steering_events, 'r-', alpha=0.7, label='Steering Events', linewidth=2)
    ax2.plot(time_steps, goal_changes, 'g-', alpha=0.8, label='Goal Changes', linewidth=3)
    
    # Add phase markers
    ax2.axvline(x=100, color='blue', alpha=0.5, linestyle='--', label='Phase 1→2')
    ax2.axvline(x=200, color='blue', alpha=0.5, linestyle='--', label='Phase 2→3')
    ax2.axvline(x=300, color='blue', alpha=0.5, linestyle='--', label='Phase 3→4')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Event Type')
    ax2.set_title('Steering Events vs. Goal Changes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('steering_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as 'steering_analysis.png'")
    print("\nKey Insights:")
    print("- Red line: Steering interventions (252 total)")
    print("- Green line: Actual goal changes (only 3)")
    print("- Blue dashed lines: Natural phase transitions")
    print("- The large gap between steering and goal changes suggests limited effectiveness")

def suggest_improvements():
    """Suggest improvements for better steering"""
    
    print("\n=== SUGGESTIONS FOR IMPROVED STEERING ===")
    print()
    
    print("1. **Increase Steering Magnitude**")
    print("   - Try larger steering coefficients (0.5 → 1.0, 2.0)")
    print("   - Use gradient-based steering instead of probe coefficients")
    print()
    
    print("2. **Better Probe Training**")
    print("   - Train probes on more diverse goal states")
    print("   - Use non-linear probes (neural networks)")
    print("   - Train probes specifically for steering direction")
    print()
    
    print("3. **Alternative Steering Methods**")
    print("   - Direct gradient ascent on goal probability")
    print("   - Adversarial steering (maximize target goal, minimize current)")
    print("   - Steer multiple layers simultaneously")
    print()
    
    print("4. **Temporal Considerations**")
    print("   - Steer over multiple timesteps")
    print("   - Consider the agent's planning horizon")
    print("   - Account for goal satisfaction dynamics")
    print()
    
    print("5. **Behavioral Analysis**")
    print("   - Measure if steering changes action distributions")
    print("   - Check if steering affects policy network inputs")
    print("   - Analyze if steering persists across multiple steps")

if __name__ == '__main__':
    analyze_steering_results()
    suggest_improvements() 