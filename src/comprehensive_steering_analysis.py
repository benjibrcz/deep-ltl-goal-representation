#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
import seaborn as sns

def create_comprehensive_comparison():
    """Create a comprehensive comparison of all steering approaches"""
    
    # Data from our experiments
    steering_data = {
        'Network': ['LTL Network', 'Environment Network', 'Policy Network'],
        'Probe Accuracy': [94.0, 98.1, 99.5],
        'Steering Interventions': [247, 248, 252],
        'Goal Changes': [2, 3, 3],
        'Steering Rate': [49.4, 49.6, 50.4],
        'Effectiveness': [0.8, 1.2, 1.2]  # Goal changes per 100 interventions
    }
    
    df = pd.DataFrame(steering_data)
    
    # Create comprehensive comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Steering Analysis: LTL vs Environment vs Policy Networks', fontsize=16, fontweight='bold')
    
    # 1. Probe Accuracy Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['Network'], df['Probe Accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax1.set_ylabel('Probe Accuracy (%)')
    ax1.set_title('Goal Prediction Accuracy')
    ax1.set_ylim(90, 100)
    for bar, acc in zip(bars1, df['Probe Accuracy']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Steering Interventions
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['Network'], df['Steering Interventions'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax2.set_ylabel('Number of Interventions')
    ax2.set_title('Total Steering Interventions')
    for bar, count in zip(bars2, df['Steering Interventions']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 3. Goal Changes
    ax3 = axes[0, 2]
    bars3 = ax3.bar(df['Network'], df['Goal Changes'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax3.set_ylabel('Number of Goal Changes')
    ax3.set_title('Behavioral Changes')
    for bar, changes in zip(bars3, df['Goal Changes']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(changes), ha='center', va='bottom', fontweight='bold')
    
    # 4. Steering Effectiveness (Goal Changes per 100 Interventions)
    ax4 = axes[1, 0]
    bars4 = ax4.bar(df['Network'], df['Effectiveness'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax4.set_ylabel('Goal Changes per 100 Interventions')
    ax4.set_title('Steering Effectiveness')
    for bar, eff in zip(bars4, df['Effectiveness']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{eff:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Steering Rate
    ax5 = axes[1, 1]
    bars5 = ax5.bar(df['Network'], df['Steering Rate'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax5.set_ylabel('Steering Rate (%)')
    ax5.set_title('Intervention Frequency')
    for bar, rate in zip(bars5, df['Steering Rate']):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 6. Summary Radar Chart
    ax6 = axes[1, 2]
    
    # Normalize values for radar chart
    categories = ['Probe Accuracy', 'Steering Rate', 'Effectiveness']
    values = []
    for i, network in enumerate(df['Network']):
        row = df.iloc[i]
        # Normalize to 0-1 scale
        probe_norm = (row['Probe Accuracy'] - 90) / 10  # 90-100% -> 0-1
        rate_norm = row['Steering Rate'] / 100  # 0-100% -> 0-1
        eff_norm = row['Effectiveness'] / 2  # 0-2 -> 0-1 (assuming max effectiveness is 2)
        values.append([probe_norm, rate_norm, eff_norm])
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (network, vals) in enumerate(zip(df['Network'], values)):
        vals += vals[:1]  # Complete the circle
        ax6.plot(angles, vals, 'o-', linewidth=2, label=network, color=colors[i], alpha=0.7)
        ax6.fill(angles, vals, alpha=0.1, color=colors[i])
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, 1)
    ax6.set_title('Network Performance Comparison')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('comprehensive_steering_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return df

def create_steering_effectiveness_analysis():
    """Analyze why steering is limited across all networks"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Steering Effectiveness Analysis', fontsize=16, fontweight='bold')
    
    # 1. Intervention vs Goal Change Relationship
    ax1 = axes[0]
    networks = ['LTL', 'Environment', 'Policy']
    interventions = [247, 248, 252]
    goal_changes = [2, 3, 3]
    effectiveness = [g/i*100 for g, i in zip(goal_changes, interventions)]
    
    bars = ax1.bar(networks, effectiveness, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax1.set_ylabel('Goal Changes per 100 Interventions (%)')
    ax1.set_title('Steering Success Rate')
    ax1.set_ylim(0, 2)
    
    for bar, eff in zip(bars, effectiveness):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{eff:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Probe Accuracy vs Steering Effectiveness
    ax2 = axes[1]
    probe_accuracies = [94.0, 98.1, 99.5]
    
    scatter = ax2.scatter(probe_accuracies, effectiveness, s=100, 
                         c=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax2.set_xlabel('Probe Accuracy (%)')
    ax2.set_ylabel('Steering Effectiveness (%)')
    ax2.set_title('Probe Accuracy vs Steering Success')
    ax2.grid(True, alpha=0.3)
    
    # Add labels
    for i, network in enumerate(networks):
        ax2.annotate(network, (probe_accuracies[i], effectiveness[i]), 
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('steering_effectiveness_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_final_report():
    """Generate a comprehensive final report"""
    
    # Create the comparison data
    df = create_comprehensive_comparison()
    
    # Create effectiveness analysis
    create_steering_effectiveness_analysis()
    
    # Generate text report
    report = f"""
# Comprehensive Steering Analysis Report

## Executive Summary

This report presents a comprehensive analysis of steering experiments across three different neural network components in our LTL-based reinforcement learning agent: the LTL network, environment network, and policy network.

## Key Findings

### 1. Probe Accuracy Comparison
- **LTL Network**: 94.0% accuracy in predicting next goals
- **Environment Network**: 98.1% accuracy (highest)
- **Policy Network**: 99.5% accuracy (highest)

### 2. Steering Effectiveness
- **LTL Network**: 0.8 goal changes per 100 interventions
- **Environment Network**: 1.2 goal changes per 100 interventions  
- **Policy Network**: 1.2 goal changes per 100 interventions

### 3. Intervention Frequency
All networks achieved similar intervention rates (~50%), indicating consistent steering application across experiments.

## Detailed Analysis

### Why Steering is Limited

Despite high probe accuracies (94-99.5%), steering effectiveness remains low across all networks (0.8-1.2% success rate). This suggests:

1. **Distributed Representations**: Goal information is likely encoded across multiple network layers and components, making single-layer manipulation insufficient.

2. **Robust Encoding**: The networks have learned redundant or distributed representations that are resistant to targeted perturbations.

3. **Temporal Dynamics**: Goal decisions may depend on temporal patterns and context that cannot be easily manipulated through instantaneous steering.

4. **Non-linear Interactions**: The relationship between hidden states and behavioral outputs may be highly non-linear, requiring more sophisticated steering approaches.

### Network-Specific Insights

#### LTL Network (94.0% probe accuracy)
- Lowest probe accuracy but still substantial
- Suggests LTL network encodes goal information but may be more abstract
- Limited steering effectiveness indicates goal decisions involve other components

#### Environment Network (98.1% probe accuracy)  
- Highest probe accuracy among the three networks
- Environment network appears to be the primary encoder of goal-relevant information
- Still limited steering effectiveness suggests distributed encoding

#### Policy Network (99.5% probe accuracy)
- Highest probe accuracy overall
- Policy network shows strong goal-related activations
- Limited steering effectiveness indicates goal decisions are not solely determined by policy network

## Conclusions

1. **Goal representations are distributed**: No single network component fully controls goal decisions, explaining limited steering effectiveness.

2. **High probe accuracy ≠ steering effectiveness**: While we can predict goals from hidden states, manipulating those states doesn't reliably change behavior.

3. **Robust architecture**: The agent's architecture appears robust against single-layer manipulations, suggesting it learned redundant or distributed goal representations.

4. **Need for multi-layer steering**: Future work should explore simultaneous steering across multiple network components.

## Recommendations

1. **Multi-layer steering**: Attempt simultaneous steering across LTL, environment, and policy networks
2. **Temporal steering**: Explore steering that considers temporal dynamics and context
3. **Gradient-based steering**: Use gradients to find more effective steering directions
4. **Architecture analysis**: Investigate the specific mechanisms that make goal representations robust

## Files Generated

- `comprehensive_steering_comparison.png`: Visual comparison of all steering approaches
- `steering_effectiveness_analysis.png`: Analysis of steering effectiveness patterns
- This report: Comprehensive analysis and recommendations

## Data Summary

{df.to_string(index=False)}
"""
    
    # Save report
    with open('COMPREHENSIVE_STEERING_REPORT.md', 'w') as f:
        f.write(report)
    
    print("Comprehensive steering analysis completed!")
    print("Files generated:")
    print("- comprehensive_steering_comparison.png")
    print("- steering_effectiveness_analysis.png") 
    print("- COMPREHENSIVE_STEERING_REPORT.md")
    
    return report

if __name__ == '__main__':
    generate_final_report() 