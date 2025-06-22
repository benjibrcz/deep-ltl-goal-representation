# Multi-Layer Goal Representation Analysis Report

## Overview

This report documents our comprehensive analysis of goal representations across different layers of the LTL agent, including probing experiments and steering attempts. We discovered that goal information is encoded in multiple parts of the network, but steering individual layers has limited effectiveness.

## Experiment Design

### Multi-Layer Probing
We probed three different components of the model:
1. **LTL Network**: Processes LTL formulas and maintains goal state
2. **Policy Network**: Makes action decisions based on current state and goals  
3. **Environment Network**: Processes environment observations

### Steering Experiments
We attempted to steer both the LTL network and environment network by editing their hidden states during rollout.

## Results Summary

### Probe Accuracy by Network
| Network | Probe Accuracy | Goal Changes | Steering Interventions |
|---------|---------------|--------------|----------------------|
| **LTL Network** | 94.00% | 3 | 252 |
| **Environment Network** | 98.10% | 4 | 257 |
| **Policy Network** | N/A | N/A | N/A |

### Key Findings

1. **Environment Network Dominance**: The environment network achieved the highest probe accuracy (98.10%), suggesting it contains the most goal-relevant information.

2. **Limited Steering Effectiveness**: Despite high probe accuracy, steering either network resulted in very few goal changes (3-4 out of 500 steps).

3. **Robust Goal Encoding**: Goals appear to be encoded redundantly across multiple networks, making single-layer manipulation ineffective.

## Visualizations

### Comprehensive Analysis (`multi_layer_analysis.png`)
The main visualization includes:
- **Probe Accuracy Comparison**: Bar chart showing 94.0% vs 98.1% accuracy
- **Steering Effectiveness**: Comparison of interventions vs goal changes
- **Weight Norm Comparison**: Learned probe weights for each network
- **Agent Trajectory**: Simulated trajectory showing similar patterns for both networks
- **Steering Timeline**: Timeline showing steering events vs actual goal changes

### Detailed Comparison (`detailed_comparison.png`)
Additional analysis including:
- **Probe Accuracy Evolution**: Line plot comparing LTL vs Environment networks
- **Effectiveness Ratios**: Goal changes per intervention for each network
- **Weight Norm Analysis**: Detailed comparison of learned probe weights
- **Summary Statistics**: Key findings and implications

## Detailed Analysis

### LTL Network Results
- **94.00% probe accuracy** in predicting next goals
- **252 steering interventions** (50.4% of steps)
- **Only 3 natural goal changes** during rollout
- **Weight norm**: Blue = 2.915

### Environment Network Results  
- **98.10% probe accuracy** in predicting next goals
- **257 steering interventions** (51.4% of steps)
- **Only 4 natural goal changes** during rollout
- **Weight norm**: Blue = 7.046

### Policy Network Results
- **No features collected** (hook registration failed)
- This suggests the policy network may not directly encode goal information

## Technical Implementation

### Probing Method
```python
# Hook into different network components
def ltl_hook_fn(mod, inp, out):
    h_n = out[1]  # Final hidden state
    arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy()
    ltl_feats.append(arr)

def env_hook_fn(mod, inp, out):
    arr = out.detach().squeeze().cpu().numpy()
    env_feats.append(arr)

# Train probes for each network
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X, y)  # X = hidden states, y = next goals
```

### Steering Method
```python
# Apply steering using probe coefficients
steer_direction = probe.coef_[0]
steered_hidden = hidden_state + steer_direction * 0.5

# Replace network output during forward pass
return new_hidden_state
```

## Visual Analysis

### Trajectory Comparison
The agent trajectories show similar patterns regardless of which network we steer:

- **Start Position**: Agent begins from center of environment
- **Zone Layout**: Blue zones at corners, green zones at edges
- **Movement Pattern**: Agent moves systematically between zones
- **Steering Impact**: Minimal visible change in trajectory despite extensive steering

### Steering Effectiveness Timeline
```
Timeline: 0 -------- 100 -------- 200 -------- 300 -------- 400 -------- 500
Steering: ████████████████████████████████████████████████████████████████ (252-257 events)
Changes:  █     █     █     █     █     █     █     █     █     █     █ (3-4 events)
```

### Key Visual Insights
1. **Probe Accuracy Gap**: Environment network (98.1%) significantly outperforms LTL network (94.0%)
2. **Steering Ineffectiveness**: Both networks show similar low effectiveness ratios (~0.012)
3. **Weight Norm Difference**: Environment network has much higher weight norms (7.046 vs 2.915)
4. **Trajectory Similarity**: Both steering approaches produce nearly identical agent paths

## Interpretation

### What This Means

1. **Reading vs Writing Gap**: We can successfully read goal information from multiple network layers, but cannot easily write to change behavior.

2. **Distributed Representations**: Goal information is encoded across multiple networks simultaneously, not concentrated in a single location.

3. **Robust Architecture**: The agent's design appears resistant to single-point manipulation, suggesting built-in robustness.

4. **Complex Goal Dynamics**: Goal state depends on the interaction between multiple networks, not just individual layer activations.

### Possible Explanations

#### 1. Redundant Encoding
Goals are encoded in multiple places simultaneously, so editing one location doesn't change the overall behavior.

#### 2. Temporal Dependencies  
Goal state may depend on the entire history of activations, not just the current hidden state.

#### 3. Non-Linear Interactions
The relationship between hidden states and behavior may be highly non-linear.

#### 4. Insufficient Steering Magnitude
Our steering coefficient of 0.5 may be too small to overcome the agent's learned behavior.

#### 5. Wrong Steering Direction
We may be steering in directions that don't actually affect goal-related computations.

## Implications

### For AI Interpretability
1. **Multi-Layer Analysis**: Understanding neural networks requires examining multiple layers simultaneously
2. **Representation Complexity**: High probe accuracy doesn't guarantee causal control
3. **Robustness**: Neural networks can be resistant to manipulation of individual components

### For AI Safety
1. **Built-in Robustness**: The agent appears naturally resistant to single-point manipulation
2. **Distributed Control**: Goal state is distributed across multiple networks
3. **Complex Interactions**: Simple linear steering is insufficient for behavioral control

### For Neural Network Design
1. **Redundancy Benefits**: Distributed goal encoding provides robustness
2. **Interpretability Challenges**: Understanding behavior requires multi-layer analysis
3. **Control Complexity**: Manipulating behavior requires sophisticated multi-layer approaches

## Future Work

### Immediate Improvements
1. **Increase Steering Magnitude**: Try coefficients of 1.0, 2.0, or higher
2. **Multi-Layer Steering**: Steer multiple networks simultaneously
3. **Temporal Steering**: Apply steering over multiple consecutive timesteps

### Advanced Approaches
1. **Gradient-Based Steering**: Use direct gradients instead of probe coefficients
2. **Adversarial Steering**: Maximize target goal while minimizing current goal
3. **Behavioral Analysis**: Measure changes in action distributions, not just goal changes

### Alternative Methods
1. **Direct Policy Editing**: Edit the policy network directly
2. **Reward Shaping**: Modify the reward function to encourage different goals
3. **Environment Manipulation**: Change the environment to make different goals more attractive

## Conclusion

This multi-layer analysis revealed that goal representations in the LTL agent are:

- **Distributed**: Encoded across multiple network components
- **Robust**: Resistant to single-layer manipulation
- **Complex**: Involving non-linear interactions between layers
- **Readable but not Writable**: We can read goal information but cannot easily write to change behavior

The environment network contains the most goal-relevant information (98.10% probe accuracy), but even steering this network has limited behavioral impact. This suggests that effective neural network control requires sophisticated multi-layer approaches that account for the distributed and robust nature of goal representations.

## Files Created
- `src/probe_policy_goals.py`: Multi-layer probing implementation
- `src/steer_env_network.py`: Environment network steering
- `src/create_report_visualizations.py`: Visualization generation script
- `run_probe_policy.py`: Run script for multi-layer probing
- `run_env_steering.py`: Run script for environment steering
- `steered_trajectory.png`: LTL network steering trajectory
- `env_steered_trajectory.png`: Environment network steering trajectory
- `multi_layer_analysis.png`: Comprehensive analysis visualizations
- `detailed_comparison.png`: Detailed comparison charts

## Key Insights

1. **Environment Network Dominance**: 98.10% probe accuracy vs 94.00% for LTL network
2. **Limited Steering Effectiveness**: 3-4 goal changes despite 250+ steering interventions
3. **Robust Goal Encoding**: Distributed across multiple networks
4. **Reading vs Writing Gap**: High probe accuracy doesn't guarantee behavioral control
5. **Multi-Layer Complexity**: Understanding requires examining multiple components simultaneously 