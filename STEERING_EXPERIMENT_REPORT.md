# Goal Steering Experiment Report

## Overview

This report documents our attempt to **steer** the LTL agent's goal representations by directly editing the hidden states of the LTL network during rollout. The goal was to change the agent's behavior from seeking blue zones to green zones (or vice versa) by manipulating the internal goal representations we discovered in the probing experiment.

## Experiment Design

### What We Did
1. **Trained Goal Probes**: Created logistic regression probes that could predict blue vs green goals from LTL hidden states with 97.5% accuracy
2. **Implemented Steering**: Used the probe coefficients as steering directions to edit hidden states
3. **Applied Interventions**: Steered 50% of the time when the agent had a clear goal (blue or green)
4. **Measured Impact**: Tracked how many times the agent actually changed its goal behavior

### Key Parameters
- **Environment**: `PointLtl2-v0` with formula `"GF blue & GF green"`
- **Steering Frequency**: 50% of steps
- **Steering Magnitude**: 0.5 × probe coefficient
- **Rollout Length**: 500 steps

## Results

### Quantitative Findings
- **252 steering interventions** (50.4% of steps)
- **Only 3 natural goal changes** during the entire rollout
- **97.5% probe accuracy** for both blue and green goal prediction
- **Limited behavioral change** despite extensive steering

### Key Insight
**The steering was largely ineffective** - even though we successfully identified goal representations in the hidden states and applied 252 steering interventions, the agent only changed its goal behavior 3 times naturally.

## Interpretation

### What This Means
1. **Probes Work, Steering Doesn't**: We can successfully *read* goal information from hidden states, but we cannot easily *write* to change behavior
2. **Complex Goal Encoding**: Goal representations are more sophisticated than simple linear mappings
3. **Redundant Mechanisms**: The agent likely has multiple pathways encoding the same goal information
4. **Temporal Dependencies**: Goal state may depend on history, not just current hidden state

### Possible Explanations

#### 1. Non-Linear Representations
The goal information might be encoded non-linearly in the hidden states, making simple linear steering insufficient.

#### 2. Redundant Encoding
The agent may have multiple neural pathways that encode the same goal, so editing one pathway doesn't change the overall behavior.

#### 3. Temporal Dynamics
Goal state might depend on the entire history of hidden states, not just the current one.

#### 4. Insufficient Steering Magnitude
Our steering coefficient of 0.5 might have been too small to overcome the agent's learned behavior.

#### 5. Wrong Steering Direction
We might have been steering in the wrong direction or targeting the wrong aspects of the representation.

## Technical Details

### Steering Method
```python
# Get steering direction from probe coefficients
steer_direction = probe.coef_[0]

# Apply steering to hidden state
steered_hidden = hidden_state + steer_direction * 0.5
```

### Hook Implementation
```python
def hook_fn(mod, inp, out):
    h_n = out[1]  # Final hidden state
    
    # Check if we should steer
    if should_steer(current_goal):
        # Edit hidden state
        new_h_n = apply_steering(h_n, current_goal)
        return (out[0], new_h_n, out[2])
    
    return out
```

## Implications

### For Interpretability Research
1. **Reading vs Writing**: Successfully reading representations doesn't guarantee the ability to write/control them
2. **Representation Complexity**: Neural representations may be more complex than linear probes suggest
3. **Causal vs Correlative**: High probe accuracy doesn't imply causal control over behavior

### For AI Safety
1. **Robustness**: The agent's goal representations appear robust to manipulation
2. **Multiple Pathways**: Redundant encoding mechanisms may make systems harder to control
3. **Temporal Aspects**: Goal state may depend on history, not just current state

## Future Work

### Immediate Improvements
1. **Increase Steering Magnitude**: Try coefficients of 1.0, 2.0, or higher
2. **Gradient-Based Steering**: Use direct gradients instead of probe coefficients
3. **Non-Linear Probes**: Train neural network probes instead of linear ones

### Advanced Approaches
1. **Adversarial Steering**: Maximize target goal probability while minimizing current goal
2. **Multi-Layer Steering**: Steer multiple layers of the network simultaneously
3. **Temporal Steering**: Apply steering over multiple consecutive timesteps
4. **Behavioral Analysis**: Measure changes in action distributions, not just goal changes

### Alternative Methods
1. **Direct Policy Editing**: Edit the policy network directly instead of LTL network
2. **Reward Shaping**: Modify the reward function to encourage different goals
3. **Environment Manipulation**: Change the environment to make different goals more attractive

## Conclusion

This experiment revealed an important distinction between **representation reading** and **representation writing** in neural networks. While we can successfully identify goal representations in the LTL network's hidden states (94% probe accuracy), directly editing these representations has limited impact on behavior.

This suggests that:
- Goal representations are more complex than linear probes indicate
- The agent has robust, redundant goal encoding mechanisms
- Simple linear steering is insufficient for behavioral control
- More sophisticated steering methods are needed for effective representation editing

The experiment provides valuable insights into the challenges of neural network interpretability and control, highlighting the gap between understanding representations and being able to manipulate them effectively.

## Files Created
- `src/steer_goals.py`: Main steering implementation
- `src/analyze_steering.py`: Analysis and visualization script
- `run_steer_goals.py`: Run script for the experiment
- `steered_trajectory.png`: Original trajectory plot
- `steering_analysis.png`: Enhanced analysis visualization 