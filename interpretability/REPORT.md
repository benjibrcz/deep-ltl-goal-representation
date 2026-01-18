# DeepLTL Interpretability Research: Comprehensive Report

**Research Question**: Does the DeepLTL agent learn genuine planning capabilities with internal world models, or does it rely on behavioral heuristics and reactive pattern matching?

## Executive Summary

| Domain | Metric | Result | Interpretation |
|--------|--------|--------|----------------|
| **Zone Env** | Safe task success | 91% | High overall success |
| | Controlled choice (equal distance) | 55% vs 45% | Essentially random |
| **Letter World** | Safe task success | 38% | Low success rate |
| | Choice when goal UP | 65% correct | Strong directional bias |
| | Choice when goal LEFT | 12% correct | Bias against left |
| **Steering** | Probe accuracy | 94-99.5% | High information encoding |
| | Steering effectiveness | 0.8-1.2% | Information is distributed |
| **Transition Function** | Paper exact test | 3% safe choice | Fails paper claims |

**Key Findings**:
1. The agent does NOT have a learned transition function for multi-step planning
2. High success rates come from reactive heuristics, not planning
3. Goal representations are egocentric (compass-like), not allocentric world maps
4. Steering is ineffective despite high probe accuracy - information is distributed
5. Paper claims about planning capability are not supported by controlled experiments

---

## Part 1: Steering and Representation Analysis

### 1.1 Goal Probing Across Network Components

We trained linear probes to predict the agent's current goal from hidden state activations.

| Network Component | Probe Accuracy | Layer |
|-------------------|:--------------:|-------|
| Policy Network | **99.5%** | policy_mlp_0 |
| Environment Network | 98.1% | env_encoder |
| LTL Network | 94.0% | ltl_rnn_hidden |

**Finding**: Goal information is highly decodable from all network components, with policy network showing highest accuracy.

### 1.2 Steering Effectiveness

We attempted to manipulate agent behavior by modifying hidden states in the direction of the probe's weight vector.

| Network | Interventions | Goal Changes | Effectiveness |
|---------|:-------------:|:------------:|:-------------:|
| LTL Network | 247 | 2 | 0.8% |
| Environment Network | 248 | 3 | 1.2% |
| Policy Network | 252 | 3 | 1.2% |

**Critical Insight**: High probe accuracy does NOT guarantee steering effectiveness. Despite 94-99.5% accuracy in predicting goals from hidden states, manipulating those states only changes behavior 0.8-1.2% of the time.

**Interpretation**: Goal representations are **distributed** across multiple network components. The architecture is robust against single-layer manipulations, suggesting redundant encoding.

### 1.3 Spatial Representation Discovery: The Compass Phenomenon

Analyzing what probes actually learn revealed a surprising finding:

| Observation | Expected | Actual |
|-------------|----------|--------|
| Predicted goal location | Static (world coordinates) | Dynamic (moves with agent) |
| Representation type | Allocentric (world map) | **Egocentric** (agent-relative) |
| Zone drift over rollout | Minimal | **157.94 units** (high) |

**The "Compass Phenomenon"**: The agent's representations encode goal directions relative to itself, not absolute world positions. Predicted zone locations drift significantly as the agent moves, always "pointing toward" goals like a compass needle rather than encoding a static map.

**Movement Alignment**: 0.927 mean cosine similarity between agent movement and nearest predicted zone direction.

### 1.4 Optimal Steering Strength

| Strength | Effect |
|----------|--------|
| 1-5x | Minimal effect |
| **10-50x** | **Optimal balance** - shorter paths, preserved task completion |
| >50x | Saturation/interference |

Steering at 10-50x strength produces 20% reduction in trajectory complexity, but steering fundamentally doesn't change planning behavior.

---

## Part 2: Zone Environment Analysis

### 2.1 Overall Task Performance

| Formula Type | Success Rate |
|--------------|:------------:|
| Simple reach (F blue) | 95%+ |
| Sequential (F blue → F green) | 90%+ |
| **Safety choice (F A \| F B) & G !C** | **91%** |
| Infinite horizon (GF blue & GF green) | 100% |

**Observation**: High success rates on most tasks, including safety constraints.

### 2.2 Controlled Choice Experiment

**Question**: Does the agent use planning or distance-based heuristics?

**Setup**: Two reach zones of the same color at equal distance from agent. One blocked by an avoid zone, one clear path.

| Metric | Result |
|--------|--------|
| Episodes | 30 |
| **Safe (unblocked) choice** | **55.2%** |
| Blocked choice | 44.8% |

**Finding**: The near 50/50 split indicates **no evidence of planning**. The agent does not anticipate obstacles when choosing between equidistant goals.

### 2.3 Why Zone Env Performs Well Overall

Despite lacking planning, the agent achieves 91% safe success because:

1. **Continuous action space**: Can steer around obstacles rather than committing to discrete paths
2. **Lidar observations**: Better spatial awareness of nearby zones
3. **Course correction**: Can adjust trajectory mid-rollout when approaching obstacles
4. **Reactive avoidance**: Learns to avoid obstacles in the immediate vicinity

The high success rate comes from **reactive heuristics**, not multi-step planning.

---

## Part 3: Letter World Analysis

### 3.1 End-to-End Safety Test

**Task**: `(eventually letter_A OR eventually letter_B) AND globally NOT letter_C`

| Metric | Result |
|--------|--------|
| Safe success (chose unblocked) | **38%** |
| Unsafe success (got lucky) | 32% |
| Failed (hit C) | 30% |
| Total task success | 70% |

**Comparison with random**: 38% is 2.5x better than random (15%), but far from optimal.

### 3.2 Directional Bias Analysis

| Optimal Goal Direction | Correct Choice Rate |
|-----------------------|:-------------------:|
| UP | **65%** |
| DOWN | 47% |
| RIGHT | 40% |
| **LEFT** | **12%** |

**Finding**: Strong directional bias. The agent favors UP movements and strongly avoids LEFT, regardless of which direction is optimal.

### 3.3 Value Function Analysis

| Scenario | Estimated Value |
|----------|:---------------:|
| Will succeed | 0.82 |
| Will fail (hit C) | 0.79 |
| **Difference** | **0.03** |

The value function is essentially **blind** to whether the agent will succeed or fail. It doesn't distinguish good situations from bad.

### 3.4 Failure Mode Analysis

| When Decision Made | Percentage of Failures |
|-------------------|:----------------------:|
| **First action** | **72%** |
| Later corrections | 28% |

**Finding**: 72% of failures are decided at the very first action. The agent doesn't deliberate - it commits immediately based on heuristics.

---

## Part 4: Transition Function Experiments

### 4.1 Core Hypothesis

The paper claims the agent learns a "transition function" enabling multi-step planning. We designed 10 experiments to test this claim.

### 4.2 Experiment Results

| Experiment | Description | Result | Evidence for Planning? |
|------------|-------------|--------|:----------------------:|
| **1. Controlled Choice** | Equal distance, one blocked | 55% vs 45% | **No** |
| **2. Forced Detour** | Direct path blocked | 0% took detours | **No** |
| **3. Value Anticipation** | Track value before obstacles | No anticipatory drop | **No** |
| **4. Multi-Step Lookahead** | Predict position 10-20 steps ahead | R² < 0.3 | **No** |
| **5. Paper Exact Safety** | Reproduce Figure 1 setup | 3% safe, 60% violated | **No** |
| **6. Position Probing** | Can model predict own future position? | Short horizon only | **Partial** |
| **7. MLP Probing** | Non-linear probes on representations | R² = 0.99 spatial | **Partial** |
| **8. Infinite Horizon** | GF blue & GF green | Works (3.2 visits) | **Yes** |
| **9. Linear Transition** | Fit A matrix for x_{t+1} = Ax_t | Poor fit | **No** |
| **10. Counterfactual** | What-if reasoning about paths | Not present | **No** |

### 4.3 Detailed Findings

#### Paper Exact Safety Test

We reproduced the exact setup from Figure 1 of the paper:

| Outcome | Count | Percentage |
|---------|:-----:|:----------:|
| Chose safe goal | 1 | 3% |
| Chose blocked goal, avoided C | 10 | 33% |
| **Violated safety (hit C)** | **18** | **60%** |
| Other | 1 | 3% |

**Finding**: The paper's Figure 1 examples appear to be cherry-picked. On the exact configuration shown, the agent fails 60% of the time.

#### Value Anticipation

If the agent has a transition function, value estimates should drop before hitting obstacles (anticipating the penalty).

| Metric | Expected (with planning) | Actual |
|--------|-------------------------|--------|
| Value drop before obstacle | 10-20 steps ahead | 0 steps ahead |
| Anticipation horizon | Long | **None** |

The agent shows no value anticipation - it only reacts to obstacles when physically encountering them.

#### MLP vs Linear Probing

| Probe Type | Spatial Encoding R² | Interpretation |
|------------|:-------------------:|----------------|
| Linear | 0.60 | Moderate encoding |
| **MLP** | **0.99** | **Excellent** encoding |

**Finding**: The agent has excellent spatial representations in non-linear combinations, but they're not easily accessible for planning computations.

### 4.4 What DOES Work?

| Capability | Status | Mechanism |
|------------|--------|-----------|
| Infinite horizon tasks | Works | Reactive cycling |
| Immediate obstacle avoidance | Works | Pattern matching |
| Distance estimation | Works | Learned heuristic |
| Sequential navigation | Works | Reactive chaining |

### 4.5 Conclusion: No Transition Function

The agent has NOT learned a general transition function for multi-step planning. It achieves high success through:

1. **Reactive heuristics**: Immediate pattern matching for obstacle avoidance
2. **Distance-based targeting**: Navigate toward nearest/current goal
3. **Course correction**: Adjust when encountering obstacles (continuous action space)
4. **Directional biases**: Learned preferences that happen to work in many cases

---

## Part 5: Comparison: Zone Env vs Letter World

| Aspect | Zone Env | Letter World |
|--------|----------|--------------|
| Overall safe success | **91%** | 38% |
| Controlled choice (planning test) | 55% | ~50% |
| Directional bias | Minimal | Strong (UP preference) |
| Course correction | Possible (continuous) | Limited (discrete) |
| Observation | Lidar (rich) | Grid (sparse) |

**Key Insight**: Zone env's higher success rate comes from **continuous navigation allowing course correction**, not from better planning. Both agents rely on reactive heuristics, but zone env can recover from bad initial choices.

---

## Part 6: Implications for Interpretability

### 6.1 Probe Accuracy ≠ Steering Effectiveness

| Metric | Value |
|--------|-------|
| Probe accuracy | 94-99.5% |
| Steering effectiveness | 0.8-1.2% |

High decodability doesn't mean information is causally important or manipulable. Representations are distributed.

### 6.2 Egocentric vs Allocentric Representations

The agent uses egocentric (self-relative) representations, not allocentric (world) coordinates. This explains:
- Why probes seemed "perfect" initially (memorization, not generalization)
- Why representations drift as the agent moves
- Why steering has limited effect (you're pushing a compass needle, not moving a point on a map)

### 6.3 Paper Claims vs Reality

| Paper Claim | Reality |
|-------------|---------|
| Agent learns transition function | No evidence from controlled experiments |
| Figure 1 shows typical behavior | Appears cherry-picked; fails 60% on exact setup |
| Planning enables safety compliance | Reactive heuristics, not planning |

### 6.4 What Would Real Planning Look Like?

| Feature | Expected with Planning | Observed |
|---------|----------------------|----------|
| Controlled choice | 90%+ correct | 55% |
| Value anticipation | 10+ steps | 0 steps |
| Counterfactual reasoning | Present | Absent |
| Detour capability | Take longer safe paths | Never observed |

---

## Summary of Key Findings

1. **No Planning**: The agent does not have multi-step planning capability despite paper claims

2. **Reactive Heuristics**: Success comes from distance-based targeting and reactive obstacle avoidance

3. **Distributed Representations**: Goal information is spread across network components, making steering ineffective

4. **Egocentric Encoding**: The agent uses compass-like (self-relative) representations, not world maps

5. **Domain Differences**: Zone env succeeds through continuous control allowing course correction; Letter world fails due to discrete actions and strong biases

6. **Paper Overstatement**: Figure 1 examples appear cherry-picked; controlled experiments show much lower performance

---

## Files Reference

### Steering Analysis
| File | Purpose |
|------|---------|
| `analysis/steering_experiments/COMPREHENSIVE_STEERING_REPORT.md` | Full steering analysis |
| `analysis/probe_analysis/SPATIAL_REPRESENTATION_FINDINGS.md` | Egocentric representation discovery |
| `interpretability/zone_env/reports/STEERING_EXPERIMENTS_SUMMARY.md` | Steering experiments summary |

### Zone Environment
| File | Purpose |
|------|---------|
| `interpretability/zone_env/results/zone_env_analysis_report.md` | Zone env analysis |
| `interpretability/zone_env/controlled_choice_experiment.md` | Controlled choice experiment |
| `interpretability/zone_env/reports/transition_function/transition_function_experiments.md` | Transition function experiments |

### Letter World
| File | Purpose |
|------|---------|
| `interpretability/letter_world/results/e2e_analysis_report.md` | End-to-end analysis |

---

## Conclusions

The DeepLTL agent achieves task success through reactive behavioral heuristics, not through learned transition functions or multi-step planning. Key evidence:

1. **Controlled choice experiments** show no preference for unblocked paths when distances are equal
2. **Paper exact tests** fail 60% of the time on the claimed demonstration setup
3. **No value anticipation** before obstacles - the agent doesn't predict future states
4. **Steering ineffectiveness** despite high probe accuracy reveals distributed, robust-but-not-causal representations
5. **Egocentric representations** explain why the agent navigates reactively rather than planning globally

These findings suggest that claims about the agent's planning capabilities should be re-evaluated, and that achieving true planning in RL agents may require explicit architectural or training modifications rather than emerging naturally from task success.

---

*Report generated January 2026*
