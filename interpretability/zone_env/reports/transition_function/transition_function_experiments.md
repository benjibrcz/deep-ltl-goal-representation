# Transition Function Experiments: Zone Environment

## Example Trajectories

| Simple Reach | Reach-Avoid | Safety |
|:---:|:---:|:---:|
| ![Simple Reach](figures/simple_reach_example.png) | ![Reach-Avoid](figures/reach_avoid_example.png) | ![Safety](figures/safety_example.png) |
| `F blue` | `!yellow U blue` | `(F green \| F yellow) & G !blue` |

| Optimality (Sequencing) | Infinite Horizon |
|:---:|:---:|
| ![Optimality](figures/optimality_example.png) | ![Infinite Horizon](figures/infinite_horizon_example.png) |
| `F (blue & F green)` | `G F blue & G F green` |

*Trajectories colored from dark (start) to bright (end). Numbers indicate zone visit order.*

---

## Summary

We ran ten experiments (seven behavioral, three probing) to test whether the zone environment agent has learned a transition function (world model) that enables planning:

| Experiment | Type | Evidence for Planning? | Key Finding |
|------------|------|----------------------|-------------|
| Controlled Choice | Behavioral | **No** | 55% vs 45% when distances equal - near random |
| Forced Detour | Behavioral | **No** | 0% took detours; agent goes straight or gets stuck |
| Sequencing | Behavioral | **Partial** | 59% correct order (better than 39% distance heuristic) |
| Value Anticipation | Behavioral | **No** | Value doesn't drop before obstacle contact |
| Position Probing (Linear) | Representational | **No** | Linear probe R²=0.61 (misleading - see non-linear) |
| Position Probing (MLP) | Representational | **YES** | MLP probe R²=0.99 - excellent spatial encoding! |
| Targeted: Optimality | Behavioral | **No** | 40% chose optimal, 60% chose greedy (closer) |
| Targeted: Safety Planning | Behavioral | **No** | 50% chose safe, 50% chose blocked (random) |
| **Exact Paper Safety** | Behavioral | **No** | 3% chose safe, 60% safety violations - fails paper claim |
| Paper: Infinite Horizon | Behavioral | **YES** | 3.2 avg visits per goal, alternating successfully |

**Overall conclusion**: The agent has NOT learned a general transition function for planning. It relies on:
1. **Distance-based heuristics** for goal selection
2. **Reactive obstacle avoidance** (not anticipatory)
3. **Some LTL goal representation** (for sequencing) but not robust
4. **No predictive world model** in hidden representations

---

## Critical Finding: Paper Claims vs Reality

### Investigation Summary

We investigated whether the paper's Figure 1 demonstrations (showing "planning" capabilities) reflect actual agent behavior or are cherry-picked examples.

### What We Found

**1. Training curriculum is identical to ours**

Checked the original paper repository at commit `958f3a8` ("ICLR camera ready"):
```python
# From the paper's ZONES_CURRICULUM - NO disjunction, NO global safety
Stage 0: F color                           # Simple reach
Stage 1: F (color1 & F color2)             # 2-step sequences
Stage 2: !avoid U reach                    # Reach-avoid (1 step)
Stage 3-6: Deeper reach-avoid sequences    # More complex sequences
```

**The training does NOT include:**
- ❌ Disjunction: `F A | F B` (reach A OR B)
- ❌ Global safety: `G !avoid` (always avoid)
- ❌ Combined: `(F A | F B) & G !C`

**2. Evaluation dataset also excludes these formulas**

From `eval_datasets/PointLtl2-v0/tasks.txt` (50 tasks):
```
(!green U (yellow & (!blue U magenta)))
F (yellow & F (blue & F yellow))
(!magenta U (blue & (!yellow U green)))
...
```
**Zero tasks contain `|` (OR) or `G` (globally).**

**3. Figure 1 configurations were hand-crafted**

The file `ltl_fixed.py` contains manually-placed zone positions specifically for Figure 1:
```python
# Safety configuration (for Figure 1c)
green at (1.2, -1.9)   # closer but BLOCKED
yellow at (1.1, 2.1)   # farther but SAFE
blue zones blocking path to green
```

**4. Agent fails on paper's exact configuration**

| Test | Paper's Claim | Our Results |
|------|---------------|-------------|
| Safety (Figure 1c) | Agent chooses safe path | **3% chose safe, 60% violated safety** |
| Optimality (Figure 1b) | Agent chooses optimal path | **40% optimal, 60% greedy** |
| In-distribution sequencing | N/A | **50% correct order** |

### Conclusion

**The paper's Figure 1 examples appear to be cherry-picked.** Evidence:

1. The agent was NOT trained on disjunction or global safety formulas
2. The agent was NOT evaluated on these formulas in standard evaluation
3. The fixed zone configurations in `ltl_fixed.py` were hand-crafted for the figures
4. When we run many trials on these exact configurations, the agent mostly fails
5. Even on in-distribution tasks, the agent shows ~50% optimal behavior

**The agent does not demonstrate robust planning.** Its apparent "planning" in the paper's figures likely represents:
- Lucky successful runs from many trials
- Scenarios where distance heuristics happen to give the right answer
- Reactive obstacle avoidance that sometimes works

---

## Experiment 1: Controlled Choice

**Question**: When two goals are equidistant and one is blocked, does the agent choose the unblocked path?

**Setup**:
- Two reach zones at equal distance (within 0.5 units)
- One has avoid zone blocking direct path
- Tests: Does agent anticipate obstacle?

**Results** (N=30):
| Choice | Percentage |
|--------|------------|
| Safe (unblocked) | 55.2% |
| Blocked | 44.8% |

**Interpretation**: Near 50/50 split indicates **no obstacle anticipation**. A planning agent would consistently choose the unblocked path.

---

## Experiment 2: Forced Detour

**Question**: Can the agent navigate around obstacles by taking a longer path?

**Setup**:
- Goal with 1-2 avoid zones blocking direct path
- Tested with `F reach` (simple) and `!avoid U reach` (reach-avoid)
- Measured: Did agent take a detour (initially move away from goal)?

**Results** (N=30):

| Formula | Success Rate | Took Detour |
|---------|--------------|-------------|
| Simple reach (F blue) | 56.6% | **0%** |
| Reach-avoid (!yellow U blue) | 20.0% | **0%** |

**Key observations**:
1. **0% detour rate**: No agent ever deliberately moved away from goal to go around
2. **Simple reach**: Agent just plows through obstacles (43% risky success)
3. **Reach-avoid**: Agent gets "stuck" - knows to avoid but can't find alternative path (80% neither)

**Interpretation**: **No evidence of transition function**. The agent:
- Cannot plan multi-step paths around obstacles
- Either ignores obstacles (simple reach) or gets paralyzed (reach-avoid)

---

## Experiment 3: Sequencing

**Question**: Can the agent visit zones in a required order?

**Setup**:
- 2-step sequences: `F (blue & F green)` - visit blue then green
- 3-step sequences: `F (blue & F (green & F yellow))`
- Compared to: distance heuristic (visit nearest first)

**Results**:

| Sequence Length | Correct Order | Distance Heuristic |
|-----------------|---------------|-------------------|
| 2-step | **59.2%** | 39.2% |
| 3-step | **26.7%** | 13.3% |

**Interpretation**:
- Agent performs **better than distance heuristic** on sequencing
- This suggests some goal order representation (possibly LTL automaton state)
- But performance degrades significantly with length (59% → 27%)
- Not evidence of transition function - just goal tracking

---

## Experiment 4: Value Anticipation

**Question**: Does the agent's value function drop BEFORE hitting an obstacle, or only AFTER contact?

**Setup**:
- Track value estimates during rollouts approaching avoid zones
- Categorize by distance: far (>3.2), medium (1.6-3.2), close (0.8-1.6), very_close (0.4-0.8), in_zone (<0.4)
- Compare value when approaching vs moving away from obstacles

**Results** (N=50 rollouts, ~5000 timesteps):

| Distance Category | Mean Value | Std |
|-------------------|------------|-----|
| far | 0.43 | 0.42 |
| medium | 0.85 | 0.19 |
| close | 0.82 | 0.21 |
| very_close | 0.86 | 0.15 |
| in_zone | 0.85 | 0.17 |

**Key observations**:
1. Value is LOW when far from goal (0.43), HIGH when closer (0.82-0.86)
2. **No anticipation**: Value doesn't drop when approaching obstacles
3. Value when approaching obstacle: 0.82, value when moving away: 0.82 (identical)
4. Value drop far→very_close: -0.44 (NEGATIVE - value increases!)
5. Value drop very_close→in_zone: 0.01 (negligible)

**Interpretation**: **NO ANTICIPATION**. The agent:
- Has no predictive model of future obstacle contact
- Value is purely a function of distance-to-goal, not path safety
- Is completely reactive to obstacles

---

## Experiment 5: Position Probing

**Question**: Can we decode position/next-position from the agent's hidden state?

**Setup**:
- Collect hidden states (96-dim embeddings) during rollouts
- Train linear probes (Ridge regression) to decode various quantities
- Compare to baseline: predicting from raw position

**Results** (N=50 rollouts, 4920 datapoints):

| Probe Target | From Hidden | Baseline | Interpretation |
|--------------|-------------|----------|----------------|
| Current position | R²=0.61 | - | Partial encoding |
| Next position | R²=0.61 | R²=0.9998 (from pos) | No improvement |
| Next position + action | R²=0.61 | R²=0.9998 | Action doesn't help |
| Velocity | R²=0.48 | - | Poor encoding |
| Distance to goal | R²=0.62 | - | Partial encoding |
| Distance to avoid | R²=0.55 | - | Poor encoding |

**Key observations**:
1. Hidden state only partially encodes current position (R²=0.61)
2. **No transition function**: Next-position prediction from hidden is no better than current position
3. Adding action to hidden doesn't improve prediction (0.61 vs 0.61)
4. Baseline shows next ≈ current (R²=0.9998) due to small timesteps
5. Velocity poorly encoded (R²=0.48)

**Interpretation**: **NO TRANSITION FUNCTION IN HIDDEN STATE**. The agent:
- Doesn't have a predictive world model in its representations
- Hidden state is more about goal-directed behavior than state prediction
- Cannot support planning even in principle

---

## Experiment 6: Non-linear Probing (Corrected)

**Important correction**: Our earlier linear probing gave misleadingly low R² scores because the lidar observations use **exponential distance encoding** (`exp(-dist)`). Linear probes cannot decode non-linear representations.

**Setup**:
- Compare linear (Ridge regression) vs MLP (2-layer neural network) probes
- Test at different layers: raw_features → env_net → ltl_net → embedding
- Probe for: position, velocity, distances

**Results** (N=50 rollouts, 4920 datapoints):

| Target | Raw Features ||| env_net ||| ltl_net ||| Embedding |||
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | Lin | MLP | | Lin | MLP | | Lin | MLP | | Lin | MLP |
| Position | 0.95 | **0.99** | | 0.57 | **0.99** | | 0.11 | 0.11 | | 0.61 | **0.99** |
| Velocity | 0.41 | **0.99** | | 0.47 | **0.99** | | 0.12 | 0.12 | | 0.48 | **0.99** |
| Dist to goal | 0.57 | **0.98** | | 0.58 | **0.99** | | 0.02 | 0.02 | | 0.62 | **0.99** |
| Dist to avoid | 0.50 | **0.98** | | 0.53 | **0.98** | | 0.00 | 0.00 | | 0.55 | **0.99** |

**Key findings**:
1. **MLP dramatically outperforms linear**: R² jumps from ~0.5-0.6 to ~0.98-0.99
2. **The model DOES encode spatial information extremely well** - just non-linearly
3. **env_net preserves all spatial information** from raw features
4. **ltl_net encodes NO spatial information** (R² ≈ 0) - only goal/formula state
5. **Information is non-linearly encoded** due to lidar's `exp(-dist)` representation

**Interpretation**: The agent has **excellent spatial representations** (position, velocity, distances all decodable with R²>0.98). The earlier linear probing results were misleading due to non-linear encoding. However, this doesn't change the behavioral findings - the agent still doesn't use these representations for planning.

---

## Experiment 7-9: Paper Capability Tests (Figure 1)

These experiments directly test the three capabilities claimed in the DeepLTL paper Figure 1.

### 7a. Optimality Test (Targeted)

**Question**: Does the agent choose globally optimal paths over greedy/myopic ones?

**Setup** (from paper Figure 1b):
- Task: `F (blue & F green)` - reach blue, then green
- Specifically filter for scenarios where:
  - Farther blue leads to shorter total path (optimal choice)
  - Closer blue leads to longer total path (greedy choice)
  - Path difference ≥ 0.5 units

**Results** (N=20 targeted scenarios):

| Metric | Value |
|--------|-------|
| Chose OPTIMAL (farther-first) | **40%** |
| Chose GREEDY (closer-first) | **60%** |
| Average path difference | 0.91 units |

**Interpretation**: **NO OPTIMAL PLANNING**. When the optimal choice requires going to a farther first target, the agent still prefers the closer one 60% of the time. This is essentially greedy behavior with slight noise, not evidence of multi-step planning.

### 7b. Safety Planning Test (Targeted)

**Question**: Does the agent choose a farther-but-safe goal over a closer-but-blocked goal?

**Setup**:
- Task: `(F green | F yellow) & G !blue` - reach green OR yellow while ALWAYS avoiding blue
- Specifically filter for scenarios where:
  - One goal has blue zones blocking the direct path
  - Other goal has a clear path
  - Tests if agent anticipates obstacle and plans around it

**Results** (N=20 targeted scenarios):

| Metric | Value |
|--------|-------|
| Chose SAFE path | **50%** |
| Chose BLOCKED path | **50%** |
| Safety violations (touched blue) | 25% |
| Scenarios where safe path is farther | 5% |

**Interpretation**: **NO SAFETY PLANNING**. When one path is blocked by blue zones and one is clear, the agent chooses randomly (50/50). This indicates the agent doesn't anticipate obstacles when choosing goals. The 25% safety violation rate in these challenging scenarios contrasts with near-perfect safety in simpler scenarios, suggesting reactive (not anticipatory) obstacle avoidance.

### 7b+. Exact Paper Safety Test (Fixed Configuration)

**Question**: Using the exact zone configuration from Figure 1c, does the agent choose the safe path?

**Setup** (exact replica of paper Figure 1c):
- Task: `(F green | F yellow) & G !blue`
- Fixed zone positions (from `ltl_fixed.py`):
  - Green at (1.2, -1.9) - distance 2.73 from agent (CLOSER, but BLOCKED)
  - Yellow at (1.1, 2.1) - distance 3.55 from agent (FARTHER, but SAFE)
  - 3 blue zones blocking path to green
- Agent starts at (-1.2, -0.6)

**Results** (N=30):

| Metric | Value |
|--------|-------|
| Chose YELLOW (safe, farther) | **3.3%** |
| Chose GREEN (blocked, closer) | **26.7%** |
| Reached neither goal | **70.0%** |
| Safety violations (touched blue) | **60.0%** |

**Interpretation**: **DRAMATIC FAILURE OF PLANNING**. In the exact paper configuration:
- Agent almost never chooses the safe path (3.3%)
- Repeatedly tries to reach closer goal and gets blocked
- 60% of runs violate safety constraint by touching blue
- 70% of runs fail to reach ANY goal

This directly contradicts the paper's claimed planning capability. The agent is clearly using **distance-based goal selection** (go to closer green) and **reactive obstacle avoidance** (gets stuck when blocked), not anticipatory planning.

**See "Critical Finding: Paper Claims vs Reality" above** for our investigation showing that the paper's training curriculum does NOT include disjunction or global safety formulas, and that the Figure 1 examples appear to be cherry-picked.

### 7c. Infinite Horizon Test

**Question**: Can the agent repeatedly visit multiple goals indefinitely?

**Setup** (from paper Figure 1a):
- Task: `G F blue & G F green` - infinitely often visit blue AND green
- Tests ω-regular task capability

**Results** (N=50, 300 steps each):

| Metric | Value |
|--------|-------|
| Average blue visits | 4.3 |
| Average green visits | 5.3 |
| Average alternating visits | **3.2** |

**Interpretation**: **INFINITE HORIZON CAPABILITY CONFIRMED**. Agent successfully alternates between visiting both colors multiple times per episode.

### Key Insight: Where Does Planning Come From?

The results reveal a critical distinction:

| Capability | Evidence? | Mechanism |
|------------|-----------|-----------|
| Safety constraint satisfaction | **Partial** (reactive) | Reactive avoidance, not anticipatory |
| Infinite horizon | **YES** (3.2 alternations) | LTL goal sequencing |
| Optimal path planning | **NO** (60% greedy) | NOT learned |
| Safety-aware goal selection | **NO** (50/50 random) | NOT learned |

**Conclusion**: The agent's "planning" capabilities come from the **LTL sequence search / automaton**, NOT from a learned transition function or world model. The neural network provides:
- Distance-based goal selection (go to closest target)
- Reactive obstacle avoidance (steer away when close)
- Value estimates for the current state

But it does NOT provide:
- Predictive world model
- Multi-step path planning
- Anticipatory obstacle avoidance
- Planning to choose safer/optimal paths over greedy ones

---

## What These Results Mean

### The Agent DOES Have:
1. **Goal-directed navigation**: Can reach single goals efficiently
2. **Reactive obstacle avoidance**: Can steer around obstacles once nearby
3. **Some goal sequencing**: Tracks which goals to visit (via LTL automaton)
4. **Excellent spatial representations**: Position, velocity, distances all decodable with R²≈0.99 (non-linear)
5. **Reactive safety compliance**: Avoids obstacles when nearby (but doesn't anticipate)

### The Agent Does NOT Have:
1. **Multi-step planning**: Cannot plan detours around obstacles (0% detour rate)
2. **Anticipatory behavior**: Value doesn't drop before obstacle contact
3. **Optimal path selection**: 60% greedy when greedy ≠ optimal
4. **Safety-aware goal selection**: 50/50 random when choosing between blocked vs clear paths
5. **Transition function USE**: Has good state representations but doesn't use them for prediction

### Why Zone Env Has 91% Success Rate Anyway

The high success rate comes from **continuous control enabling course correction**, not planning:
- Agent heads toward goal using distance heuristic
- When near obstacle, reactive avoidance kicks in
- Continuous action space allows steering around
- This works well when obstacles don't completely block path

---

## Probing Results Summary

**Important**: Linear probes gave misleadingly low scores. With MLP (non-linear) probes:

| Representation | Linear R² | MLP R² | Verdict |
|----------------|-----------|--------|---------|
| Current position | 0.61 | **0.99** | Excellent (non-linear) |
| Velocity | 0.48 | **0.99** | Excellent (non-linear) |
| Distance to goal | 0.62 | **0.99** | Excellent (non-linear) |
| Distance to avoid | 0.55 | **0.99** | Excellent (non-linear) |

**Layer-wise analysis**:
- **env_net**: Encodes all spatial information excellently (R²≈0.99)
- **ltl_net**: Encodes NO spatial information (R²≈0.01) - only goal state
- **embedding**: Combines both, spatial info preserved

**Conclusion**: The hidden state **DOES encode spatial information extremely well** - just non-linearly due to lidar's `exp(-dist)` encoding. The agent has all the information needed for planning, but behavioral tests show it doesn't use this information for multi-step planning or anticipation.

**Key insight**: Good representations ≠ Good planning. The agent represents space well but uses it reactively, not predictively.

---

## Files

**Behavioral experiments** (in `interpretability/zone_env/results/`):
- `controlled_choice/` - Equal-distance zone choice experiment
- `forced_detour/` - Obstacle detour experiment
- `sequencing/` - Multi-goal sequencing experiment
- `value_anticipation/` - Value anticipation analysis
- `paper_capabilities/` - Paper Figure 1 capability tests
- `paper_planning_tests/` - Targeted optimality and safety planning tests
- `paper_exact_safety/` - Exact paper Figure 1c replication (fixed zone positions)

**Probing experiments** (in `interpretability/zone_env/results/`):
- `position_probing/` - Position/next-position linear probes
- `nonlinear_probing/` - Layer-wise MLP probes (corrected results)

**Scripts** (in `interpretability/zone_env/working_scripts/analysis/`):
- `controlled_choice_e2e.py` - Controlled choice experiment
- `forced_detour_e2e.py` - Forced detour experiment
- `sequencing_e2e.py` - Sequencing experiment
- `value_anticipation.py` - Value anticipation analysis
- `position_probing.py` - Position probing experiment (linear)
- `nonlinear_probing.py` - Layer-wise MLP probing (corrected)
- `paper_capabilities_test.py` - Paper Figure 1 tests (optimality, safety, infinite horizon)
- `paper_planning_tests.py` - Targeted planning tests (optimality: farther-first, safety: blocked-vs-clear)
- `paper_exact_safety_test.py` - Exact paper Figure 1c replication (fixed zone positions)
- `generate_trajectory_plots.py` - Generate example trajectory visualizations

**Figures** (in `figures/`):
- `simple_reach_example.png` - Simple reach task trajectory
- `reach_avoid_example.png` - Reach-avoid task trajectory
- `safety_example.png` - Safety constraint task trajectory
- `optimality_example.png` - Two-goal sequencing trajectory
- `infinite_horizon_example.png` - Infinite horizon (repeated visits) trajectory

**Related reports**:
- `training_analysis.md` - Detailed comparison of training curriculum vs paper claims

---
*Updated January 2026*
