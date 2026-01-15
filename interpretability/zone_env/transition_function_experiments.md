# Transition Function Experiments: Zone Environment

## Summary

We ran three behavioral experiments to test whether the zone environment agent has learned a transition function (world model) that enables planning:

| Experiment | Evidence for Planning? | Key Finding |
|------------|----------------------|-------------|
| Controlled Choice | **No** | 55% vs 45% when distances equal - near random |
| Forced Detour | **No** | 0% took detours; agent goes straight or gets stuck |
| Sequencing | **Partial** | 59% correct order (better than 39% distance heuristic) |

**Overall conclusion**: The agent has NOT learned a general transition function for planning. It relies on:
1. **Distance-based heuristics** for goal selection
2. **Reactive obstacle avoidance** (not anticipatory)
3. **Some LTL goal representation** (for sequencing) but not robust

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

## What These Results Mean

### The Agent DOES Have:
1. **Goal-directed navigation**: Can reach single goals efficiently
2. **Reactive obstacle avoidance**: Can steer around obstacles once nearby
3. **Some goal sequencing**: Tracks which goals to visit (via LTL automaton)

### The Agent Does NOT Have:
1. **Transition function/world model**: Cannot predict future states
2. **Multi-step planning**: Cannot plan detours around obstacles
3. **Anticipatory behavior**: Doesn't avoid paths that will be blocked

### Why Zone Env Has 91% Success Rate Anyway

The high success rate comes from **continuous control enabling course correction**, not planning:
- Agent heads toward goal using distance heuristic
- When near obstacle, reactive avoidance kicks in
- Continuous action space allows steering around
- This works well when obstacles don't completely block path

---

## Implications for Probing

Given these behavioral findings, probing for transition functions is **unlikely to find strong evidence**. However, we might find:

1. **Distance representations**: Agent clearly uses distance to goal
2. **Obstacle proximity**: Reactive avoidance suggests some obstacle representation
3. **LTL automaton state**: Goal sequencing suggests tracking of formula satisfaction

**Recommended probing targets**:
- Decode distance-to-goal from hidden states
- Decode direction-to-goal from hidden states
- Decode obstacle proximity (lidar-like representation)
- Check if hidden state changes when LTL automaton transitions

---

## Files

- `controlled_choice/` - Equal-distance zone choice experiment
- `forced_detour/` - Obstacle detour experiment
- `sequencing/` - Multi-goal sequencing experiment

---
*Generated January 2026*
