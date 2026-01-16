# Training Analysis: Why the Agent Fails Safety Tests

## Current Agent Training (big_test)

The agent was trained using `run_zones.py` with the following configuration:

```bash
python src/train/train_ppo.py \
    --env PointLtl2-v0 \
    --curriculum PointLtl2-v0 \
    --num_steps 15_000_000 \
    --lr 0.0003 \
    --discount 0.998 \
    --entropy_coef 0.003
```

**Training curriculum (ZONES_CURRICULUM):**

| Stage | Task Type | Example Formula | Threshold |
|-------|-----------|-----------------|-----------|
| 0 | Simple reach | `F blue` | 80% min |
| 1 | 2-step reach | `F (blue & F green)` | 95% mean |
| 2 | Reach-avoid (1 step) | `!yellow U blue` | 95% mean |
| 3 | 2-step reach-avoid | `(!yellow U blue) & F (green)` | 90% mean |
| 4 | Mixed (40% reach-avoid, 60% reach-stay) | `!avoid U reach` or `G F reach` | 90% mean |
| 5 | Mixed (80% reach-avoid, 20% reach-stay) | Similar, deeper sequences | 90% mean |
| 6 | Mixed (80% reach-avoid, 20% reach-stay) | 3-step sequences | None (final) |

---

## What's Missing from Training

The paper's safety test uses: `(F green | F yellow) & G !blue`

**This formula contains TWO constructs NOT in the training distribution:**

### 1. Disjunction (OR between goals)
- **Training**: Never uses `F A | F B` (reach A OR B)
- **Test formula**: `F green | F yellow` (reach green OR yellow)
- **Impact**: Agent doesn't know how to handle "either goal is acceptable"

### 2. Global Safety Constraint
- **Training**: Uses `!avoid U reach` (avoid UNTIL reach)
- **Test formula**: Uses `G !blue` (GLOBALLY avoid blue - forever)
- **Impact**: Agent learned "avoid until goal reached", not "avoid forever while pursuing goal"

### Comparison Table

| Construct | Training | Paper Safety Test |
|-----------|----------|-------------------|
| Reach goal | ✅ `F blue` | ✅ `F green` |
| Avoid until | ✅ `!yellow U blue` | ❌ Not used |
| OR between goals | ❌ Never trained | ✅ `F green \| F yellow` |
| Global avoid | ❌ Never trained | ✅ `G !blue` |
| Combined | ❌ | ✅ `(F A \| F B) & G !C` |

---

## Why This Matters

The agent's failures make sense given the training:

1. **60% safety violations**: Agent learned `!avoid U reach`, meaning "avoid is temporary until I reach the goal." With `G !blue`, the constraint is permanent - the agent doesn't understand this.

2. **Goes toward blocked goal**: Agent uses distance-based heuristic (go to nearest goal). It never learned that sometimes you need to consider path safety when CHOOSING which goal to pursue.

3. **3.3% chose safe path**: Essentially random - the agent has no concept of "either goal works, pick the safe one."

---

## How to Train an Agent That CAN Solve These

### Option 1: Add Disjunction and Global Safety to Curriculum

Create a new curriculum stage that includes:

```python
# New formula types needed:
"F green | F yellow"                    # Disjunction
"G !blue"                               # Global safety
"(F green | F yellow) & G !blue"        # Combined
"F green & G !blue"                     # Single goal with global safety
```

### Option 2: Modify the Sequence Sampler

Add new samplers to `sequence/samplers/sequence_samplers.py`:

```python
def sample_disjunction_safety(
    num_goals: int | tuple[int, int],
    num_avoid: int | tuple[int, int],
) -> Callable[[list[str]], LDBASequence]:
    """Sample (F A | F B | ...) & G !C formulas"""
    # Implementation needed
    pass
```

### Option 3: Use a Different Training Distribution

The paper may have used a different training distribution that includes these formula types. Check:
1. If there's a different curriculum in the original repo
2. If the paper describes their exact training formulas
3. If there are alternative sequence samplers

---

## Recommended Changes

### Immediate Fix: Add to Curriculum Stage 4+

Modify `src/sequence/samplers/curriculum.py`:

```python
# Add to stages 4-6:
RandomCurriculumStage(
    sampler=sample_disjunction_with_global_safety((2, 3), (1, 2)),
    threshold=None,
    threshold_type=None
),
```

### New Sampler Implementation

```python
def sample_disjunction_with_global_safety(
    num_goals: tuple[int, int],
    num_avoid: tuple[int, int],
) -> Callable[[list[str]], LDBASequence]:
    """
    Samples formulas like: (F green | F yellow) & G !blue

    This creates an LDBA where:
    - Multiple goal states are accepting (any goal satisfies)
    - Avoid states cause immediate rejection (global constraint)
    """
    def wrapper(propositions: list[str]) -> LDBASequence:
        ng = random.randint(*num_goals)
        na = random.randint(*num_avoid)

        goals = random.sample(propositions, ng)
        available = [p for p in propositions if p not in goals]
        avoids = random.sample(available, min(na, len(available)))

        # Create LDBA representation
        # Goal: reach ANY of the goals
        # Constraint: NEVER touch avoid zones
        goal_assignments = frozenset([
            Assignment.single_proposition(g, propositions).to_frozen()
            for g in goals
        ])
        avoid_assignments = frozenset([
            Assignment.single_proposition(a, propositions).to_frozen()
            for a in avoids
        ])

        # This is a single-step task with disjunctive goals and global avoid
        return LDBASequence([
            (goal_assignments, avoid_assignments)
        ], global_avoid=avoid_assignments)  # Need to add global_avoid support

    return wrapper
```

### Required LDBA Changes

The `LDBASequence` class may need modification to support:
1. Disjunctive goals (multiple accepting states)
2. Global safety constraints (permanent avoid, not just until-reach)

---

## Training Time Estimate

To train an agent with these additional capabilities:

- **Additional training steps**: ~5-10M steps (on top of existing 15M)
- **New curriculum stages**: 2-3 stages with disjunction/safety formulas
- **Suggested approach**:
  1. Start from existing checkpoint (curriculum stage 6)
  2. Add new stages 7-8 with disjunction/safety formulas
  3. Train until convergence

---

## Summary

| Question | Answer |
|----------|--------|
| Why does agent fail safety test? | Formula type not in training distribution |
| Can current agent solve it? | No - never trained on OR or G-safety |
| Is this a planning failure? | Partially - also an out-of-distribution failure |
| Can we fix with retraining? | Yes - add disjunction and global safety to curriculum |
| Effort required | Modify samplers + 5-10M additional training steps |
