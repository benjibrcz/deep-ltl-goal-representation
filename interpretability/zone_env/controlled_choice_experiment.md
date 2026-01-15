# Controlled Choice Experiment: Zone Environment

## Overview

This experiment tests whether the zone environment agent uses **planning** or **distance-based heuristics** when navigating to goal zones. We create controlled scenarios where two reach zones are equidistant from the agent, but one has an avoid zone blocking the direct path.

## Key Finding

**The agent shows no significant preference for unblocked paths when distances are equal (~55% safe vs ~45% blocked), suggesting distance-based heuristics rather than multi-step planning.**

## Experimental Setup

### Scenario Design

Each scenario has:
- **Two REACH zones** of the same color at equal distance from the agent (within 0.5 unit tolerance)
- **One AVOID zone** blocking the direct path to one of the reach zones
- The other reach zone has a clear, unobstructed path

### Validity Criteria

Scenarios are rejected if:
- Reach and avoid zones overlap (min separation: 0.8 units)
- The two reach zones are too close together (min separation: 1.0 unit)
- Agent starts too close to any zone (min distance: 0.6 units)

### Parameters

| Parameter | Value |
|-----------|-------|
| Distance tolerance | 0.5 units |
| Blocking threshold | 0.6 units (perpendicular distance) |
| Max steps | 240 |
| Zone radius | 0.4 units |

## Results

### Agent Choice (N=30 scenarios, 29 reached a zone)

| Choice | Count | Percentage |
|--------|-------|------------|
| Safe (unblocked) zone | 16 | 55.2% |
| Blocked zone | 13 | 44.8% |
| Neither | 1 | 3.3% |

### Outcome Distribution

| Outcome | Count | Percentage |
|---------|-------|------------|
| Safe success | 13 | 43.3% |
| Risky success (touched avoid) | 9 | 30.0% |
| Unsafe success (chose blocked, no avoid contact) | 7 | 23.3% |
| Fail | 1 | 3.3% |

### Avoid Zone Contact

- Overall: 33.3% touched avoid zone
- When chose safe: 18.8% touched avoid
- When chose blocked: 46.2% touched avoid

## Interpretation

The near 50/50 split between safe and blocked choices indicates:

1. **No evidence of planning**: A planning agent would consistently choose the unblocked path
2. **Distance-based heuristics**: The agent appears to navigate toward the nearest goal without considering obstacles in the path
3. **Reactive obstacle avoidance**: The agent can navigate around obstacles reactively (hence 91% overall success), but doesn't anticipate them when choosing which goal to pursue

## Comparison with Letter World

| Metric | Zone Env | Letter World |
|--------|----------|--------------|
| Overall safe success | 91% | 38% |
| Controlled choice (safe) | 55% | ~50% (similar) |
| Planning evidence | Weak/None | Weak/None |

**Key insight**: Zone env's higher success rate comes from **continuous navigation allowing course correction**, not from better planning. Both agents rely on reactive heuristics.

## Why Zone Env Performs Better Overall

1. **Continuous action space**: Can steer around obstacles rather than committing to discrete paths
2. **Lidar observations**: Better spatial awareness of nearby zones
3. **Course correction**: Can adjust trajectory mid-rollout when approaching obstacles

## Files

- `summary.csv` - Per-scenario results
- `scenarios.json` - Scenario configurations
- `results.pkl` - Full results with trajectories
- `plots/` - Trajectory visualizations

## Reproducing

```bash
PYTHONPATH=src python interpretability/zone_env/working_scripts/analysis/controlled_choice_e2e.py \
    --n_target 30 \
    --max_attempts 1000 \
    --distance_tolerance 0.5 \
    --blocking_threshold 0.6 \
    --out_dir interpretability/zone_env/results/controlled_choice
```

## Example Visualizations

See `plots/` directory for trajectory visualizations showing:
- Orange diamond: Agent start position
- Green line: Agent trajectory
- Red squares: Position markers every 20 steps
- Zone labels: SAFE, BLOCKED, AVOID

---
*Generated January 2026*
