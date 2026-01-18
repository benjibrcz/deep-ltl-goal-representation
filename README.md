## Interpretability of DeepLTL Goal Representation

This repository includes comprehensive experiments on probing and steering the DeepLTL goal-conditioned agent system to understand how goal information is encoded and whether it can be steered.

Note: this repository has been cloned from https://github.com/mathiasj33/deep-ltl.

## Key Finding

**The agent does NOT have a learned transition function for multi-step planning.** High task success rates come from reactive heuristics and continuous control allowing course correction, not from genuine planning capability.

See [interpretability/REPORT.md](interpretability/REPORT.md) for the full research report.

## Summary Results

| Domain | Metric | Result |
|--------|--------|--------|
| Zone Env | Safe task success | 91% |
| Zone Env | Controlled choice (planning test) | 55% vs 45% (random) |
| Letter World | Safe task success | 38% |
| Letter World | Paper exact Figure 1 test | 3% safe, 60% violated |
| Steering | Probe accuracy | 94-99.5% |
| Steering | Effectiveness | 0.8-1.2% |

## Research Areas

- **Steering Experiments**: Probing and manipulating goal representations across network components
- **Spatial Representations**: Discovery of egocentric "compass" encoding vs allocentric world maps
- **Transition Function Tests**: 10 experiments testing multi-step planning capability
- **Cross-Domain Comparison**: Zone env vs Letter world analysis
