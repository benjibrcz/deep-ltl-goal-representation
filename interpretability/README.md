# DeepLTL Interpretability Research

This folder contains experiments investigating whether the DeepLTL agent learns genuine planning capabilities or relies on behavioral heuristics.

## Key Finding

The agent exhibits **reactive heuristics** (pattern matching and distance-based navigation) but fails at **multi-step planning** (requires simulating future states). Paper claims about transition function learning are not supported by controlled experiments.

See [REPORT.md](REPORT.md) for the full write-up.

## Folder Structure

```
interpretability/
├── REPORT.md                    # Main comprehensive report
│
├── zone_env/                    # Zone environment experiments
│   ├── working_scripts/         # Analysis scripts
│   ├── probing/                 # Probe experiments
│   ├── reports/                 # Individual report files
│   │   ├── transition_function/ # Transition function experiments
│   │   └── *.md                 # Various analysis reports
│   ├── results/                 # Generated results
│   └── controlled_choice_experiment.md
│
└── letter_world/                # Letter world experiments
    ├── results/
    │   └── e2e_analysis_report.md
    └── ...
```

## Key Results

| Test | Metric | Zone Env | Letter World |
|------|--------|----------|--------------|
| Safe task success | Completion with safety | 91% | 38% |
| Controlled choice | Planning vs heuristic | 55% | ~50% |
| Probe accuracy | Goal decoding | 94-99% | - |
| Steering effectiveness | Behavior change | 0.8-1.2% | - |

## Research Areas

### 1. Steering Experiments
- Probing goal representations across LTL, environment, and policy networks
- Attempting to manipulate agent behavior through representation steering
- Finding: High probe accuracy (99%) but low steering effectiveness (1%)

### 2. Spatial Representations
- Discovery of egocentric "compass" encoding vs allocentric world maps
- Predicted goal locations move dynamically with agent
- Representations are distributed, not localized

### 3. Transition Function Tests
- 10 experiments testing multi-step planning capability
- Paper exact Figure 1 test: 3% safe choice, 60% violated safety
- No evidence of value anticipation or counterfactual reasoning

### 4. Cross-Domain Comparison
- Zone env succeeds through continuous control allowing course correction
- Letter world fails due to discrete actions and strong directional biases
- Both rely on reactive heuristics, not planning

## Completed Experiments

1. **Goal Probing**: 94-99.5% accuracy across network components
2. **Steering**: Limited effectiveness (0.8-1.2%) despite high probe accuracy
3. **Controlled Choice**: 55% vs 45% when distances equal (essentially random)
4. **Paper Exact Test**: 60% violated safety on Figure 1 setup
5. **Value Anticipation**: No anticipatory value drop before obstacles
6. **MLP Probing**: R²=0.99 spatial encoding in non-linear combinations
7. **Infinite Horizon**: Works via reactive cycling (not planning)
