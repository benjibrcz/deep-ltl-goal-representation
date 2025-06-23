# Interpretability Analysis Directory

This directory contains interpretability analysis reports and findings organized by research area.

## Directory Structure

### `trajectory_analysis/`
Analysis of trajectory plots and steering effects:
- `TRAJECTORY_ANALYSIS_REPORT.md` - Comprehensive report on trajectory analysis
- Contains insights about steering effectiveness, optimal strengths, and behavioral changes

### `steering_experiments/`
Analysis of steering experiment results:
- `COMPREHENSIVE_STEERING_REPORT.md` - Detailed steering experiment analysis
- Contains findings about steering strength effects and optimization

### `probe_analysis/`
Analysis of neural network probing results:
- `SPATIAL_REPRESENTATION_FINDINGS.md` - Findings about spatial representations
- Contains insights about how the neural network represents space and goals

### `zone_analysis/`
Analysis of zone-related behavior:
- Zone location and behavior analysis
- Zone prediction accuracy studies
- Zone steering effectiveness

## Key Findings

### Trajectory Analysis
- Steering is effective with 10-50x strength providing optimal balance
- File size reductions indicate shorter, more efficient paths
- 20% reduction in trajectory complexity with strong steering
- Saturation effect beyond 50x steering strength

### Steering Experiments
- 99.2% probe accuracy on policy_mlp_0 layer
- Clear evidence of path optimization
- Task completion preserved across steering strengths
- Sweet spot identified at 10-50x strength

### Probe Analysis
- Neural network features are highly predictive of goals
- Strong spatial representations in environment network layers
- Reliable steering signals for behavior modification
- Excellent feature quality for steering applications

## Usage

These interpretability analysis reports provide comprehensive insights into the steering experiments and can be used to:
- Understand steering effectiveness
- Optimize steering parameters
- Design future experiments
- Guide practical applications

## Related Scripts

The analysis in these reports was generated using scripts in `scripts/analysis/` and can be reproduced or extended using those tools. 