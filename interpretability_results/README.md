# Interpretability Results Directory

This directory contains **interpretability and analysis results** from our steering experiments, separate from the main codebase results.

## Directory Structure

### `trajectory_plots/`
Trajectory visualization plots from steering experiments:
- `trajectories_steering_strength_*.png` - Trajectory plots for different steering strengths (0.0, 10.0, 50.0, 100.0, 500.0)
- `trajectories_selected_worlds*.png` - Trajectory plots for selected world configurations
- `trajectories_steered.png` / `trajectories_unsteered.png` - Comparison plots

### `steering_results/`
Text files containing detailed steering experiment results:
- `steer_subgoals_*.txt` - Detailed output from steering experiments
- Contains step-by-step trajectory data, probe accuracies, and completion statistics

### `probe_results/`
Results from neural network probing experiments:
- `agent_location_probe*.png` - Agent location probe visualizations
- `zone_*_probe_weights*.png` - Zone probe weight visualizations
- `probe_dynamics*.png` - Probe dynamics over time
- `probe_correlation_analysis.png` - Correlation analysis results
- `probe_decision_boundary.png` - Decision boundary visualizations

### `analysis_plots/`
General analysis and visualization plots:
- `steering_debug_analysis.png` - Steering effectiveness analysis
- `compass_accuracy.png` - Spatial representation analysis
- `zone_distance_true_vs_pred.png` - Zone prediction accuracy
- `zone_direction_true_vs_pred_*.png` - Zone direction predictions by color
- `sequence_generation_steering_*.png` - Sequence generation analysis
- `*phenomenon*.png` / `*phenomenon*.gif` - Compass phenomenon visualizations
- `*dynamics*.png` / `*dynamics*.gif` - Dynamics analysis plots

## Note: Original Results

The original experiment results (CSV files) are stored in the `results/` directory:
- `FlatWorld-v0.csv`
- `LetterEnv-v0.csv` 
- `PointLtl2-v0.csv`
- `generalisation.csv`
- `safety.csv`

This `interpretability_results/` directory contains only the **interpretability and analysis outputs** from our steering experiments.

## File Naming Conventions

- `*_steering_strength_X.X.png` - Trajectory plots with specific steering strength
- `*_probe_*.png` - Probe-related visualizations
- `*_true_vs_pred_*.png` - True vs predicted comparison plots
- `*_weights_*.png` - Neural network weight visualizations

## Analysis

These results can be analyzed using the scripts in the `scripts/analysis/` directory. The trajectory plots show the effectiveness of steering at different strengths, while the probe results reveal insights into the neural network's internal representations. 