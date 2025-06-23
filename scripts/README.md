# Scripts Directory

This directory contains all the scripts organized by their purpose.

## Directory Structure

### `steering/`
Scripts related to neural network steering experiments:
- `steer_subgoals.py` - Main steering script for LTL-based navigation
- `ablate_probe_direction.py` - Ablation studies for probe direction
- `ablate_probe_direction_current_goal.py` - Ablation with current goal context
- `ablate_nonlinear_probe_direction_current_goal.py` - Non-linear probe ablation

### `probes/`
Scripts for neural network probing and analysis:
- `probe_goal.py` - Goal prediction probe analysis
- `run_probe_color_locations.py` - Color location probe experiments

### `analysis/`
Scripts for analyzing experiment results:
- `analyze_trajectories.py` - Basic trajectory plot analysis
- `detailed_trajectory_analysis.py` - Detailed trajectory metrics extraction
- `trajectory_analysis_summary.py` - Comprehensive trajectory analysis summary

### `visualization/`
Scripts for creating visualizations and plots:
- Various visualization utilities for trajectory and probe analysis

## Usage

Each script can be run independently. Most scripts are designed to analyze specific aspects of the steering experiments and generate reports or visualizations.

## Dependencies

Most scripts require:
- Python 3.x
- NumPy
- Matplotlib
- PyTorch
- Scikit-learn
- The deep-ltl environment and model files 