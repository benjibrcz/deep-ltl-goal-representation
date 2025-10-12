# Robust Agent Location Probing

This directory contains a new, properly designed agent location probe that addresses the overfitting issues identified in previous probing experiments.

## Overview

The `probe_current_agent_location.py` script implements a comprehensive agent location probe with proper evaluation methodology to avoid the data leakage and overfitting problems found in earlier experiments.

## Key Features

### 1. Cross-World Generalization Testing
- **Train on some worlds, test on completely different worlds**
- Prevents memorization of world-specific patterns
- Tests true generalization across different environment configurations

### 2. Temporal Generalization Testing
- **Train on early steps, test on later steps**
- Evaluates whether the probe can predict future behavior from early behavior
- Tests temporal consistency of spatial representations

### 3. Dimensionality Reduction
- **PCA with 95% variance explained** (configurable)
- Reduces feature dimensionality to prevent overfitting
- Maintains most important information while reducing noise

### 4. Baseline Comparisons
- **Previous Position**: Predicts current position as the previous position (trivial baseline for smooth trajectories)
- **Mean Position**: Always predicts the mean position from training data
- **Linear Extrapolation**: Uses velocity from last 2 positions to predict current position
- **Random Position**: Predicts random positions within the observed range
- **Ridge on Raw Features**: Linear regression without dimensionality reduction (tests for overfitting)
- Provides context for interpreting probe performance

### 5. Proper Train/Test Splits
- **Adequate sample sizes** for reliable evaluation
- **No data leakage** between train and test sets
- **Multiple evaluation metrics**: R², MSE, MAE

## Usage

```bash
python interpretability/probing/agent_location/probe_current_agent_location.py \
    --layer env_net.mlp.2 \
    --n-train-worlds 8 \
    --n-test-worlds 2 \
    --n-rollouts 10 \
    --max-steps 200
```

### Arguments

- `--layer`: Neural network layer to probe (required)
- `--n-train-worlds`: Number of worlds for training (default: 8)
- `--n-test-worlds`: Number of worlds for testing (default: 2)
- `--n-rollouts`: Number of rollouts per world (default: 10)
- `--max-steps`: Maximum steps per rollout (default: 200)
- `--n-components`: Number of PCA components (default: None, uses 95% variance)
- `--output-dir`: Output directory for results
- `--seed`: Random seed for reproducibility

## Output

The script generates:

1. **Console output** with detailed results and comparisons
2. **CSV file** (`results_summary.csv`) with all metrics
3. **Visualization plots**:
   - `performance_comparison.png`: R²/MSE comparisons and error distributions
   - `temporal_generalization.png`: Temporal generalization results
   - `pca_analysis.png`: PCA variance explained analysis

## Addressing Previous Issues

### Problem 1: Data Leakage
**Previous**: Train and test on similar trajectories from the same world
**Solution**: Cross-world testing with completely different world configurations

### Problem 2: Feature Overfitting
**Previous**: 640 features vs 190 samples (3.37:1 ratio)
**Solution**: PCA dimensionality reduction to ~50-100 components

### Problem 3: Inadequate Test Sets
**Previous**: 2-3 samples per step, 38 samples total
**Solution**: Hundreds of test samples from different worlds

### Problem 4: No Baseline Comparisons
**Previous**: Only reported probe performance
**Solution**: Multiple baselines for context and interpretation

### Problem 5: No Temporal Testing
**Previous**: Only tested on similar time periods
**Solution**: Train on early steps, test on later steps

## Expected Results

With proper evaluation, we expect:

1. **Lower R² scores** than previous experiments (0.3-0.8 instead of 0.99+)
2. **Meaningful baseline comparisons** showing probe improvement over simple baselines
3. **Temporal degradation** indicating limits of spatial representation
4. **Cross-world consistency** showing genuine generalization

## Interpretation Guidelines

### Good Results
- R² > 0.5 with proper cross-world testing
- Probe significantly outperforms baselines
- Reasonable temporal generalization (R² > 0.3)

### Concerning Results
- R² < 0.2 (poor spatial representation)
- Probe performs worse than baselines
- Large drop in temporal generalization

### Overfitting Indicators
- R² > 0.9 with cross-world testing (suspicious)
- Large gap between train and test performance
- Poor temporal generalization despite good cross-world performance

## Comparison with Previous Experiments

| Aspect | Previous Experiments | New Robust Probe |
|--------|---------------------|------------------|
| Train/Test Split | Same world, different rollouts | Different worlds |
| Feature Dimensionality | 640 features | ~50-100 (PCA) |
| Test Set Size | 2-38 samples | 200+ samples |
| Baseline Comparisons | None | Multiple baselines |
| Temporal Testing | None | Early vs late steps |
| Expected R² | 0.99+ (overfitted) | 0.3-0.8 (realistic) |

## Future Extensions

1. **Multi-layer comparison**: Test different network layers
2. **Ablation studies**: Remove different components to understand contributions
3. **Cross-task generalization**: Test on different LTL formulas
4. **Non-linear probes**: MLP probes for comparison
5. **Attention analysis**: Analyze which features are most important 