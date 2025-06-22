# Running probe_goal.py

This guide explains how to run the `probe_goal.py` script, which analyzes the LTL (Linear Temporal Logic) network's ability to predict the next proposition in a sequence.

## Prerequisites

1. **Python Environment**: Make sure you have Python 3.8+ installed
2. **Dependencies**: Install the required packages

### Installing Dependencies

```bash
# Install the required packages
pip install -r requirements.txt
```

The key dependencies include:
- `torch>=1.9.0` - PyTorch for deep learning
- `scikit-learn>=1.0.0` - For logistic regression
- `gymnasium>=0.28.0` - For environment interactions
- `numpy`, `tqdm`, `matplotlib` - For data processing and visualization

## Running the Script

### Method 1: Using the run script (Recommended)

```bash
python run_probe_goal.py
```

### Method 2: Running directly

```bash
# Set the Python path to include the src directory
export PYTHONPATH=src/
python src/probe_goal.py
```

## What the Script Does

The script performs the following steps:

1. **Environment Setup**: Creates a PointLtl2-v0 environment with the formula "F green & F yellow"
2. **Model Loading**: Loads a pre-trained model from the `big_test` experiment (seed 0)
3. **Feature Extraction**: Hooks into the LTL RNN to extract hidden states during rollout
4. **Data Collection**: Runs a 200-step rollout, collecting:
   - RNN hidden states (features)
   - Next proposition labels (targets)
5. **Analysis**: Trains a logistic regression probe to predict next propositions from hidden states
6. **Results**: Reports accuracy and weight norms for each proposition

## Configuration

You can modify the script parameters at the top of `src/probe_goal.py`:

```python
ENV       = "PointLtl2-v0"    # Environment name
EXP       = "big_test"        # Experiment name
SEED      = 0                 # Model seed
MAX_STEPS = 200               # Number of rollout steps
FORMULA   = "F green & F yellow"  # LTL formula to test
```

## Expected Output

The script should output something like:

```
Detected propositions: ['blue', 'green', 'yellow']
Collected 45 valid next‐prop labels

Next-prop probe accuracy: 78.23%

     blue   ‖w‖ = 0.234
    green   ‖w‖ = 0.456
   yellow   ‖w‖ = 0.189
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure `PYTHONPATH` includes the `src/` directory
2. **Missing Model**: Ensure the trained model exists at `experiments/ppo/PointLtl2-v0/big_test/0/`
3. **CUDA Issues**: The script runs on CPU by default. If you have CUDA issues, the model is loaded with `map_location="cpu"`

### Model Requirements

The script expects these files in the experiment directory:
- `status.pth` - Model weights and training status
- `vocab.pkl` - Vocabulary for LTL parsing

## Understanding the Results

- **Accuracy**: How well the LTL network's hidden states can predict the next proposition
- **Weight Norms**: The magnitude of weights for each proposition, indicating their importance in the prediction
- **Valid Labels**: Number of steps where a single next proposition was identified (vs. multiple or none)

## Customization

You can modify the script to:
- Test different LTL formulas
- Use different environments (LetterEnv-v0, FlatWorld-v0)
- Analyze different model checkpoints
- Change the probe type (e.g., use Ridge regression instead of logistic regression) 