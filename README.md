# DeepLTL: Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL

[![Python: 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper [DeepLTL: Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL](https://openreview.net/pdf?id=9pW2J49flQ) (ICLR'25 Oral).

## Installation
The code requires Python 3.10 with a working installation of PyTorch (tested with version 2.2.2). In order to use the _ZoneEnv_ environment, use the following command to install the required dependencies:
```bash
# activate the virtual environment, e.g. using conda
conda activate deepltl
cd src/envs/zones/safety-gymnasium
pip install -e .
```
This will also take care of installing the required versions of `mujoco`, `gymnasium` and `gymnasium-robotics`. To install the remaining dependencies, run
```bash
pip install -r requirements.txt
```
We rely on [Rabinizer 4](https://www7.in.tum.de/~kretinsk/rabinizer4.html) for the conversion of LTL formulae into LDBAs. Download the program using the following [link](https://www7.in.tum.de/~kretinsk/rabinizer4.zip) and unzip it into the project directory. Rabinizer requires Java 11 to be installed on your system and `$JAVA_HOME` to be set accordingly. To test the installation, run
```bash
./rabinizer4/bin/ltl2ldba -h
```
which should print a help message.

### Installing with Docker
Alternatively, you can use the provided Dockerfile to build a Docker image with all dependencies installed. To build the image, run
```bash
docker build -t deepltl .
```
To run the image while preserving trained models and logs, you can use the following command to mount the `experiments` directory:
```bash
mkdir experiments
docker run -it --mount type=bind,src="$(pwd)/experiments",target=/deep-ltl/experiments deepltl
```

## Training

To train a model on an environment, run the `train_ppo.py` file in `src/train`. We provide convenience scripts to train a model with the default parameters in our evaluation environments (_LetterWorld_, _ZoneEnv_, and _FlatWorld_). For example, to train a model on the _ZoneEnv_ environment, run
```bash
PYTHONPATH=src/ python run_zones.py --device cpu --name test --seed 1
```
The resulting logs and model files will be saved in `experiments/ppo/PointLtl2-v0/test/1` (where `PointLtl2-v0` is the internal name for the _ZoneEnv_ environment).

## Evaluation

We provide several evaluation scripts in `src/evaluation`. To simulate a trained model with a given LTL formula and output several statistics such as success rate (SR) and average number of steps (μ), run
```bash
PYTHONPATH=src/ python src/evaluation/simulate.py --env PointLtl2-v0 --exp test --seed 1 --formula '(!blue U green) & F yellow' --finite --deterministic
```

Note that we generally evaluate the deterministic policy in _ZoneEnv_ (by taking the mean of the predicted action distribution), whereas we evaluate stochastic policies in _LetterWorld_ and _FlatWorld_.

The script also supports a `--render` flag to visualise the simulation in real time. Alternatively, we provide the scripts `draw_zone_trajectories.py` and `draw_flat_trajectories.py` to visualise the trajectories of the agents in the _ZoneEnv_ and _FlatWorld_ environments, respectively.

For a more comprehensive evaluation, we provide the scripts `eval_test_tasks_finite.py` and `eval_test_tasks_infinite.py` to evaluate the performance of a model on a set of test tasks. The former evaluates the model on a set of finite-horizon tasks, while the latter evaluates the model on a set of infinite-horizon tasks. The default tasks specified in the scripts match the tasks from our evaluation in Table 1.

Finally, we provide the script `eval_over_time.py` which evaluates the performance of a model over training on a fixed dataset of specifications sampled from the _reach/avoid_ task space. To plot the resulting training curves, run `plot_training_curves.py`.

## Steering Analysis Experiments

This repository includes comprehensive experiments on steering neural network representations to understand how goal information is encoded and whether it can be manipulated. The experiments explore three key network components:

### Goal Probing
We trained linear probes to predict the agent's current goal from hidden states of different network components:
- **LTL Network**: 94.0% accuracy in predicting next goals
- **Environment Network**: 98.1% accuracy (highest)
- **Policy Network**: 99.5% accuracy (highest)

### Steering Experiments
We attempted to steer the agent's behavior by manipulating hidden states during rollouts:
- **LTL Network Steering**: 0.8 goal changes per 100 interventions
- **Environment Network Steering**: 1.2 goal changes per 100 interventions
- **Policy Network Steering**: 1.2 goal changes per 100 interventions

### Key Findings
Despite high probe accuracies (94-99.5%), steering effectiveness remains low across all networks (0.8-1.2% success rate). This suggests that goal representations are distributed across multiple network components and are robust against single-layer manipulations.

### Running the Experiments
```bash
# Run goal probing
python run_probe_goal.py

# Run LTL network steering
python run_steer_goals.py

# Run environment network steering  
python run_env_steering.py

# Run policy network steering
python run_policy_steering.py

# Generate comprehensive analysis
python run_comprehensive_analysis.py
```

### Reports and Visualizations
- `PROBE_GOAL_REPORT.md`: Detailed analysis of goal probing results
- `STEERING_EXPERIMENT_REPORT.md`: Analysis of LTL and environment network steering
- `COMPREHENSIVE_STEERING_REPORT.md`: Complete comparison of all steering approaches
- Various PNG files showing trajectories, probe accuracies, and steering effectiveness

## License
This project is licensed under the [MIT License](LICENSE). We use the following third-party libraries:
- [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) (Apache License 2.0)
- [torch_ac](https://github.com/lcswillems/torch-ac) (MIT License)

We thank the authors of [LTL2Action](https://github.com/LTL2Action/LTL2Action) for providing a starting point for our implementation.

## Citation
If you find this code useful in your research, please consider citing our paper:
```bibtex
@inproceedings{deepltl,
    title     = {Deep{LTL}: Learning to Efficiently Satisfy Complex {LTL} Specifications for Multi-Task {RL}},
    author    = {Mathias Jackermeier and Alessandro Abate},
    booktitle = {International Conference on Learning Representations ({ICLR})},
    year      = {2025}
}
```
