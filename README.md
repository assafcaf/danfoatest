# DanfoaTest

A reinforcement learning research project for training agents in multi-agent environments using various reward modeling techniques.

## Overview

This project implements and evaluates different reward modeling approaches for multi-agent reinforcement learning:

- **PRM** (Preference-based Reward Model)
- **NRP** (Neural Reward Predictor)
- **CRM** (Counterfactual Reward Model)

## Project Structure

```
DanfoaTest/
├── src/
│   ├── agents/          # Agent implementations
│   ├── env/             # Environment configurations
│   ├── learners/        # Learning algorithms
│   ├── reward_predictor/# Reward prediction models
│   ├── rl_agents/       # RL agent implementations
│   ├── experiment_runner/# Experiment orchestration
│   ├── teach_prm.py     # PRM training script
│   ├── teach_nrp.py     # NRP training script
│   └── teach_crm.py     # CRM training script
├── results/             # Training results and logs
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run experiments using the training scripts:

```bash
# Train with PRM
python src/teach_prm.py --config <config_file>

# Train with NRP
python src/teach_nrp.py --config <config_file>

# Train with CRM
python src/teach_crm.py --config <config_file>
```

## Requirements

- Python 3.8+
- PyTorch
- Stable Baselines3
- Gymnasium/Gym
- PettingZoo
- See `requirements.txt` for complete list

## Results

Experiment results are saved in the `results/` directory with subdirectories organized by algorithm, environment, and agent count.
