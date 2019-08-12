# Competitive Training Methods
Competitive training methods for reinforcement learning using RLLIB

## Requirements
- Python 3.6


## Installation
1. Optionally create a Python3 virtualenv called `venv` (separate project dependencies)

        virtualenv -p python3 venv

1. Activate the virtualenv (if you created one)

        source venv/bin/activate

1. Install dependencies

        pip install -r requirements.txt


## Usage
Activate the virtualenv (if used)

    source venv/bin/activate

Run using main.py and configure via optional command line arguments

    python main.py {options}

### Command Line Options

| Flag | Parameters | Description | Required | Default Value | 
| ---- | ---------- | ----------- | -------- | ------------- |
| run | str | An optional name of this experiment run | N | '' |
| epochs | int | The number of experiment epochs to run | N | 1 |
| env | str | The game environment to use | N | Connect4 |
| agent | str | The agent types to use | N | Deep Q-Network (DQN) |
| trainer | str | The agent training method to use | N | DefaultTrainer |
| save-agent | bool | Whether to save all agents weights/memory/progress. | N | False |
| human-player | bool | Allows you to compete against trained agents. | N | False |
| visualise | bool | Whether to show a visualisation of the game. | N | False |
