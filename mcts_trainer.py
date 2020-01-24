"""Train a RL policy against a customisable MCTS agent."""

import argparse
import random

import ray
from ray import tune
from ray.tune.registry import register_env

from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy
from src.utils import get_worker_config, get_model_config, get_policy_config, get_mcts_policy_configs


parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='PPO')
parser.add_argument('--use-cnn', action='store_true')
parser.add_argument('--num-rollouts', type=int, default=128)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--human', action='store_true')
parser.add_argument('--save-experience', action='store_true')
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()

ray.init(local_mode=args.debug)
tune_config = get_worker_config(args)
tune_config.update(get_policy_config(args.policy))
model_config, env_cls = get_model_config(args.use_cnn)
register_env('c4', lambda cfg: env_cls(cfg))
env = env_cls()
obs_space, action_space = env.observation_space, env.action_space

if args.save_experience:
    tune_config['output'] = 'experience'

train_pid = 'learned00'
mcts_pid = f'mcts{args.num_rollouts:04d}'
mcts_rollouts = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
mcts_policies = get_mcts_policy_configs(mcts_rollouts, obs_space, action_space)
policies = [train_pid, 'human'] if args.human else [train_pid, mcts_pid]


def name_trial(trial):
    """Give trials a more readable name in terminal & Tensorboard."""
    network = 'CNN' if args.use_cnn else 'MLP'
    rollouts = args.num_rollouts
    return f'MCTS{rollouts}vs{trial.trainable_name}-{network}'


tune.run(
    args.policy,
    name='mcts_trainer',
    trial_name_creator=name_trial,
    stop={f'policy_reward_mean/{train_pid}': 0.6},
    config=dict({
        'env': 'c4',
        'multiagent': {
            'policies_to_train': [train_pid],
            'policy_mapping_fn': lambda _: random.sample(policies, k=2),
            'policies': {
                train_pid: (None, obs_space, action_space, {'model': model_config}),
                **mcts_policies,
                'human': (HumanPolicy, obs_space, action_space, {}),
                'random': (RandomPolicy, obs_space, action_space, {}),
                'rollouts': (MCTSPolicy, obs_space, action_space, {'max_rollouts': 3}),
            },
        },
    }, **tune_config),
    # checkpoint_freq=20,
    checkpoint_at_end=True,
    resume=args.resume
)
