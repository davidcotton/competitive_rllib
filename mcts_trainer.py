"""Train a RL policy against a customisable MCTS agent."""

import argparse
import logging
import random

import ray
from ray import tune
from ray.tune.registry import register_env

from src.callbacks import mcts_on_episode_end
from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy
from src.utils import get_debug_config, get_model_config

logger = logging.getLogger('ray.rllib')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save-experience', action='store_true')
    args = parser.parse_args()

    ray.init(local_mode=args.debug)
    tune_config = get_debug_config(args.debug)

    model_config, env_cls = get_model_config(args.use_cnn)
    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space = env.observation_space
    action_space = env.action_space

    if args.policy == 'DQN':
        tune_config.update({
            'hiddens': [],
            'dueling': False,
        })

    if args.save_experience:
        tune_config['output'] = 'experience'

    policies = ['learned00', 'mcts']
    # policies = ['learned00', 'human']

    def random_policy_mapping_fn(info):
        return random.sample(policies, k=2)

    tune.run(
        args.policy,
        name='mcts_trainer',
        stop={
            # 'policy_reward_mean': {'learned00': 0.95},
            'policy_reward_mean': {'learned00': 0.8},  # ray==0.7.3
            # 'policy_reward_mean/learned00': 0.8,  # ray==0.7.5
        },
        config=dict({
            'env': 'c4',
            'env_config': {},
            'lr': 0.001,
            'gamma': 0.995,
            'lambda': 0.95,
            'clip_param': 0.2,
            # 'kl_coeff': 1.0,
            'multiagent': {
                'policies_to_train': ['learned00'],
                'policy_mapping_fn': tune.function(random_policy_mapping_fn),
                'policies': {
                    'learned00': (None, obs_space, action_space, {'model': model_config}),
                    'random': (RandomPolicy, obs_space, action_space, {}),
                    'mcts': (MCTSPolicy, obs_space, action_space, {
                        # 'max_rollouts': 10000,
                        # 'rollouts_timeout': 0.001,  # ~ 2 rollouts/action
                        # 'rollouts_timeout': 0.01,  # ~20 rollouts/action
                        # 'rollouts_timeout': 0.1,  # ~200 rollouts/action
                        # 'rollouts_timeout': 0.5,  # ~1k rollouts/action
                        # 'rollouts_timeout': 1.0,  # ~2k rollouts/action
                        'rollouts_timeout': 1.0,
                        # 'max_rollouts': 32,
                        # 'max_rollouts': 64,
                        # 'max_rollouts': 96,
                        'max_rollouts': 128,
                    }),
                    'human': (HumanPolicy, obs_space, action_space, {}),
                },
            },
            'callbacks': {'on_episode_end': tune.function(mcts_on_episode_end)},
            # 'evaluation_interval': 100,
            # 'evaluation_num_episodes': 10,
            # 'evaluation_config': {
            #     'entropy_coeff': 0.0,  # just copy in defaults to trick Trainer._evaluate()
            #     'entropy_coeff_schedule': None,
            # },
        }, **tune_config),
        checkpoint_at_end=True,
        # resume=True
    )
