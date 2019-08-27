import argparse
import logging

import numpy as np
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from src.envs import Connect4Env, SquareConnect4Env
from src.models import ParametricActionsMLP, ParametricActionsCNN
from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy

logger = logging.getLogger('ray.rllib')


def on_episode_end(info):
    """Add custom metrics to track MCTS rollouts"""
    if 'mcts' in info['policy']:
        episode = info['episode']
        mcts_policy = info['policy']['mcts']
        episode.custom_metrics['num_rollouts'] = np.mean(mcts_policy.metrics['num_rollouts'])
        mcts_policy.metrics['num_rollouts'].clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="PG")
    parser.add_argument("--use-cnn", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tune_config = {}

    if args.use_cnn:
        env_cls = SquareConnect4Env
        ModelCatalog.register_custom_model('parametric_model', ParametricActionsCNN)
        model_config = {
            'custom_model': 'parametric_model',
            'conv_filters': [[16, [2, 2], 1], [32, [2, 2], 1], [64, [3, 3], 2]],
            'conv_activation': 'leaky_relu',
            'fcnet_hiddens': [256, 256],
            'fcnet_activation': 'leaky_relu',
        }
    else:
        env_cls = Connect4Env
        ModelCatalog.register_custom_model('parametric_model', ParametricActionsMLP)
        model_config = {
            'custom_model': 'parametric_model',
            'fcnet_hiddens': [256, 256],
            'fcnet_activation': 'leaky_relu',
        }

    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space = env.observation_space
    action_space = env.action_space

    if args.policy == 'DQN':
        tune_config.update({
            'hiddens': [],
            'dueling': False,
        })

    if args.debug:
        tune_config['log_level'] = 'DEBUG'

    tune.run(
        args.policy,
        stop={
            # 'timesteps_total': int(50e3),
            # 'timesteps_total': int(500e3),
            # 'timesteps_total': int(10e6),
            # 'timesteps_total': int(10e6),
            'policy_reward_mean': {'learned': 0.95},
            # 'policy_reward_mean': {'learned': 0.8},
            # 'policy_reward_mean': {'learned': 0.5},
            # 'policy_reward_mean': {'learned': 0.8, 'learned2': 0.8},
        },
        config=dict({
            'env': 'c4',
            'gamma': 0.9,
            # 'num_workers': 0,
            'num_workers': 20,
            # 'num_gpus': 1,
            'multiagent': {
                'policies_to_train': ['learned'],
                # 'policies_to_train': ['learned', 'learned2'],
                # 'policy_mapping_fn': tune.function(select_policy),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'random'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'learned2'][agent_id % 2]),
                'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'mcts'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'human'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['mcts', 'human'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda _: 'random'),
                'policies': {
                    'learned': (None, obs_space, action_space, {'model': model_config}),
                    'learned2': (None, obs_space, action_space, {'model': model_config}),
                    'random': (RandomPolicy, obs_space, action_space, {}),
                    'mcts': (MCTSPolicy, obs_space, action_space, {
                        'player_id': 1,
                        'max_rollouts': 10000,
                        'rollouts_timeout': 0.001,  # ~ 2 rollouts/action
                        # 'rollouts_timeout': 0.01,  # ~20 rollouts/action
                        # 'rollouts_timeout': 0.1,  # ~200 rollouts/action
                        # 'rollouts_timeout': 0.5,  # ~1k rollouts/action
                        # 'rollouts_timeout': 1.0,  # ~2k rollouts/action
                    }),
                    'human': (HumanPolicy, obs_space, action_space, {
                        'player_id': 1,
                    }),
                },
            },
            'callbacks': {
                'on_episode_end': tune.function(on_episode_end),
            },
        }, **tune_config),
        # checkpoint_freq=100,
        # checkpoint_at_end=True,
        # restore='/home/dave/ray_results/PPO/PPO_SquareConnect4Env_0_2019-08-25_08-34-579oxf4ods/checkpoint_119/checkpoint-119',
        # restore='/home/dave/ray_results/PPO/PPO_SquareConnect4Env_0_2019-08-26_15-48-13f_i8pykb/checkpoint_2500/checkpoint-2500',
        # resume=True
    )
