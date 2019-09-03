"""Train a RL policy against a customisable MCTS agent."""

import argparse
import logging
import random

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from src.callbacks import on_episode_end
from src.envs import Connect4Env, SquareConnect4Env
from src.models import ParametricActionsMLP, ParametricActionsCNN
from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy

logger = logging.getLogger('ray.rllib')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="PG")
    parser.add_argument("--use-cnn", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    ray.init(local_mode=args.debug)
    tune_config = {}

    if args.use_cnn:
        env_cls = SquareConnect4Env
        ModelCatalog.register_custom_model('parametric_actions_model', ParametricActionsCNN)
        model_config = {
            'custom_model': 'parametric_actions_model',
            'conv_filters': [[16, [2, 2], 1], [32, [2, 2], 1], [64, [3, 3], 2]],
            'conv_activation': 'leaky_relu',
            'fcnet_hiddens': [256, 256],
            'fcnet_activation': 'leaky_relu',
        }
    else:
        env_cls = Connect4Env
        ModelCatalog.register_custom_model('parametric_actions_model', ParametricActionsMLP)
        model_config = {
            'custom_model': 'parametric_actions_model',
            # 'fcnet_hiddens': [256, 256],
            'fcnet_hiddens': [128, 128],
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

    policy = None
    policies = ['learned', 'mcts']

    def policy_mapping_fn(agent_id):
        global policy
        if agent_id == 0:
            assert policy is None
            policy = random.choice(policies)
            return policy
        else:
            value = policies[0] if policy == policies[1] else policies[1]
            policy = None
            return value

    tune.run(
        args.policy,
        name='mcts_trainer',
        stop={
            # 'timesteps_total': int(50e3),
            # 'timesteps_total': int(10e6),
            # 'policy_reward_mean': {'learned': 0.95},
            'policy_reward_mean': {'learned': 0.8},
            # 'policy_reward_mean': {'learned': 0.8, 'learned2': 0.8},
        },
        config=dict({
            'env': 'c4',
            'env_config': {},
            'gamma': 0.9,
            # 'num_workers': 0,
            'num_workers': 20,
            'num_gpus': 0,
            # 'num_gpus': 1,
            'multiagent': {
                'policies_to_train': ['learned'],
                # 'policies_to_train': ['learned', 'learned2'],
                'policy_mapping_fn': tune.function(policy_mapping_fn),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'random'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'learned2'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'mcts'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'human'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['mcts', 'human'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda _: 'random'),
                'policies': {
                    'learned': (None, obs_space, action_space, {'model': model_config}),
                    'learned2': (None, obs_space, action_space, {'model': model_config}),
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
                        'max_rollouts': 64,
                        # 'max_rollouts': 96,
                    }),
                    'human': (HumanPolicy, obs_space, action_space, {}),
                },
            },
            'callbacks': {'on_episode_end': tune.function(on_episode_end)},
            # 'evaluation_interval': 100,
            # 'evaluation_num_episodes': 10,
            # 'evaluation_config': {
            #     'entropy_coeff': 0.0,  # just copy in defaults to trick Trainer._evaluate()
            #     'entropy_coeff_schedule': None,
            # },
        }, **tune_config),
        # checkpoint_freq=100,
        checkpoint_at_end=True,
        # restore='/home/dave/ray_results/selfplay/PPO_c4_0_2019-08-30_20-15-16tvle8xqv/checkpoint_13486/checkpoint-13486',
        # resume=True
    )
