import argparse
import logging

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from src.envs import Connect4Env, SquareConnect4Env
from src.models import ParametricActionsMLP, ParametricActionsCNN
from src.models.preprocessors import SquareObsPreprocessor, FlattenObsPreprocessor
from src.policies import RandomPolicy

logger = logging.getLogger('ray.rllib')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="PG")
    parser.add_argument("--use-cnn", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tune_config = {}
    ray.init(local_mode=args.debug)

    # ModelCatalog.register_custom_preprocessor('flatten_obs_preprocessor', FlattenObsPreprocessor)

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
        name='main',
        stop={
            # 'timesteps_total': int(500e3),
            'timesteps_total': int(250e3),
            'policy_reward_mean': {'learned': 0.99},
        },
        config=dict({
            'env': 'c4',
            'gamma': 0.9,
            # 'gamma': tune.grid_search([0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 1.0]),
            # 'num_workers': 20,
            # 'num_workers': tune.grid_search([0, 1, 2, 4, 8, 16, 20]),
            # 'num_envs_per_worker': 4,
            'num_workers': 0,
            # 'num_envs_per_worker': 1,
            # 'num_gpus': 1,
            # 'sample_batch_size': 10,
            # 'train_batch_size': 200,
            'multiagent': {
                'policies_to_train': ['learned'],
                'policies': {
                    'random': (RandomPolicy, obs_space, action_space, {}),
                    'learned': (None, obs_space, action_space, {'model': model_config}),
                    'learned2': (None, obs_space, action_space, {'model': model_config}),
                },
                # 'policy_mapping_fn': tune.function(select_policy),
                'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'random'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'learned2'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda _: 'random'),
            },
        }, **tune_config))
