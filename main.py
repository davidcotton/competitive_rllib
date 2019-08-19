import argparse
import logging

from gym import spaces
import numpy as np
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from src.envs import Connect4Env, FlattenedConnect4Env, SquareConnect4Env
from src.models import MaskedMLPModel, MaskedCNNModel, ParametricActionsMLP, ParametricActionsCNN
from src.models.preprocessors import SquareObsPreprocessor, FlattenObsPreprocessor
from src.policies import RandomPolicy

logger = logging.getLogger('ray.rllib')
tf = try_import_tf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="PG")
    args = parser.parse_args()

    def select_policy(agent_id):
        if agent_id == 0:
            return 'learned'
        else:
            # return random.choice(['always_same', 'beat_last'])
            return 'random'

    ray.init()

    # ModelCatalog.register_custom_preprocessor('flatten_obs_preprocessor', FlattenObsPreprocessor)
    ModelCatalog.register_custom_model('masked_mlp_model', MaskedMLPModel)
    ModelCatalog.register_custom_model('masked_cnn_model', MaskedCNNModel)
    ModelCatalog.register_custom_model('parametric_mlp', ParametricActionsMLP)
    ModelCatalog.register_custom_model('parametric_cnn', ParametricActionsCNN)

    obs_space = spaces.Dict({
        # 'board': spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.uint8),
        # 'board': spaces.Box(low=0, high=2, shape=(6 * 7,), dtype=np.uint8),
        'board': spaces.Box(low=0, high=3, shape=(7, 7), dtype=np.uint8),
        'action_mask': spaces.Box(low=0, high=1, shape=(7,), dtype=np.uint8),
    })
    action_space = spaces.Discrete(7)

    if args.policy == 'DQN':
        config = {
            'hiddens': [],
            'dueling': False,
        }
    else:
        config = {}

    tune.run(
        # trainer,
        args.policy,
        stop={
            # 'timesteps_total': 1000,
            # 'timesteps_total': int(500e3),
            'timesteps_total': int(250e3),
            'policy_reward_mean': {'learned': 0.99},
        },
        config=dict({
            # 'env': Connect4Env,
            # 'env': FlattenedConnect4Env,
            'env': SquareConnect4Env,
            'log_level': 'DEBUG',
            'gamma': 0.9,
            # 'gamma': tune.grid_search([0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 1.0]),
            # 'num_workers': 16,
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
                    'learned': (None, obs_space, action_space, {
                        'model': {
                            # 'custom_model': 'masked_mlp_model',
                            # 'custom_model': 'masked_cnn_model',
                            'custom_model': 'parametric_mlp',
                            # 'custom_model': 'parametric_cnn',
                            'conv_filters': [[16, [2, 2], 1], [32, [2, 2], 1], [64, [3, 3], 2]],
                            # 'conv_filters': [[7, [2, 2], 1], [49, [3, 3], 1], [512, [2, 2], 1]],
                            'conv_activation': 'leaky_relu',
                            'fcnet_hiddens': [256, 256],
                            'fcnet_activation': 'leaky_relu',
                            # 'custom_preprocessor': 'flatten_obs_preprocessor',
                            # 'custom_options': {},
                        }
                    }),
                },
                # 'policy_mapping_fn': tune.function(select_policy),
                'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'random'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda _: 'random'),
            },
        }, **config))
