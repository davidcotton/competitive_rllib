import logging

from gym import spaces
import numpy as np
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from src.envs import Connect4Env, FlattenedConnect4Env, SquareConnect4Env
from src.models import MaskedMLPModel, MaskedCNNModel
from src.models.preprocessors import SquareObsPreprocessor, FlattenObsPreprocessor
from src.policies import RandomPolicy

logger = logging.getLogger('ray.rllib')
tf = try_import_tf()
POLICY = 'PG'


if __name__ == '__main__':
    def select_policy(agent_id):
        if agent_id == 0:
            return 'learned'
        else:
            # return random.choice(['always_same', 'beat_last'])
            return 'random'

    ray.init()

    # ModelCatalog.register_custom_preprocessor('square_obs_preprocessor', SquareObsPreprocessor)
    # ModelCatalog.register_custom_preprocessor('flatten_obs_preprocessor', FlattenObsPreprocessor)
    # ModelCatalog.register_custom_model('masked_mlp_model', JasonMLPModel)
    ModelCatalog.register_custom_model('masked_mlp_model', MaskedCNNModel)
    ModelCatalog.register_custom_model('masked_cnn_model', MaskedCNNModel)

    obs_space = spaces.Dict({
        # 'board': spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.uint8),
        # 'board': spaces.Box(low=0, high=2, shape=(6 * 7,), dtype=np.uint8),
        'board': spaces.Box(low=0, high=3, shape=(7, 7), dtype=np.uint8),
        'action_mask': spaces.Box(low=0, high=1, shape=(7,), dtype=np.uint8),
    })
    action_space = spaces.Discrete(7)

    tune.run(
        # trainer,
        POLICY,
        stop={
            # 'timesteps_total': 1000,
            'timesteps_total': 500000,
            'policy_reward_mean': {'learned': 0.99},
        },
        config={
            # 'env': Connect4Env,
            # 'env': FlattenedConnect4Env,
            'env': SquareConnect4Env,
            'log_level': 'DEBUG',
            # 'gamma': 0.9,
            # 'num_workers': 4,
            # 'num_envs_per_worker': 4,
            'num_workers': 0,
            'num_envs_per_worker': 1,
            'num_gpus': 1,
            # 'sample_batch_size': 10,
            # 'train_batch_size': 200,
            'multiagent': {
                'policies_to_train': ['learned'],
                'policies': {
                    'random': (RandomPolicy, obs_space, action_space, {
                        # 'model': {
                        #     'custom_preprocessor': 'flatten_obs_preprocessor',
                        # },
                    }),
                    'learned': (None, obs_space, action_space, {
                        'model': {
                            # 'custom_model': 'masked_mlp_model',
                            'custom_model': 'masked_mlp_model',
                            # 'custom_model': 'masked_cnn_model',
                            # 'conv_filters': [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]],
                            'conv_filters': [[7, [2, 2], 1], [49, [3, 3], 1], [512, [2, 2], 1]],
                            # 'conv_filters': [[16, [2, 2], 1],],
                            'fcnet_hiddens': [256, 256],
                            # "fcnet_hiddens": [256, 128, 64, 32],
                            'fcnet_activation': 'leaky_relu',
                            # 'fcnet_activation': 'relu',
                            # 'fcnet_activation': 'tanh',
                            # 'custom_preprocessor': 'square_obs_preprocessor',
                            # 'custom_preprocessor': 'flatten_obs_preprocessor',
                            # 'custom_options': {},  # extra options to pass to your preprocessor
                            'custom_options': {
                                'parent_policy': POLICY,
                            },
                        }
                    }),
                },
                # 'policy_mapping_fn': tune.function(select_policy),
                'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'random'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda _: 'random'),
            },
        })
