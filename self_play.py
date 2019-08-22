import argparse
import logging

from gym import spaces
import numpy as np
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from src.envs import Connect4Env, FlattenedConnect4Env, SquareConnect4Env
from src.models import ParametricActionsMLP, ParametricActionsCNN
from src.policies import MCTSPolicy, RandomPolicy

logger = logging.getLogger('ray.rllib')
tf = try_import_tf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="PG")
    args = parser.parse_args()

    ModelCatalog.register_custom_model('parametric_mlp', ParametricActionsMLP)
    ModelCatalog.register_custom_model('parametric_cnn', ParametricActionsCNN)

    obs_space = spaces.Dict({
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
        args.policy,
        stop={
            # 'timesteps_total': int(50e3),
            # 'timesteps_total': int(500e3),
            # 'timesteps_total': int(10e6),
            'policy_reward_mean': {'learned': 0.80},
        },
        config=dict({
            # 'env': Connect4Env,
            # 'env': FlattenedConnect4Env,
            'env': SquareConnect4Env,
            'log_level': 'DEBUG',
            'gamma': 0.9,
            # 'num_workers': 0,
            'num_workers': 20,
            # 'num_gpus': 1,
            'multiagent': {
                # 'policies_to_train': ['learned', 'learned2'],
                'policies_to_train': ['learned'],
                # 'policy_mapping_fn': tune.function(select_policy),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'random'][agent_id % 2]),
                'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'mcts'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'learned2'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'learned'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda _: 'random'),
                'policies': {
                    'learned': (None, obs_space, action_space, {
                        'model': {
                            'custom_model': 'parametric_mlp',
                            'fcnet_hiddens': [256, 256],
                            'fcnet_activation': 'leaky_relu',
                        }
                    }),
                    # 'learned2': (None, obs_space, action_space, {
                    #     'model': {
                    #         'custom_model': 'parametric_mlp',
                    #         'fcnet_hiddens': [256, 256],
                    #         'fcnet_activation': 'leaky_relu',
                    #     }
                    # }),
                    'random': (RandomPolicy, obs_space, action_space, {}),
                    'mcts': (MCTSPolicy, obs_space, action_space, {
                        'max_rollouts': 1000,
                        'rollouts_timeout': 0.001,
                        # 'rollouts_timeout': 0.01,
                        # 'rollouts_timeout': 0.1,
                    }),
                },
            },
        }, **config),
        # checkpoint_freq=10,
        # checkpoint_at_end=True,
    )
