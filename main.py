import argparse
import logging
import random

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

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
    policies = {
        'learned1': (None, obs_space, action_space, {'model': model_config}),
        'learned2': (None, obs_space, action_space, {'model': model_config}),
        'learned3': (None, obs_space, action_space, {'model': model_config}),
        'learned4': (None, obs_space, action_space, {'model': model_config}),
        # 'learned5': (None, obs_space, action_space, {'model': model_config}),
        # 'learned6': (None, obs_space, action_space, {'model': model_config}),
        # 'learned7': (None, obs_space, action_space, {'model': model_config}),
        # 'learned8': (None, obs_space, action_space, {'model': model_config}),
    }
    # policies = ['learned', 'human']

    def policy_mapping_fn(agent_id):
        global policy
        if agent_id == 0:
            assert policy is None
            policy = random.choice([*policies])
            return policy
        else:
            remaining = [*policies]
            remaining.remove(policy)
            policy = None
            return random.choice(remaining)

    tune.run(
        args.policy,
        name='main',
        stop={
            # 'timesteps_total': int(500e3),
            # 'timesteps_total': int(1e9),
            'timesteps_total': int(100e6),
            # 'policy_reward_mean': {'learned': 0.99},
            # 'policy_reward_mean': {'learned': 0.8, 'learned2': 0.8},
        },
        config=dict({
            'env': 'c4',
            'env_config': {},
            # 'gamma': 0.9,
            # 'gamma': tune.grid_search([0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 1.0]),
            # 'num_workers': 0,
            # 'num_gpus': 0,
            'num_workers': 20,
            'num_gpus': 1,

            # ------------------------
            # PPO customisations
            'lr': 0.001,
            'clip_param': 0.2,
            'gamma': 0.995,
            'lambda': 0.95,
            # 'kl_coeff': 1.0,
            'train_batch_size': 65536,
            'sgd_minibatch_size': 4096,
            'num_sgd_iter': 6,
            'num_envs_per_worker': 32,
            # ------------------------

            'multiagent': {
                'policies_to_train': [*policies],
                'policy_mapping_fn': tune.function(policy_mapping_fn),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'random'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda _: 'random'),
                'policies': dict({
                    'random': (RandomPolicy, obs_space, action_space, {}),
                    'human': (HumanPolicy, obs_space, action_space, {}),
                    'mcts': (MCTSPolicy, obs_space, action_space, {'max_rollouts': 1000, 'rollouts_timeout': 2.0}),
                }, **policies),
            },
        }, **tune_config),
        # checkpoint_freq=100,
        checkpoint_at_end=True,
        # restore='/home/dave/ray_results_old/mcts_trainer/PPO_c4_0_2019-09-03_21-50-47nggyraoy/checkpoint_453/checkpoint-453',
        # resume=True,
    )
