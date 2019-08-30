import argparse
import logging

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
    parser.add_argument('--policy', type=str, default='PG')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--debug', action='store_true')
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

    def get_policy_by_time(rollouts_time):
        return {
            'policies_to_train': ['learned'],
            'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'mcts'][agent_id % 2]),
            'policies': {
                'learned': (None, obs_space, action_space, {'model': model_config}),
                'mcts': (MCTSPolicy, obs_space, action_space, {
                    'player_id': 1,
                    'max_rollouts': 10000,
                    'rollouts_timeout': rollouts_time,
                }),
            },
        }
    mcts_rollout_times = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    # mcts_rollout_times = [0.001, 0.005, 0.01]

    def get_policy_by_num(num_rollouts):
        return {
            'policies_to_train': ['learned'],
            'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'mcts'][agent_id % 2]),
            'policies': {
                'learned': (None, obs_space, action_space, {'model': model_config}),
                'learned2': (None, obs_space, action_space, {'model': model_config}),
                'random': (RandomPolicy, obs_space, action_space, {}),
                'mcts': (MCTSPolicy, obs_space, action_space, {
                    'player_id': 1,
                    'max_rollouts': num_rollouts,
                    'rollouts_timeout': 2.0,
                }),
                'human': (HumanPolicy, obs_space, action_space, {'player_id': 1}),
            },
        }
    mcts_num_rollouts = [4, 8, 16, 32, 64, 128, 256, 512]
    # mcts_num_rollouts = [5, 10, 50, 100, 250, 500, 1000]

    ray.init(local_mode=args.debug)

    tune.run(
        args.policy,
        name='EvalTrainer',
        stop={
            # 'timesteps_total': int(10e3),
            # 'episodes_total': 400,
            'episodes_total': 1000,
        },
        config=dict({
            'env': 'c4',
            'env_config': {},
            # 'log_level': 'DEBUG',
            'gamma': 0.9,
            # 'num_workers': 0,
            'num_workers': 20,
            # 'num_gpus': 1,
            'num_gpus': 0,
            # 'multiagent': tune.grid_search([get_policy_by_time(t) for t in mcts_rollout_times]),
            'multiagent': tune.grid_search([get_policy_by_num(n) for n in mcts_num_rollouts]),
            'callbacks': {'on_episode_end': tune.function(on_episode_end)},
            # 'evaluation_interval': 1,
            # 'evaluation_num_episodes': 10,
            # 'evaluation_config': {
            #     # 'entropy_coeff': 0.0,  # just copy in defaults to trick Trainer._evaluate()
            #     # 'entropy_coeff_schedule': None,
            #     # 'rollouts_timeout': 0.5
            #     'exploration_fraction': 0,
            #     'exploration_final_eps': 0.5,
            # },
        }, **tune_config),
        # restore='/home/dave/ray_results/selfplay/PPO_c4_0_2019-08-30_12-59-08rq0qg5nh/checkpoint_74/checkpoint-74',
        # resume=True
    )
