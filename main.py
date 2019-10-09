import argparse
import logging
import random

import ray
from ray import tune
from ray.tune.registry import register_env

from src.callbacks import win_matrix_on_episode_end
from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy
from src.utils import get_debug_config, get_learner_policy_configs, get_model_config

logger = logging.getLogger('ray.rllib')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--num-learners', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    ray.init(local_mode=args.debug)
    tune_config = get_debug_config(args.debug)

    model_config, env_cls = get_model_config(args.use_cnn)
    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space = env.observation_space
    action_space = env.action_space
    policies = get_learner_policy_configs(args.num_learners, obs_space, action_space, model_config)

    if args.policy == 'DQN':
        tune_config.update({
            'hiddens': [],
            'dueling': False,
        })

    player1, player2 = None, None

    def policy_mapping_fn(agent_id):
        global player1, player2
        if agent_id == 0:
            player1, player2 = random.sample([*policies], k=2)
            return player1
        else:
            return player2

    tune.run(
        args.policy,
        name='main',
        stop={
            'timesteps_total': int(100e6),
            # 'timesteps_total': int(1e9),
        },
        config=dict({
            'env': 'c4',
            'env_config': {},
            'lr': 0.001,
            'gamma': 0.995,
            # 'gamma': 0.9,
            # 'gamma': tune.grid_search([0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 1.0]),
            'clip_param': 0.2,
            'lambda': 0.95,
            # 'kl_coeff': 1.0,
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
            'callbacks': {
                'on_episode_end': tune.function(win_matrix_on_episode_end),
            },
        }, **tune_config),
        # checkpoint_freq=100,
        checkpoint_at_end=True,
        # resume=True,
        # restore='/home/dave/ray_results_old/mcts_trainer/PPO_c4_0_2019-09-03_21-50-47nggyraoy/checkpoint_453/checkpoint-453',
    )
