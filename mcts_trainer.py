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

    player1, player2 = None, None
    policies = ['learned00', 'mcts']
    # policies = ['learned00', 'human']

    def policy_mapping_fn(agent_id):
        global player1, player2
        if agent_id == 0:
            player1, player2 = random.sample(policies, k=2)
            return player1
        else:
            return player2

    tune.run(
        args.policy,
        name='mcts_trainer',
        stop={
            # 'policy_reward_mean': {'learned00': 0.95},
            'policy_reward_mean': {'learned00': 0.8},
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
                'policy_mapping_fn': tune.function(policy_mapping_fn),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned00', 'mcts'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['learned00', 'human'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['mcts', 'human'][agent_id % 2]),
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
        # checkpoint_freq=10,
        checkpoint_at_end=True,
        # restore='/home/dave/ray_results/selfplay/PPO_c4_0_2019-08-30_20-15-16tvle8xqv/checkpoint_13486/checkpoint-13486',
        # restore='/home/dave/ray_results_old/mcts_trainer/PPO_c4_0_2019-09-03_21-50-47nggyraoy/checkpoint_453/checkpoint-453',
        # resume=True
    )
