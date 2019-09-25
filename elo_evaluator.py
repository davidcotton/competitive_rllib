"""Evaluate a trained agent against different MCTS strengths."""

import argparse
import logging
import random

import ray
from ray import tune
from ray.tune.registry import register_env

from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy
from src.utils import get_debug_config, get_learner_policy_configs, get_model_config, EloRater

logger = logging.getLogger('ray.rllib')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--num-learners', type=int, default=2)
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    ray.init(local_mode=args.debug)
    tune_config = {}

    tune_config.update(get_debug_config(args.debug))

    model_config, env_cls = get_model_config(args.use_cnn)
    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space = env.observation_space
    action_space = env.action_space
    policies = get_learner_policy_configs(args.num_learners, obs_space, action_space, model_config)

    player1, player2 = None, None

    def policy_mapping_fn(agent_id):
        global player1, player2
        if agent_id == 0:
            player1, player2 = random.sample([*policies], k=2)
            return player1
        else:
            return player2


    elo_rater = EloRater(policies.keys())

    def elo_on_episode_end(info):
        global elo_rater
        agents = [agent[1] for agent in info['episode'].agent_rewards.keys()]
        rewards = list(info['episode'].agent_rewards.values())
        if rewards[0] == rewards[1]:
            winner = 'draw'
        elif rewards[0] > rewards[1]:
            winner = agents[0]
        else:
            winner = agents[1]
        ratings = elo_rater.rate(*agents, winner)
        info['episode'].custom_metrics.update({'elo_' + k: v for k, v in ratings.items()})


    tune.run(
        args.policy,
        name='elo_evaluator',
        stop={
            # 'episodes_total': 1000,
            'timesteps_total': int(100e6),
        },
        config=dict({
            'env': 'c4',
            'env_config': {},
            'lr': 0.001,
            'gamma': 0.995,
            'lambda': 0.95,
            'clip_param': 0.2,
            'multiagent': {
                'policies_to_train': [*policies],
                'policy_mapping_fn': tune.function(policy_mapping_fn),
                'policies': dict({
                    'random': (RandomPolicy, obs_space, action_space, {}),
                    'human': (HumanPolicy, obs_space, action_space, {}),
                    'mcts': (MCTSPolicy, obs_space, action_space, {'max_rollouts': 1000, 'rollouts_timeout': 2.0}),
                }, **policies),
            },
            'callbacks': {'on_episode_end': tune.function(elo_on_episode_end)},
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
        # restore='/home/dave/ray_results/mcts_trainer/PPO_c4_0_2019-09-14_07-14-15ydsrlhcr/checkpoint_241/checkpoint-241',
        # restore='/home/dave/ray_results/main/PPO_c4_0_2019-09-23_16-17-45z9x1oc9j/checkpoint_782/checkpoint-782',
    )
