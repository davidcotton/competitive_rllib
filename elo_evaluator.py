"""Evaluate a trained agent against different MCTS strengths."""

import argparse
import random

import ray
from ray import tune
from ray.tune.registry import register_env

from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy
from src.utils import get_worker_config, get_learner_policy_configs, get_model_config, EloRater


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--num-learners', type=int, default=2)
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--debug', action='store_true')
    # e.g. --restore="/home/dave/ray_results/main/PPO_c4_0_2019-09-23_16-17-45z9x1oc9j/checkpoint_782/checkpoint-782"
    parser.add_argument('--restore', type=str)
    parser.add_argument('--human', action='store_true')
    args = parser.parse_args()

    ray.init(local_mode=args.debug)
    tune_config = get_worker_config(args)

    model_config, env_cls = get_model_config(args.use_cnn)
    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space, action_space = env.observation_space, env.action_space
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
                'policy_mapping_fn': policy_mapping_fn,
                'policies': {
                    **policies,
                    'mcts': (MCTSPolicy, obs_space, action_space, {}),
                    'human': (HumanPolicy, obs_space, action_space, {}),
                    'random': (RandomPolicy, obs_space, action_space, {}),
                },
            },
            'callbacks': {'on_episode_end': elo_on_episode_end},
        }, **tune_config),
        restore=args.restore,
        checkpoint_at_end=True,
    )
