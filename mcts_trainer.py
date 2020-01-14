"""Train a RL policy against a customisable MCTS agent."""

import argparse
import random

import ray
from ray import tune
from ray.tune.registry import register_env

from src.callbacks import mcts_metrics_on_episode_end
from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy
from src.utils import get_worker_config, get_model_config, get_policy_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--num-rollouts', type=int, default=128)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--human', action='store_true')
    parser.add_argument('--save-experience', action='store_true')
    args = parser.parse_args()

    ray.init(local_mode=args.debug)
    tune_config = get_worker_config(args)
    tune_config.update(get_policy_config(args.policy))

    model_config, env_cls = get_model_config(args.use_cnn)
    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space, action_space = env.observation_space, env.action_space

    if args.save_experience:
        tune_config['output'] = 'experience'

    train_pid = 'learned00'
    mcts_pid = f'mcts{args.num_rollouts:03d}'
    policies = [train_pid, 'human'] if args.human else [train_pid, mcts_pid]

    def random_policy_mapping_fn(info):
        return random.sample(policies, k=2)

    def name_trial(trial):
        """Give trials a more readable name in terminal & Tensorboard."""
        network = 'CNN' if args.use_cnn else 'MLP'
        rollouts = args.num_rollouts
        return f'MCTS{rollouts}vs{trial.trainable_name}-{network}'

    tune.run(
        args.policy,
        name='mcts_trainer',
        trial_name_creator=name_trial,
        stop={f'policy_reward_mean/{train_pid}': 0.6},
        config=dict({
            'env': 'c4',
            'env_config': {},
            'multiagent': {
                'policies_to_train': [train_pid],
                'policy_mapping_fn': random_policy_mapping_fn,
                'policies': {
                    train_pid: (None, obs_space, action_space, {'model': model_config}),
                    mcts_pid: (MCTSPolicy, obs_space, action_space, {'max_rollouts': args.num_rollouts}),
                    'human': (HumanPolicy, obs_space, action_space, {}),
                    'random': (RandomPolicy, obs_space, action_space, {}),
                },
            },
            'callbacks': {'on_episode_end': mcts_metrics_on_episode_end},
        }, **tune_config),
        checkpoint_freq=100,
        checkpoint_at_end=True,
        resume=True
    )
