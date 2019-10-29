import argparse
import math

import ray
from ray import tune
from ray.tune.registry import register_env

from src.callbacks import mcts_eval_policy_mapping_fn, random_policy_mapping_fn
from src.utils import get_debug_config, get_model_config, get_learner_policy_configs, get_mcts_policy_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--num-learners', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--human', action='store_true')
    args = parser.parse_args()

    ray.init(local_mode=args.debug)
    tune_config = get_debug_config(args)

    model_config, env_cls = get_model_config(args.use_cnn)
    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space, action_space = env.observation_space, env.action_space
    trainable_policies = get_learner_policy_configs(args.num_learners, obs_space, action_space, model_config)
    # mcts_policies = get_mcts_policy_configs([64, 128, 256, 512], obs_space, action_space)
    # mcts_policies = get_mcts_policy_configs([64, 128, 256], obs_space, action_space)
    # mcts_policies = get_mcts_policy_configs([32, 64], obs_space, action_space)
    # mcts_policies = get_mcts_policy_configs([8, 16, 32, 64], obs_space, action_space)
    mcts_eval_policies = get_mcts_policy_configs([8, 16], obs_space, action_space)
    # mcts_policies = get_mcts_policy_configs([1, 2], obs_space, action_space)
    mcts_train_policies = get_mcts_policy_configs([8, 16, 32, 64, 128, 256, 512], obs_space, action_space)

    def on_episode_start(info):
        episode = info['episode']
        episode.user_data['trainable_policies'] = [*trainable_policies]
        episode.user_data['mcts_policies'] = [*mcts_eval_policies]

    def name_trial(trial):
        """Give trials a more readable name in terminal & Tensorboard."""
        return f'{args.num_learners}x{trial.trainable_name}'


    policy_mapping_fn = random_policy_mapping_fn if args.num_learners > 1 else lambda _: ('learned00', 'learned00')

    tune.run(
        args.policy,
        name='trainer_evaluator',
        trial_name_creator=tune.function(name_trial),
        stop={
            'timesteps_total': int(100e6),
            # 'timesteps_total': int(1e9),
        },
        config=dict({
            'env': 'c4',
            'env_config': {},
            # 'lr': 0.001,
            'lr': 5e-5,
            # 'gamma': 0.995,
            'gamma': 0.9,
            'clip_param': 0.2,
            'lambda': 0.95,
            # 'kl_coeff': 1.0,
            'entropy_coeff': 0.01,
            'multiagent': {
                'policies_to_train': [*trainable_policies],
                'policy_mapping_fn': tune.function(policy_mapping_fn),
                'policies': {**trainable_policies, **mcts_train_policies},
            },
            'callbacks': {
                'on_episode_start': tune.function(on_episode_start),
            },
            'evaluation_interval': 10,
            # 'evaluation_num_episodes': 1,
            'evaluation_num_episodes': 1 if args.debug else math.ceil(args.num_learners / 2),
            'evaluation_config': {'multiagent': {'policy_mapping_fn': tune.function(mcts_eval_policy_mapping_fn)}},
        }, **tune_config),
        checkpoint_at_end=True,
    )
