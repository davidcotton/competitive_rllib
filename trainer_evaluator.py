import argparse
import random

import ray
from ray import tune
from ray.tune.registry import register_env

from src.callbacks import win_matrix_on_episode_end, mcts_metrics_on_episode_end, mcts_eval_policy_mapping_fn, \
    random_policy_mapping_fn
from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy
from src.utils import get_debug_config, get_model_config, get_learner_policy_configs, get_mcts_policy_configs


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
    obs_space, action_space = env.observation_space, env.action_space
    trainable_policies = get_learner_policy_configs(args.num_learners, obs_space, action_space, model_config)
    mcts_policies = get_mcts_policy_configs([1, 2, 4], obs_space, action_space)

    # def random_policy_mapping_fn(info):
    #     return random.sample([*trainable_policies], k=2)
    #
    # def eval_policy_mapping_fn(info):
    #     eval_policies = [random.choice([*trainable_policies]), random.choice([*mcts_policies])]
    #     random.shuffle(eval_policies)
    #     return eval_policies

    def on_episode_start(info):
        episode = info['episode']
        episode.user_data['trainable_policies'] = [*trainable_policies]
        episode.user_data['mcts_policies'] = [*mcts_policies]

    tune.run(
        args.policy,
        name='main',
        stop={
            'timesteps_total': int(1e6),
            # 'timesteps_total': int(100e6),
            # 'timesteps_total': int(1e9),
        },
        config=dict({
            'env': 'c4',
            'env_config': {},
            'lr': 0.001,
            'gamma': 0.995,
            'clip_param': 0.2,
            'lambda': 0.95,
            # 'kl_coeff': 1.0,
            'multiagent': {
                'policies_to_train': [*trainable_policies],
                'policy_mapping_fn': tune.function(random_policy_mapping_fn),
                # 'policies': dict({
                #     'human': (HumanPolicy, obs_space, action_space, {}),
                #     'mcts': (MCTSPolicy, obs_space, action_space, {}),
                # }, **trainable_policies),
                'policies': {**trainable_policies, **mcts_policies},
            },
            'callbacks': {
                'on_episode_start': tune.function(on_episode_start),
                # 'on_episode_end': tune.function(win_matrix_on_episode_end),
                # 'on_episode_end': tune.function(mcts_metrics_on_episode_end),
            },
            'evaluation_interval': 1,
            'evaluation_num_episodes': 10,
            'evaluation_config': {
                'multiagent': {'policy_mapping_fn': tune.function(mcts_eval_policy_mapping_fn)},
                'callbacks': {
                    # 'on_episode_start': tune.function(on_episode_start),
                    # 'on_episode_end': tune.function(win_matrix_on_episode_end),
                    # 'on_episode_end': tune.function(mcts_metrics_on_episode_end),
                },
            },
        }, **tune_config),
        checkpoint_at_end=True,
    )
