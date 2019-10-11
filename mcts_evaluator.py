"""Evaluate a trained agent against different MCTS strengths."""

import argparse

import ray
from ray import tune
from ray.tune.registry import register_env

from src.callbacks import mcts_metrics_on_episode_end
from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy
from src.utils import get_debug_config, get_learner_policy_configs, get_model_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--num-learners', type=int, default=2)
    # e.g. --restore="/home/dave/ray_results/main/PPO_c4_0_2019-09-23_16-17-45z9x1oc9j/checkpoint_782/checkpoint-782"
    parser.add_argument('--restore', type=str)
    # e.g. --eval-policy=learned06
    parser.add_argument('--eval-policy', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    ray.init(local_mode=args.debug)
    tune_config = get_debug_config(args.debug)

    model_config, env_cls = get_model_config(args.use_cnn)
    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space, action_space = env.observation_space, env.action_space
    trainable_policies = get_learner_policy_configs(args.num_learners, obs_space, action_space, model_config)

    def get_policy_by_num(num_rollouts):
        return {
            'policies_to_train': [*trainable_policies],
            # 'policy_mapping_fn': tune.function(lambda agent_id: [args.eval_policy, 'mcts'][agent_id % 2]),
            'policy_mapping_fn': tune.function(lambda _: (args.eval_policy, 'mcts')),
            'policies': dict({
                'random': (RandomPolicy, obs_space, action_space, {}),
                'mcts': (MCTSPolicy, obs_space, action_space, {'max_rollouts': num_rollouts, 'rollouts_timeout': 2.0}),
                'human': (HumanPolicy, obs_space, action_space, {}),
            }, **trainable_policies),
        }
    mcts_num_rollouts = [4, 8, 16, 32, 64, 128, 256, 512]
    # mcts_num_rollouts = [128, 256, 512, 1024, 2048]

    def name_trial(trial):
        """Give trials a more readable name in terminal & Tensorboard."""
        num_mcts_rollouts = trial.config['multiagent']['policies']['mcts'][3]['max_rollouts']
        return f'{trial.trainable_name}_MCTS({num_mcts_rollouts})'

    tune.run(
        args.policy,
        name='mcts_evaluator',
        trial_name_creator=tune.function(name_trial),
        stop={
            'episodes_total': 1000,
        },
        config=dict({
            'env': 'c4',
            'env_config': {},
            'lr': 0.001,
            'gamma': 0.995,
            'lambda': 0.95,
            'clip_param': 0.2,
            # 'kl_coeff': 1.0,
            'multiagent': tune.grid_search([get_policy_by_num(n) for n in mcts_num_rollouts]),
            'callbacks': {'on_episode_end': tune.function(mcts_metrics_on_episode_end)},
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
        restore=args.restore,
    )
