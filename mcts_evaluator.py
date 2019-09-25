"""Evaluate a trained agent against different MCTS strengths."""

import argparse
import logging

import ray
from ray import tune
from ray.tune.registry import register_env

from src.callbacks import mcts_on_episode_end
from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy
from src.utils import get_debug_config, get_learner_policy_configs, get_model_config

logger = logging.getLogger('ray.rllib')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    ray.init(local_mode=args.debug)
    tune_config = {}

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

    num_learners = 2
    policies = get_learner_policy_configs(num_learners, obs_space, action_space, model_config)
    test_policy = 'learned06'

    def get_policy_by_time(rollouts_time):
        return {
            'policies_to_train': [*policies],
            'policy_mapping_fn': tune.function(lambda agent_id: [test_policy, 'mcts'][agent_id % 2]),
            'policies': dict({
                'mcts': (MCTSPolicy, obs_space, action_space, {
                    'max_rollouts': 10000,
                    'rollouts_timeout': rollouts_time,
                }),
            }, **policies),
        }
    mcts_rollout_times = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    # mcts_rollout_times = [0.001, 0.005, 0.01]

    def get_policy_by_num(num_rollouts):
        return {
            'policies_to_train': [*policies],
            'policy_mapping_fn': tune.function(lambda agent_id: [test_policy, 'mcts'][agent_id % 2]),
            'policies': dict({
                'random': (RandomPolicy, obs_space, action_space, {}),
                'mcts': (MCTSPolicy, obs_space, action_space, {'max_rollouts': num_rollouts, 'rollouts_timeout': 2.0}),
                'human': (HumanPolicy, obs_space, action_space, {}),
            }, **policies),
        }
    mcts_num_rollouts = [4, 8, 16, 32, 64, 128, 256, 512]
    # mcts_num_rollouts = [128, 256, 512, 1024, 2048]
    # mcts_num_rollouts = [5, 10, 50, 100, 250, 500, 1000]

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
            # 'log_level': 'DEBUG',
            # 'gamma': 0.9,
            # 'num_workers': 0,
            # 'num_gpus': 0,
            'num_workers': 20,
            'num_gpus': 1,

            # PPO customisations
            'lr': 0.001,
            'clip_param': 0.2,
            'gamma': 0.995,
            'lambda': 0.95,
            # # 'kl_coeff': 1.0,
            # 'train_batch_size': 65536,
            # 'sgd_minibatch_size': 4096,
            # 'num_sgd_iter': 6,
            # 'num_envs_per_worker': 32,

            # 'multiagent': tune.grid_search([get_policy_by_time(t) for t in mcts_rollout_times]),
            'multiagent': tune.grid_search([get_policy_by_num(n) for n in mcts_num_rollouts]),
            'callbacks': {'on_episode_end': tune.function(mcts_on_episode_end)},
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
        # restore='/home/dave/ray_results_old/mcts_trainer/PPO_c4_0_2019-09-03_21-50-47nggyraoy/checkpoint_453/checkpoint-453',
        # restore='/home/dave/ray_results/main/PPO_c4_0_2019-09-06_11-52-59zfekt1j_/checkpoint_79/checkpoint-79',
        # restore='/home/dave/ray_results/main/PPO_c4_0_2019-09-06_12-24-03jjal0_ts/checkpoint_782/checkpoint-782',
        # restore='/home/dave/ray_results/main/PPO_c4_0_2019-09-06_14-22-37v_bjhds7/checkpoint_782/checkpoint-782',
        # restore='/home/dave/ray_results/main/PPO_c4_0_2019-09-07_09-42-2148shgivu/checkpoint_7813/checkpoint-7813',
        # restore='/home/dave/ray_results/PPO_c4_2019-09-13_15-59-311qrvopuq/checkpoint_1/checkpoint-1',
        # restore='/home/dave/ray_results/mcts_trainer/PPO_c4_0_2019-09-14_07-14-15ydsrlhcr/checkpoint_241/checkpoint-241',
        restore='/home/dave/ray_results/main/PPO_c4_0_2019-09-23_16-17-45z9x1oc9j/checkpoint_782/checkpoint-782',
        # resume=True
    )
