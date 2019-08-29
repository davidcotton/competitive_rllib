import argparse
import logging

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy
from ray.rllib.agents.dqn.dqn import check_config_and_setup_param_noise, get_initial_state, make_optimizer, \
    setup_exploration, update_worker_explorations, update_target_if_needed, add_trainer_metrics, disable_exploration
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import choose_policy_optimizer, update_kl, warn_about_bad_reward_scales, validate_config
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from src.callbacks import on_episode_end
from src.envs import Connect4Env, SquareConnect4Env
from src.models import ParametricActionsMLP, ParametricActionsCNN
from src.policies import MCTSPolicy, RandomPolicy

logger = logging.getLogger('ray.rllib')


# def my_train_fn(config, reporter):
#     # Train for 100 iterations with high LR
#     agent1 = PPOTrainer(env=Connect4Env, config=config)
#     for _ in range(10):
#         result = agent1.train()
#         result['phase'] = 1
#         reporter(**result)
#         phase1_time = result['timesteps_total']
#     state = agent1.save()
#     agent1.stop()
#
#     # Train for 100 iterations with low LR
#     config['lr'] = 0.0001
#     agent2 = PPOTrainer(env=Connect4Env, config=config)
#     agent2.restore(state)
#     for _ in range(10):
#         result = agent2.train()
#         result['phase'] = 2
#         result['timesteps_total'] += phase1_time  # keep time moving forward
#         reporter(**result)
#     agent2.stop()
#
#
# class MCTSEvalMixin:
#     def _evaluate(self):
#         """Evaluates current policy under `evaluation_config` settings.
#
#         Note that this default implementation does not do anything beyond
#         merging evaluation_config with the normal trainer config.
#         """
#
#         print('\n\n\n$$$$$$$$$$$$$$$$$$$$$$\nEVALUATE\n$$$$$$$$$$$$$$$$$$$$\n\n\n')
#
#         if not self.config['evaluation_config']:
#             raise ValueError(
#                 'No evaluation_config specified. It doesn't make sense '
#                 'to enable evaluation without specifying any config '
#                 'overrides, since the results will be the '
#                 'same as reported during normal policy evaluation.')
#
#         logger.info('Evaluating current policy for {} episodes'.format(self.config['evaluation_num_episodes']))
#         self._before_evaluate()
#         self.evaluation_workers.local_worker().restore(self.workers.local_worker().save())
#         for _ in range(self.config['evaluation_num_episodes']):
#             self.evaluation_workers.local_worker().sample()
#
#         metrics = collect_metrics(self.evaluation_workers.local_worker())
#         return {'evaluation': metrics}
#
#
# MyPPOTrainer = build_trainer(
#     name='PPO',
#     default_config=PPO_DEFAULT_CONFIG,
#     default_policy=PPOTFPolicy,
#     make_policy_optimizer=choose_policy_optimizer,
#     validate_config=validate_config,
#     after_optimizer_step=update_kl,
#     after_train_result=warn_about_bad_reward_scales,
#     mixins=[MCTSEvalMixin]
# )
#
# MyDQNTrainer = build_trainer(
#     name='DQN',
#     default_policy=DQNTFPolicy,
#     default_config=DQN_DEFAULT_CONFIG,
#     validate_config=check_config_and_setup_param_noise,
#     get_initial_state=get_initial_state,
#     make_policy_optimizer=make_optimizer,
#     before_init=setup_exploration,
#     before_train_step=update_worker_explorations,
#     after_optimizer_step=update_target_if_needed,
#     after_train_result=add_trainer_metrics,
#     collect_metrics_fn=collect_metrics,
#     before_evaluate_fn=disable_exploration)


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
                'mcts': (MCTSPolicy, obs_space, action_space, {
                    'player_id': 1,
                    'max_rollouts': num_rollouts,
                    'rollouts_timeout': 2.0,
                }),
            },
        }
    mcts_num_rollouts = [4, 8, 16, 32, 64, 128, 256, 512]
    # mcts_num_rollouts = [5, 10, 50, 100, 250, 500, 1000]

    # if args.debug:
    #     tune_config['log_level'] = 'DEBUG'

    ray.init(local_mode=args.debug)

    tune.run(
        args.policy,
        # my_train_fn,
        # MyPPOTrainer,
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
            # 'multiagent': {
            #     'policies_to_train': ['learned'],
            #     'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'mcts'][agent_id % 2]),
            #     'policies': get_policy(mcts_rollout_times[0]),
            # },
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
        # restore='/home/dave/ray_results/PPO/PPO_SquareConnect4Env_0_2019-08-26_15-48-13f_i8pykb/checkpoint_2500/checkpoint-2500',
        # resume=True
    )
