import argparse
import logging
import random

import ray
from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_TRAINER_CONFIG, choose_policy_optimizer, \
    validate_config, update_kl, warn_about_bad_reward_scales
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, KLCoeffMixin, ValueNetworkMixin, ppo_surrogate_loss, \
    kl_and_loss_stats, vf_preds_and_logits_fetches, postprocess_ppo_gae, clip_gradients, setup_config, setup_mixins
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.tune.trainable import Trainable
from ray.tune.registry import register_env

from src.envs import Connect4Env, SquareConnect4Env
from src.models import ParametricActionsMLP, ParametricActionsCNN
from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy

logger = logging.getLogger('ray.rllib')


MyPPOTFPolicy = build_tf_policy(
    name='MyPPOTFPolicy',
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=vf_preds_and_logits_fetches,
    postprocess_fn=postprocess_ppo_gae,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])
MyPPOTrainer = build_trainer(
    name='MyPPOTrainer',
    default_config=PPO_TRAINER_CONFIG,
    # default_policy=PPOTFPolicy,
    default_policy=MyPPOTFPolicy,
    # get_policy_class=my_get_policy_class,
    make_policy_optimizer=choose_policy_optimizer,
    validate_config=validate_config,
    after_optimizer_step=update_kl,
    after_train_result=warn_about_bad_reward_scales)


active_policy = None
trainable_policies = []
threshold = 0.7
trainer_updates = []


def my_train_fn(config, reporter):
    global trainable_policies, active_policy, trainer_updates
    # ppo_trainer = MyPPOTrainer(env='c4', config=config)
    ppo_trainer = PPOTrainer(env='c4', config=config)

    trainable_policies = ppo_trainer.workers.foreach_worker(lambda w: w.policies_to_train)[0][:]
    # trainable_policies = ppo_trainer.workers.foreach_worker(
    #     lambda w: w.foreach_trainable_policy(lambda p, i: (i, p))
    # )

    while True:
        result = ppo_trainer.train()
        reporter(**result)

        # timestep = result['timesteps_total']
        # print('\n')
        # print('$$$$$$$$$$$$$$$$$$$$$$$')
        # print('timestep: {:,}'.format(timestep))
        # print('trainable_policies: %s' % trainable_policies)
        # if active_policy is None and timestep > int(5e6):
        # # if active_policy is None and timestep > int(25e4):
        #     active_policy = trainable_policies[0]
        #     # ppo_trainer.workers.foreach_worker(
        #     #     lambda w: w.foreach_trainable_policy(lambda p, i: (i, p))
        #     # )
        #     ppo_trainer.workers.foreach_worker(
        #         lambda w: w.policies_to_train.remove(trainable_policies[1])
        #     )
        #     trainer_updates.append(timestep)
        # elif active_policy == trainable_policies[0] \
        #         and result['policy_reward_mean'][trainable_policies[0]] > threshold:
        #     active_policy = trainable_policies[1]
        #     ppo_trainer.workers.foreach_worker(
        #         lambda w: w.policies_to_train.remove(trainable_policies[0])
        #     )
        #     ppo_trainer.workers.foreach_worker(
        #         lambda w: w.policies_to_train.append(trainable_policies[1])
        #     )
        #     trainer_updates.append(timestep)
        # elif active_policy == trainable_policies[1] \
        #         and result['policy_reward_mean'][trainable_policies[1]] > threshold:
        #     active_policy = trainable_policies[0]
        #     ppo_trainer.workers.foreach_worker(
        #         lambda w: w.policies_to_train.remove(trainable_policies[1])
        #     )
        #     ppo_trainer.workers.foreach_worker(
        #         lambda w: w.policies_to_train.append(trainable_policies[0])
        #     )
        #     trainer_updates.append(timestep)
        #
        # print('active_policy: %s' % active_policy)
        # print('worker TPs: %s' % ppo_trainer.workers.foreach_worker(lambda w: w.policies_to_train)[0])
        # print('trainer updates: %s' % '\n  - ' + '\n  - '.join('{:,}'.format(tu) for tu in trainer_updates))
        # print('$$$$$$$$$$$$$$$$$$$$$$$')
        # print('\n')

        # if result["episode_reward_mean"] > 200:
        #     phase = 2
        # elif result["episode_reward_mean"] > 100:
        #     phase = 1
        # else:
        #     phase = 0
        # ppo_trainer.workers.foreach_worker(
        #     lambda ev: ev.foreach_env(
        #         lambda env: env.set_phase(phase)))


class MyTrainable(Trainable):
    def _train(self):
        ppo_trainer = PPOTrainer(env='c4', config=self.config)
        while True:
            result = ppo_trainer.train()
            reporter(**result)


class MyTrainer(Trainer):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="PG")
    parser.add_argument("--use-cnn", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    ray.init(local_mode=args.debug)
    tune_config = {}

    if args.use_cnn:
        env_cls = SquareConnect4Env
        ModelCatalog.register_custom_model('parametric_actions_model', ParametricActionsCNN)
        model_config = {
            'custom_model': 'parametric_actions_model',
            'conv_filters': [[16, [2, 2], 1], [32, [2, 2], 1], [64, [3, 3], 2]],
            'conv_activation': 'leaky_relu',
            'fcnet_hiddens': [256, 256],
            'fcnet_activation': 'leaky_relu',
        }
    else:
        env_cls = Connect4Env
        ModelCatalog.register_custom_model('parametric_actions_model', ParametricActionsMLP)
        model_config = {
            'custom_model': 'parametric_actions_model',
            'fcnet_hiddens': [256, 256],
            'fcnet_activation': 'leaky_relu',
        }

    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space = env.observation_space
    action_space = env.action_space

    if args.debug:
        tune_config.update({
            'log_level': 'DEBUG',
            'num_workers': 1,
        })
    else:
        tune_config.update({
            'num_workers': 20,
            'num_gpus': 1,
            'train_batch_size': 65536,
            'sgd_minibatch_size': 4096,
            'num_sgd_iter': 6,
            'num_envs_per_worker': 32,
        })

    player1, player2 = None, None
    policies = {
        'learned1': (None, obs_space, action_space, {'model': model_config}),
        'learned2': (None, obs_space, action_space, {'model': model_config}),
        # 'learned3': (None, obs_space, action_space, {'model': model_config}),
        # 'learned4': (None, obs_space, action_space, {'model': model_config}),
        # 'learned5': (None, obs_space, action_space, {'model': model_config}),
        # 'learned6': (None, obs_space, action_space, {'model': model_config}),
        # 'learned7': (None, obs_space, action_space, {'model': model_config}),
        # 'learned8': (None, obs_space, action_space, {'model': model_config}),
    }

    def policy_mapping_fn(agent_id):
        global player1, player2
        if agent_id == 0:
            player1, player2 = random.sample([*policies], k=2)
            return player1
        else:
            return player2


    resources = PPOTrainer.default_resource_request(tune_config).to_json()
    tune.run(
        # MyPPOTrainer,
        my_train_fn,
        # MyTrainable,
        # MyTrainer,
        name='competitive_trg',
        stop={
            # 'timesteps_total': int(50e3),
            'timesteps_total': int(100e6),
            # 'timesteps_total': int(1e9),
        },
        config=dict({
            'env': 'c4',
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
        }, **tune_config),
        resources_per_trial=resources,
        # checkpoint_freq=100,
        checkpoint_at_end=True,
        # resume=True
    )
