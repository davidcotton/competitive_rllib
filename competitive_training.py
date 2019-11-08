import argparse
import logging
import time
import random

import ray
from ray import tune
from ray.exceptions import RayError
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_TRAINER_CONFIG, choose_policy_optimizer, \
    validate_config, update_kl, warn_about_bad_reward_scales
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, KLCoeffMixin, ValueNetworkMixin, ppo_surrogate_loss, \
    kl_and_loss_stats, vf_preds_and_logits_fetches, postprocess_ppo_gae, clip_gradients, setup_config, setup_mixins
from ray.rllib.agents.trainer import Trainer, MAX_WORKER_FAILURE_RETRIES
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import FilterManager
from ray.rllib.utils.annotations import override, PublicAPI
from ray.tune.trainable import Trainable
from ray.tune.registry import register_env

from src.bandits import Exp3Bandit
from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy
from src.utils import get_worker_config, get_learner_policy_configs, get_model_config

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


class MyTrainable(Trainable):
    def _train(self):
        ppo_trainer = PPOTrainer(env='c4', config=self.config)
        while True:
            result = ppo_trainer.train()
            # reporter(**result)
            print('ran iteration')


class MyTrainer(Trainer):
    @override(Trainable)
    @PublicAPI
    def train(self):
        """Overrides super.train to synchronize global vars."""

        if self._has_policy_optimizer():
            self.global_vars['timestep'] = self.optimizer.num_steps_sampled
            self.optimizer.workers.local_worker().set_global_vars(
                self.global_vars)
            for w in self.optimizer.workers.remote_workers():
                w.set_global_vars.remote(self.global_vars)
            logger.debug('updated global vars: {}'.format(self.global_vars))

        result = None
        for _ in range(1 + MAX_WORKER_FAILURE_RETRIES):
            try:
                result = Trainable.train(self)
            except RayError as e:
                if self.config['ignore_worker_failures']:
                    logger.exception(
                        'Error in train call, attempting to recover')
                    self._try_recover()
                else:
                    logger.info(
                        'Worker crashed during call to train(). To attempt to '
                        'continue training without the failed worker, set '
                        '`\'ignore_worker_failures\': True`.')
                    raise e
            except Exception as e:
                time.sleep(0.5)  # allow logs messages to propagate
                raise e
            else:
                break
        if result is None:
            raise RuntimeError('Failed to recover from worker crash')

        if (self.config.get('observation_filter', 'NoFilter') != 'NoFilter'
                and hasattr(self, 'workers')
                and isinstance(self.workers, WorkerSet)):
            FilterManager.synchronize(
                self.workers.local_worker().filters,
                self.workers.remote_workers(),
                update_remote=self.config['synchronize_filters'])
            logger.debug('synchronized filters: {}'.format(
                self.workers.local_worker().filters))

        if self._has_policy_optimizer():
            result['num_healthy_workers'] = len(
                self.optimizer.workers.remote_workers())

        if self.config['evaluation_interval']:
            if self._iteration % self.config['evaluation_interval'] == 0:
                evaluation_metrics = self._evaluate()
                assert isinstance(evaluation_metrics, dict), \
                    '_evaluate() needs to return a dict.'
                result.update(evaluation_metrics)

        return result

    # def _train(self):
    #     foo = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--num-learners', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    ray.init(local_mode=args.debug)
    tune_config = get_worker_config(args)

    model_config, env_cls = get_model_config(args.use_cnn)
    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space, action_space = env.observation_space, env.action_space
    trainable_policies = get_learner_policy_configs(args.num_learners, obs_space, action_space, model_config)

    def random_policy_mapping_fn(episode_id):
        return random.sample([*trainable_policies], k=2)

    def my_train_fn(config, reporter):
        active_policy = None
        threshold = 0.7
        trainer_updates = []

        # ppo_trainer = MyPPOTrainer(env='c4', config=config)
        ppo_trainer = PPOTrainer(env='c4', config=config)
        bandit = Exp3Bandit(len(trainable_policies))

        def func(worker):
            worker.sampler.policy_mapping_fn = learned_vs_random_mapping_fn
            foo = 1

        # ppo_trainer.workers.foreach_worker(lambda w: w.sampler.policy_mapping_fn)
        ppo_trainer.workers.foreach_worker(func)

        # trainable_policies = ppo_trainer.workers.foreach_worker(lambda w: w.policies_to_train)[0][:]
        # trainable_policies = ppo_trainer.workers.foreach_worker(
        #     lambda w: w.foreach_trainable_policy(lambda p, i: (i, p))
        # )

        while True:
            result = ppo_trainer.train()
            reporter(**result)

            foo = 1

            timestep = result['timesteps_total']
            training_iteration = result['training_iteration']
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

            # if timestep > int(1e6):
            # if training_iteration >= 20:
            #     break

            # if result['episode_reward_mean'] > 200:
            #     phase = 2
            # elif result['episode_reward_mean'] > 100:
            #     phase = 1
            # else:
            #     phase = 0
            # ppo_trainer.workers.foreach_worker(
            #     lambda ev: ev.foreach_env(
            #         lambda env: env.set_phase(phase)))

        state = ppo_trainer.save()
        ppo_trainer.stop()

    # resources = PPOTrainer.default_resource_request(tune_config).to_json()
    tune.run(
        # MyPPOTrainer,
        my_train_fn,
        # MyTrainable,
        # MyTrainer,
        name='competitive_trg',
        stop={
            'training_iteration': 2,
            # 'timesteps_total': int(1e6),
            # 'timesteps_total': int(100e6),
            # 'timesteps_total': int(1e9),
        },
        config=dict({
            'env': 'c4',
            'lr': 0.001,
            'gamma': 0.995,
            'lambda': 0.95,
            'clip_param': 0.2,
            'multiagent': {
                'policies_to_train': [*trainable_policies],
                'policy_mapping_fn': random_policy_mapping_fn,
                'policies': {
                    **trainable_policies,
                    'mcts': (MCTSPolicy, obs_space, action_space, {}),
                    'human': (HumanPolicy, obs_space, action_space, {}),
                    'random': (RandomPolicy, obs_space, action_space, {}),
                },
            },
            'callbacks': {
                # 'on_episode_start': on_episode_start,
            },
        }, **tune_config),
        # resources_per_trial=resources,
        # checkpoint_freq=100,
        # checkpoint_at_end=True,
        # resume=True
    )
