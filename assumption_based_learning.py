import argparse
import random

import numpy as np
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from ray.rllib.agents.ppo.ppo_policy import ppo_surrogate_loss, kl_and_loss_stats, vf_preds_and_logits_fetches, \
    postprocess_ppo_gae, clip_gradients, KLCoeffMixin, ValueNetworkMixin, LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf

from src.policies import HumanPolicy, RandomPolicy
from src.utils import get_worker_config, get_learner_policy_configs, get_mcts_policy_configs, get_model_config

tf = try_import_tf()


class AssistMixin:
    def __init__(self) -> None:
        self.memory = {}

    def compute_actions(self,
                        obs_batches,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """Select actions to take from a batch of observations.

        :param obs_batches: A batch of observations.
        :param state_batches: A list of RNN state, if any.
        :param prev_action_batch: Previous action values, if any.
        :param prev_reward_batch: Previous reward, if any.
        :param info_batch: Info object, if any.
        :param episodes: The episode state.
        :param kwargs:
        :return: A tuple containing:
            a list of actions to take,
            a list of RNN state and,
            a dict of info.
        """

        # 1) compute assisted actions
        action_overrides = []
        for flattened_obs in obs_batches:
            if flattened_obs[7] == 1:
                action_overrides.append({})
                continue  # skip other player's turn
            action_mask = flattened_obs[:8]
            obs = flattened_obs[8:50]
            action_override = {}
            for action, action_available in enumerate(action_mask[:-1]):
                if action_available == 0:
                    continue  # skip masked actions
                value, attempts = self.get_value(obs, action)
                if attempts < 0 and value < -0.99:
                    flattened_obs[action] = 0  # mask the action
                    print('assist masked action: %s, value: %s, attempts: %s' % (action, value, attempts))
                if value > 0.999:
                    reshaped_obs = obs.reshape((6, 7))
                    action_override.update({
                        'action': action,
                        'value': value,
                        'attempts': attempts,
                        'obs': reshaped_obs
                    })
                    # action_override = action
                    # print(reshaped_obs)
                    # print('best_value', value, 'action_override', action, 'best_attempts', attempts)
            action_overrides.append(action_override)

        # 2) compute neural network actions
        fetches = super().compute_actions(obs_batches, state_batches, prev_action_batch, prev_reward_batch, info_batch,
                                          episodes, **kwargs)
        actions, state_outs, actions_info = fetches

        # 3) merge/compare assisted actions
        actions_info['action_overridden'] = []
        for i, action_override in enumerate(action_overrides):
            # is_overridden = action_override is not None
            is_overridden = 'action' in action_override
            if is_overridden:
                print('replaced action: %s with %s' % (actions[i], action_override['action']))
                print(action_override['obs'])
                print('value:', action_override['value'], 'attempts:', action_override['attempts'])
                actions[i] = action_override['action']
            actions_info['action_overridden'].append(int(is_overridden))
        actions_info['action_overridden'] = np.array(actions_info['action_overridden'])

        return actions, state_outs, actions_info

    def update_assist(self, sample_batch, other_agent_batches, episode):
        if episode is None:  # RLlib startup, skip
            return
        steps = 0
        reward = sample_batch[SampleBatch.REWARDS][-1]
        for i in range(sample_batch.count - 1, -1, -1):
            obs = sample_batch[SampleBatch.CUR_OBS][i]
            if obs[7] == 1:
                continue  # skip "pass action" (other player's turn)
            obs = obs[8:50]
            reshaped_obs = obs.reshape((6, 7))
            action = sample_batch[SampleBatch.ACTIONS][i]
            # reward = sample_batch[SampleBatch.REWARDS][i]
            value, attempts = self.get_value(obs, action)
            best_value = reward

            if steps == 0:
                if attempts > 0:
                    attempts = 0
                new_value = best_value - attempts * value
                attempts -= 1
                new_value /= float(-attempts)
            else:
                if attempts < 0:
                    new_value = value
                else:
                    attempts += 1
                    new_value = 0

            self.memory[self.get_key(obs, action)] = new_value, attempts
            reward = 0.0
            steps += 1

    def get_key(self, obs, action):
        return tuple(obs), action

    def get_value(self, obs, action):
        key = self.get_key(obs, action)
        if key not in self.memory:
            return 0, 0
        value = self.memory[key]
        return value


def postprocess_fn(policy, sample_batch, other_agent_batches=None, episode=None):
    policy.update_assist(sample_batch, other_agent_batches, episode)
    return postprocess_ppo_gae(policy, sample_batch, other_agent_batches, episode)


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    AssistMixin.__init__(policy)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


AssumptionPPOTFPolicy = build_tf_policy(
    name="AssumptionPPOTFPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=vf_preds_and_logits_fetches,
    postprocess_fn=postprocess_fn,
    gradients_fn=clip_gradients,
    before_init=ray.rllib.agents.ppo.ppo_policy.setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin,
        AssistMixin
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--num-learners', type=int, default=2)
    parser.add_argument('--assist', action='store_true')
    parser.add_argument('--mcts', type=int, default=0)
    # e.g. --restore="/home/dave/ray_results/main/PPO_c4_0_2019-09-23_16-17-45z9x1oc9j/checkpoint_782/checkpoint-782"
    parser.add_argument('--restore', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--human', action='store_true')
    args = parser.parse_args()

    ray.init(local_mode=args.debug)
    tune_config = get_worker_config(args)

    model_config, env_cls = get_model_config(args.use_cnn)
    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space, action_space = env.observation_space, env.action_space

    num_learners = 1 if args.human or args.mcts > 0 else args.num_learners
    learner_policy = None
    if args.policy == 'PPO':
        tune_config.update({
            'lr': 0.001,
            'gamma': 0.995,
            'clip_param': 0.2,
            'lambda': 0.95,
            'kl_coeff': 1.0,
        })
        if args.assist:
            print('Using assumption-based learning policy: AssumptionPPOTFPolicy')
            learner_policy = AssumptionPPOTFPolicy
    elif args.policy == 'DQN':
        tune_config.update({
            'hiddens': [],
            'dueling': False,
        })
        if args.assist:
            raise ValueError('DQN assist not implemented yet')
    trainable_policies = {
        f'learned{i:02d}': (learner_policy, obs_space, action_space, {'model': model_config}) for i in
        range(num_learners)}
    mcts_policies = get_mcts_policy_configs([2, 8, 16, 32, 64, 128, 256, 512], obs_space, action_space)
    stop_criterion = ('policy_reward_mean/learned00', 0.6) if args.mcts > 0 else ('timesteps_total', int(100e6))

    def random_policy_mapping_fn(info):
        if args.human:
            return random.sample(['learned00', 'human'], k=2)
        elif args.mcts > 0:
            return random.sample(['learned00', f'mcts{args.mcts:03d}'], k=2)
        elif num_learners == 1:
            return ['learned00', 'learned00']
        else:
            return random.sample([*trainable_policies], k=2)

    def name_trial(trial):
        """Give trials a more readable name in terminal & Tensorboard."""
        assist = 'Assist' if args.assist else 'NoAssist'
        mcts = f'vsMCTS{args.mcts}' if args.mcts > 0 else ''
        debug = '-debug' if args.debug else ''
        return f'{num_learners}x{trial.trainable_name}{mcts}-{assist}{debug}'

    tune.run(
        args.policy,
        name='ABL',
        trial_name_creator=name_trial,
        stop={
            stop_criterion[0]: stop_criterion[1]
        },
        config=dict({
            'env': 'c4',
            'lr': 0.001,
            'gamma': 0.995,
            'multiagent': {
                'policies_to_train': [*trainable_policies],
                'policy_mapping_fn': random_policy_mapping_fn,
                'policies': {
                    **trainable_policies,
                    **mcts_policies,
                    'human': (HumanPolicy, obs_space, action_space, {}),
                    'random': (RandomPolicy, obs_space, action_space, {}),
                },
            },
        }, **tune_config),
        # checkpoint_freq=100,
        checkpoint_at_end=True,
        # resume=True,
        restore=args.restore,
    )
