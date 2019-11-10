import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class SimpleAssumptionMixin:
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
            # reshaped_obs = obs.reshape((6, 7))
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
