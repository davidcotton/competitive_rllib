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
                action_overrides.append({})  # skip other player's turn
                continue

            action_mask = flattened_obs[:7]  # ignore "pass" action (8)
            obs = flattened_obs[8:50]
            reshaped_obs = obs.reshape((6, 7))
            action_override = self._compute_assisted_actions(obs, action_mask, reshaped_obs)
            action_overrides.append(action_override)

        # action_overrides = []
        # for flattened_obs in obs_batches:
        #     if flattened_obs[7] == 1:
        #         action_overrides.append({})
        #         continue  # skip other player's turn
        #     action_mask = flattened_obs[:8]
        #     obs = flattened_obs[8:50]
        #     action_override = {}
        #     for action, action_available in enumerate(action_mask[:-1]):
        #         if action_available == 0:
        #             continue  # skip masked actions
        #         value, attempts = self.get_value(obs, action)
        #         if attempts < 0 and value < -0.99:
        #             flattened_obs[action] = 0  # mask the action
        #             print('assist masked action: %s, value: %s, attempts: %s' % (action, value, attempts))
        #         if value > 0.999:
        #             reshaped_obs = obs.reshape((6, 7))
        #             action_override.update({
        #                 'action': action,
        #                 'value': value,
        #                 'attempts': attempts,
        #                 'obs': reshaped_obs
        #             })
        #     action_overrides.append(action_override)

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
                # print('replaced action: %s with %s' % (actions[i], action_override['action']))
                # print(action_override['reshaped_obs'])
                # print('value:', action_override['value'], 'attempts:', action_override['attempts'])
                actions[i] = action_override['action']
            actions_info['action_overridden'].append(int(is_overridden))
        actions_info['action_overridden'] = np.array(actions_info['action_overridden'])

        return actions, state_outs, actions_info

    def _compute_assisted_actions(self, obs, action_mask, reshaped_obs, limit=0):
        action_override = {}
        for action, action_available in enumerate(action_mask):
            if action_available == 0:
                continue  # skip masked actions

            value, attempts = self.get_value(obs, action)
            if attempts < 0 and value < -0.99:
                # flattened_obs[action] = 0  # mask the action
                action_mask[action] = 0  # mask the action
                # print('assist masked action: %s, value: %s, attempts: %s' % (action, value, attempts))
            elif value > 0.99:
                action_override.update({
                    'action': action,
                    'value': value,
                    'attempts': attempts,
                    'reshaped_obs': reshaped_obs
                })

        return action_override

    def update_assist(self, sample_batch, other_agent_batches, episode):
        if episode is None:
            return  # RLlib startup, skip

        step = 0
        reward = sample_batch[SampleBatch.REWARDS][-1]
        for i in range(sample_batch.count - 1, -1, -1):
            obs = sample_batch[SampleBatch.CUR_OBS][i]
            if obs[7] == 1:
                continue  # skip "pass action" (other player's turn)

            obs = obs[8:50]
            action = sample_batch[SampleBatch.ACTIONS][i]
            reshaped_obs = obs.reshape((6, 7))
            self._update_step(step, obs, action, reward, reshaped_obs)

            reward = 0.0
            step += 1

    def _update_step(self, step, obs, action, reward, reshaped_obs):
        value, attempts = self.get_value(obs, action)
        best_value = reward
        if step == 0:
            if attempts > 0:  # if this branch was previous considered non-terminal
                attempts = 0
            new_value = best_value - attempts * value
            attempts -= 1
            new_value /= float(-attempts)
        else:
            if attempts < 0:  # terminal, don't update branches closer to root
                new_value = value
            else:
                attempts += 1
                new_value = 0
        self.memory[self.get_key(obs, action)] = new_value, attempts

        return reward

    def get_key(self, obs, action):
        return tuple(obs), action

    def get_value(self, obs, action):
        key = self.get_key(obs, action)
        if key not in self.memory:
            return 0, 0
        value = self.memory[key]
        return value


class WindowedAssumptionMixin(SimpleAssumptionMixin):
    def __init__(self) -> None:
        super().__init__()
        self.obs_height = 6
        self.obs_width = 7
        self.window_size = 4

    def _compute_assisted_actions(self, obs, action_mask, reshaped_obs, limit=0):
        action_override = super()._compute_assisted_actions(obs, action_mask, reshaped_obs, limit)
        if not action_override:
            for py in range(self.obs_height - self.window_size + 1):
                for px in range(self.obs_width - self.window_size + 1):
                    window_obs = self._obs_sub_window(obs, px, py)
                    window_actions = [action_mask[i + px] for i in range(self.window_size)]
                    reshaped_window_obs = np.array(window_obs).reshape((self.window_size, self.window_size))
                    if any(window_actions):
                        action_override = super()._compute_assisted_actions(window_obs, window_actions, reshaped_window_obs, 5)
                        if action_override:
                            print('\n##################')
                            print('overriding windowed actions')
                            from pprint import pprint
                            pprint(action_override)
                            print('##################\n')
                            # best_action_id_ = best_action_id_ + px
                            # return best_action_id_, best_q_t1_val_, filtered_actions
                            return action_override

        return action_override

    def _update_step(self, step, obs, action, reward, reshaped_obs):
        reward = super()._update_step(step, obs, action, reward, reshaped_obs)
        for py in range(self.obs_height - self.window_size + 1):
            for px in range(self.obs_width - self.window_size + 1):
                window_action = action - px
                if 0 <= window_action < self.window_size:
                    window_obs = self._obs_sub_window(obs, px, py)
                    reshaped_window_obs = np.array(window_obs).reshape((self.window_size, self.window_size))
                    super()._update_step(step, window_obs, window_action, reward, reshaped_obs)
        return reward

    def _obs_sub_window(self, obs, px, py):
        window = [0] * (self.window_size ** 2)
        for y in range(self.window_size):
            for x in range(self.window_size):
                k = y * self.window_size + x
                if px + x >= self.obs_width or py + y >= self.obs_height:
                    window[k] = 3
                else:
                    v = (px + x) + (py + y) * self.obs_width
                    window[k] = obs[v]
        return window
