import random

import numpy as np
from ray.rllib.policy.policy import Policy


class RandomPolicy(Policy):
    """Pick actions uniformly random from available actions each turn."""

    def __init__(self, observation_space, action_space, config) -> None:
        super().__init__(observation_space, action_space, config)
        self.board_start = np.prod(observation_space.original_space['action_mask'].shape).item()
        self.board_len = np.prod(observation_space.original_space['board'].shape).item()
        self.num_actions = action_space.n - 1  # last action is the "pass" action

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """Compute actions for the current policy.

        Arguments:
            obs_batch (np.ndarray): batch of observations
            state_batches (list): list of RNN state input batches, if any
            prev_action_batch (np.ndarray): batch of previous action values
            prev_reward_batch (np.ndarray): batch of previous rewards
            info_batch (info): batch of info objects
            episodes (list): MultiAgentEpisode for each obs in obs_batch.
                This provides access to all of the internal episode state,
                which may be useful for model-based or multiagent algorithms.
            kwargs: forward compatibility placeholder

        Returns:
            actions (np.ndarray): batch of output actions, with shape like [BATCH_SIZE, ACTION_SHAPE].
            state_outs (list): list of RNN state output batches, if any, with shape like [STATE_SIZE, BATCH_SIZE].
            info (dict): dictionary of extra feature batches, if any, with shape like
                {"f1": [BATCH_SIZE, ...], "f2": [BATCH_SIZE, ...]}.
        """

        actions = []
        board_end = self.board_start + self.board_len
        for obs in obs_batch:
            action_mask, board = obs[:self.board_start], obs[self.board_start:board_end]
            current_player, player_id = obs[board_end:board_end + 1].item(), obs[board_end + 1:].item()
            if current_player == player_id:
                legal_actions = [i for i in range(self.num_actions) if action_mask[i]]
                actions.append(random.choice(legal_actions))
            else:
                actions.append(self.num_actions)  # "pass" action

        return np.array(actions), state_batches, {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
