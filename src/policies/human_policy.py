import numpy as np
from ray.rllib.policy.policy import Policy


class HumanPolicy(Policy):
    """Allow a human player to play against AI."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.board_shape = observation_space.original_space['board'].shape
        self.board_len = np.prod(self.board_shape).item()
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

        assert len(obs_batch) == 1  # too hard for human to play parallel games
        obs = obs_batch[0]
        board_start = self.action_space.n
        board_end = self.action_space.n + self.board_len
        action_mask, board = obs[:board_start], obs[board_start:board_end]
        current_player, player_id = int(obs[board_end:board_end + 1]), int(obs[board_end + 1:])
        board = board.reshape(self.board_shape).astype(np.uint8)
        self._render_board(board)

        valid_actions = [i for i in range(self.num_actions) if action_mask[i]]
        if current_player == player_id:
            action = None
            while action not in list(valid_actions):
                try:
                    action = int(input('What is your action?')) - 1
                except ValueError:
                    pass
        else:
            action = self.num_actions  # "pass" action

        return np.array([action]), state_batches, {}

    def _render_board(self, obs):
        print('  1 2 3 4 5 6 7')
        print(' ---------------')
        print(obs)
        print(' ---------------')
        print('  1 2 3 4 5 6 7')
        print()

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
