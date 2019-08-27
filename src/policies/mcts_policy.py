import logging
import random
import time

import numpy as np
from ray.rllib.policy.policy import Policy

from src.envs import Connect4

logger = logging.getLogger('ray.rllib')


class MCTSPolicy(Policy):
    """Monte-Carlo Tree Search."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        mcts_config = config['multiagent']['policies']['mcts']
        obs_space = mcts_config[1]
        self.board_shape = obs_space['board'].shape
        self.board_size = np.prod(obs_space['board'].shape)
        self.player = mcts_config[3]['player_id']
        self.max_rollouts = mcts_config[3]['max_rollouts']
        self.rollouts_timeout = mcts_config[3]['rollouts_timeout']
        self.metrics = {'num_rollouts': []}

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
        board_start = self.action_space.n
        board_end = self.action_space.n + self.board_size
        for obs in obs_batch:
            action_mask, board, player = obs[:board_start], obs[board_start:board_end], obs[board_end:].item()
            if player == self.player:
                board = np.flip(board.reshape(self.board_shape), axis=0).astype(np.uint8)
                if self.board_shape == (7, 7):  # if square obs, cut off the filler `3`s at the top
                    board = board[1:]
                game = Connect4(game_state={'board': board, 'player': 1})
                action, metrics = mcts(game, self.max_rollouts, self.rollouts_timeout)
                actions.append(action)
                self.metrics['num_rollouts'].append(metrics['num_rollouts'])
            else:
                actions.append(0)  # dummy action as not our turn

        return np.array(actions), state_batches, {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


def mcts(current_state: Connect4, max_rollouts=10000, rollouts_timeout=100):
    root = Node(state=current_state)
    start = time.clock()
    for i in range(max_rollouts):
        node = root
        state = current_state.clone()

        # selection
        # keep going down the tree based on best UCT values until terminal or unexpanded node
        while len(node.untried_moves) == 0 and len(node.children):
            node = node.selection()
            state.move(node.move)

        # expand
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            state.move(move)
            node = node.expand(move, state)

        # rollout
        while state.get_moves():
            state.move(random.choice(state.get_moves()))

        # backpropagate
        while node is not None:
            node.update(state.get_reward(node.player))
            node = node.parent

        duration = time.clock() - start
        if duration > rollouts_timeout:
            break

    def score(x):
        return x.wins / x.visits

    sorted_children = sorted(root.children, key=score)[::-1]
    metrics = {'num_rollouts': i}

    return sorted_children[0].move, metrics


class Node:
    def __init__(self, move=None, parent=None, state=None):
        self.state = state.clone()
        self.parent = parent
        self.move = move
        self.untried_moves = state.get_moves()
        self.children = []
        self.wins = 0
        self.visits = 0
        self.player = state.player

    def selection(self):
        # return child with largest UCT value
        def uct(x):
            return x.wins / x.visits + np.sqrt(2 * np.log(self.visits) / x.visits)

        return sorted(self.children, key=uct)[-1]

    def expand(self, move, state):
        # return child when move is taken
        # remove move from current node
        child = Node(move=move, parent=self, state=state)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        self.wins += result
        self.visits += 1
