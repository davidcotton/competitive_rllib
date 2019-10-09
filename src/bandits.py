from itertools import permutations
import math
import random

import numpy as np


class Exp3Bandit:
    def __init__(self, num_agents, gamma=None) -> None:
        super().__init__()
        # build a dict of all possible agent pairings (without replacement)
        # to map arm: (player1, player2)
        # note: a, b != b, a as player1 has a positional advantage over player2
        self.agent_pairs = {i: v for i, v in enumerate(list(permutations(range(num_agents), r=2)))}
        if gamma is None:
            gamma = np.sqrt(np.log(num_agents) / self.num_arms)
        self.gamma = float(gamma)
        self.weights = [1.0] * self.num_arms
        # dict of in-progress selections as episodes run in parallel
        self.actions = {}

    def select(self, episode_id) -> tuple:
        if episode_id not in self.actions:
            probs = self._make_distribution(self.weights)
            arm = self._draw(probs)
            p1, p2 = self.agent_pairs[arm]
            self.actions[episode_id] = arm, (p1, p2)
        policy_ids = self.actions[episode_id][1]
        return policy_ids

    def update(self, episode_id, reward) -> None:
        assert 0 <= reward <= 1
        reward -= np.mean(self.weights)  # stop the weights blowing up
        arm, _ = self.actions.pop(episode_id)
        probs = self._make_distribution(self.weights)
        estimated_reward = reward / probs[arm]
        self.weights[arm] *= math.exp(self.gamma * estimated_reward / self.num_arms)
        # self.weights[arm] += math.exp(self.gamma * estimated_reward / self.num_arms)

    def _make_distribution(self, weights):
        sum_weights = sum(weights)
        return [(1 - self.gamma) * (w / sum_weights) + (self.gamma / len(weights)) for w in weights]

    def _draw(self, weights):
        rand = random.random()
        cumulative_weights = 0.0
        for i, weight in enumerate(weights):
            cumulative_weights += weight
            if cumulative_weights > rand:
                return i
        return len(weights) - 1

    @property
    def num_arms(self):
        return len(self.agent_pairs)

    def get_weights(self):
        return self.weights

    def get_probs(self):
        probs = self._make_distribution(self.weights)
        assert 0.99 < sum(probs) < 1.01, print(f'sum_probs: {sum(probs)}, probs: {probs}, weights: {self.weights}')
        metrics = {f'bandit{a}': pr for a, pr in zip(self.agent_pairs.values(), probs)}
        metrics['sum_weights'] = sum(self.weights)
        return metrics
