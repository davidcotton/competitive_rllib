from itertools import permutations
import math
import random

import numpy as np


class Exp3Bandit:
    def __init__(self, num_agents, gamma=None) -> None:
        super().__init__()
        if gamma is None:
            gamma = np.sqrt(np.log(num_agents) / num_agents)
        self.gamma = float(gamma)
        self.num_arms = num_agents
        self.lookup = {i: v for i, v in enumerate(list(permutations(range(num_agents), r=2)))}
        self.weights = [1.0] * len(self.lookup)
        self.actions = {}

    def select(self, episode_id) -> tuple:
        if episode_id not in self.actions:
            probs = self._make_distribution(self.weights, self.gamma)
            arm = self._categorical_draw(probs)
            p1, p2 = self.lookup[arm]
            self.actions[episode_id] = arm, (p1, p2)
        policy_ids = self.actions[episode_id][1]
        return policy_ids

    def update(self, episode_id, reward) -> None:
        assert 0 <= reward <= 1
        arm, _ = self.actions.pop(episode_id)
        probs = self._make_distribution(self.weights, self.gamma)
        x = reward / probs[arm]
        # self.weights[arm] *= math.exp((self.gamma / self.num_arms) * x)
        self.weights[arm] *= math.exp(x * self.gamma / self.num_arms)

    def _make_distribution(self, weights, gamma):
        sum_weights = sum(weights)
        return [(1 - gamma) * (w / sum_weights) + (gamma / len(weights)) for w in weights]

    def _categorical_draw(self, probs):
        # z = random.random()
        z = random.uniform(0, sum(probs))
        cumulative_probs = 0.0
        for i in range(len(probs)):
            cumulative_probs += probs[i]
            if cumulative_probs > z:
                return i
        return len(probs) - 1

    def get_weights(self):
        return self.weights
