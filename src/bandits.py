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
        self.awaiting = {}
        self.updated = 0

    def select(self, episode_id) -> tuple:
        if episode_id not in self.actions:
            probs = self._make_distribution(self.weights, self.gamma)
            arm = self._categorical_draw(probs)
            p1, p2 = self.lookup[arm]
            self.actions[episode_id] = arm, (p1, p2)
        return self.actions[episode_id][1]

    def update(self, episode_id, reward) -> None:
        self.awaiting[episode_id] = reward
        to_update = [ep_id for ep_id in self.awaiting.keys() if ep_id in self.actions]
        for episode_id in to_update:
            reward = self.awaiting.pop(episode_id)
            self._update(episode_id, reward)
            self.updated += 1

    def _update(self, episode_id, reward) -> None:
        assert 0 <= reward <= 1
        arm, _ = self.actions.pop(episode_id)
        probs = self._make_distribution(self.weights, self.gamma)
        x = reward / probs[arm]
        # self.weights[arm] *= math.exp((self.gamma / self.num_arms) * x)
        self.weights[arm] *= math.exp(x * self.gamma / self.num_arms)

    def check_queue(self, info):
        self.actions.clear()  # clear unfinished episodes

    @staticmethod
    def _make_distribution(weights, gamma):
        sum_weights = sum(weights)
        return [(1 - gamma) * (w / sum_weights) + (gamma / len(weights)) for w in weights]

    @staticmethod
    def _categorical_draw(probs):
        # z = random.random()
        z = random.uniform(0, sum(probs))
        cumulative_probs = 0.0
        for i in range(len(probs)):
            cumulative_probs += probs[i]
            if cumulative_probs > z:
                return i
        return len(probs) - 1


