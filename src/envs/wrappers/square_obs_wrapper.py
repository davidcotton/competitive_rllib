import gym
import numpy as np


class SquareObsWrapper(gym.ObservationWrapper):
    """Flatten an observation into a vector."""
    def __init__(self, env):
        super().__init__(env)
        self.new_shape = (7, 7)
        self.observation_space.shape = self.new_shape
        self.new_max_obs = np.max(self.env.observation_space.high) + 1

    def observation(self, obs):
        for k, v in obs:
            if isinstance(v, np.array):
                v = np.append(v, np.full((1, 7), self.new_max_obs), axis=0)
        # sq_obs1 = np.append(obs[0], np.full((1, 7), self.new_max_obs), axis=0)
        # sq_obs2 = np.append(obs[1], np.full((1, 7), self.new_max_obs), axis=0)
        # return sq_obs1, sq_obs2
        return obs
