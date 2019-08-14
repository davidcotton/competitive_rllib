import logging

from gym import spaces
import numpy as np
from ray.rllib.models.preprocessors import Preprocessor, get_preprocessor

logger = logging.getLogger(__name__)


class SquareObsPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        # print('obs_space; SquareObsPreprocessor: ', obs_space)
        new_shape = (1, 7, 7)
        return new_shape  # can vary depending on inputs

    def transform(self, observation):
        print('SquareObsPreprocessor-observation:', observation)
        for k, v in observation:
            if k == 'board':
                sq_obs = np.append(v, np.full((1, 7), self.new_max_obs), axis=0)
                expd_obs = np.expand_dims(sq_obs, axis=0)
                observation[k] = expd_obs
        return observation


class FlattenObsPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        print('FlattenObsPreprocessor._init_shape().obs_space ->', obs_space)
        assert isinstance(self._obs_space, spaces.Dict)
        # new_shape = (42,)
        # return new_shape  # can vary depending on inputs
        size = 0
        self.preprocessors = []
        for space in self._obs_space.spaces.values():
            logger.debug("Creating sub-preprocessor for {}".format(space))
            preprocessor = get_preprocessor(space)(space, self._options)
            self.preprocessors.append(preprocessor)
            size += preprocessor.size
        return size,

    def transform(self, observation):
        print('FlattenObsPreprocessor.transform().observation: ->', observation)
        # flat_obs = np.ravel(observation)
        # return flat_obs
        for k, v in observation.items():
            if k == 'board':
                observation[k] = np.ravel(v)
        return observation
