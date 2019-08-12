import argparse

from gym import spaces
import numpy as np
import ray
from ray import tune
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models.tf.misc import get_activation_fn, flatten
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

from src.envs import Connect4Env
from src.policies import RandomPolicy

tf = try_import_tf()
POLICY = 'PG'


class SquareObsPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        # print('obs_space; SquareObsPreprocessor: ', obs_space)
        new_shape = (1, 7, 7)
        return new_shape  # can vary depending on inputs

    def transform(self, observation):
        print('SquareObsPreprocessor-observation:', observation)
        for k, v in observation:
            if isinstance(v, np.array):
                sq_obs = np.append(v, np.full((1, 7), self.new_max_obs), axis=0)
                expd_obs = np.expand_dims(sq_obs, axis=0)
                observation[k] = expd_obs
        return observation


class FlattenObsPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        new_shape = (42,)
        return new_shape  # can vary depending on inputs

    def transform(self, observation):
        print('FlattenObsPreprocessor-observation:', observation)
        # flat_obs = np.ravel(observation)
        # return flat_obs
        for k, v in observation:
            if k == 'board':
                observation[k] = np.ravel(v)
        return observation


class MyVisionNetwork(Model):
    """Generic vision network."""

    @override(Model)
    def _build_layers_v2(self, input_dict, num_outputs, options):
        inputs = input_dict["obs"]
        filters = options.get("conv_filters")
        # if not filters:
        #     filters = _get_filter_config(inputs.shape.as_list()[1:])

        activation = get_activation_fn(options.get("conv_activation"))

        with tf.name_scope("vision_net"):
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                inputs = tf.layers.conv2d(
                    inputs,
                    out_size,
                    kernel,
                    stride,
                    activation=activation,
                    padding="same",
                    name="conv{}".format(i))
            out_size, kernel, stride = filters[-1]

            # skip final linear layer
            if options.get("no_final_linear"):
                fc_out = tf.layers.conv2d(
                    inputs,
                    num_outputs,
                    kernel,
                    stride,
                    activation=activation,
                    padding="valid",
                    name="fc_out")
                return flatten(fc_out), flatten(fc_out)

            fc1 = tf.layers.conv2d(
                inputs,
                out_size,
                kernel,
                stride,
                activation=activation,
                padding="valid",
                name="fc1")
            fc2 = tf.layers.conv2d(
                fc1,
                num_outputs, [1, 1],
                activation=None,
                padding="same",
                name="fc2")
            return flatten(fc2), flatten(fc1)


class ParametricActionsModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        print('\n\n')
        print('_build_layers_v2().input_dict ->', input_dict)
        print('_build_layers_v2().num_outputs ->', num_outputs)
        print('_build_layers_v2().options ->', options)
        action_mask = input_dict['obs']['action_mask']
        print('_build_layers_v2().action_mask ->', action_mask)

        # Standard FC net component.
        last_layer = input_dict['obs']['board']
        # last_layer = input_dict['obs']
        print('_build_layers_v2().last_layer ->', last_layer, '\n\n')
        hiddens = [256, 128, 64, 32]
        for i, size in enumerate(hiddens):
            label = "fc{}".format(i)
            last_layer = tf.layers.dense(
                last_layer,
                size,
                # kernel_initializer=normc_initializer(1.0),
                activation=tf.nn.relu,
                name=label)
        output = tf.layers.dense(
            last_layer,
            num_outputs,
            # kernel_initializer=normc_initializer(0.01),
            activation=None,
            name="fc_out")

        intent_vector = tf.expand_dims(output, 1)

        action_logits = tf.reduce_sum(intent_vector, axis=2)

        if POLICY == 'DQN':  # For DQN set invalid actions to -10 as a loss
            masked_logits = tf.where(action_mask < 0.5, -tf.ones_like(output) * 10, output)
        else:
            masked_logits = tf.where(action_mask < 0.5, -tf.ones_like(output) * (2 ** 18), output)

        # print ('debug',masked_logits.shape, num_outputs,action_mask.shape,output.shape, '\n\n\n\n\n' )
        return masked_logits, last_layer
        # return action_logits, last_layer


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    use_lstm = False

    def select_policy(agent_id):
        if agent_id == 0:
            return "learned"
        else:
            # return random.choice(["always_same", "beat_last"])
            return 'random'

    # ray.init()

    ModelCatalog.register_custom_preprocessor("square_obs", SquareObsPreprocessor)
    ModelCatalog.register_custom_preprocessor("flatten_obs", FlattenObsPreprocessor)
    ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)

    # obs_space = spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.uint8)
    # obs_space = spaces.Box(low=0, high=2, shape=(6 * 7,), dtype=np.uint8)
    obs_space = spaces.Dict({
        # 'board': spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.uint8),
        'board': spaces.Box(low=0, high=2, shape=(6 * 7,), dtype=np.uint8),
        'action_mask': spaces.Box(low=0, high=1, shape=(7,), dtype=np.uint8),
    })
    action_space = spaces.Discrete(7)

    tune.run(
        # trainer,
        POLICY,
        stop={
            # 'timesteps_total': 400000
            # 'timesteps_total': 1000,
            'policy_reward_mean': {'learned': 0.95},
        },
        config={
            'env': Connect4Env,
            # "gamma": 0.9,
            # "num_workers": 4,
            # 'num_envs_per_worker': 4,
            'num_workers': 0,
            'num_envs_per_worker': 1,
            "num_gpus": 1,
            # 'sample_batch_size': 10,
            # 'train_batch_size': 200,
            'multiagent': {
                'policies_to_train': ['learned'],
                'policies': {
                    'random': (RandomPolicy, obs_space, action_space, {}),
                    'random2': (RandomPolicy, obs_space, action_space, {}),
                    'learned': (None, obs_space, action_space, {
                        'model': {
                            'custom_model': 'pa_model',
                            # 'use_lstm': use_lstm,
                            # 'dim': 7,
                            # # 'conv_filters': [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]],
                            # # 'conv_filters': [[16, [2, 2], 1], [42, [3, 3], 1], [512, [2, 2], 1]],
                            # "conv_filters": [[16, [2, 2], 1],],
                            # 'custom_preprocessor': 'square_obs',
                            # 'custom_preprocessor': 'flatten_obs',
                            # 'custom_options': {},  # extra options to pass to your preprocessor
                        }
                    }),
                },
                # 'policy_mapping_fn': tune.function(select_policy),
                'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'random'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda agent_id: ['random', 'random2'][agent_id % 2]),
            },
        })
