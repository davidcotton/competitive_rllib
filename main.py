import logging

from gym import spaces
import numpy as np
import ray
from ray import tune
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.models.tf.misc import get_activation_fn, flatten, normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet_v1 import VisionNetwork, _get_filter_config
from ray.rllib.utils import try_import_tf

from src.envs import Connect4Env, FlattenedConnect4Env, SquareConnect4Env
from src.models.preprocessors import SquareObsPreprocessor, FlattenObsPreprocessor
from src.policies import RandomPolicy

logger = logging.getLogger('ray.rllib')
tf = try_import_tf()
POLICY = 'PG'


class MyVisionNetwork(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs", "prev_action", "prev_reward",
                "is_training".
            num_outputs (int): Output tensor must be of size [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs] and [BATCH_SIZE, desired_feature_size].
        """

        inputs = input_dict['obs']['board']
        inputs = tf.reshape(inputs, (-1, 1, *inputs.shape))  # add channels dim
        filters = options.get('conv_filters')
        if not filters:
            filters = _get_filter_config(inputs.shape.as_list()[1:])
        activation = get_activation_fn(options.get('conv_activation'))

        logger.debug('\n\n#######################')
        logger.debug('input_dict:%s' % input_dict)
        logger.debug('inputs:%s' % inputs)
        logger.debug('filters:%s' % filters)
        logger.debug('\n#######################\n\n')

        with tf.name_scope('vision_net'):
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                logger.debug('i:%s, out_size:%s, kernel:%s, stride:%s' % (i, out_size, kernel, stride))
                inputs = tf.layers.conv2d(
                    inputs,
                    out_size,
                    kernel,
                    stride,
                    activation=activation,
                    padding='same',
                    name='conv{}'.format(i))
            out_size, kernel, stride = filters[-1]

            fc1 = tf.layers.conv2d(inputs, out_size, kernel, stride, activation=activation, padding='valid', name='fc1')
            fc2 = tf.layers.conv2d(fc1, num_outputs, [1, 1], activation=None, padding='same', name='fc2')
            fc1 = flatten(fc1)
            fc2 = flatten(fc2)

            logger.debug('\n\n#######################')
            logger.debug('fc1:%s' % fc1)
            logger.debug('fc2:%s' % fc2)
            logger.debug('\n#######################\n\n')

            action_mask = input_dict['obs']['action_mask']
            if POLICY == 'DQN':  # For DQN set invalid actions to -10 as a loss
                # masked_logits = tf.where(action_mask < 0.5, -tf.ones_like(output) * 10, output)
                masked_logits = tf.where(action_mask < 0.5, tf.ones_like(output) * tf.float32.min, output)
            else:
                masked_logits = tf.where(action_mask < 0.5, -tf.ones_like(output) * (2 ** 18), output)

            # return flatten(fc2), flatten(fc1)
            return masked_logits, inputs


class JasonMLPModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs", "prev_action", "prev_reward",
                "is_training".
            num_outputs (int): Output tensor must be of size [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs] and [BATCH_SIZE, desired_feature_size].
        """
        inputs = input_dict['obs']['board']
        inputs = tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))  # flatten
        action_mask = input_dict['obs']['action_mask']
        hiddens = [256, 128, 64, 32]
        for i, size in enumerate(hiddens):
            label = 'fc{}'.format(i)
            inputs = tf.layers.dense(
                inputs,
                size,
                # kernel_initializer=normc_initializer(1.0),
                activation=tf.nn.relu,
                name=label)
        output = tf.layers.dense(
            inputs,
            num_outputs,
            # kernel_initializer=normc_initializer(0.01),
            activation=None,
            name='fc_out')
        logger.debug('output:%s' % output)

        if POLICY == 'DQN':
            masked_logits = tf.where(action_mask < 0.5, tf.ones_like(output) * tf.float32.min, output)
        else:
            masked_logits = tf.where(action_mask < 0.5, -tf.ones_like(output) * (2 ** 18), output)

        return masked_logits, inputs


class MaskedCNNModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MaskedCNNModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        logger.debug('\n\n######## MyKerasModel.init() #########')
        logger.debug('obs_space:%s' % obs_space)
        logger.debug('action_space:%s' % action_space)
        logger.debug('model_config:%s' % model_config)
        logger.debug('########################\n')

        hiddens = model_config['fcnet_hiddens']
        activation = get_activation_fn(model_config['fcnet_activation'])
        self.inputs = tf.keras.layers.Input(shape=(7 * 7,), name="observations")
        last_layer = self.inputs
        for i, size in enumerate(hiddens, 1):
            last_layer = tf.keras.layers.Dense(
                size,
                name='fc{}'.format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(last_layer)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(last_layer)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = flatten(input_dict['obs']['board'])
        logger.debug('>>>>>>>> obs:%s' % obs)
        model_out, self._value_out = self.base_model(obs)
        action_mask = input_dict['obs']['action_mask']
        if POLICY == 'DQN':
            masked_logits = tf.where(action_mask < 0.5, tf.ones_like(model_out) * tf.float32.min, model_out)
        else:
            masked_logits = tf.where(action_mask < 0.5, -tf.ones_like(model_out) * (2 ** 18), model_out)

        return masked_logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


if __name__ == '__main__':
    def select_policy(agent_id):
        if agent_id == 0:
            return 'learned'
        else:
            # return random.choice(['always_same', 'beat_last'])
            return 'random'

    ray.init()

    # ModelCatalog.register_custom_preprocessor('square_obs_preprocessor', SquareObsPreprocessor)
    # ModelCatalog.register_custom_preprocessor('flatten_obs_preprocessor', FlattenObsPreprocessor)
    # ModelCatalog.register_custom_model('masked_mlp_model', JasonMLPModel)
    # ModelCatalog.register_custom_model('my_vision_model', MyVisionNetwork)
    ModelCatalog.register_custom_model('masked_cnn_model', MaskedCNNModel)

    obs_space = spaces.Dict({
        # 'board': spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.uint8),
        # 'board': spaces.Box(low=0, high=2, shape=(6 * 7,), dtype=np.uint8),
        'board': spaces.Box(low=0, high=3, shape=(7, 7), dtype=np.uint8),
        'action_mask': spaces.Box(low=0, high=1, shape=(7,), dtype=np.uint8),
    })
    action_space = spaces.Discrete(7)

    tune.run(
        # trainer,
        POLICY,
        stop={
            # 'timesteps_total': 1000,
            'timesteps_total': 500000,
            'policy_reward_mean': {'learned': 0.95},
        },
        config={
            # 'env': Connect4Env,
            # 'env': FlattenedConnect4Env,
            'env': SquareConnect4Env,
            'log_level': 'DEBUG',
            # 'gamma': 0.9,
            # 'num_workers': 4,
            # 'num_envs_per_worker': 4,
            'num_workers': 0,
            'num_envs_per_worker': 1,
            'num_gpus': 1,
            # 'sample_batch_size': 10,
            # 'train_batch_size': 200,
            'multiagent': {
                'policies_to_train': ['learned'],
                'policies': {
                    'random': (RandomPolicy, obs_space, action_space, {
                        # 'model': {
                        #     'custom_preprocessor': 'flatten_obs_preprocessor',
                        # },
                    }),
                    'learned': (None, obs_space, action_space, {
                        'model': {
                            # 'custom_model': 'masked_mlp_model',
                            # 'custom_model': 'my_vision_model',
                            'custom_model': 'masked_cnn_model',
                            # 'conv_filters': [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]],
                            'conv_filters': [[16, [2, 2], 1], [42, [3, 3], 1], [512, [2, 2], 1]],
                            # 'conv_filters': [[16, [2, 2], 1],],
                            'fcnet_hiddens': [256, 256],
                            # "fcnet_hiddens": [256, 128, 64, 32],
                            'fcnet_activation': 'relu',
                            # 'fcnet_activation': 'tanh',
                            # 'custom_preprocessor': 'square_obs_preprocessor',
                            # 'custom_preprocessor': 'flatten_obs_preprocessor',
                            # 'custom_options': {},  # extra options to pass to your preprocessor
                        }
                    }),
                },
                # 'policy_mapping_fn': tune.function(select_policy),
                'policy_mapping_fn': tune.function(lambda agent_id: ['learned', 'random'][agent_id % 2]),
                # 'policy_mapping_fn': tune.function(lambda _: 'random'),
            },
        })
