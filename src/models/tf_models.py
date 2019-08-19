import logging

from gym import spaces
import numpy as np
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from ray.rllib.models import Model
from ray.rllib.models.model import restore_original_dimensions, _unpack_obs
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.models.tf.misc import get_activation_fn, flatten, normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

logger = logging.getLogger('ray.rllib')
tf = try_import_tf()


class MaskedMLPModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        logger.debug('\n\n######## MaskedMLPModel.init() #########')
        logger.debug('obs_space:%s' % obs_space)
        logger.debug('action_space:%s' % action_space)
        logger.debug('num_outputs:%s' % num_outputs)
        logger.debug('model_config:%s' % model_config)
        logger.debug('########################\n')

        self.inputs = tf.keras.layers.Input(shape=(7 * 7,), name='observations')
        last_layer = self.inputs
        fc_activation = get_activation_fn(model_config['fcnet_activation'])
        for i, size in enumerate(model_config['fcnet_hiddens'], 1):
            last_layer = tf.keras.layers.Dense(
                size,
                name='fc{}'.format(i),
                activation=fc_activation,
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
        from pprint import pprint
        print('\n\n\n<<<<<<< input_dict >>>>>>>>>')
        pprint(input_dict)
        print('-----------------------------------\n\n\n')

        obs = flatten(input_dict['obs']['board'])
        model_out, self._value_out = self.base_model(obs)
        action_mask = input_dict['obs']['action_mask']
        masked_logits = tf.where(action_mask < 0.5, -tf.ones_like(model_out) * (2 ** 18), model_out)
        return masked_logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MaskedCNNModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        logger.debug('\n\n######## MaskedCNNModel.init() #########')
        logger.debug('obs_space:%s' % obs_space)
        logger.debug('action_space:%s' % action_space)
        logger.debug('model_config:%s' % model_config)
        logger.debug('########################\n')

        self.inputs = tf.keras.layers.Input(shape=(7, 7, 1), name='observations')
        last_layer = self.inputs
        conv_filters = model_config['conv_filters']
        conv_activation = get_activation_fn(model_config['conv_activation'])
        for i, (filters, kernel_size, stride) in enumerate(conv_filters, 1):
            last_layer = tf.keras.layers.Conv2D(
                filters,
                kernel_size,
                stride,
                name="conv{}".format(i),
                activation=conv_activation,
                padding='same')(last_layer)
        last_layer = tf.keras.layers.Flatten()(last_layer)

        fc_activation = get_activation_fn(model_config['fcnet_activation'])
        for i, size in enumerate(model_config['fcnet_hiddens'], 1):
            last_layer = tf.keras.layers.Dense(
                size,
                name='fc{}'.format(i),
                activation=fc_activation,
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
        obs = input_dict['obs']['board']
        obs = tf.reshape(obs, (-1, *obs.shape[1:], 1))
        model_out, self._value_out = self.base_model(obs)
        action_mask = input_dict['obs']['action_mask']
        masked_logits = tf.where(action_mask < 0.5, -tf.ones_like(model_out) * (2 ** 18), model_out)
        return masked_logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class ParametricActionsMLP(DistributionalQModel, TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)

        # DictFlatteningPreprocessor, combines all obs components together
        # the obs.shape for our MLP should be a flattened game board obs
        original_space = obs_space.original_space['board']
        flat_obs_space = spaces.Box(low=np.min(original_space.low), high=np.max(original_space.high),
                                    shape=(np.prod(original_space.shape),))

        logger.debug('\n\n########## ParametricActionsMLP.init() ##########')
        logger.debug('obs_space:%s' % obs_space)
        logger.debug('obs_space.original_space:%s' % obs_space.original_space)
        logger.debug('flat_obs_space:%s' % flat_obs_space)
        logger.debug('action_space:%s' % action_space)
        logger.debug('num_outputs:%s' % num_outputs)
        logger.debug('model_config:%s' % model_config)
        logger.debug('kwargs:%s' % kwargs)
        logger.debug('\n')

        self.mlp = FullyConnectedNetwork(flat_obs_space, action_space, num_outputs, model_config, name)
        self.register_variables(self.mlp.variables())

    def forward(self, input_dict, state, seq_lens):
        obs = flatten(input_dict['obs']['board'])
        action_mask = tf.maximum(tf.log(input_dict['obs']['action_mask']), tf.float32.min)
        model_out, _ = self.mlp({'obs': obs})
        return action_mask + model_out, state

    def value_function(self):
        return self.mlp.value_function()


class ParametricActionsCNN(DistributionalQModel, TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)

        conv_filters = model_config['conv_filters']
        self.is_conv = bool(conv_filters)
        orig_shape = obs_space.original_space['board']
        new_shape = orig_shape.shape + (1,) if self.is_conv else (np.prod(orig_shape.shape),)

        logger.debug('\n\n########## ParametricActionsCNN.init() ##########')
        logger.debug('obs_space:%s' % obs_space)
        logger.debug('new_shape:%s' % str(new_shape))
        logger.debug('action_space:%s' % action_space)
        logger.debug('num_outputs:%s' % num_outputs)
        logger.debug('model_config:%s' % model_config)
        logger.debug('kwargs:%s' % kwargs)
        logger.debug('\n')

        self.inputs = tf.keras.layers.Input(shape=new_shape, name='observations')
        last_layer = self.inputs

        if self.is_conv:
            conv_activation = get_activation_fn(model_config['conv_activation'])
            for i, (filters, kernel_size, stride) in enumerate(conv_filters, 1):
                last_layer = tf.keras.layers.Conv2D(
                    filters,
                    kernel_size,
                    stride,
                    name="conv{}".format(i),
                    activation=conv_activation,
                    padding='same')(last_layer)
            last_layer = tf.keras.layers.Flatten()(last_layer)

        fc_activation = get_activation_fn(model_config['fcnet_activation'])
        for i, size in enumerate(model_config['fcnet_hiddens'], 1):
            last_layer = tf.keras.layers.Dense(
                size,
                name='fc{}'.format(i),
                activation=fc_activation,
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
        obs = input_dict['obs']['board']
        obs = tf.expand_dims(obs, -1) if self.is_conv else flatten(obs)
        action_mask = tf.maximum(tf.log(input_dict['obs']['action_mask']), tf.float32.min)
        model_out, self._value_out = self.base_model(obs)
        return action_mask + model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


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
