"""A collection of training utilities."""

from ray.rllib.models import ModelCatalog

from src.envs import Connect4Env
from src.models import MaskedActionsMLP, MaskedActionsCNN
from src.policies import MCTSPolicy


def get_worker_config(args):
    """When debugging use a single worker otherwise app is too slow and hard to debug.

    :param args: Command line arguments
    :return: Tune configuration.
    """
    if args.debug and args.policy in ['PPO', 'APEX']:
        return {
            'log_level': 'DEBUG',
            # 'num_workers': 0,
            # 'num_workers': 1,
            'num_workers': 2,
        }
    elif args.human:
        # if human player, keep it single env
        return {
            'num_workers': 1,
        }
    elif args.policy == 'PPO':
        # scale right up for PPO
        return {
            'num_workers': 20,
            'num_gpus': 1,
            'num_envs_per_worker': 32,
            'train_batch_size': 65536,
            'sgd_minibatch_size': 4096,
            'num_sgd_iter': 6,
        }
    elif args.policy == 'APEX':
        return {
            'num_workers': 20,
            'num_gpus': 1,
            'num_envs_per_worker': 32,
        }
    else:
        return {
            'num_workers': 0,
            'num_gpus': 1,
        }


def get_model_config(use_cnn, fc_hiddens=None, fc_activation=None, conv_filters=None, conv_activation=None):
    """Build the model and environment configuration based on the agent config.

    We use a separate environment class instead of custom post-processing filters
    as filters muck up RLlibs ability to reconstruct flattened observation space.

    :param use_cnn: Whether to use a CNN.
    :param fc_hiddens: Optional fully-connected hidden layer configuration.
    :param fc_activation: Optional fully-connected activation function.
    :param conv_filters: Optional convolutional filter configuration.
    :param conv_activation: Optional convolutional activation function.
    :return: A tuple containing: the neural network model config, and the environment class to use.
    """
    fc_hiddens = fc_hiddens or [128, 128]
    fc_activation = fc_activation or 'leaky_relu'
    conv_filters = conv_filters or [[16, [2, 2], 1], [32, [2, 2], 1], [64, [3, 3], 2]]
    conv_activation = conv_activation or 'leaky_relu'
    env_cls = Connect4Env
    if use_cnn:
        ModelCatalog.register_custom_model('masked_actions_model', MaskedActionsCNN)
        model_config = {
            'custom_model': 'masked_actions_model',
            'conv_filters': conv_filters,
            'conv_activation': conv_activation,
            'fcnet_hiddens': fc_hiddens,
            'fcnet_activation': fc_activation,
        }
    else:
        ModelCatalog.register_custom_model('masked_actions_model', MaskedActionsMLP)
        model_config = {
            'custom_model': 'masked_actions_model',
            'fcnet_hiddens': fc_hiddens,
            'fcnet_activation': fc_activation,
        }

    return model_config, env_cls


def get_policy_config(policy) -> dict:
    if policy == 'PPO':
        return {
            'lr': 0.001,
            'gamma': 0.995,
            'lambda': 0.95,
            'clip_param': 0.2,
            'kl_coeff': 1.0,
        }
    elif policy in ['DQN', 'APEX']:
        config = {
            # 'num_atoms': 51,
            # 'noisy': True,
            'dueling': False,
            'hiddens': [],
        }
        if policy == 'APEX':
            config.update({
                # 'buffer_size': 50000,
                # 'n_step': 1,
                # 'learning_starts': 1000,
                # 'train_batch_size': 10000,
            })
        return config
    else:
        return {}


def get_learner_policy_configs(num_learners, obs_space, action_space, model_config):
    """Build a dictionary of learner policy configurations.

    :param num_learners: The number of policy configurations to generate.
    :param obs_space: The environment observation space.
    :param action_space: The environment action space.
    :param model_config: The learner neural network configuration.
    :return: A dict of trainable policy configs, e.g. {policy_key: policy_constructor_params}
    """
    return {f'learned{i:02d}': (None, obs_space, action_space, {'model': model_config}) for i in range(num_learners)}


def get_mcts_policy_configs(rollouts, obs_space, action_space):
    """Build a dictionary of MCTS opponent policy configurations.

    :param rollouts: A list of integers for each MCTS rollout.
    :param obs_space: The environment observation space.
    :param action_space: The environment action space.
    :return: A dict of MCTS policy configs, e.g. {policy_key: policy_constructor_params}
    """
    return {f'mcts{i:03d}': (MCTSPolicy, obs_space, action_space, {'max_rollouts': i}) for i in rollouts}


class EloRater:
    def __init__(self, agent_ids=None, k_factor=32, start_value=1200.0) -> None:
        """Calculate and track of the ELO ratings of a group of agents.

        :param agent_ids: An optional list of agent IDs to initialise the ratings pool with.
        :param k_factor: Low k-factors means too stable ratings, high means wild fluctuations.
        :param start_value: A new player's initial starting rating.
        """
        super().__init__()
        self.ratings = {a_id: start_value for a_id in agent_ids}
        self.k = k_factor
        self.start_value = start_value

    def add_agent(self, agent_id) -> None:
        self.ratings[agent_id] = self.start_value

    def rate(self, player_a, player_b, winner=None):
        rating_a = self.ratings[player_a]
        rating_b = self.ratings[player_b]
        win_prob_a = self.win_probability(rating_b, rating_a)
        win_prob_b = self.win_probability(rating_a, rating_b)

        if winner == player_a:
            score_a = 1.0
            score_b = 0.0
        elif winner == player_b:
            score_a = 0.0
            score_b = 1.0
        elif winner == 'draw':
            score_a = 0.5
            score_b = 0.5
        else:
            raise ValueError('Invalid winner')

        self.ratings[player_a] += self.k * (score_a - win_prob_a)
        self.ratings[player_b] += self.k * (score_b - win_prob_b)

        return self.ratings.copy()

    @staticmethod
    def win_probability(rating1, rating2):
        return (1 + 10 ** ((rating1 - rating2) / 400)) ** -1
