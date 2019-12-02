import ray
from ray.rllib.agents.dqn.simple_q_policy import ExplorationStateMixin, TargetNetworkMixin
from ray.rllib.agents.dqn.dqn_policy import ComputeTDErrorMixin, postprocess_trajectory
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy

from src.policies.assumption.assumption_mixin import SimpleAssumptionMixin, WindowedAssumptionMixin


def postprocess_fn(policy, sample_batch, other_agent_batches=None, episode=None):
    policy.update_assist(sample_batch, other_agent_batches, episode)
    return postprocess_trajectory(policy, sample_batch, other_agent_batches, episode)


def simple_setup_mixins(policy, obs_space, action_space, config):
    ray.rllib.agents.dqn.dqn_policy.setup_early_mixins(policy, obs_space, action_space, config)
    SimpleAssumptionMixin.__init__(policy)


def window_setup_mixins(policy, obs_space, action_space, config):
    ray.rllib.agents.dqn.dqn_policy.setup_early_mixins(policy, obs_space, action_space, config)
    WindowedAssumptionMixin.__init__(policy)


SimpleAssumptionDQNTFPolicy = build_tf_policy(
    name='SimpleAssumptionDQNTFPolicy',
    get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    make_model=ray.rllib.agents.dqn.dqn_policy.build_q_model,
    action_sampler_fn=ray.rllib.agents.dqn.dqn_policy.build_q_networks,
    loss_fn=ray.rllib.agents.dqn.dqn_policy.build_q_losses,
    stats_fn=ray.rllib.agents.dqn.dqn_policy.build_q_stats,
    postprocess_fn=postprocess_fn,
    optimizer_fn=ray.rllib.agents.dqn.dqn_policy.adam_optimizer,
    gradients_fn=ray.rllib.agents.dqn.dqn_policy.clip_gradients,
    extra_action_fetches_fn=lambda policy: {'q_values': policy.q_values},
    extra_learn_fetches_fn=lambda policy: {'td_error': policy.q_loss.td_error},
    before_init=simple_setup_mixins,
    before_loss_init=ray.rllib.agents.dqn.dqn_policy.setup_mid_mixins,
    after_init=ray.rllib.agents.dqn.dqn_policy.setup_late_mixins,
    obs_include_prev_action_reward=False,
    mixins=[
        ExplorationStateMixin, TargetNetworkMixin, ComputeTDErrorMixin, LearningRateSchedule,
        SimpleAssumptionMixin,
    ])


WindowedAssumptionDQNTFPolicy = build_tf_policy(
    name='WindowedAssumptionDQNTFPolicy',
    get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    make_model=ray.rllib.agents.dqn.dqn_policy.build_q_model,
    action_sampler_fn=ray.rllib.agents.dqn.dqn_policy.build_q_networks,
    loss_fn=ray.rllib.agents.dqn.dqn_policy.build_q_losses,
    stats_fn=ray.rllib.agents.dqn.dqn_policy.build_q_stats,
    postprocess_fn=postprocess_fn,
    optimizer_fn=ray.rllib.agents.dqn.dqn_policy.adam_optimizer,
    gradients_fn=ray.rllib.agents.dqn.dqn_policy.clip_gradients,
    extra_action_fetches_fn=lambda policy: {'q_values': policy.q_values},
    extra_learn_fetches_fn=lambda policy: {'td_error': policy.q_loss.td_error},
    before_init=simple_setup_mixins,
    before_loss_init=ray.rllib.agents.dqn.dqn_policy.setup_mid_mixins,
    after_init=ray.rllib.agents.dqn.dqn_policy.setup_late_mixins,
    obs_include_prev_action_reward=False,
    mixins=[
        ExplorationStateMixin, TargetNetworkMixin, ComputeTDErrorMixin, LearningRateSchedule,
        WindowedAssumptionMixin,
    ])
