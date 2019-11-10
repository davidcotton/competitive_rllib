import ray
from ray.rllib.agents.ppo.ppo_policy import KLCoeffMixin, ValueNetworkMixin, LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy

from src.policies.assumption.assumption_mixin import SimpleAssumptionMixin


def postprocess_fn(policy, sample_batch, other_agent_batches=None, episode=None):
    policy.update_assist(sample_batch, other_agent_batches, episode)
    return ray.rllib.agents.ppo.ppo_policy.postprocess_ppo_gae(policy, sample_batch, other_agent_batches, episode)


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    SimpleAssumptionMixin.__init__(policy)
    EntropyCoeffSchedule.__init__(policy, config['entropy_coeff'], config['entropy_coeff_schedule'])
    LearningRateSchedule.__init__(policy, config['lr'], config['lr_schedule'])


AssumptionPPOTFPolicy = build_tf_policy(
    name='AssumptionPPOTFPolicy',
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ray.rllib.agents.ppo.ppo_policy.ppo_surrogate_loss,
    stats_fn=ray.rllib.agents.ppo.ppo_policy.kl_and_loss_stats,
    extra_action_fetches_fn=ray.rllib.agents.ppo.ppo_policy.vf_preds_and_logits_fetches,
    postprocess_fn=postprocess_fn,
    gradients_fn=ray.rllib.agents.ppo.ppo_policy.clip_gradients,
    before_init=ray.rllib.agents.ppo.ppo_policy.setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin,
        SimpleAssumptionMixin
    ])
