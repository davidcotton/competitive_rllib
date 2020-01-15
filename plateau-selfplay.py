import argparse
import random

import ray
from ray import tune
from ray.tune.registry import register_env

from src.policies import HumanPolicy, RandomPolicy
from src.utils import get_worker_config, get_learner_policy_configs, get_mcts_policy_configs, get_model_config, \
    get_policy_config


parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='PPO')
parser.add_argument('--use-cnn', action='store_true')
# e.g. --restore="/home/dave/ray_results/main/PPO_c4_0_2019-09-23_16-17-45z9x1oc9j/checkpoint_782/checkpoint-782"
parser.add_argument('--restore', type=str)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--human', action='store_true')
args = parser.parse_args()

ray.init(local_mode=args.debug)

tune_config = get_worker_config(args)
tune_config.update(get_policy_config(args.policy))
model_config, env_cls = get_model_config(args.use_cnn)
register_env('c4', lambda cfg: env_cls(cfg))
env = env_cls()
obs_space, action_space = env.observation_space, env.action_space

trainable_policies = get_learner_policy_configs(1, obs_space, action_space, model_config)
mcts_train_policies = get_mcts_policy_configs([8, 16, 32, 64, 128, 256, 512], obs_space, action_space)
# mcts_eval_rollouts = [8, 16]
mcts_eval_rollouts = [32, 64, 128]
mcts_eval_policies = get_mcts_policy_configs(mcts_eval_rollouts, obs_space, action_space)


def eval_policy_mapping_fn(info):
    eval_policies = ['learned00', random.choice([*mcts_eval_policies])]
    random.shuffle(eval_policies)
    return eval_policies


def name_trial(trial):
    """Give trials a more readable name in terminal & Tensorboard."""
    return f'selfplay-{trial.trainable_name}'


tune.run(
    args.policy,
    name='plateau',
    trial_name_creator=name_trial,
    stop={
        'timesteps_total': int(100e6),
        # 'timesteps_total': int(1e9),
    },
    config=dict({
        'env': 'c4',
        # 'env': tune.grid_search(['c4', 'c4', 'c4']),
        'multiagent': {
            'policies_to_train': [*trainable_policies],
            'policy_mapping_fn': lambda _: ['learned00', 'learned00'],
            'policies': {
                **trainable_policies,
                **mcts_train_policies,
                'human': (HumanPolicy, obs_space, action_space, {}),
                'random': (RandomPolicy, obs_space, action_space, {}),
            },
        },
        'evaluation_interval': 10,
        # 'evaluation_interval': 100,
        'evaluation_num_episodes': 1,
        'evaluation_config': {'multiagent': {'policy_mapping_fn': eval_policy_mapping_fn}},
    }, **tune_config),
    # checkpoint_freq=100,
    checkpoint_at_end=True,
    # resume=True,
    restore=args.restore,
)
