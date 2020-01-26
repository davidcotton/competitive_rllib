"""Evaluate a trained agent against different MCTS strengths."""

import argparse
import random

import ray
from ray import tune
from ray.tune.registry import register_env

from src.policies import HumanPolicy, MCTSPolicy, RandomPolicy
from src.utils import get_worker_config, get_learner_policy_configs, get_model_config, get_policy_config, \
    get_mcts_policy_configs


parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, default='PPO')
parser.add_argument('--use-cnn', action='store_true')
parser.add_argument('--num-learners', type=int, default=2)
# e.g. --restore="/home/dave/ray_results/main/PPO_c4_0_2019-09-23_16-17-45z9x1oc9j/checkpoint_782/checkpoint-782"
parser.add_argument('--restore', type=str)
# e.g. --eval-policy=learned06
parser.add_argument('--eval-policy', type=str)
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
trainable_policies = get_learner_policy_configs(args.num_learners, obs_space, action_space, model_config)
mcts_train_rollouts = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# mcts_eval_rollouts = [4, 8, 16, 32, 64, 128]
mcts_eval_rollouts = [128, 256, 512, 1024, 2048]
mcts_policies = get_mcts_policy_configs(mcts_train_rollouts, obs_space, action_space)


def get_policy_by_num(num_rollouts):
    return {
        'policies_to_train': [*trainable_policies],
        'policy_mapping_fn': lambda _: random.sample([args.eval_policy, f'mcts{num_rollouts:04d}'], k=2),
        'policies': {
            **trainable_policies,
            **mcts_policies,
            'human': (HumanPolicy, obs_space, action_space, {}),
            'random': (RandomPolicy, obs_space, action_space, {}),
            'rollouts': (MCTSPolicy, obs_space, action_space, {'max_rollouts': num_rollouts}),
        },
    }


def name_trial(trial):
    """Give trials a more readable name in terminal & Tensorboard."""
    pids = trial.config['multiagent']['policies']
    for pid, policy in pids.items():
        if pid == 'rollouts':
            break
    num_mcts_rollouts = policy[3]['max_rollouts']
    return f'{trial.trainable_name}vsMCTS{num_mcts_rollouts}'


tune.run(
    args.policy,
    name='mcts_evaluator',
    trial_name_creator=name_trial,
    stop={
        'episodes_this_iter': 1000,
    },
    config=dict({
        'env': 'c4',
        'env_config': {},
        'multiagent': tune.grid_search([get_policy_by_num(n) for n in mcts_eval_rollouts]),
    }, **tune_config),
    restore=args.restore,
)
