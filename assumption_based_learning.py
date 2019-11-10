import argparse
import random

import ray
from ray import tune
from ray.tune.registry import register_env

from src.policies import HumanPolicy, RandomPolicy
from src.policies.assumption.assumption_dqn_policy import AssumptionDQNTFPolicy
from src.policies.assumption.assumption_ppo_policy import AssumptionPPOTFPolicy
from src.utils import get_worker_config, get_mcts_policy_configs, get_model_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--num-learners', type=int, default=2)
    parser.add_argument('--assist', action='store_true')
    parser.add_argument('--mcts', type=int, default=0)
    # e.g. --restore="/home/dave/ray_results/main/PPO_c4_0_2019-09-23_16-17-45z9x1oc9j/checkpoint_782/checkpoint-782"
    parser.add_argument('--restore', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--human', action='store_true')
    args = parser.parse_args()

    ray.init(local_mode=args.debug)
    tune_config = get_worker_config(args)

    model_config, env_cls = get_model_config(args.use_cnn)
    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space, action_space = env.observation_space, env.action_space

    num_learners = 1 if args.human or args.mcts > 0 else args.num_learners
    learner_policy = None
    if args.policy == 'PPO':
        tune_config.update({
            'lr': 0.001,
            'gamma': 0.995,
            'clip_param': 0.2,
            'lambda': 0.95,
            'kl_coeff': 1.0,
        })
        if args.assist:
            print('\n#############################################################')
            print('Using assumption-based learning policy: AssumptionPPOTFPolicy')
            print('#############################################################\n')
            learner_policy = AssumptionPPOTFPolicy
    elif args.policy in ['DQN', 'APEX']:
        tune_config.update({
            'dueling': False,  # not supported with parametric actions
            'hiddens': [],     # not supported with parametric actions
            # 'n_step': 4,
        })
        if args.policy == 'APEX':
            tune_config.update({
                # 'buffer_size': 1000000,
                'buffer_size': 50000,
                'n_step': 1,
                'learning_starts': 1000,
                'train_batch_size': 10000,
            })
        if args.assist:
            print('\n#############################################################')
            print('Using assumption-based learning policy: AssumptionDQNTFPolicy')
            print('#############################################################\n')
            learner_policy = AssumptionDQNTFPolicy

    trainable_policies = {
        f'learned{i:02d}': (learner_policy, obs_space, action_space, {'model': model_config}) for i in
        range(num_learners)}
    mcts_policies = get_mcts_policy_configs([2, 8, 16, 32, 64, 128, 256, 512], obs_space, action_space)
    stop_criterion = ('policy_reward_mean/learned00', 0.6) if args.mcts > 0 else ('timesteps_total', int(100e6))

    def random_policy_mapping_fn(info):
        if args.human:
            return random.sample(['learned00', 'human'], k=2)
        elif args.mcts > 0:
            return random.sample(['learned00', f'mcts{args.mcts:03d}'], k=2)
        elif num_learners == 1:
            return ['learned00', 'learned00']
        else:
            return random.sample([*trainable_policies], k=2)

    def name_trial(trial):
        """Give trials a more readable name in terminal & Tensorboard."""
        assist = 'Assist' if args.assist else 'NoAssist'
        mcts = f'vsMCTS{args.mcts}' if args.mcts > 0 else ''
        debug = '-debug' if args.debug else ''
        return f'{num_learners}x{trial.trainable_name}{mcts}-{assist}{debug}'

    tune.run(
        args.policy,
        name='ABL',
        trial_name_creator=name_trial,
        stop={
            stop_criterion[0]: stop_criterion[1]
        },
        config=dict({
            'env': 'c4',
            'multiagent': {
                'policies_to_train': [*trainable_policies],
                'policy_mapping_fn': random_policy_mapping_fn,
                'policies': {
                    **trainable_policies,
                    **mcts_policies,
                    'human': (HumanPolicy, obs_space, action_space, {}),
                    'random': (RandomPolicy, obs_space, action_space, {}),
                },
            },
        }, **tune_config),
        checkpoint_at_end=True,
        # resume=True,
        restore=args.restore,
    )
