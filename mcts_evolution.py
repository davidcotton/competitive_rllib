import argparse
from collections import Counter
import random

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune.registry import register_env
from scipy.special import softmax

from src.callbacks import win_matrix_on_episode_end, mcts_eval_policy_mapping_fn, random_policy_mapping_fn
from src.utils import get_worker_config, get_model_config, get_learner_policy_configs, get_mcts_policy_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--num-learners', type=int, default=4)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--human', action='store_true')
    args = parser.parse_args()

    ray.init(local_mode=args.debug)
    tune_config = get_worker_config(args)

    model_config, env_cls = get_model_config(args.use_cnn)
    register_env('c4', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space, action_space = env.observation_space, env.action_space
    trainable_policies = get_learner_policy_configs(args.num_learners, obs_space, action_space, model_config)
    mcts_rollouts = [1, 2] if args.debug else [8, 16]
    mcts_policies = get_mcts_policy_configs(mcts_rollouts, obs_space, action_space)

    def on_episode_start(info):
        episode = info['episode']
        episode.user_data['trainable_policies'] = [*trainable_policies]
        episode.user_data['mcts_policies'] = [*mcts_policies]

    # tau = 0.2
    tau = 0.4
    # tau = 1.0

    def name_trial(trial):
        """Give trials a more readable name in terminal & Tensorboard."""
        return f'(new_weights-tau:{tau})-{args.num_learners}xPPO-MCTS_Evolution'
        # return f'(np.zeros)-{args.num_learners}xPPO-MCTS_Evolution'
        # return f'(np.ones)-{args.num_learners}xPPO-MCTS_Evolution'
        # return f'(np.ones*-100)-{args.num_learners}xPPO-MCTS_Evolution'
        # return f'(np.rand)-{args.num_learners}xPPO-MCTS_Evolution'
        # return f'(NoWeightChanges)-{args.num_learners}xPPO-MCTS_Evolution'
        # return f'(RandomWeightOrder)-{args.num_learners}xPPO-MCTS_Evolution'

    def my_train_fn(config, reporter):
        assert args.num_learners >= 4, 'Requires 4 or more trainable agents'
        ppo_trainer = PPOTrainer(env='c4', config=config)
        while True:
            result = ppo_trainer.train()
            if 'evaluation' in result:
                train_policies = config['multiagent']['policies_to_train']
                scores = {k: v for k, v in result['evaluation']['policy_reward_mean'].items() if k in train_policies}

                scores_dist = softmax(np.array(list(scores.values())) / tau)
                new_trainables = random.choices(list(scores.keys()), scores_dist, k=len(scores))
                # new_trainables = train_policies
                # random.shuffle(new_trainables)

                weights = ppo_trainer.get_weights()
                new_weights = {old_pid: weights[new_pid] for old_pid, new_pid in zip(weights.keys(), new_trainables)}
                # new_weights = {pid: np.zeros_like(wt) for pid, wt in weights.items() if wt is not None}
                # new_weights = {pid: np.ones_like(wt)*-100 for pid, wt in weights.items() if wt is not None}
                # new_weights = {pid: np.random.rand(*wt.shape) for pid, wt in weights.items() if wt is not None}

                print('\n\n################\nSETTING WEIGHTS\n################\n\n')
                ppo_trainer.set_weights(new_weights)

                num_metrics = 4
                c = Counter(new_trainables)
                result['custom_metrics'].update(
                    {f'most_common{i:02d}': v[1] for i, v in enumerate(c.most_common(num_metrics))})
                result['custom_metrics'].update(
                    {f'scores_dist{i:02d}': v for i, v in enumerate(sorted(scores_dist, reverse=True)[:num_metrics])})
                print('scores_dist', scores_dist)
                # result['custom_metrics'].update(
                #     {f'new_agent{i:02d}': int(v[-2:]) for i, v in enumerate(new_trainables)})
            reporter(**result)

    tune.run(
        my_train_fn,
        name='main',
        trial_name_creator=name_trial,
        stop={
            # 'timesteps_total': int(100e6) * args.num_learners,
            'timesteps_total': int(1e9),
        },
        config=dict({
            'env': 'c4',
            'lr': 5e-5,
            'gamma': 0.9,
            'clip_param': 0.2,
            'lambda': 0.95,
            'entropy_coeff': 0.01,
            'vf_share_layers': True,
            'vf_loss_coeff': 1.0,
            'multiagent': {
                'policies_to_train': [*trainable_policies],
                'policy_mapping_fn': random_policy_mapping_fn,
                'policies': {**trainable_policies, **mcts_policies},
            },
            'callbacks': {
                'on_episode_start': on_episode_start,
                # 'on_episode_end': win_matrix_on_episode_end,
            },
            'evaluation_interval': 1 if args.debug else 10,
            'evaluation_num_episodes': 10 if args.debug else 1,
            'evaluation_config': {'multiagent': {'policy_mapping_fn': mcts_eval_policy_mapping_fn}},
        }, **tune_config),
        # checkpoint_freq=250,
        # checkpoint_at_end=True,
    )
