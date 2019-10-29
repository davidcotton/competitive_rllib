import itertools
import random

import numpy as np
import ray


def mcts_metrics_on_episode_end(info):
    """Add custom metrics to track MCTS rollouts.

    :param info: A dictionary of callback info.
    """
    if 'mcts' in info['policy']:
        episode = info['episode']
        mcts_policy = info['policy']['mcts']
        episode.custom_metrics['num_rollouts'] = np.mean(mcts_policy.metrics['num_rollouts'])
        mcts_policy.metrics['num_rollouts'].clear()


def win_matrix_on_episode_end(info):
    """Build a matrix to track in Tensorboard the win rate of agent pairs.

    :param info: A dictionary of callback info.
    """
    env = info['env'].envs[0]
    agent_ids = [int(agent[1][-2:]) for agent in info['episode'].agent_rewards.keys()]
    agent_perms = [f'win{perm}' for perm in itertools.permutations(agent_ids)]
    rewards = tuple(info['episode'].agent_rewards.values())
    for ids, reward in zip(agent_perms, rewards):
        if reward == env.reward_win:
            value = 1.0
        elif reward == env.reward_lose:
            value = -1.0
        elif reward == env.reward_draw:
            value = 0.0
        else:
            raise ValueError('Invalid reward')
        info['episode'].custom_metrics[ids] = value


def random_policy_mapping_fn(info):
    trainable_policies = info['user_data']['trainable_policies']
    return random.sample(trainable_policies, k=2)


def mcts_eval_policy_mapping_fn(info):
    user_data = info['user_data']
    eval_policies = [random.choice(user_data['trainable_policies']), random.choice(user_data['mcts_policies'])]
    random.shuffle(eval_policies)
    return eval_policies


def bandit_on_episode_start(info):
    episode = info['episode']
    episode.user_data['env'] = info['env'].get_unwrapped()[0]


def bandit_policy_mapping_fn(info):
    bandit = info['user_data']['env'].bandit
    future = bandit.select.remote(info['episode_id'])
    policy_ids = ray.get(future)
    policy_names = tuple([f'learned{pid:02d}' for pid in policy_ids])
    return policy_names


def bandit_on_episode_end(info):
    episode = info['episode']
    reward = (42 - episode.length) / 42
    bandit = info['env'].get_unwrapped()[0].bandit
    bandit.update.remote(episode.episode_id, reward)
    episode.custom_metrics.update(ray.get(bandit.get_probs.remote()))
