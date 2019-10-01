import itertools

import numpy as np


def mcts_on_episode_end(info):
    """Add custom metrics to track MCTS rollouts.

    :param info: A dictionary of callback info.
    """
    if 'mcts' in info['policy']:
        episode = info['episode']
        mcts_policy = info['policy']['mcts']
        episode.custom_metrics['num_rollouts'] = np.mean(mcts_policy.metrics['num_rollouts'])
        mcts_policy.metrics['num_rollouts'].clear()


def win_matrix_on_episode_end(info):
    env = info['env'].envs[0]
    agents = [int(agent[1][-2:]) for agent in info['episode'].agent_rewards.keys()]
    agent_perms = [f'win({perm})' for perm in itertools.permutations(agents)]
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
