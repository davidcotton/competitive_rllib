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


def elo_on_episode_end(info):
    """Update agent Elo ratings.

    :param info: A dictionary of callback info.
    """

    global elo_rater
    agents = [agent[1] for agent in info['episode'].agent_rewards.keys()]
    rewards = list(info['episode'].agent_rewards.values())
    if rewards[0] == rewards[1]:  # draw
        winner = 'draw'
    elif rewards[0] > rewards[1]:
        winner = agents[0]
    else:
        winner = agents[1]

    ratings = elo_rater.rate(*agents, winner)
    ratings = {'elo_' + k: v for k, v in ratings.items()}
    info['episode'].custom_metrics.update(ratings)
