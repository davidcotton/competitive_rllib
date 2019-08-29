import numpy as np


def on_episode_end(info):
    """Add custom metrics to track MCTS rollouts.

    :param info: A dictionary of callback info.
    """
    if 'mcts' in info['policy']:
        episode = info['episode']
        mcts_policy = info['policy']['mcts']
        episode.custom_metrics['num_rollouts'] = np.mean(mcts_policy.metrics['num_rollouts'])
        mcts_policy.metrics['num_rollouts'].clear()
