import random

trainable_policies = {'learned00': 'Policy'}
player1, player2 = None, None


def policy_mapping_fn(agent_id):
    global player1, player2
    if agent_id == 0:
        player1, player2 = random.sample([*trainable_policies], k=2)
        return player1
    else:
        return player2
