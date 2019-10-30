_DUMMY_AGENT_ID = 1


def policy_for(self, agent_id=_DUMMY_AGENT_ID):
    """Returns the policy for the specified agent.

    If the agent is new, the policy mapping fn will be called to bind the
    agent to a policy for the duration of the episode.
    """

    if self.episode_id not in self._agent_to_policy:
        map_fn_data = {'episode_id': self.episode_id, 'user_data': self.user_data}
        self._agent_to_policy[self.episode_id] = self._policy_mapping_fn(map_fn_data)
    return self._agent_to_policy[self.episode_id][agent_id]

    # if agent_id not in self._agent_to_policy:
    #     self._agent_to_policy[agent_id] = self._policy_mapping_fn(agent_id)
    # return self._agent_to_policy[agent_id]
