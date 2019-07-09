from .abstract import Agent

class GreyActorCritic(Agent):
    def __init__(self, v, policy, gamma=1):
        self.v = v
        self.policy = policy
        self.gamma = gamma
        self._return = 0
        self._previous_reward = 0
        self._previous_state = None

    def act(self, state, reward):
        if not self._previous_state or self._previous_state.done:
            return self._initial(state)
        if not state.done:
            return self._act(state, reward)
        return self._terminal(state, reward)

    def _initial(self, state):
        self._return = 0
        self._previous_reward = 0
        self._previous_state = state
        return self.policy(state)

    def _act(self, state, reward):
        td_error = (
            reward
            + self.v.eval(state)
            - self._previous_reward
            - self.v(self._previous_state)
        )
        self.v.reinforce(td_error)
        self.policy.reinforce(td_error)
        self._previous_state = state
        self._previous_reward = reward
        self._return += reward
        return self.policy(state)

    def _terminal(self, state, reward):
        self._return += reward
        td_error = (
            self._return
            - self._previous_reward
            -self.v(self._previous_state)
        )
        self.v.reinforce(td_error)
        self.policy.reinforce(td_error)
        self._previous_state = state
        self._previous_reward = 0
        self._return = 0
        return
