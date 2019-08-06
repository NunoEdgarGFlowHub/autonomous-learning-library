from all.memory import NStepAdvantageBuffer
from ._agent import Agent


class TA2C(Agent):
    def __init__(
            self,
            features,
            v,
            plus,
            policy,
            beta=1, # mixing factor
            n_envs=None,
            n_steps=4,
            discount_factor=0.99
    ):
        if n_envs is None:
            raise RuntimeError("Must specify n_envs.")
        self.features = features
        self.v = v
        self.plus = plus
        self.policy = policy
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.discount_factor = discount_factor
        self.beta = beta
        self._states = None
        self._actions = None
        self._batch_size = n_envs * n_steps
        self._buffer = self._make_buffer(v, discount_factor)
        self._plus_buffer = self._make_buffer(plus, 1)
        self._features = []

    def act(self, states, rewards):
        features = self.features.eval(states)
        values = self.v.eval(features)
        self._store_transitions(rewards, values)
        self._train(states)
        self._states = states
        self._actions = self.policy.eval(features)
        return self._actions

    def _store_transitions(self, rewards, values):
        if self._states:
            self._buffer.store(self._states, self._actions, rewards)
            self._plus_buffer.store(self._states, self._actions, values)

    def _train(self, states):
        if len(self._buffer) >= self._batch_size:
            _, _, plus_advantages = self._plus_buffer.advantages(states)
            states, actions, advantages = self._buffer.advantages(states)
            augmented_advantage = (
                advantages +
                self.beta * (1 - self.discount_factor) *
                plus_advantages
            )
            # forward pass
            features = self.features(states)
            self.v(features)
            self.plus(features)
            self.policy(features, actions)
            # backward pass
            self.v.reinforce(advantages)
            self.plus.reinforce(plus_advantages)
            self.policy.reinforce(augmented_advantage)
            self.features.reinforce()

    def _make_buffer(self, v, discount_factor):
        return NStepAdvantageBuffer(
            v,
            self.features,
            self.n_steps,
            self.n_envs,
            discount_factor=discount_factor
        )
 