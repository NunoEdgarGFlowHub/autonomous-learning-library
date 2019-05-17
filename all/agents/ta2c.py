import torch
from all.memory import NStepBatchBuffer
from .abstract import Agent


class TA2C(Agent):
    def __init__(
            self,
            features,
            v,
            v_plus,
            policy,
            n_envs=None,
            n_steps=4,
            discount_factor=0.99
    ):
        if n_envs is None:
            raise RuntimeError("Must specify n_envs.")
        self.features = features
        self.v = v
        self.v_plus = v_plus
        self.policy = policy
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.discount_factor = discount_factor
        self._batch_size = n_envs * n_steps
        self._buffer = self._make_buffer(discount_factor)
        self._long_buffer = self._make_buffer(1)

    def act(self, states, rewards, info=None):
        # store transition and train BEFORE choosing action
        # Do not need to know actions, so pass in empy array
        self._buffer.store(states, torch.zeros(self.n_envs), rewards)
        self._long_buffer.store(
            states,
            torch.zeros(self.n_envs),
            self.v.eval(self.features.eval(states))
        )
        while len(self._buffer) >= self._batch_size:
            self._train()
        return self.policy(self.features(states))

    def _train(self):
        states, _, next_states, returns, rollout_lengths = self._buffer.sample(self._batch_size)
        _, _, _, v_sum, _ = self._long_buffer.sample(self._batch_size)
        features = self.features(states)
        next_features = self.features.eval(next_states)
        td_errors = (
            returns
            + (self.discount_factor ** rollout_lengths)
            * self.v.eval(next_features)
            - self.v(features)
        )
        td_plus = (
            v_sum
            + self.v_plus.eval(next_features)
            - self.v_plus(features)
        )
        self.v.reinforce(td_errors, retain_graph=True)
        self.v_plus.reinforce(td_plus, retain_graph=True)
        self.policy.reinforce(td_errors + (1 - self.discount_factor) * td_plus)
        self.features.reinforce()

    def _make_buffer(self, discount_factor):
        return NStepBatchBuffer(
            self.n_steps,
            self.n_envs,
            discount_factor=discount_factor
        )
