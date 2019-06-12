import torch
from all.environments import State
from all.memory import NStepBatchBuffer
from .abstract import Agent


class RA2C(Agent):
    def __init__(
            self,
            features,
            r,
            v,
            policy,
            n_envs=None,
            n_steps=4,
            discount_factor=0.99
    ):
        if n_envs is None:
            raise RuntimeError("Must specify n_envs.")
        self.features = features
        self.r = r
        self.v = v
        self.policy = policy
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.discount_factor = discount_factor
        self._batch_size = n_envs * n_steps
        self._buffer = self._make_buffer()
        self._features = []

    def act(self, states, rewards):
        self._buffer.store(states, torch.zeros(self.n_envs), rewards)
        self._train()
        features = self.features(states)
        self._features.append(features)
        return self.policy(features)

    def _train(self):
        if len(self._buffer) >= self._batch_size:
            _, _, returns, next_states, rollout_lengths = self._buffer.sample(self._batch_size)
            features = State.from_list(self._features)
            next_features = self.features.eval(next_states)
            values = self.v(features)
            next_values = self.v.eval(next_features)

            reward_errors = (
                values.detach()
                - self.discount_factor * next_values
                - self.r(State(torch.cat((features.raw.detach(), next_features.raw), dim=1)))
            )

            td_errors = (
                returns
                + (self.discount_factor ** rollout_lengths)
                * next_values
                - values
            )

            self.r.reinforce(reward_errors)
            self.v.reinforce(td_errors)
            self.policy.reinforce(td_errors)
            self.features.reinforce()
            self._features = []

    def _make_buffer(self):
        return NStepBatchBuffer(
            self.n_steps,
            self.n_envs,
            discount_factor=self.discount_factor
        )
 