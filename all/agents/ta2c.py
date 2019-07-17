import torch
from all.experiments import DummyWriter
from all.environments import State
from all.memory import NStepAdvantageBuffer
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
            discount_factor=0.99,
            final_discount_frame=1e7,
            writer=DummyWriter()
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
        self._writer = writer
        self._frames = 0.
        self._state = None
        self._last_values = None
        self._features = None
        self._final_discount_frame = final_discount_frame
        self._batch_size = n_envs * n_steps
        self._buffer = self._make_buffer()
        self._buffer_plus = self._make_buffer_plus()
        self._dummy_actions = torch.zeros(self.n_envs)
        self._features = []

    def act(self, states, rewards):
        # have to hackily reach back in time
        # in order to update values in old plus buffer
        if self._last_values is not None:
            self._last_values[:] = self.v.eval(self.features.eval(states))
        self._last_values = torch.zeros(self.n_envs, device=self.v_plus.device)
        self._buffer.store(states, self._dummy_actions, rewards)
        self._buffer_plus.store(states, self._dummy_actions, self._last_values)
        self._train()
        features = self.features(states)
        self._features.append(features)
        self._frames += 1
        return self.policy(features)

    def _train(self):
        if len(self._buffer) >= self._batch_size:
            _, _, a = self._buffer.sample(self._batch_size)
            _, _, a_plus = self._buffer_plus.sample(self._batch_size)
            features = State.from_list(self._features)
            future_discount = min(self._frames / self._final_discount_frame, 1)
            self._writer.add_loss('future_discount', future_discount)
            self.v(features)
            self.v.reinforce(a)
            self.v_plus(features)
            self.v_plus.reinforce(a_plus)
            self.policy.reinforce(a + future_discount * (1 - self.discount_factor) * a_plus)
            self.features.reinforce()
            self._features = []

    def _make_buffer(self):
        return NStepAdvantageBuffer(
            self.v,
            self.features,
            self.n_steps,
            self.n_envs,
            discount_factor=self.discount_factor
        )

    def _make_buffer_plus(self):
        return NStepAdvantageBuffer(
            self.v_plus,
            self.features,
            self.n_steps,
            self.n_envs,
            discount_factor=1
        )
