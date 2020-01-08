from torch.nn.functional import mse_loss
from all.logging import DummyWriter
from all.memory import NStepAdvantageBuffer
from ._agent import Agent


class A2C(Agent):
    def __init__(
            self,
            features,
            v,
            policy,
            discount_factor=0.99,
            entropy_loss_scaling=0.01,
            n_envs=None,
            n_steps=4,
            writer=DummyWriter()
    ):
        if n_envs is None:
            raise RuntimeError("Must specify n_envs.")
        # objects
        self.features = features
        self.v = v
        self.policy = policy
        self.writer = writer
        # hyperparameters
        self.discount_factor = discount_factor
        self.entropy_loss_scaling = entropy_loss_scaling
        self.n_envs = n_envs
        self.n_steps = n_steps
        # private
        self._states = None
        self._actions = None
        self._batch_size = n_envs * n_steps
        self._buffer = self._make_buffer()

    def act(self, states, rewards):
        self._buffer.store(self._states, self._actions, rewards)
        self._train(states)
        self._states = states
        self._actions = self.policy.eval(self.features.eval(states)).sample()
        return self._actions

    def _train(self, states):
        if len(self._buffer) >= self._batch_size:
            # load trajectories from buffer
            states, actions, advantages = self._buffer.advantages(states)

            # forward pass
            features = self.features(states)
            values = self.v(features)
            distribution = self.policy(features)

            # compute targets
            targets = values.detach() + advantages

            # compute losses
            value_loss = mse_loss(values, targets)
            policy_gradient_loss = -(distribution.log_prob(actions) * advantages).mean()
            entropy_loss = -distribution.entropy().mean()
            policy_loss = policy_gradient_loss + self.entropy_loss_scaling * entropy_loss

            # backward pass
            self.v.reinforce(value_loss)
            self.policy.reinforce(policy_loss)
            self.features.reinforce()

            # debugging
            self.writer.add_loss('policy_gradient', policy_gradient_loss.detach())
            self.writer.add_loss('entropy', entropy_loss.detach())

    def _make_buffer(self):
        return NStepAdvantageBuffer(
            self.v,
            self.features,
            self.n_steps,
            self.n_envs,
            discount_factor=self.discount_factor
        )
 