import torch
from all.environments import State
from all.memory import NStepBatchBuffer
from .abstract import Agent


class RecurrentA2C(Agent):
    def __init__(
            self,
            features,
            v,
            policy,
            rnn,
            hidden_features=None,
            n_envs=None,
            n_steps=4,
            discount_factor=0.99
    ):
        if n_envs is None:
            raise RuntimeError("Must specify n_envs.")
        if hidden_features is None:
            raise RuntimeError("Must specifiy number of recurrent features.")
        self.features = features
        self.v = v
        self.policy = policy
        self.rnn = rnn
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.discount_factor = discount_factor
        self._hidden = torch.zeros((hidden_features), device=rnn.device)
        self._batch_size = n_envs * n_steps
        self._buffer = self._make_buffer()

    def act(self, states, rewards):
        # TODO tweak this implementation to match a2c better
        features = self.features(states)
        # mask hidden state on episode resets
        self._hidden = self._hidden * states.mask.float().unsqueeze(-1)
        # compute new hidden features
        self._hidden = self.rnn(cat_features(features, self._hidden))
        # combine regular and hidden features
        features = cat_features(features, self._hidden)
        # store combined features
        self._buffer.store(features, torch.zeros(self.n_envs), rewards)
        # train if possible
        self._train()
        # choose next action
        return self.policy(features)

    def _train(self):
        if len(self._buffer) >= self._batch_size:
            states, _, returns, next_states, rollout_lengths = self._buffer.sample(self._batch_size)
            td_errors = (
                returns
                + (self.discount_factor ** rollout_lengths)
                * self.v.eval(next_states)
                - self.v(states)
            )
            self.v.reinforce(td_errors)
            self.policy.reinforce(td_errors)
            self.rnn.reinforce(td_errors) # Async CPGT
            self.features.reinforce()

    def _make_buffer(self):
        return NStepBatchBuffer(
            self.n_steps,
            self.n_envs,
            discount_factor=self.discount_factor
        )
 
def cat_features(features, hidden):
    raw = torch.cat((features.raw, hidden), dim=1)
    return State(raw, mask=features.mask)
