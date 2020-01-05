import torch
from all.approximation import Approximation
from all.nn import ListNetwork


class DeterministicPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            space,
            name='policy',
            **kwargs
    ):
        model = DeterministicPolicyNetwork(model, space)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

class DeterministicPolicyNetwork(ListNetwork):
    def __init__(self, model, space):
        super().__init__(model)
        self._action_dim = space.shape[0]
        self._tanh_scale = torch.tensor((space.high - space.low) / 2).to(self.device)
        self._tanh_mean = torch.tensor((space.high + space.low) / 2).to(self.device)

    def forward(self, state):
        return self._squash(super().forward(state))

    def _squash(self, x):
        return torch.tanh(x) * self._tanh_scale + self._tanh_mean
