import torch
from torch import nn
from torch.nn import *  # export everthing
from torch.nn import functional as F
import numpy as np
from all.environments import State
from .noisy import NoisyLinear, NoisyFactorizedLinear


class ListNetwork(nn.Module):
    """
    Wraps a network such that States can be given as input.
    """

    def __init__(self, model, _=None):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, state):
        return self.model(state.features.float()) * state.mask.float().unsqueeze(-1)


class ListToList(nn.Module):
    """
    Wraps a network such that States can be given as inputs, and are received as output.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, state):
        return State(self.model(state.features.float()), state.mask, state.info)


class Aggregation(nn.Module):
    """len()
    Aggregation layer for the Dueling architecture.

    https://arxiv.org/abs/1511.06581
    This layer computes a Q function by combining
    an estimate of V with an estimate of the advantage.
    The advantage is normalized by substracting the average
    advantage so that we can propertly
    """

    def forward(self, value, advantages):
        return value + advantages - torch.mean(advantages, dim=1, keepdim=True)


class Dueling(nn.Module):
    """
    Implementation of the head for the Dueling architecture.

    https://arxiv.org/abs/1511.06581
    This module computes a Q function by computing
    an estimate of V, and estimate of the advantage,
    and combining them with a special Aggregation layer.
    """

    def __init__(self, value_model, advantage_model):
        super(Dueling, self).__init__()
        self.value_model = value_model
        self.advantage_model = advantage_model
        self.aggregation = Aggregation()

    def forward(self, features):
        value = self.value_model(features)
        advantages = self.advantage_model(features)
        return self.aggregation(value, advantages)


class CategoricalDueling(nn.Module):
    """Dueling architecture for C51/Rainbow"""

    def __init__(self, value_model, advantage_model):
        super(CategoricalDueling, self).__init__()
        self.value_model = value_model
        self.advantage_model = advantage_model

    def forward(self, features):
        batch_size = len(features)
        value_dist = self.value_model(features)
        atoms = value_dist.shape[1]
        advantage_dist = self.advantage_model(features).view((batch_size, -1, atoms))
        advantage_mean = advantage_dist.mean(dim=1, keepdim=True)
        return (
            value_dist.view((batch_size, 1, atoms)) + advantage_dist - advantage_mean
        ).view((batch_size, -1))


class Flatten(nn.Module):  # pylint: disable=function-redefined
    """
    Flatten a tensor, e.g., between conv2d and linear layers.

    The maintainers FINALLY added this to torch.nn, but I am
    leaving it in for compatible for the moment.
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)

class Linear0(nn.Linear):
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class TanhActionBound(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.register_buffer(
            "weight", torch.tensor((action_space.high - action_space.low) / 2)
        )
        self.register_buffer(
            "bias", torch.tensor((action_space.high + action_space.low) / 2)
        )

    def forward(self, x):
        return torch.tanh(x) * self.weight + self.bias


class VModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = ListNetwork(model, (1,))

    def forward(self, states):
        return self.model(states).squeeze(-1)


class QModuleContinuous(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = model

    def forward(self, states, actions):
        x = torch.cat((states.features.float(), actions), dim=1)
        return self.model(x).squeeze(-1) * states.mask.float()


def td_loss(loss):
    def _loss(estimates, errors):
        return loss(estimates, errors + estimates.detach())

    return _loss
