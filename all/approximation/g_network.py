import torch
from torch.nn.functional import mse_loss
from all.nn import Module, td_loss
from .approximation import Approximation

class GModule(Module):
    def __init__(self, model):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = model

    def forward(self, states, rewards):
        x = torch.cat((states.features, rewards.unsqueeze(1)), dim=1)
        return self.model(x).squeeze(-1) * states.mask.float()

class GNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            loss=mse_loss,
            name='g',
            **kwargs
    ):
        model = GModule(model)
        loss = td_loss(loss)
        super().__init__(
            model,
            optimizer,
            loss=loss,
            name=name,
            **kwargs
        )
