# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from all.agents import MEDQN
from all.approximation import QNetwork, PolyakTarget
from all.experiments import DummyWriter
from all.memory import ExperienceReplayBuffer
from .models import fc_relu_q

def medqn(
        minibatch_size=32,
        replay_buffer_size=20000,
        discount_factor=0.99,
        update_frequency=1,
        lr=7e-4,
        temperature=10,
        replay_start_size=1000,
        polyak_rate=0.01,
        device=torch.device('cpu')
):
    def _medqn(env, writer=DummyWriter()):
        model = fc_relu_q(env).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QNetwork(
            model,
            optimizer,
            env.action_space.n,
            target=PolyakTarget(polyak_rate),
            loss=mse_loss,
            writer=writer
        )
        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size, device=device)
        return MEDQN(
            q,
            replay_buffer,
            discount_factor=discount_factor,
            replay_start_size=replay_start_size,
            update_frequency=update_frequency,
            minibatch_size=minibatch_size,
            temperature=temperature,
            writer=writer
        )
    return _medqn


__all__ = ["medqn"]
