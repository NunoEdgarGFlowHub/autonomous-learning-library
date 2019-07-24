# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import RMSprop
from torch.nn.functional import smooth_l1_loss
from all.approximation import QNetwork, PolyakTarget
from all.agents import MEDQN
from all.bodies import DeepmindAtariBody
from all.experiments import DummyWriter
from all.memory import ExperienceReplayBuffer
from .models import nature_dqn


def medqn(
        # Taken from Extended Data Table 1
        # in https://www.nature.com/articles/nature14236
        # except where noted.
        minibatch_size=32,
        replay_buffer_size=200000, # originally 1e6
        agent_history_length=4,
        polyak_rate=0.001, # polyak averaging of target network
        discount_factor=0.99,
        action_repeat=4,
        update_frequency=4,
        # RMSprop settings
        lr=2.5e-4,
        gradient_momentum=0.95,
        squared_gradient_momentum=0.95,
        min_squared_gradient=0.01,
        # Exporation parameter
        temperature=0.01,
        # Other
        replay_start_size=50000,
        noop_max=30,
        device=torch.device('cpu')
):
    # counted by number of updates rather than number of frame
    replay_start_size /= action_repeat

    def _medqn(env, writer=DummyWriter()):
        model = nature_dqn(env).to(device)
        optimizer = RMSprop(
            model.parameters(),
            lr=lr,
            momentum=gradient_momentum,
            alpha=squared_gradient_momentum,
            eps=min_squared_gradient,
        )
        q = QNetwork(
            model,
            optimizer,
            env.action_space.n,
            target=PolyakTarget(polyak_rate),
            loss=smooth_l1_loss,
            writer=writer
        )
        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        )
        return DeepmindAtariBody(
            MEDQN(
                q,
                replay_buffer,
                discount_factor=discount_factor,
                minibatch_size=minibatch_size,
                replay_start_size=replay_start_size,
                update_frequency=update_frequency,
                temperature=temperature,
                writer=writer
            ),
            env,
            action_repeat=action_repeat,
            frame_stack=agent_history_length,
            noop_max=noop_max
        )
    return _medqn
