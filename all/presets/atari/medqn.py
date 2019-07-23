# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
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
        replay_buffer_size=100000, # originally 1e6
        agent_history_length=4,
        discount_factor=0.99,
        action_repeat=4,
        update_frequency=4,
        lr=7e-4, # lr for Adam: Deepmind used RMSprop
        eps=1.5e-4, # stability parameter for Adam
        polyak_rate=0.001,
        final_exploration_frame=1000000,
        replay_start_size=50000,
        temperature=0.01,
        noop_max=30,
        device=torch.device('cpu')
):
    # counted by number of updates rather than number of frame
    final_exploration_frame /= action_repeat
    replay_start_size /= action_repeat

    def _medqn(env, writer=DummyWriter()):
        _model = nature_dqn(env).to(device)
        _optimizer = Adam(
            _model.parameters(),
            lr=lr,
            eps=eps
        )
        q = QNetwork(
            _model,
            _optimizer,
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
