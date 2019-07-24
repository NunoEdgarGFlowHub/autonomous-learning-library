# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import RMSprop
from torch.nn.functional import smooth_l1_loss
from all.approximation import QNetwork, FixedTarget
from all.agents import DQN
from all.bodies import DeepmindAtariBody
from all.experiments import DummyWriter
from all.policies import GreedyPolicy
from all.memory import ExperienceReplayBuffer
from .models import nature_dqn


def dqn(
        # Taken from Extended Data Table 1
        # in https://www.nature.com/articles/nature14236
        # except where noted.
        minibatch_size=32,
        replay_buffer_size=200000, # originally 1e6
        agent_history_length=4,
        target_update_frequency=10000,
        discount_factor=0.99,
        action_repeat=4,
        update_frequency=4,
        # RMSprop settings
        lr=2.5e-4,
        gradient_momentum=0.95,
        squared_gradient_momentum=0.95,
        min_squared_gradient=0.01,
        # Epsilon Greedy settings
        initial_exploration=1.,
        final_exploration=0.1,
        final_exploration_frame=1000000,
        # Other
        replay_start_size=50000,
        noop_max=30,
        device=torch.device('cpu')
):
    # counted by number of updates rather than number of frame
    final_exploration_frame /= action_repeat
    replay_start_size /= action_repeat

    def _dqn(env, writer=DummyWriter()):
        _model = nature_dqn(env).to(device)
        _optimizer = RMSprop(
            _model.parameters(),
            lr=lr,
            momentum=gradient_momentum,
            alpha=squared_gradient_momentum,
            eps=min_squared_gradient,
        )
        q = QNetwork(
            _model,
            _optimizer,
            env.action_space.n,
            target=FixedTarget(target_update_frequency),
            loss=smooth_l1_loss,
            writer=writer
        )
        policy = GreedyPolicy(q,
                              env.action_space.n,
                              annealing_start=replay_start_size,
                              annealing_time=final_exploration_frame - replay_start_size,
                              initial_epsilon=initial_exploration,
                              final_epsilon=final_exploration
                              )
        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        )
        return DeepmindAtariBody(
            DQN(q, policy, replay_buffer,
                discount_factor=discount_factor,
                minibatch_size=minibatch_size,
                replay_start_size=replay_start_size,
                update_frequency=update_frequency,
                ),
            env,
            action_repeat=action_repeat,
            frame_stack=agent_history_length,
            noop_max=noop_max
        )
    return _dqn
