import torch
from torch import nn
from torch.optim import RMSprop
from all.agents import BBPG
from all.bodies import DeepmindAtariBody
from all.experiments import DummyWriter
from all.layers import Flatten, Linear0
from all.policies import SoftmaxPolicy


def policy_net(env):
    return nn.Sequential(
        nn.Conv2d(4, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(3456, 512),
        nn.ReLU(),
        Linear0(512, env.action_space.n)
    )

def bbpg(
        # match a2c hypers
        clip_grad=0.5,
        lr_r=0.01,
        lr_pi=7e-6,    # RMSprop learning rate
        alpha=0.99, # RMSprop momentum decay
        eps=1e-4,   # RMSprop stability
        entropy_loss_scaling=0.001,
        n_episodes=5,
        device=torch.device('cpu')
):
    def _bbpg_atari(env, writer=DummyWriter()):
        policy_model = policy_net(env).to(device)
        policy_optimizer = RMSprop(
            policy_model.parameters(),
            alpha=alpha,
            lr=lr_pi,
            eps=eps
        )
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer,
        )

        return DeepmindAtariBody(
            BBPG(policy, lr=lr_r, n_episodes=n_episodes),
            env
        )
    return _bbpg_atari


__all__ = ["bbpg"]
