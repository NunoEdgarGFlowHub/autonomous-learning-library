# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch import nn
from torch.optim import SGD
from all.layers import Flatten
from all.agents import BBPG
from all.experiments import DummyWriter
from all.policies import SoftmaxPolicy

def fc_policy(env):
    return nn.Sequential(
        Flatten(),
        nn.Linear(env.state_space.shape[0], 256),
        nn.ReLU(),
        nn.Linear(256, env.action_space.n)
    )

def bbpg(
        clip_grad=0,
        entropy_loss_scaling=0.001,
        lr_r=0.01,
        lr_pi=1e-3,
        n_episodes=5,
        device=torch.device('cpu')
):
    def _bbpg(env, writer=DummyWriter()):
        policy_model = fc_policy(env).to(device)
        policy_optimizer = SGD(policy_model.parameters(), lr=lr_pi)
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer
        )
        return BBPG(policy, lr=lr_r, n_episodes=n_episodes, writer=writer)
    return _bbpg


__all__ = ["bbpg"]
