# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch import nn
from torch.optim import RMSprop
from all.layers import Flatten, Linear0
from all.agents.ra2c import RecurrentA2C
from all.bodies import ParallelAtariBody
from all.approximation import ValueNetwork, FeatureNetwork
from all.experiments import DummyWriter
from all.policies import SoftmaxPolicy, GaussianPolicy


def conv_features():
    return nn.Sequential(
        nn.Conv2d(1, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        Flatten(),
    )

def hidden_net(features):
    return nn.Sequential(nn.Linear(3456 + features, 512), nn.ReLU(), nn.Linear(512, features * 2))

def value_net(features):
    return nn.Sequential(nn.Linear(3456 + features, 512), nn.ReLU(), Linear0(512, 1))

def policy_net(env, features):
    return nn.Sequential(
        nn.Linear(3456 + features, 512), nn.ReLU(), Linear0(512, env.action_space.n)
    )

def ra2c(
        # modified from stable baselines hyperparameters
        clip_grad=0.1,
        discount_factor=0.99,
        lr=7e-4,    # RMSprop learning rate
        alpha=0.99, # RMSprop momentum decay
        eps=1e-4,   # RMSprop stability
        entropy_loss_scaling=0.01,
        # hidden_loss_scaling=1,
        value_loss_scaling=0.25,
        feature_lr_scaling=1,
        n_envs=64,
        n_steps=5,
        hidden_features=64,
        device=torch.device("cpu"),
):
    def _ra2c(envs, writer=DummyWriter()):
        env = envs[0]

        feature_model = conv_features().to(device)
        hidden_model = hidden_net(hidden_features).to(device)
        value_model = value_net(hidden_features).to(device)
        policy_model = policy_net(env, hidden_features).to(device)

        feature_optimizer = RMSprop(
            feature_model.parameters(),
            alpha=alpha,
            lr=lr * feature_lr_scaling,
            eps=eps
        )
        hidden_optimizer = RMSprop(
            hidden_model.parameters(),
            alpha=alpha,
            lr=lr,
            eps=eps
        )
        value_optimizer = RMSprop(
            value_model.parameters(),
            alpha=alpha,
            lr=lr,
            eps=eps
        )
        policy_optimizer = RMSprop(
            policy_model.parameters(),
            alpha=alpha,
            lr=lr,
            eps=eps
        )

        features = FeatureNetwork(
            feature_model,
            feature_optimizer,
            clip_grad=clip_grad
        )
        hidden = GaussianPolicy(
            hidden_model,
            hidden_optimizer,
            action_dim=hidden_features,
            clip_grad=clip_grad,
            writer=writer
        )
        v = ValueNetwork(
            value_model,
            value_optimizer,
            loss_scaling=value_loss_scaling,
            clip_grad=clip_grad,
            writer=writer
        )
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer,
        )

        return ParallelAtariBody(
            RecurrentA2C(
                features,
                v,
                policy,
                hidden,
                hidden_features=hidden_features,
                n_envs=n_envs,
                n_steps=n_steps,
                discount_factor=discount_factor,
            ),
            envs,
            frame_stack=1,
        )

    return _ra2c, n_envs


__all__ = ["ra2c"]
