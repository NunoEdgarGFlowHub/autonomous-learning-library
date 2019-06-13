# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch import nn
from torch.optim import RMSprop
from all.layers import Flatten, Linear0
from all.agents.ra2c import RA2C
from all.bodies import ParallelAtariBody
from all.approximation import ValueNetwork, FeatureNetwork
from all.experiments import DummyWriter
from all.policies import SoftmaxPolicy


def conv_features():
    return nn.Sequential(
        nn.Conv2d(4, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        Flatten(),
    )

def reward_net(latent_features):
    return nn.Sequential(
        nn.Linear(3456 * 2, 512),
        nn.ReLU(),
        nn.Linear(512, latent_features),
        nn.BatchNorm1d(latent_features),
        nn.Sigmoid(),
        Linear0(latent_features, 1, bias=False)
    )

def value_net():
    return nn.Sequential(nn.Linear(3456, 512), nn.ReLU(), Linear0(512, 1))

def policy_net(env):
    return nn.Sequential(
        nn.Linear(3456, 512), nn.ReLU(), Linear0(512, env.action_space.n)
    )

def ra2c(
        # based on stable baselines hyperparameters
        clip_grad=0.1,
        discount_factor=0.99,
        lr=7e-4,    # RMSprop learning rate
        alpha=0.99, # RMSprop momentum decay
        eps=1e-4,   # RMSprop stability
        entropy_loss_scaling=0.01,
        value_loss_scaling=0.25,
        feature_lr_scaling=1,
        save_frequency=200,
        n_envs=50,
        n_steps=5,
        latent_features=4,
        device=torch.device("cpu"),
):
    def _ra2c(envs, writer=DummyWriter()):
        env = envs[0]

        feature_model = conv_features().to(device)
        reward_model = reward_net(latent_features).to(device)
        value_model = value_net().to(device)
        policy_model = policy_net(env).to(device)

        feature_optimizer = RMSprop(
            feature_model.parameters(),
            alpha=alpha,
            lr=lr * feature_lr_scaling,
            eps=eps
        )
        reward_optimizer = RMSprop(
            reward_model.parameters(),
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
            clip_grad=clip_grad,
            save_frequency=save_frequency
        )
        r = ValueNetwork(
            reward_model,
            reward_optimizer,
            clip_grad=clip_grad,
            writer=writer,
            name='reward',
            save_frequency=save_frequency
        )
        v = ValueNetwork(
            value_model,
            value_optimizer,
            loss_scaling=value_loss_scaling,
            clip_grad=clip_grad,
            writer=writer,
            save_frequency=save_frequency
        )
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer,
            save_frequency=save_frequency
        )

        return ParallelAtariBody(
            RA2C(
                features,
                r,
                v,
                policy,
                n_envs=n_envs,
                n_steps=n_steps,
                discount_factor=discount_factor,
            ),
            envs,
        )

    return _ra2c, n_envs


__all__ = ["ra2c"]
