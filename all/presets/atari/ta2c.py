# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import RMSprop
from all import nn
from all.agents import TA2C
from all.bodies import ParallelAtariBody
from all.approximation import VNetwork, FeatureNetwork
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
        nn.Flatten(),
    )


def value_net():
    return nn.Sequential(nn.Linear(3456, 512), nn.ReLU(), nn.Linear0(512, 1))


def policy_net(env):
    return nn.Sequential(
        nn.Linear(3456, 512), nn.ReLU(), nn.Linear0(512, env.action_space.n)
    )


def ta2c(
        # modified from stable baselines hyperparameters
        clip_grad=0.1,
        discount_factor=0.99,
        lr=7e-4,    # RMSprop learning rate
        alpha=0.99, # RMSprop momentum decay
        eps=1e-4,   # RMSprop stability
        entropy_loss_scaling=0.01,
        value_loss_scaling=0.25,
        plus_loss_scaling=1e-9,
        feature_lr_scaling=1,
        n_envs=64,
        n_steps=5,
        final_discount_frame=100e6,
        device=torch.device("cpu"),
):
    def _ta2c(envs, writer=DummyWriter()):
        env = envs[0]

        feature_model = conv_features().to(device)
        value_model = value_net().to(device)
        plus_model = value_net().to(device)
        policy_model = policy_net(env).to(device)

        feature_optimizer = RMSprop(
            feature_model.parameters(),
            alpha=alpha,
            lr=lr * feature_lr_scaling / 4,
            eps=eps
        )
        value_optimizer = RMSprop(
            value_model.parameters(),
            alpha=alpha,
            lr=lr,
            eps=eps
        )
        plus_optimizer = RMSprop(
            plus_model.parameters(),
            alpha=alpha,
            lr=lr / 2,
            eps=eps
        )
        policy_optimizer = RMSprop(
            policy_model.parameters(),
            alpha=alpha,
            lr=lr / 3,
            eps=eps
        )

        features = FeatureNetwork(
            feature_model,
            feature_optimizer,
            clip_grad=clip_grad
        )
        v = VNetwork(
            value_model,
            value_optimizer,
            loss_scaling=value_loss_scaling,
            clip_grad=clip_grad,
            writer=writer
        )
        v_plus = VNetwork(
            plus_model,
            plus_optimizer,
            name='v_plus',
            loss_scaling=plus_loss_scaling,
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
            TA2C(
                features,
                v,
                v_plus,
                policy,
                n_envs=n_envs,
                n_steps=n_steps,
                discount_factor=discount_factor,
                final_discount_frame=final_discount_frame / 4 / n_envs,
                writer=writer
            ),
            envs,
        )

    return _ta2c, n_envs
