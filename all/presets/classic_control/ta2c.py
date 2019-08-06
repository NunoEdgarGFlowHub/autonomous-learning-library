# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all.agents import TA2C
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import fc_relu_features, fc_policy_head, fc_value_head


def ta2c(
        clip_grad=0.1,
        discount_factor=0.99,
        entropy_loss_scaling=0.001,
        lr=3e-3,
        n_envs=4,
        n_steps=32,
        device=torch.device('cpu')
):
    def _ta2c(envs, writer=DummyWriter()):
        env = envs[0]
        feature_model = fc_relu_features(env).to(device)
        value_model = fc_value_head().to(device)
        plus_model = fc_value_head().to(device)
        policy_model = fc_policy_head(env).to(device)

        feature_optimizer = Adam(feature_model.parameters(), lr=lr)
        value_optimizer = Adam(value_model.parameters(), lr=lr)
        plus_optimizer = Adam(value_model.parameters(), lr=lr)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr)

        features = FeatureNetwork(
            feature_model, feature_optimizer, clip_grad=clip_grad)
        v = VNetwork(
            value_model,
            value_optimizer,
            clip_grad=clip_grad,
            writer=writer
        )
        plus = VNetwork(
            plus_model,
            plus_optimizer,
            clip_grad=clip_grad,
            writer=writer,
            name="plus"
        )
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer
        )
        return TA2C(
            features,
            v,
            plus,
            policy,
            n_envs=n_envs,
            n_steps=n_steps,
            discount_factor=discount_factor
        )
    return _ta2c, n_envs


__all__ = ["ta2c"]
