# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all.agents import GPG
from all.approximation import GNetwork, FeatureNetwork
from all.experiments import DummyWriter
from all.policies import SoftmaxPolicy
from .models import fc_relu_features, fc_policy_head, fc_value_head


def gpg(
        clip_grad=0,
        entropy_loss_scaling=0.001,
        gamma=0.99,
        lr_g=1e-2,
        lr_pi=3e-3,
        lr_fe=1e-3,
        min_batch_size=500,
        device=torch.device('cpu')
):
    def _gpg(env, writer=DummyWriter()):
        feature_model = fc_relu_features(env).to(device)
        value_model = fc_value_head(hidden=65).to(device) # include extra reward parameter
        policy_model = fc_policy_head(env).to(device)

        value_optimizer = Adam(value_model.parameters(), lr=lr_g)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        feature_optimizer = Adam(feature_model.parameters(), lr=lr_fe)

        features = FeatureNetwork(
            feature_model, feature_optimizer, clip_grad=clip_grad)
        g = GNetwork(
            value_model,
            value_optimizer,
            clip_grad=clip_grad,
            writer=writer
        )
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer
        )
        return GPG(features, g, policy, gamma=gamma, min_batch_size=min_batch_size, writer=writer)
    return _gpg


__all__ = ["gpg"]
