# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all.agents.pppo import PPPO
from all.approximation import VNetwork, FeatureNetwork, PolyakTarget
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import fc_relu_features, fc_policy_head, fc_value_head

def pppo(
        clip_grad=0.1,
        discount_factor=0.99,
        entropy_loss_scaling=0.001,
        lr=1e-3,
        epochs=4,
        n_envs=8,
        n_steps=8,
        polyak=0.001,
        device=torch.device('cpu')
):
    def _pppo(envs, writer=DummyWriter()):
        env = envs[0]
        feature_model = fc_relu_features(env).to(device)
        value_model = fc_value_head().to(device)
        policy_model = fc_policy_head(env).to(device)

        feature_optimizer = Adam(feature_model.parameters(), lr=lr)
        value_optimizer = Adam(value_model.parameters(), lr=lr)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr)

        features = FeatureNetwork(
            feature_model,
            feature_optimizer,
            clip_grad=clip_grad,
            target=PolyakTarget(polyak)
        )
        v = VNetwork(
            value_model,
            value_optimizer,
            clip_grad=clip_grad,
            writer=writer,
            target=PolyakTarget(polyak)
        )
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer,
            target=PolyakTarget(polyak)
        )
        return PPPO(
            features,
            v,
            policy,
            epochs=epochs,
            n_envs=n_envs,
            n_steps=n_steps,
            discount_factor=discount_factor
        )
    return _pppo, n_envs

__all__ = ["pppo"]
