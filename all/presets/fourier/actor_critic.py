from all.approximation.value.state import LinearStateValue
from all.approximation.bases import FourierBasis
# from all.approximation.traces import AccumulatingTraces
from all.policies import SoftmaxLinear
from all.agents import ActorCritic


def actor_critic(env, lr_v=0.001, lr_pi=0.001, order=1):
    num_actions = env.env.action_space.n
    basis = FourierBasis(env.env.observation_space, order)
    v = LinearStateValue(lr_v, basis)
    policy = SoftmaxLinear(lr_pi, basis, num_actions)
    return ActorCritic(v, policy)
