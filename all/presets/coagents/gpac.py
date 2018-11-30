from all.approximation.value.state import LinearStateValue
from all.approximation.bases import FourierBasis
from all.policies import SoftmaxLinear
from all.agents import ActorCritic
from all.coagents.gated_policy_actor_critic import GatedPolicyActorCritic
from all.coagents.linear_gate import LinearGate


def gated_policy_actor_critic(env, alpha=0.001, order=1):
    num_actions = env.action_space.n
    basis = FourierBasis(env.env.observation_space, order)
    v = LinearStateValue(alpha, basis)
    policy = SoftmaxLinear(alpha, basis, num_actions)
    gate = LinearGate(alpha, basis, env.action_space)
    return GatedPolicyActorCritic(v, policy, gate)
