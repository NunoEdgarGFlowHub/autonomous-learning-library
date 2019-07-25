import torch
from all.environments import State
from ._agent import Agent

class GPG(Agent):
    '''Returns Policy Gradient'''
    def __init__(
            self,
            features,
            g,
            policy,
            gamma=0.99,
            # run complete episodes until we have
            # seen at least min_batch_size states
            min_batch_size=1,
            writer=None
    ):
        self.features = features
        self.g = g
        self.policy = policy
        self.gamma = gamma
        self.min_batch_size = min_batch_size
        self._current_batch_size = 0
        self._trajectories = []
        self._features = []
        self._rewards = []
        self._writer = writer

    def act(self, state, reward):
        if not self._features:
            return self._initial(state)
        if not state.done:
            return self._act(state, reward)
        return self._terminal(reward)

    def _initial(self, state):
        features = self.features(state)
        self._features = [features.features]
        return self.policy(features)

    def _act(self, state, reward):
        features = self.features(state)
        self._features.append(features.features)
        self._rewards.append(reward)
        return self.policy(features)

    def _terminal(self, reward):
        self._rewards.append(reward)
        features = torch.cat(self._features)
        rewards = torch.tensor(self._rewards, device=features.device)
        self._trajectories.append((features, rewards))
        self._current_batch_size += len(features)
        self._features = []
        self._rewards = []

        if self._current_batch_size >= self.min_batch_size:
            self._train()

    def _train(self):
        advantages = torch.cat([
            self._compute_advantages(features, rewards)
            for (features, rewards)
            in self._trajectories
        ])
        self.g.reinforce(advantages)
        self.policy.reinforce(advantages)
        self.features.reinforce()
        self._trajectories = []
        self._current_batch_size = 0

    def _compute_advantages(self, features, rewards):
        returns = rewards.sum()
        rewards_so_far = rewards.cumsum(0)
        rewards_so_far[1:] = rewards_so_far[0:-1]
        rewards_so_far[0] = 0
        values = self.g(State(features), rewards_so_far)
        self._writer.add_scalar('rewards_so_far', rewards_so_far.mean())
        self._writer.add_scalar('values', values.mean())
        self._writer.add_scalar('returns', returns.mean())
        return returns - values

    def _compute_discounted_returns(self, rewards):
        returns = rewards.clone()
        t = len(returns) - 1
        discounted_return = 0
        for reward in torch.flip(rewards, dims=(0,)):
            discounted_return = reward + self.gamma * discounted_return
            returns[t] = discounted_return
            t -= 1
        return returns
