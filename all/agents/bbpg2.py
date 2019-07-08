import torch
import numpy as np
from all.experiments import DummyWriter
from .abstract import Agent

class BBPG2(Agent):
    def __init__(
            self,
            policy,
            lr=0.01,
            n_episodes=1,
            writer=DummyWriter()
    ):
        self.policy = policy
        self.lr = lr
        self.n_episodes = n_episodes
        self._rewards = []
        self._trajectories = []
        self._gamma = 1
        self._actions_taken = 0
        self._mean_returns = 0
        self._writer = writer

    def act(self, state, reward):
        if self._actions_taken == 0:
            return self._initial(state)
        if not state.done:
            return self._act(state, reward)
        return self._terminal(reward)

    def _initial(self, state):
        self._rewards = []
        self._actions_taken = 1
        return self.policy(state)

    def _act(self, state, reward):
        self._rewards.append(reward)
        self._actions_taken += 1
        return self.policy(state)

    def _terminal(self, reward):
        self._rewards.append(reward)
        rewards = torch.tensor(self._rewards, device=self.policy.device)
        self._trajectories.append((self._actions_taken, rewards))
        self._actions_taken = 0
        if len(self._trajectories) >= self.n_episodes:
            self._train()

    def _train(self):
        advantages = torch.cat([
            self._compute_advantages(actions_taken, rewards)
            for (actions_taken, rewards)
            in self._trajectories
        ])
        self._writer.add_loss('value', advantages.var().item())
        mean_returns = np.mean([rewards.sum().item() for (actions_taken, rewards) in self._trajectories])
        self._mean_returns += self.lr * (mean_returns - self._mean_returns)
        self.policy.reinforce(advantages)
        self._trajectories = []

    def _compute_advantages(self, _, rewards):
        returns = self._compute_discounted_returns(rewards)
        values = self._compute_expected_returns(rewards)
        return returns - values

    def _compute_discounted_returns(self, rewards):
        returns = rewards.clone()
        t = len(returns) - 1
        discounted_return = 0
        for reward in torch.flip(rewards, dims=(0,)):
            discounted_return = reward + self._gamma * discounted_return
            returns[t] = discounted_return
            t -= 1
        return returns

    def _compute_expected_returns(self, rewards):
        expected_returns = rewards.clone()
        expected_return = self._mean_returns
        for t, reward in enumerate(rewards):
            expected_returns[t] = expected_return
            expected_return -= reward
        return expected_returns
