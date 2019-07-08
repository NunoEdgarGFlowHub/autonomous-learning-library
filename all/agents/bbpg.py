import torch
import numpy as np
from all.experiments import DummyWriter
from .abstract import Agent

class BBPG(Agent):
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
        self._trajectories = []
        self._actions_taken = 0
        self._rewards = 0
        self._mean_returns = 0
        self._writer = writer

    def act(self, state, reward):
        if self._actions_taken == 0:
            return self._initial(state)
        if not state.done:
            return self._act(state, reward)
        return self._terminal(reward)

    def _initial(self, state):
        self._rewards = 0
        self._actions_taken = 1
        return self.policy(state)

    def _act(self, state, reward):
        self._rewards += reward
        self._actions_taken += 1
        return self.policy(state)

    def _terminal(self, reward):
        self._rewards += reward
        self._trajectories.append((self._actions_taken, self._rewards))
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
        mean_returns = np.mean([rewards for (actions_taken, rewards) in self._trajectories])
        self._mean_returns += self.lr * (mean_returns - self._mean_returns)
        self.policy.reinforce(advantages)
        self._trajectories = []

    def _compute_advantages(self, actions_taken, returns):
        advantage = returns - self._mean_returns
        return torch.tensor([]).new_full(
            (actions_taken,),
            advantage,
            device=self.policy.device,
            dtype=torch.float
        )
