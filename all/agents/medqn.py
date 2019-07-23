import torch
from all.experiments import DummyWriter
from ._agent import Agent


class MEDQN(Agent):
    def __init__(self,
                 q,
                 replay_buffer,
                 discount_factor=0.99,
                 minibatch_size=32,
                 replay_start_size=5000,
                 update_frequency=1,
                 temperature=0.1,
                 writer=DummyWriter()
                 ):
        # objects
        self.q = q
        self.replay_buffer = replay_buffer
        self.writer = writer
        # hyperparameters
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # data
        self.env = None
        self.state = None
        self.action = None
        self.frames_seen = 0
        self.temperature = temperature

    def act(self, state, reward):
        self._store_transition(state, reward)
        self._train()
        self.state = state
        self.action = self._policy(state).sample()
        return self.action

    def _policy(self, state):
        return self._dist(self.q.eval(state))

    def _dist(self, q_values):
        probs = torch.nn.functional.softmax(q_values / self.temperature, dim=1)
        return torch.distributions.categorical.Categorical(probs=probs)

    def _store_transition(self, state, reward):
        if self.state and not self.state.done:
            self.frames_seen += 1
            self.replay_buffer.store(self.state, self.action, reward, state)

    def _train(self):
        if self._should_train():
            (states, actions, rewards, next_states, weights) = self.replay_buffer.sample(
                self.minibatch_size)
            next_q_values = self.q.eval(next_states)
            next_probs = torch.nn.functional.softmax(next_q_values / self.temperature, dim=1)
            next_policy = torch.distributions.categorical.Categorical(probs=next_probs)
            next_values = (self.discount_factor * next_q_values * next_probs).sum(dim=1)
            entropy = next_policy.entropy()

            self.writer.add_loss('entropy', entropy.mean())
            self.writer.add_scalar('v', next_values.mean())

            targets = rewards + next_values + self.temperature * entropy

            td_errors = targets - self.q(states, actions)
            self.q.reinforce(weights * td_errors)
            self.replay_buffer.update_priorities(td_errors)

    def _should_train(self):
        return (self.frames_seen > self.replay_start_size and
                self.frames_seen % self.update_frequency == 0)
