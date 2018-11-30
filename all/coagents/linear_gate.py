import numpy as np
from all.policies.policy import Policy

class LinearGate(Policy):
    def __init__(self, learning_rate, basis, action_space):
        self.learning_rate = learning_rate
        self.basis = basis
        self.weights = np.zeros((action_space.n, 2, self.basis.num_features))

    def call(self, state, action):
        features = self.basis.features(state)
        probabilities = self.probabilities(features, action)
        return np.random.choice(probabilities.shape[0], p=probabilities)

    def update(self, error, state, action, gate_opened):
        self.weights[action] += self.learning_rate * error * self.gradient(state, action, gate_opened)

    def gradient(self, state, action, gate_opened):
        features = self.basis.features(state)
        neg_probabilities = -self.probabilities(features, action)
        neg_probabilities[gate_opened] += 1
        return np.outer(neg_probabilities, features)

    def apply(self, gradient):
        self.weights += self.learning_rate * gradient

    @property
    def parameters(self):
        return self.weights

    @parameters.setter
    def parameters(self, parameters):
        self.weights = parameters

    def probabilities(self, features, action):
        action_scores = np.exp(self.weights[action].dot(features))
        return action_scores / np.sum(action_scores)
