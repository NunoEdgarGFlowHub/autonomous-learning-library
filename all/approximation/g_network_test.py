import unittest
import torch
from torch import nn
import torch_testing as tt
from all.approximation.g_network import GNetwork
from all.environments import State

STATE_DIM = 2


class TestVNetwork(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(nn.Linear(STATE_DIM + 1, 1))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.g = GNetwork(self.model, optimizer)

    def test_reinforce_list(self):
        states = State(torch.randn(5, STATE_DIM), mask=torch.tensor([1, 1, 0, 1, 0]))
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        result = self.g(states, rewards)
        tt.assert_almost_equal(
            result, torch.tensor([0.2520876, 0.3541654, 0.0, 0.2201912, 0.0])
        )
        self.g.reinforce(torch.tensor([1, -1, 1, 1, 1]).float())
        result = self.g(states, rewards)
        tt.assert_almost_equal(
            result, torch.tensor([0.4323483, 0.4017012, 0.0, 0.3950523, 0.0])
        )

if __name__ == "__main__":
    unittest.main()
