import unittest
import torch
from torch import nn
import torch_testing as tt
from all.environments import State
from all.approximation.q_network import QNetwork


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self._shape = shape

    def forward(self, x):
        return x.view(self._shape)


STATE_DIM = 2
REWARD_DIM = 2
ACTIONS = 3

# pylint: disable=bad-whitespace
class TestMultiDimQNetwork(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, ACTIONS * REWARD_DIM), View((-1, ACTIONS, REWARD_DIM))
        )
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.q = QNetwork(self.model, optimizer, ACTIONS)

    def test_forward(self):
        state = State(torch.randn(2, STATE_DIM))
        results = self.q(state)
        expected = torch.tensor([
            [
                [-0.2371,  0.4967],
                [-0.0580,  0.2703],
                [ 0.1734, -0.4500]
            ], [
                [-0.0341,  0.5139],
                [-0.3381,  1.0650],
                [ 1.0020, -0.5296]
            ]
        ])
        tt.assert_almost_equal(results, expected, decimal=3)

    def test_mask(self):
        state = State(torch.randn(2, STATE_DIM), torch.tensor([0, 1]))
        results = self.q(state)
        expected = torch.tensor([
            [
                [0, 0],
                [0, 0],
                [0, 0]
            ], [
                [-0.0341,  0.5139],
                [-0.3381,  1.0650],
                [ 1.0020, -0.5296]
            ]
        ])
        tt.assert_almost_equal(results, expected, decimal=3)

    def test_choose_action(self):
        state = State(torch.randn(2, STATE_DIM))
        actions = torch.tensor([1, 2])
        results = self.q(state, actions)
        expected = torch.tensor([
            [-0.0580,  0.2703],
            [ 1.0020, -0.5296]
        ])
        tt.assert_almost_equal(results, expected, decimal=3)

    def test_backwards(self):
        # generate initial output
        state = State(torch.randn(3, STATE_DIM))
        actions = torch.tensor([0, 1, 2])
        out = self.q.eval(state)
        # create eligibility and update
        self.q(state, actions)
        self.q.reinforce(torch.tensor([[1, 1], [0, 0], [-1, -1]]).float())
        # compare new output
        new_out = self.q.eval(state)
        expected_diff = torch.tensor(
            [
                [[0.0390, 0.0390], [0.0000, 0.0000], [-0.0206, -0.0206]],
                [[0.0368, 0.0368], [0.0000, 0.0000], [-0.0866, -0.0866]],
                [[0.0206, 0.0206], [0.0000, 0.0000], [-0.1213, -0.1213]],
            ]
        )
        tt.assert_almost_equal(new_out - out, expected_diff, decimal = 3)


if __name__ == "__main__":
    unittest.main()
