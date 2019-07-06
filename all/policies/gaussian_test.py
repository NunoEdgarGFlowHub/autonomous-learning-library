import unittest
import torch
from torch import nn
import torch_testing as tt
from all.environments import State
from all.policies import GaussianPolicy

STATE_DIM = 2
ACTION_DIM = 3

class TestSoftmax(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, ACTION_DIM * 2)
        )
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.01)
        self.policy = GaussianPolicy(self.model, optimizer, ACTION_DIM)

    def test_output_shape(self):
        state = State(torch.randn(1, STATE_DIM))
        action = self.policy(state)
        self.assertEqual(action.shape, (1, ACTION_DIM))
        state = State(torch.randn(5, STATE_DIM))
        action = self.policy(state)
        self.assertEqual(action.shape, (5, ACTION_DIM))

    def test_reinforce_one(self):
        state = State(torch.randn(1, STATE_DIM))
        self.policy(state)
        self.policy.reinforce(torch.tensor([1]).float())

    def test_converge(self):
        state = State(torch.randn(1, STATE_DIM))
        target = torch.tensor([1., 2., -1.])

        for _ in range(0, 1000):
            action = self.policy(state)
            loss = torch.abs(target - action).mean()
            self.policy.reinforce(-loss)

        self.assertTrue(loss < 1)

    # def test_multi_action(self):
    #     states = State(torch.randn(3, STATE_DIM))
    #     actions = self.policy(states)
    #     tt.assert_equal(actions, torch.tensor([2, 2, 0]))
    #     self.policy.reinforce(torch.tensor([[1, 2, 3]]).float())

    # def test_multi_batch_reinforce(self):
    #     self.policy(State(torch.randn(2, STATE_DIM)))
    #     self.policy(State(torch.randn(2, STATE_DIM)))
    #     self.policy(State(torch.randn(2, STATE_DIM)))
    #     self.policy.reinforce(torch.tensor([1, 2, 3, 4]).float())
    #     self.policy.reinforce(torch.tensor([1, 2]).float())
    #     with self.assertRaises(Exception):
    #         self.policy.reinforce(torch.tensor([1, 2]).float())

    # def test_list(self):
    #     torch.manual_seed(1)
    #     states = State(torch.randn(3, STATE_DIM), torch.tensor([1, 0, 1]))
    #     actions = self.policy(states)
    #     tt.assert_equal(actions, torch.tensor([1, 2, 1]))
    #     self.policy.reinforce(torch.tensor([[1, 2, 3]]).float())

    # def test_action_prob(self):
    #     torch.manual_seed(1)
    #     states = State(torch.randn(3, STATE_DIM), torch.tensor([1, 0, 1]))
    #     with torch.no_grad():
    #         actions = self.policy(states)
    #     probs = self.policy(states, action=actions)
    #     tt.assert_almost_equal(probs, torch.tensor([0.204, 0.333, 0.217]), decimal=3)


if __name__ == '__main__':
    unittest.main()
