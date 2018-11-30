from all.agents.agent import Agent

class GatedPolicyActorCritic(Agent):
    def __init__(self, v, policy, gate):
        self.v = v
        self.policy = policy
        self.gate = gate

        self.env = None
        self.opened_state = None
        self.gate_opened = False
        self.state = None
        self.action = None
        self.next_state = None
        self.accumulated_td_error = 0

    def new_episode(self, env):
        self.env = env
        self.gate_opened = False
        self.state = self.env.state
        self.opened_state = self.state
        self.action = self.policy.call(self.state)
        self.next_state = None
        self.accumulated_td_error = 0

    def act(self):
        self.gate_opened = self.gate.call(self.state, self.action)
        if self.gate_opened == 1:
            self.update_policy()
            self.opened_state = self.state
            self.action = self.policy.call(self.state)
        self.env.step(self.action)
        self.next_state = self.env.state
        self.update()
        self.state = self.next_state

    def update(self):
        td_error = (self.env.reward
            + (self.v.call(self.next_state)
                if not self.env.done else 0)
            - self.v.call(self.state))
        self.accumulated_td_error += td_error
        self.gate.update(td_error, self.state, self.action, self.gate_opened)

    def update_policy(self):
        self.policy.update(self.accumulated_td_error, self.opened_state, self.action)
        self.v.update(self.accumulated_td_error, self.opened_state)
        self.accumulated_td_error = 0
