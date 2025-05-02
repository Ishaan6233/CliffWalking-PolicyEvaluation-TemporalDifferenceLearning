import numpy as np
from rl_glue import RLGlue
from Agent import BaseAgent
from Environment import BaseEnvironment
from manager import Manager

# -------------------------------
# Environment
# -------------------------------

class CliffWalkEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        self.grid_h = env_info.get("grid_height", 4)
        self.grid_w = env_info.get("grid_width", 12)
        self.start_loc = (self.grid_h - 1, 0)
        self.goal_loc = (self.grid_h - 1, self.grid_w - 1)
        self.cliff = [(self.grid_h - 1, i) for i in range(1, self.grid_w - 1)]
        self.reward_state_term = (None, None, None)

    def state(self, loc):
        x, y = loc
        return self.grid_w * x + y

    def env_start(self):
        self.agent_loc = self.start_loc
        state = self.state(self.agent_loc)
        self.reward_state_term = (0, state, False)
        return state

    def env_step(self, action):
        x, y = self.agent_loc
        if action == 0: x -= 1
        elif action == 1: y -= 1
        elif action == 2: x += 1
        elif action == 3: y += 1

        if not (0 <= x < self.grid_h and 0 <= y < self.grid_w):
            x, y = self.agent_loc

        self.agent_loc = (x, y)
        reward = -1
        terminal = False

        if self.agent_loc in self.cliff:
            reward = -100
            self.agent_loc = self.start_loc
        elif self.agent_loc == self.goal_loc:
            terminal = True

        self.reward_state_term = (reward, self.state(self.agent_loc), terminal)
        return self.reward_state_term

    def env_cleanup(self):
        self.agent_loc = self.start_loc


# -------------------------------
# Agent
# -------------------------------

class TDAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.policy = agent_info.get("policy")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")
        self.values = np.zeros((self.policy.shape[0],))

    def agent_start(self, state):
        self.last_state = state
        return self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])

    def agent_step(self, reward, state):
        target = reward + self.discount * self.values[state]
        self.values[self.last_state] += self.step_size * (target - self.values[self.last_state])
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.last_state = state
        return action

    def agent_end(self, reward):
        target = reward
        self.values[self.last_state] += self.step_size * (target - self.values[self.last_state])

    def agent_cleanup(self):
        self.last_state = None

    def agent_message(self, message):
        if message == "get_values":
            return self.values
        else:
            raise Exception("TDAgent.agent_message(): Message not understood!")


# -------------------------------
# Experiment Runner
# -------------------------------

def run_experiment(env_info, agent_info, num_episodes=5000, experiment_name=None, plot_freq=100, true_values_file=None):
    rl_glue = RLGlue(CliffWalkEnvironment, TDAgent)
    rl_glue.rl_init(agent_info, env_info)
    manager = Manager(env_info, agent_info, true_values_file=true_values_file, experiment_name=experiment_name)

    for episode in range(1, num_episodes + 1):
        rl_glue.rl_episode(0)
        if episode % plot_freq == 0:
            values = rl_glue.agent.agent_message("get_values")
            manager.visualize(values, episode)

    return rl_glue.agent.agent_message("get_values")


# -------------------------------
# Example Policy and Experiment
# -------------------------------

if __name__ == "__main__":
    env_info = {"grid_height": 4, "grid_width": 12, "seed": 0}
    agent_info = {"discount": 1, "step_size": 0.01, "seed": 0}

    policy = np.ones((48, 4)) * 0.25
    policy[36] = [1, 0, 0, 0]
    for i in range(24, 35):
        policy[i] = [0, 0, 0, 1]
    policy[35] = [0, 0, 1, 0]

    agent_info.update({"policy": policy})
    run_experiment(env_info, agent_info, num_episodes=5000, experiment_name="TD Policy Evaluation")
