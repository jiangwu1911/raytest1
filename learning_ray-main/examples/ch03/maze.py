import random
import os
import time
import numpy as np

class Discrete:
    def __init__(self, num_actions: int):
        """ Discrete action space for num_actions.
        Discrete(4) can be used as encoding moving in
        one of the cardinal directions.
        """
        self.n = num_actions

    def sample(self):
        return random.randint(0, self.n - 1)


class Environment:
    def __init__(self,  *args, **kwargs):
        self.seeker, self.goal = (0, 0), (4, 4)
        self.info = {'seeker': self.seeker, 'goal': self.goal}

        self.action_space = Discrete(4)
        self.observation_space = Discrete(5*5)

    def reset(self):
        """Reset seeker position and return observations."""
        self.seeker = (0, 0)

        return self.get_observation()

    def get_observation(self):
        """Encode the seeker position as integer"""
        return 5 * self.seeker[0] + self.seeker[1]

    def get_reward(self):
        """Reward finding the goal"""
        return 1 if self.seeker == self.goal else 0

    def is_done(self):
        """We're done if we found the goal"""
        return self.seeker == self.goal

    def step(self, action):
        """Take a step in a direction and return all available information."""
        if action == 0:  # move down
            self.seeker = (min(self.seeker[0] + 1, 4), self.seeker[1])
        elif action == 1:  # move left
            self.seeker = (self.seeker[0], max(self.seeker[1] - 1, 0))
        elif action == 2:  # move up
            self.seeker = (max(self.seeker[0] - 1, 0), self.seeker[1])
        elif action == 3:  # move right
            self.seeker = (self.seeker[0], min(self.seeker[1] + 1, 4))
        else:
            raise ValueError("Invalid action")

        obs = self.get_observation()
        rew = self.get_reward()
        done = self.is_done()
        return obs, rew, done, self.info

    def render(self, *args, **kwargs):
        """We override this method here so clear the output in Jupyter notebooks.
        The previous implementation works well in the terminal, but does not clear
        the screen in interactive environments.
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except Exception:
            pass
        grid = [['| ' for _ in range(5)] + ["|\n"] for _ in range(5)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.seeker[0]][self.seeker[1]] = '|S'
        print(''.join([''.join(grid_row) for grid_row in grid]))


class Policy:
    def __init__(self, env):
        """A Policy suggests actions based on the current state.
        We do this by tracking the value of each state-action pair.
        """
        self.state_action_table = [
            [0 for _ in range(env.action_space.n)]
            for _ in range(env.observation_space.n)
        ]
        self.action_space = env.action_space

    def get_action(self, state, explore=True, epsilon=0.1):
        """Explore randomly or exploit the best value currently available."""
        if explore and random.uniform(0, 1) < epsilon:
            return self.action_space.sample()
        return np.argmax(self.state_action_table[state])


class Simulation(object):
    def __init__(self, env):
        """Simulates rollouts of an environment, given a policy to follow."""
        self.env = env

    def rollout(self, policy, render=False, explore=True, epsilon=0.1):
        """Returns experiences for a policy rollout."""
        experiences = []
        state = self.env.reset()
        done = False
        while not done:
            action = policy.get_action(state, explore, epsilon)
            next_state, reward, done, info = self.env.step(action)
            experiences.append([state, action, reward, next_state])
            state = next_state
            if render:
                time.sleep(0.05)
                self.env.render()

        return experiences
