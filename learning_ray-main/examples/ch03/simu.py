from maze import Environment
from maze import Policy
from maze import Simulation

environment = Environment()

untrained_policy = Policy(environment)
sim = Simulation(environment)

exp = sim.rollout(untrained_policy, render=True, epsilon=1.0)
for row in untrained_policy.state_action_table:
    print(row)
