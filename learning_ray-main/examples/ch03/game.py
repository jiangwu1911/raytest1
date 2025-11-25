from maze import Environment
import time

environment = Environment()

while not environment.is_done():
    random_action = environment.action_space.sample()
    environment.step(random_action)
    time.sleep(0.1)
    environment.render()
