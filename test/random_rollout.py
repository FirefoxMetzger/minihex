import gym
import minihex
import numpy as np


def random_policy(state):
    # never surrender :)
    idx = minihex.empty_tiles(state)
    choice = np.random.randint(len(idx))
    return idx[choice]


env = gym.make("hex-v0", opponent_policy=random_policy)
state = env.reset()

done = False
while not done:
    action = random_policy(state)
    state, reward, done, _ = env.step(action)

env.render()

if reward == -1:
    print("Player (Black) Lost")
elif reward == 1:
    print("Player (Black) Won")
else:
    print("Draw")
