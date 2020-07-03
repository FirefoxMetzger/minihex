import gym
import minihex
from tqdm import tqdm

env = gym.make("hex-v0",
               opponent_policy=minihex.random_policy,
               board_size=5)


for _ in tqdm(range(10000)):
    state, info = env.reset()
    done = False
    while not done:
        board, player = state
        action = minihex.random_policy(board, player, info)
        state, reward, done, info = env.step(action)
