import gym
import minihex


env = gym.make("hex-v0",
               opponent_policy=minihex.random_policy,
               board_size=11)

import tqdm
for _ in tqdm.tqdm(range(1000)):
    state, info = env.reset()
    done = False
    while not done:
        board, player = state
        action = minihex.random_policy(board, player, info)
        state, reward, done, info = env.step(action)

env.render()

if reward == -1:
    print("Player (Black) Lost")
elif reward == 1:
    print("Player (Black) Won")
else:
    print("Draw")
