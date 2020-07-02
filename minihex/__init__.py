from minihex.HexGame import player
from gym.envs.registration import register
from minihex.HexGame import HexGame
import numpy as np
import random


def random_policy(board, player, info):
    actions = np.arange(board.shape[0] * board.shape[1])
    valid_actions = actions[board.flatten() == player.EMPTY]
    choice = int(random.random() * len(valid_actions))
    return valid_actions[choice]


register(
    id='hex-v0',
    entry_point='minihex.HexGame:HexEnv'
)
