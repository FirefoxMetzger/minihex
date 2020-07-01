from minihex.HexGame import player as player
from gym.envs.registration import register
from minihex.HexGame import HexGame
import numpy as np


def random_policy(board, player, info):
    coords = np.where(board[2, ...] == 1)
    idx = np.ravel_multi_index(coords, board.shape[1:])
    choice = np.random.randint(len(idx))
    return idx[choice]


register(
    id='hex-v0',
    entry_point='minihex.HexGame:HexEnv'
)
