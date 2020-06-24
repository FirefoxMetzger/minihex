from minihex.HexGame import player as player
from minihex.HexGame import empty_tiles
from gym.envs.registration import register


# def random_policy(state):
#     possible_moves = game.get_possible_actions()
#     action_idx = np.random.randint(len(possible_moves))
#     return possible_moves[action_idx]


register(
    id='hex-v0',
    entry_point='minihex.HexGame:HexEnv'
)
