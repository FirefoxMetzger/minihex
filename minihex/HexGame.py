import gym
from gym import spaces
import numpy as np
from enum import IntEnum
from copy import deepcopy


class player(IntEnum):
    BLACK = 0
    WHITE = 1


def empty_tiles(board):
    coords = np.where(board[2, ...] == 1)
    idx = np.ravel_multi_index(coords, board.shape[1:])
    return idx


class HexGame(object):
    """
    Hex Game Environment.
    """

    def __init__(self, active_player, board,
                 focus_player, connected_stones=None):
        self.board = board
        # track number of empty feelds for speed
        self.empty_fields = np.count_nonzero(board[2, ...])

        # self.special_moves = IntEnum("SpecialMoves", {
        #     "RESIGN": self.board_size ** 2,
        #     "SWAP": self.board_size ** 2 + 1
        # })

        if connected_stones is None:
            self.regions = {
                player.BLACK: np.pad(np.zeros_like(self.board[0, ...]), 1),
                player.WHITE: np.pad(np.zeros_like(self.board[0, ...]), 1)
            }
            self.regions[player.WHITE][:, 0] = 1
            self.regions[player.BLACK][0, :] = 1
            self.regions[player.WHITE][:, self.board_size + 1] = 2
            self.regions[player.BLACK][self.board_size + 1, :] = 2
        else:
            self.regions = connected_stones

        self.region_counter = {
            player.BLACK: np.max(self.regions[player.BLACK]) + 1,
            player.WHITE: np.max(self.regions[player.WHITE]) + 1
        }

        if connected_stones is None:
            self.active_player = player.WHITE
            for y, row in enumerate(board[player.WHITE, ...]):
                for x, value in enumerate(row):
                    if value == 1:
                        self.flood_fill((y, x))

            self.active_player = player.BLACK
            for y, row in enumerate(board[player.BLACK, ...]):
                for x, value in enumerate(row):
                    if value == 1:
                        self.flood_fill((y, x))

        self.active_player = active_player
        self.player = focus_player
        self.done = False
        self.winner = None

    @property
    def board_size(self):
        return self.board.shape[1]

    def is_valid_move(self, action):
        coords = self.action_to_coordinate(action)
        if self.board[2, coords[0], coords[1]] == 1:
            return True
        else:
            return False

    def make_move(self, action):
        # # currently resigning is not a possible option
        # if action == self.special_moves.RESIGN:
        #     self.done = True
        #     self.winner = (self.active_player + 1) % 2
        #     return (self.active_player + 1) % 2

        if not self.is_valid_move(action):
            raise Exception(("Illegal move "
                             f"{self.action_to_coordinate(action)}"))

        position = self.action_to_coordinate(action)
        self.board[(2, *position)] = 0
        self.board[(self.active_player, *position)] = 1
        self.empty_fields -= 1

        winner = None
        self.flood_fill(position)

        regions = self.regions[self.active_player]
        if regions[-1, -1] == 1:
            self.done = True
            winner = self.active_player
        self.winner = winner

        if self.empty_fields <= 0:
            self.done = True
            winner = None

        self.active_player = (self.active_player + 1) % 2
        return winner

    def coordinate_to_action(self, coords):
        return np.ravel_multi_index(coords, (self.board_size, self.board_size))

    def action_to_coordinate(self, action):
        return np.unravel_index(action, (self.board_size, self.board_size))

    def get_possible_actions(self):
        coords = np.where(self.board[2, ...] == 1)
        move_actions = self.coordinate_to_action(coords)
        return move_actions

    def flood_fill(self, position):
        regions = self.regions[self.active_player]

        y, x = (position[0] + 1, position[1] + 1)
        neighborhood = regions[(y - 1):(y + 2), (x - 1):(x + 2)].copy()
        neighborhood[0, 0] = 0
        neighborhood[2, 2] = 0
        adjacent_regions = sorted(set(neighborhood.flatten().tolist()))

        # region label = 0 is always present, but not a region
        adjacent_regions.pop(0)

        if len(adjacent_regions) == 0:
            regions[y, x] = self.region_counter[self.active_player]
            self.region_counter[self.active_player] += 1
        else:
            new_region_label = adjacent_regions.pop(0)
            regions[y, x] = new_region_label
            for label in adjacent_regions:
                regions[regions == label] = new_region_label


class HexEnv(gym.Env):
    """
    Hex environment. Play against a fixed opponent.
    """

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, opponent_policy,
                 player_color=player.BLACK,
                 active_player=player.BLACK,
                 board=None,
                 board_size=5):
        self.opponent_policy = opponent_policy
        self.action_space = spaces.Discrete(board_size ** 2)

        if board is None:
            board = np.zeros((3, board_size, board_size))
            board[2, ...] = 1

        self.initial_board = board
        self.active_player = active_player
        self.player = player_color
        self.simulator = None
        self.winner = None
        self.previous_opponent_move = None

        # cache initial connection matrix (approx +100 games/s)
        self.initial_regions = None

    @property
    def opponent(self):
        return (self.player + 1) % 2

    def reset(self):
        if self.initial_regions is None:
            self.simulator = HexGame(self.active_player,
                                     self.initial_board.copy(),
                                     self.player)
            regions = deepcopy(self.simulator.regions)
            self.initial_regions = regions
        else:
            regions = deepcopy(self.initial_regions)
            self.simulator = HexGame(self.active_player,
                                     self.initial_board.copy(),
                                     self.player,
                                     connected_stones=regions)

        self.previous_opponent_move = None

        if self.player != self.active_player:
            info_opponent = {
                'state': self.simulator.board,
                'last_move_opponent': None,
                'last_move_player': None
            }
            self.opponent_move(info_opponent)

        info = {
            'state': self.simulator.board,
            'last_move_opponent': self.previous_opponent_move,
            'last_move_player': None
        }

        return (self.simulator.board, self.active_player), info

    def step(self, action):
        if not self.simulator.done:
            self.winner = self.simulator.make_move(action)

        opponent_action = None

        if not self.simulator.done:
            info_opponent = {
                'state': self.simulator.board,
                'last_move_opponent': action,
                'last_move_player': self.previous_opponent_move
            }
            opponent_action = self.opponent_move(info_opponent)

        if self.winner == self.player:
            reward = 1
        elif self.winner == self.opponent:
            reward = -1
        else:
            reward = 0

        info = {
            'state': self.simulator.board,
            'last_move_opponent': opponent_action,
            'last_move_player': action
        }

        return ((self.simulator.board, self.active_player), reward,
                self.simulator.done, info)

    def render(self, mode='ansi', close=False):
        board = self.simulator.board
        print(" " * 6, end="")
        for j in range(board.shape[1]):
            print(" ", j + 1, " ", end="")
            print("|", end="")
        print("")
        print(" " * 5, end="")
        print("-" * (board.shape[1] * 6 - 1), end="")
        print("")
        for i in range(board.shape[1]):
            print(" " * (1 + i * 3), i + 1, " ", end="")
            print("|", end="")
            for j in range(board.shape[1]):
                if board[2, i, j] == 1:
                    print("  O  ", end="")
                elif board[0, i, j] == 1:
                    print("  B  ", end="")
                else:
                    print("  W  ", end="")
                print("|", end="")
            print("")
            print(" " * (i * 3 + 1), end="")
            print("-" * (board.shape[1] * 7 - 1), end="")
            print("")

    def opponent_move(self, info):
        opponent_action = self.opponent_policy(self.simulator.board,
                                               self.opponent,
                                               info)
        self.winner = self.simulator.make_move(opponent_action)
        self.previous_opponent_move = opponent_action
        return opponent_action
