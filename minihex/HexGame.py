import gym
from gym import spaces
import numpy as np
from enum import IntEnum
from copy import deepcopy


class player(IntEnum):
    BLACK = 0
    WHITE = 1


class Side(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


def empty_tiles(board):
    coords = np.where(board[2, ...] == 1)
    idx = np.ravel_multi_index(coords, board.shape[1:])
    return idx


class HexGame(object):
    """
    Hex Game Environment.
    """

    def __init__(self, active_player, board, focus_player, connected_stones=None):
        self.board = board

        self.special_moves = IntEnum("SpecialMoves", {
            "RESIGN": self.board_size ** 2,
            "SWAP": self.board_size ** 2 + 1
        })

        if connected_stones is None:
            self.connected_stones = {
                player.WHITE: np.pad(np.zeros_like(self.board[0, ...]), 1),
                player.BLACK: np.pad(np.zeros_like(self.board[0, ...]), 1)
            }
            self.connected_stones[player.WHITE][:, 0] = 1
            self.connected_stones[player.BLACK][0, :] = 1

            self.active_player = player.WHITE
            self.flood_fill((0, 0))
            self.active_player = player.BLACK
            self.flood_fill((0, 0))
        else:
            self.connected_stones = connected_stones

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
        if action == self.special_moves.RESIGN:
            self.done = True
            self.winner = (self.active_player + 1) % 2
            return (self.active_player + 1) % 2

        if not self.is_valid_move(action):
            raise Exception(("Illegal move "
                             f"{self.action_to_coordinate(action)}"))

        coords = self.action_to_coordinate(action)
        self.board[(2, *coords)] = 0
        self.board[(self.active_player, *coords)] = 1

        winner = None
        position = self.action_to_coordinate(action)
        self.flood_fill(position)
        if self.active_player == player.BLACK:
            conn = self.connected_stones[player.BLACK]
            if np.any(conn[-2, :] == 1):
                self.done = True
                winner = player.BLACK
        else:
            conn = self.connected_stones[player.WHITE]
            if np.any(conn[:, -2] == 1):
                self.done = True
                winner = player.WHITE
        self.winner = winner

        if np.all(self.board[2, :, :] == 0):
            self.done = True

        self.active_player = (self.active_player + 1) % 2
        # if winner != None:
        #     import pdb; pdb.set_trace()
        return winner

    def coordinate_to_action(self, coords):
        return np.ravel_multi_index(coords, (self.board_size, self.board_size))

    def action_to_coordinate(self, action):
        return np.unravel_index(action, (self.board_size, self.board_size))

    def get_possible_actions(self):
        coords = np.where(self.board[2, ...] == 1)
        move_actions = self.coordinate_to_action(coords)
        return move_actions  #+ [self.special_moves.RESIGN]

    # @profile
    def flood_fill(self, position):
        board = np.zeros((self.board_size+2, self.board_size+2))
        board[1:self.board_size+1, 1:self.board_size+1] = self.board[self.active_player, ...]
        side = Side.NORTH if self.active_player == player.BLACK else Side.WEST
        connected_stones = self.connected_stones[self.active_player]

        connections = np.array([[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]])

        positions_to_test = [np.array(position) + np.array([1, 1])]
        while len(positions_to_test) > 0:
            current_position = positions_to_test.pop()
            current_position_tuple = tuple(current_position)

            if connected_stones[current_position_tuple] == 1:
                continue

            neighbour_positions = current_position[np.newaxis, ...] + connections
            ny = neighbour_positions[:, 0]
            nx = neighbour_positions[:, 1]
            adjacent_connections = connected_stones[ny, nx]
            adjacent_stones = board[ny, nx]

            connected_stones[current_position_tuple] = np.max(adjacent_connections)
            if connected_stones[current_position_tuple] == 1:
                neighbours_to_test = (adjacent_connections == 0) & (adjacent_stones == 1)
                for neighbour in neighbour_positions[neighbours_to_test]:
                    positions_to_test.append(neighbour)


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
        self.action_space = spaces.Discrete(board_size ** 2 + 1)

        if board is None:
            board = np.zeros((3, board_size+4, board_size+4))
            board[player.BLACK, :2, :] = 1
            board[player.BLACK, -2:, :] = 1
            board[player.WHITE, :, :2] = 1
            board[player.WHITE, :, -2:] = 1
            board[2, 2:-2, 2:-2] = 1

        self.initial_board = board
        self.active_player = active_player
        self.player = player_color
        self.simulator = None
        self.winner = None
        self.previous_opponent_move = None

        # cache initial connection matrix (approx +100 games/s)
        self.initial_connections = None


    @property
    def opponent(self):
        return (self.player + 1) % 2

    def reset(self):
        if self.initial_connections is None:
            self.simulator = HexGame(self.active_player,
                                     self.initial_board.copy(),
                                     self.player)
            con_copy = deepcopy(self.simulator.connected_stones)
            self.initial_connections = con_copy
            # import pdb; pdb.set_trace()
        else:
            con_copy = deepcopy(self.initial_connections)
            self.simulator = HexGame(self.active_player,
                                     self.initial_board.copy(),
                                     self.player,
                                     connected_stones=con_copy)

        if self.player != self.active_player:
            info_opponent = {
                'state': self.simulator.board,
                'last_move_opponent': action,
                'last_move_player': self.previous_opponent_move
            }
            self.opponent_move(info_opponent)
            # TODO: properly communicate previous moves

        return (self.simulator.board, self.active_player)

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
        board = self.simulator.board[:, 2:-2, 2:-2]
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
