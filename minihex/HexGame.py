import gym
from gym import spaces
import numpy as np
from gym import error
from enum import IntEnum


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

    def __init__(self, active_player, board, focus_player):
        self.board = board

        self.special_moves = IntEnum("SpecialMoves", {
            "RESIGN": self.board_size ** 2,
            "SWAP": self.board_size ** 2 + 1
        })

        self.connected_stones = {
            player.WHITE: np.zeros_like(self.board[0, ...]),
            player.BLACK: np.zeros_like(self.board[0, ...])
        }
        self.always_connected = np.zeros((len(Side), *board[0, ...].shape))
        self.always_connected[Side.NORTH, 0, :] = 1
        self.always_connected[Side.EAST, :, -1] = 1
        self.always_connected[Side.SOUTH, -1, :] = 1
        self.always_connected[Side.WEST, :, 0] = 1

        self.active_player = player.WHITE
        self.flood_fill((0, 0))
        self.active_player = player.BLACK
        self.flood_fill((0, 0))

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
            return (self.active_player + 1) % 2

        if not self.is_valid_move(action):
            raise Exception(("Illegal move "
                             f"{self.action_to_coordinate(action)}"))

        coords = self.action_to_coordinate(action)
        self.board[(2, *coords)] = 0
        self.board[(self.active_player, *coords)] = 1

        winner = self.update_front(action)
        if np.all(self.board[2, :, :] == 0):
            self.done = True
        self.active_player = (self.active_player + 1) % 2
        self.winner = winner
        return winner

    def coordinate_to_action(self, coords):
        return np.ravel_multi_index(coords, (self.board_size, self.board_size))

    def action_to_coordinate(self, action):
        return np.unravel_index(action, (self.board_size, self.board_size))

    def get_possible_actions(self):
        coords = np.where(self.board[2, ...] == 1)
        move_actions = self.coordinate_to_action(coords)
        return move_actions  #+ [self.special_moves.RESIGN]

    def update_front(self, action):
        position = self.action_to_coordinate(action)
        self.flood_fill(position)
        if self.active_player == player.BLACK:
            conn = self.connected_stones[player.BLACK]
            if np.any(conn[-1, :] == 1):
                return player.BLACK
        else:
            conn = self.connected_stones[player.WHITE]
            if np.any(conn[:, -1] == 1):
                return player.WHITE
        return None

    def neighbours(self, position):
        max_val = self.board_size - 1
        if position[0] == 0 and position[1] == 0:
            connections = np.array([[0, 1], [1, 0]])
        elif position[0] == max_val and position[1] == max_val:
            connections = np.array([[-1, 0], [0, -1]])
        elif position[0] == 0 and position[1] == max_val:
            connections = np.array([[0, -1], [1, -1], [1, 0]])
        elif position[0] == max_val and position[1] == 0:
            connections = np.array([[-1, 0], [-1, 1], [0, 1]])
        elif position[0] == 0:
            connections = np.array([[0, -1], [0, 1], [1, -1], [1, 0]])
        elif position[0] == max_val:
            connections = np.array([[-1, 0], [-1, 1], [0, -1], [0, 1]])
        elif position[1] == 0:
            connections = np.array([[-1, 0], [-1, 1], [0, 1], [1, 0]])
        elif position[1] == max_val:
            connections = np.array([[-1, 0], [0, -1], [1, -1], [1, 0]])
        else:
            connections = np.array([[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]])
        return connections

    def flood_fill(self, position):
        board = self.board[self.active_player, ...]
        side = Side.NORTH if self.active_player == player.BLACK else Side.WEST
        connected_stones = self.connected_stones[self.active_player]
        connections = self.neighbours(position)
        always_connected = self.always_connected[side, ...]

        positions_to_test = [np.array(position)]
        while len(positions_to_test) > 0:
            current_position = positions_to_test.pop()
            current_position_tuple = tuple(current_position)

            if board[current_position_tuple] == 0:
                continue

            if connected_stones[current_position_tuple] == 1:
                continue

            neighbour_positions = current_position[np.newaxis, ...] + self.neighbours(current_position)
            ny = neighbour_positions[:, 0]
            nx = neighbour_positions[:, 1]
            adjacent_connections = connected_stones[ny, nx]
            adjacent_stones = board[ny, nx]

            if (np.any(adjacent_connections == 1) or
                    always_connected[current_position_tuple] == 1):
                connected_stones[current_position_tuple] = 1
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

    @property
    def opponent(self):
        return (self.player + 1) % 2

    def reset(self):
        self.simulator = HexGame(self.active_player,
                                 self.initial_board.copy(),
                                 self.player)

        if self.player != self.active_player:
            self.opponent_move()

        return (self.simulator.board, self.active_player)

    def step(self, action):
        if not self.simulator.done:
            self.winner = self.simulator.make_move(action)

        opponent_action = None
        if not self.simulator.done:
            opponent_action = self.opponent_move()

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
                self.simulator.done, {'state': self.simulator.board})

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

    def opponent_move(self):
        opponent_action = self.opponent_policy(self.simulator.board,
                                               self.opponent)
        self.winner = self.simulator.make_move(opponent_action)
        return opponent_action
