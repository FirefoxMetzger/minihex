import gym
from gym import spaces
import numpy as np
from gym import error
from enum import IntEnum


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

    def __init__(self, active_player, board, player):
        self.board = board

        self.front = {player.WHITE: set((y, x)
                                        for y in range(self.board_size)
                                        for x in range(2)),
                      player.BLACK: set((y, x)
                                        for y in range(2)
                                        for x in range(self.board_size))}
        self.active_player = active_player
        self.player = player
        self.done = False

    @property
    def board_size(self):
        return self.board.shape[1]

    def is_reisgn_move(self, action):
        return action == self.board.shape[1] ** 2

    def is_valid_move(self, action):
        coords = self.action_to_coordinate(action)
        if self.board[2, coords[0], coords[1]] == 1:
            return True
        else:
            return False

    def make_move(self, action):
        if self.is_reisgn_move(action):
            self.done = True
            return (self.active_player + 1) % 2

        if not self.is_valid_move(action):
            raise Exception(("Illegal move "
                             f"{self.action_to_coordinate(action)}"))

        coords = self.action_to_coordinate(action)
        self.board[(2, *coords)] = 0
        self.board[(self.active_player, *coords)] = 1

        winner = self.update_front(action)
        self.active_player = (self.active_player + 1) % 2
        return winner

    def coordinate_to_action(self, coords):
        return np.ravel_multi_index(coords, (self.board_size, self.board_size))

    def action_to_coordinate(self, action):
        return np.unravel_index(action, (self.board_size, self.board_size))

    def get_possible_actions(self):
        coords = np.where(self.board[2, ...] == 1)
        return self.coordinate_to_action(coords)

    def update_front(self, action):
        active_player = self.active_player
        connections = np.array([[-1, -1,  0,  0,  1,  1],
                                [0,   1, -1,  1, -1,  0]])

        position = np.array(self.action_to_coordinate(action))

        positions_to_test = [position]
        while len(positions_to_test) > 0:
            current_position = positions_to_test.pop()

            if self.board[(active_player, *position)] == 0:
                continue

            if tuple(current_position) in self.front[active_player]:
                continue

            neighbours = list()
            check_neighbours = False
            for direction in connections.T:
                neighbour_position = current_position + direction

                if neighbour_position[0] >= self.board_size:
                    if active_player == player.WHITE:
                        self.done = True
                        return player.WHITE
                    else:
                        continue

                if neighbour_position[1] >= self.board_size:
                    if active_player == player.BLACK:
                        self.done = True
                        return player.BLACK
                    else:
                        continue

                if np.any(neighbour_position < 0):
                    continue

                if self.board[(active_player, *neighbour_position)] == 1:
                    neighbours.append(neighbour_position)

                if tuple(neighbour_position) in self.front[active_player]:
                    self.front[active_player].add(tuple(current_position))
                    check_neighbours = True

            if check_neighbours:
                positions_to_test += neighbours

        if np.all(self.board[2, :, :] == 0):
            self.done = True
        return None

    # function for introspection
    def front_array(self, player):
        front_array = np.zeros((self.board_size, self.board_size))
        for pos in self.front[player]:
            front_array[pos] = 1.0
        return front_array


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

        return self.simulator.board

    def step(self, action):
        if not self.simulator.done:
            self.winner = self.simulator.make_move(action)

        if not self.simulator.done:
            action = self.opponent_policy(self.simulator.board)
            self.winner = self.simulator.make_move(action)

        if self.winner == self.player:
            reward = 1
        elif self.winner == self.opponent:
            reward = -1
        else:
            reward = 0

        return (self.simulator.board, reward,
                self.simulator.done, {'state': self.simulator.board})

    def render(self, mode='ascii', close=False):
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
