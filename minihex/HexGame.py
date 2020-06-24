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


class HexEnv(gym.Env):
    """
    Hex environment. Play against a fixed opponent.
    """

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, opponent_policy,
                 player_color=player.BLACK,
                 board_size=5):

        self.board_size = board_size+4
        self.playable_board_size = board_size
        self.player_color = player_color
        self.opponent_policy = opponent_policy

        self.action_space = spaces.Discrete(self.board_size ** 2 + 1)

        # state space
        self.board = np.zeros((3, self.board_size, self.board_size))
        self.move_count = 0
        self.to_play_color = player.BLACK

        # speed optimization
        self.front = {player.BLACK: set(), player.WHITE: set()}

    def reset(self):
        self.board = np.zeros((3, self.board_size, self.board_size))
        self.board[player.BLACK, :2, :] = 1
        self.board[player.BLACK, -2:, :] = 1
        self.board[player.WHITE, :, :2] = 1
        self.board[player.WHITE, :, -2:] = 1
        self.board[2, 2:-2, 2:-2] = 1

        self.front = {player.BLACK: set(), player.WHITE: set()}
        self.front[player.BLACK] = set((y, x)
                                       for x in range(self.board_size)
                                       for y in [0, 1])
        self.front[player.WHITE] = set((y, x)
                                       for x in [0, 1]
                                       for y in range(self.board_size))

        self.to_play_color = player.BLACK
        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play_color:
            self.opponent_move()

        return self.board

    def step(self, action):
        reward = self.make_move(action)

        if not self.done:
            action = self.opponent_policy(self.board)
            reward = self.make_move(action)

        return self.board, reward, self.done, {'state': self.board}

    def render(self, mode='asi', close=False):
        board = self.board
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

    def is_reisgn_move(self, action):
        return action == self.board_size ** 2

    def is_valid_move(self, action):
        coords = self.action_to_coordinate(action)
        if self.board[2, coords[0], coords[1]] == 1:
            return True
        else:
            return False

    def make_move(self, action):
        if self.is_reisgn_move(action):
            self.done = True
            if self.to_play_color == self.player_color:
                return self.board, 1, True, {'state': self.board}
            else:
                return self.board, -1, True, {'state': self.board}

        if not self.is_valid_move(action):
            if self.to_play_color != self.player_color:
                raise Exception(("Opponent policy played illegal move"
                                 f" {self.action_to_coordinate(action)}"))
            else:
                raise Exception(("Illegal move "
                                 f"{self.action_to_coordinate(action)}"))

        coords = self.action_to_coordinate(action)
        self.board[(2, *coords)] = 0
        self.board[(self.to_play_color, *coords)] = 1

        reward = self.game_finished(action)
        self.to_play_color = (self.to_play_color + 1) % 2
        return reward

    def coordinate_to_action(self, coords):
        return np.ravel_multi_index(coords, (self.board_size, self.board_size))

    def action_to_coordinate(self, action):
        return np.unravel_index(action, (self.board_size, self.board_size))

    def get_possible_actions(self):
        coords = np.where(self.board[2, ...] == 1)
        return self.coordinate_to_action(coords)

    def game_finished(self, action):
        active_player = self.to_play_color
        connections = np.array([[-1, -1,  0,  0,  1,  1],
                                [0,   1, -1,  1, -1,  0]])

        position = np.array(self.action_to_coordinate(action))

        positions_to_test = [position]
        while len(positions_to_test) > 0:
            current_position = positions_to_test.pop()

            if tuple(current_position) in self.front[active_player]:
                continue

            neighbours = list()
            check_neighbours = False
            for direction in connections.T:
                neighbour_position = current_position + direction

                if neighbour_position[0] >= self.board_size:
                    if active_player == player.WHITE:
                        self.done = True
                        return 1
                    else:
                        continue

                if neighbour_position[1] >= self.board_size:
                    if active_player == player.BLACK:
                        self.done = True
                        return -1
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
        return 0

    def front_array(self, player):
        front_array = np.zeros((self.board_size, self.board_size))
        for pos in self.front[player]:
            front_array[pos] = 1.0
        return front_array


if __name__ == "__main__":
    game = HexEnv()
    action_space = game.action_space

    def random_policy(state):
        possible_moves = game.get_possible_actions()
        action_idx = np.random.randint(len(possible_moves))
        return possible_moves[action_idx]

    game.opponent_policy = random_policy

    import tqdm
    for rollout in tqdm.tqdm(range(2000)):
        state = game.reset()
        done = False
        while not done:
            action = random_policy(state)
            state, _, done, _ = game.step(action)
