import numpy as np
import random
import copy

class QLearn:
    """params
        grid: np.array, map of environment/states
        actions: list, actions agent can take in each state
            assumes actions>=4, and first four indeces: [up,down,left,right,...]
        start: tuple, starting position
    """
    def __init__(self, grid, actions, start):
        self.grid = grid
        self.actions = actions
        self.moves = 0
        self.attempts = 0
        self.location = self.start = start
        self.table = np.array([])
        for i in range(np.shape(grid)[0]):
            for j in range(np.shape(grid)[1]):
                self.table[i][j] = len(self.actions)*[0]
        # TODO: check what typical values are for alpha and gamma
        self.alpha = 0.9
        self.gamma = 0.9
        self.epsilon = 1

    def can_move(self, action):
        # if self.is_win: return False
        if action == 0 and self.location[0] <= 0: return False
        if action == 1 and self.location[0] >= self.nrows - 1: return False
        if action == 2 and self.location[1] <= 0: return False
        if action == 3 and self.location[1] >= self.ncols - 1: return False
        return True

    def move(self, action):
        # self.moves += 1
        if action == 0:
            return (self.location[0]-1, self.location[1])
        elif action == 1:
            return (self.location[0]+1, self.location[1])
        elif action == 2:
            return (self.location[0], self.location[1]-1)
        else:
            return (self.location[0], self.location[1]+1)

    """params
        location: tuple, location in state table
        action:, integer, corresponding to index in actions
    """
    def q_func(self, current_val, action):
        if can_move(action):
            next_location = move(action)
            next_state = get_state(next_location)
            next_projection = max(next_state)
            next_reward = self.grid[next_location[0]][next_location[1]]
            return current_val + self.alpha*(next_reward+(self.gamma*next_projection)-current_val)
        return current_val

    def get_state(self, location):
        return copy.deepcopy(self.table[location[0]][location[1]])

    def learn(self):
        if random.randuniform(0, 1.0) > self.epsilon: # exploitation
            action = get_state().index(max(get_state()))
        else: # choose random action
            action = random.randint(0, len(self.actions))
        self.table[location[0]][location[1]] = q_func(get_state()[action], action) # update Q(s,a)

    def display(self):
        return 0

DEATH = -100
SUCCESS = 100
TILE = 1

my_grid = np.array([[TILE, DEATH, SUCCESS],
                    [TILE, TILE, TILE],
                    [TILE, TILE, DEATH]])