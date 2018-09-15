import numpy as np
import copy
import random
from matplotlib import pyplot as plt


class Frogger:
    def __init__(self, nrows, ncols, is_win=False):
        self.nrows = nrows
        self.ncols = ncols
        self.is_win = is_win
        self.frog = 50
        self.car = 255
        self.nmoves = 0
        self.time = 0
        self.attempts = []
        self.location = (self.nrows - 1, int(self.ncols / 2))
        self.grid = np.zeros((self.nrows, self.ncols))
        # set cars...
        for row in range(1, self.nrows - 1):  # exclude bottom & top row
            self.grid[row][random.randint(0, self.ncols - 1)] = self.car

    def update_grid(self):
        # TODO: evaluate if frog has landed on a car space
        # TODO: don't want frog to roll w/ cars during updates -- don't actually place in grid
        if self.grid[self.location[0]][self.location[1]] == self.car:
            self.attempts.append((self.time, self.nmoves))
            self.time = 0
            self.nmoves = 0
            self.location = (self.nrows - 1, int(self.ncols / 2))
        self.time += 1
        for row in range(1, self.nrows - 1):
            if row % 2 == 0:
                self.grid[row] = np.roll(self.grid[row], 1)
            else:
                self.grid[row] = np.roll(self.grid[row], -1)

    def can_move(self, direction):
        if self.is_win: return False
        if direction == 'down' and self.location[0] >= self.nrows - 1: return False
        if direction == 'left' and self.location[1] <= 0: return False
        if direction == 'right' and self.location[1] >= self.ncols - 1: return False
        return True

    def move(self, direction):
        self.nmoves += 1
        if direction == 'down':
            self.location[0] += 1
        elif direction == 'up':
            self.location[0] -= 1
        elif direction == 'right':
            self.location[1] += 1
        else:
            self.location[1] -= 1

    """ for showing frog on board """

    def display(self, axis, pause_time):
        board = copy.deepcopy(self.grid)
        board[self.location[0]][self.location[1]] = 50
        axis.imshow(board)
        axis.set_title("Attempt: " + str(len(self.attempts)) + "; Time: " + str(copy.deepcopy(self.time)) + "; Moves " +
                       str(copy.deepcopy(self.nmoves)))
        plt.pause(pause_time)


""" From CSE 415 """


class Operator:
    def __init__(self, name, precond, state_transf):
        self.name = name
        self.precond = precond
        self.state_transf = state_transf

    def is_applicable(self, s):
        return self.precond(s)

    def apply(self, s):
        return self.state_transf(s)


movements = ['up', 'down', 'left', 'right']
OPERATORS = [Operator("Move " + movement,
                      lambda s, m=movement: s.can_move(m),
                      lambda s, m=movement: s.move(m))
             for movement in movements]

frogger = Frogger(nrows=10, ncols=10)
fig, ax = plt.subplots()
while True:
    frogger.update_grid()
    frogger.display(axis=ax, pause_time=0.05)
plt.show()
