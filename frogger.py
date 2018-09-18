import numpy as np
import random
import copy
from matplotlib import pyplot as plt

CAR = -100
WIN = 200

class QLearn:
    """params
        grid: np.array, map of environment/states
        actions: list, actions agent can take in each state
            assumes actions>=4, and first four indeces: [up,down,left,right,...]
        start: tuple, starting position
        states: dictionary of all states in the environment and their corresponding Q-states
    """

    def __init__(self, environment, actions, states, start):
        self.memorygrid = 255*np.ones(environment.get_shape())
        self.environment = environment
        self.grid_shape = environment.get_shape()
        self.actions = actions
        self.moves = 0
        self.attempts = 0
        self.wins = 0
        self.location = self.start = start
        self.states = states
        #self.table = np.ndarray(shape=(self.grid_shape[0],self.grid_shape[1],len(self.actions)))
        #for i in range(self.grid_shape[0]):
        #    for j in range(self.grid_shape[1]):
        #        self.table[i][j] = len(self.actions) * [0]
        # TODO: check what typical values are for alpha and gamma
        self.state = self.get_state(self.location)
        self.alpha = 0.9
        self.gamma = 0.9
        self.epsilon = 1

    def get_state_key(self, location):
        key = ""
        grid = self.environment.get_grid()
        key+=str(int(grid[location[0]-1][location[1]-1] < 0))
        key+=str(int(grid[location[0] - 1][location[1]] < 0))
        key+=str(int(grid[location[0] - 1][(location[1] + 1)%self.grid_shape[1]] < 0))
        key+=str(int(grid[location[0]][location[1] - 1] < 0))
        key+=str(int(grid[location[0]][(location[1] + 1)%self.grid_shape[1]] < 0))
        key+=str(int(grid[(location[0]+1)%self.grid_shape[0]][location[1] - 1] < 0))
        key+=str(int(grid[(location[0]+1)%self.grid_shape[0]][location[1]] < 0))
        key+=str(int(grid[(location[0]+1)%self.grid_shape[0]][(location[1] + 1)%self.grid_shape[1]] < 0))
        return key

    def get_state(self, location):
        key = self.get_state_key(location)
        #print("STATE: " + str(self.states[key]))
        return copy.deepcopy(self.states[key])
        #return copy.deepcopy(self.table[location[0]][location[1]])

    def can_move(self, action):
        # if self.is_win: return False
        if action == 0 and self.location[0] <= 0: return False
        if action == 1 and self.location[0] >= self.grid_shape[0] - 1: return False
        if action == 2 and self.location[1] <= 0: return False
        if action == 3 and self.location[1] >= self.grid_shape[1] - 1: return False
        return True

    def move(self, action):
        # self.moves += 1
        if action == 0:
            return (self.location[0] - 1, self.location[1])
        elif action == 1:
            return (self.location[0] + 1, self.location[1])
        elif action == 2:
            return (self.location[0], self.location[1] - 1)
        else:
            return (self.location[0], self.location[1] + 1)

    """params
        location: tuple, location in state table
        action:, integer, corresponding to index in actions
    """

    # TODO: fix -> the way next states are evaluated contradicts grid updates!!
    def learn(self):
        fig,ax = plt.subplots()
        while(True):
            action = 0
            self.display(axis=ax)
            if random.uniform(0, 1.0) > self.epsilon:  # exploitation
                action = np.argmax(self.get_state(self.location))#[0]
            else:  # choose random action
                action = random.randint(0, len(self.actions)-1)
            #print(self.actions[action])
            if self.can_move(action):
                # q function
                grid = self.environment.get_grid()
                next_location = self.move(action)
                next_state = self.get_state(next_location)
                next_projection = max(next_state)
                next_reward = grid[next_location[0]][next_location[1]]
                current_key = self.get_state_key(self.location)
                self.states[current_key][action] = self.state[action] + self.alpha * (
                    next_reward + (self.gamma * next_projection) - self.state[action])
                self.display(axis=ax)
                self.memorygrid[self.location[0]][self.location[1]] *= 0.9
                self.location = next_location
                self.state = next_state
                self.moves += 1
                if self.epsilon > 0.1: self.epsilon -= 0.01
                # update to see if collision with car
                self.environment.update()
                if self.environment.get_grid()[self.location[0]][self.location[1]] == CAR:
                    self.location = self.start
                    self.state = self.get_state(self.location)
                    self.attempts += 1
                    self.moves = 0
                if self.location[0] == 0:
                    self.wins += 1
                    self.attempts += 1
                    self.moves = 0
                    self.location = self.start
                    self.state = self.get_state(self.location)
        plt.show()

    def display(self, axis):
        board = self.environment.get_grid()
        board[self.location[0]][self.location[1]] = 25
        axis.imshow(board)
        plt.suptitle("Wins: " + str(self.wins) + "; Attempt: " + str(self.attempts) + "; Moves: " + str(self.moves))# + "; Q-state: " +
                     #str(self.state))#' '.join(str(e) for e in self.state) + "]")
        plt.pause(0.001)

class Frogger_Environment:

    def __init__(self, shape):
        self.grid = np.ones(shape=shape)
        self.grid[0] *= 200
        for row in range(1, np.shape(self.grid)[0] - 1):  # exclude bottom & top row
            self.grid[row][random.randint(0, np.shape(self.grid)[1] - 1)] = -100

    def update(self):
        for row in range(1, np.shape(self.grid)[0] - 1):
            if row % 2 == 0:
                self.grid[row] = np.roll(self.grid[row], 1)
            else:
                self.grid[row] = np.roll(self.grid[row], -1)

    def get_grid(self):
        return copy.deepcopy(self.grid)

    def get_shape(self):
        return np.shape(self.grid)

frogger_states = {}
for i in range(0, 256):
    frogger_states[str('{0:08b}'.format(i))] = [0,0,0,0]

#print(frogger_states.keys())

agent = QLearn(environment=Frogger_Environment(shape=(5,5)), actions=['up', 'down', 'left', 'right'],
               start=(4, 2), states=frogger_states)
agent.learn()
