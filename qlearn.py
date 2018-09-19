import numpy as np
import random
import copy
from matplotlib import pyplot as plt

class QLearn:
    
    """params:
            agent: class, need following defined functions: 
                get_actions(), get_location(), get_state(), get_state_key(), get_value(), update_state(), update_location(), 
                display_state(), can_move(), move()
            environment: class, need following defined functions:
                update(), get_grid(), get_shape(), display()
            start: tuple, starting position for agent
       returns: none
       initializes class
    """
    def __init__(self, agent, environment, start):
        self.memorygrid = 255 * np.ones(environment.get_shape())
        self.agent = agent
        self.environment = environment
        self.grid_shape = environment.get_shape()
        self.moves = 0
        self.attempts = 0
        self.wins = 0
        self.start = start
        # TODO: check what typical values are for alpha and gamma
        self.alpha = 0.9
        self.gamma = 0.9
        self.epsilon = 1

    """ params: none
        returns: none
        runs q-learning algorithm, updates agent location and q-states, displays environment and relevant data
    """
    def learn(self):
        fig, ax = plt.subplots(2, 2)
        while (True):
            action = 0
            self.display(axis=ax)
            curr_location = self.agent.get_location()
            curr_state = self.agent.get_state(curr_location)
            if random.uniform(0, 1.0) > self.epsilon:  # exploitation
                action = np.argmax(curr_state)
            else:  # choose random action
                action = random.randint(0, len(self.agent.get_actions()) - 1)
            if self.agent.can_move(action):
                grid = self.environment.get_grid()
                next_location = self.agent.move(action)
                next_state = self.agent.get_state(next_location, self.environment)
                next_projection = max(next_state)
                next_reward = grid[next_location[0]][next_location[1]]
                current_key = self.agent.get_state_key(curr_location, self.environment)
                q = curr_state[action] + self.alpha * (next_reward + (self.gamma * next_projection) -
                                                                   curr_state[action])
                self.agent.update_state(current_key, action, q)
                self.display(axis=ax)
                self.memorygrid[curr_location[0]][curr_location[1]] *= 0.9
                self.agent.update_location(next_location)
                self.moves += 1
                if self.epsilon > 0.1: self.epsilon -= 0.01
                if grid[next_location[0]][next_location[1]] == DEATH:
                    self.agent.update_location(self.start)
                    self.attempts += 1
                    self.moves = 0
                # TODO: improve abstraction of success check
                if next_location[0] == 0:
                    self.wins += 1
                    self.attempts += 1
                    self.moves = 0
                    self.agent.update_location(self.start)
                self.environment.update()
                # plt.show()

    """ params: axis
        returns: none
        displays environment grid, memory grid, state, and other relevant visual information
    """
    def display(self, axis):
        plt.suptitle("Wins: " + str(self.wins) + "; Attempt: " + str(self.attempts) + "; Moves: " + str(
            self.moves) + "; Q-state: [" + ' '.join(
            str(round(e, 2)) for e in self.agent.get_state(self.agent.get_location())) + "]")
        axis[0, 0].imshow(self.environment.display(self.agent.get_location(), self.agent.get_value()))
        axis[0, 1].imshow(
            self.agent.display_state(self.agent.get_state_key(self.agent.get_location(), self.environment)))
        axis[1, 0].imshow(self.memorygrid)
        plt.savefig(fname=str(self.attempts) + "_" + str(self.moves))
        # plt.pause(0.001)


class Frogger_Agent:
    
    """params:
            actions: list, set of all possible actions for agent
            location: tuple, initial agent XY location
       returns: none
       initializes agent
    """
    def __init__(self, actions, location):
        self.actions = actions
        self.location = location
        self.states = {}
        for i in range(0, 256):
            self.states[str('{0:08b}'.format(i))] = [0, 0, 0, 0]
        self.value = 25

    # returns a copy of the agent's actions
    def get_actions(self):
        return copy.deepcopy(self.actions)

    # returns a copy of the agent's location as a tuple
    def get_location(self):
        return copy.deepcopy(self.location)

    # returns the key associated with the current location and environment
    def get_state_key(self, location, environment):
        key = ""
        grid = environment.get_grid()
        grid_shape = np.shape(grid)
        key += str(int(grid[location[0] - 1][location[1] - 1] < 0))
        key += str(int(grid[location[0] - 1][location[1]] < 0))
        key += str(int(grid[location[0] - 1][(location[1] + 1) % grid_shape[1]] < 0))
        key += str(int(grid[location[0]][location[1] - 1] < 0))
        key += str(int(grid[location[0]][(location[1] + 1) % grid_shape[1]] < 0))
        key += str(int(grid[(location[0] + 1) % grid_shape[0]][location[1] - 1] < 0))
        key += str(int(grid[(location[0] + 1) % grid_shape[0]][location[1]] < 0))
        key += str(int(grid[(location[0] + 1) % grid_shape[0]][(location[1] + 1) % grid_shape[1]] < 0))
        return key

    # returns the state associated with the current location and environment
    def get_state(self, location, environment):
        key = self.get_state_key(location, environment)
        return copy.deepcopy(self.states[key])

    # returns a copy of the agent's value
    def get_value(self):
        return copy.deepcopy(self.value)

    # returns a visual representation of the key state
    def display_state(self, key):
        state_display = np.zeros(shape=(3, 3))
        state_display[0][0] = int(key[0])
        state_display[0][1] = int(key[1])
        state_display[0][2] = int(key[2])
        state_display[1][0] = int(key[3])
        state_display[1][2] = int(key[4])
        state_display[2][0] = int(key[5])
        state_display[2][1] = int(key[6])
        state_display[2][2] = int(key[7])
        state_display *= DEATH
        state_display[1][1] = self.value
        return state_display

    # updates the agent's location
    def update_location(self, location):
        self.location = location

    # updates the action in the state specified by key to the given value
    def update_state(self, key, action, value):
        self.states[key][action] = value
        
    # returns True if the action, given the environment and current agent location, is permissible; False otherwise
    def can_move(self, action, environment):
        grid = environment.get_grid()
        if action == 0 and self.location[0] <= 0: return False
        if action == 1 and self.location[0] >= np.shape(grid)[0] - 1: return False
        if action == 2 and (self.location[1] <= 0 or (self.location[0] % 2 == 0 and
                                                              grid[self.location[0]][self.location[1]] == DEATH)):
            return False
        if action == 3 and (self.location[1] >= np.shape(grid)[1] - 1 or (self.location[0] % 2 > 0 and
                                                                              grid[self.location[0]]
                                                                              [self.location[1]]) == DEATH):
            return False
        return True

    # returns a tuple: the location determined by taking the specified action from the agent's current location
    def move(self, action):
        if action == 0:
            return (self.location[0] - 1, self.location[1])
        elif action == 1:
            return (self.location[0] + 1, self.location[1])
        elif action == 2:
            return (self.location[0], self.location[1] - 1)
        else:
            return (self.location[0], self.location[1] + 1)


class Frogger_Environment:

    """params:
            shape: tuple, shape of environm
    """
    def __init__(self, shape):
        self.grid = np.zeros(shape=shape)
        self.grid[0] = 200
        for row in range(1, np.shape(self.grid)[0] - 1):  # exclude bottom & top row
            self.grid[row] += (np.shape(self.grid)[0] - 1 - row)
            self.grid[row][random.randint(0, np.shape(self.grid)[1] - 1)] = DEATH

    # updates the rows with cars 
    def update(self):
        for row in range(1, np.shape(self.grid)[0] - 1):
            if row % 2 == 0:
                self.grid[row] = np.roll(self.grid[row], 1)
            else:
                self.grid[row] = np.roll(self.grid[row], -1)

    # returns a copy of the environment grid
    def get_grid(self):
        return copy.deepcopy(self.grid)

    # returns the shape of the environment grid as a tuple
    def get_shape(self):
        return np.shape(self.grid)

    # returns a display of the environmnet grid
    def display(self, agent_loc, agent_val):
        # board display shows cars one step behind for purposes of state identification vs visual updates
        board = self.get_grid()
        for row in range(1, np.shape(self.grid)[0] - 1):
            if row % 2 == 0:
                board[row] = np.roll(board[row], -1)
            else:
                board[row] = np.roll(board[row], 1)
        board[agent_loc[0]][agent_loc[1]] = agent_val
        return board


DEATH = -100
SUCCESS = 255

plt.gray()

import os
os.chdir("D:/frogger")

ENVIRONMENT_WIDTH = 5
ENVIRONMENT_HEIGHT = 5

START_X = int(ENVIRONMENT_WIDTH / 2)
START_Y = ENVIRONMENT_HEIGHT - 1

start = (START_X, START_Y)
shape = (ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)

frogger_q = QLearn(agent=Frogger_Agent(actions=['up', 'down', 'left', 'right'], location=start),
               environment=Frogger_Environment(shape=shape),
               start=start)
frogger_q.learn()
