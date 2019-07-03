import numpy as np
import random
import copy
from matplotlib import pyplot as plt
import os
import sys

class QLearn:
    """params:
            agent: class, need following defined functions: 
                get_actions(), get_location(), get_state(), get_state_key(), get_value(), update_state(), update_location(), 
                display_state(), can_move(), move(), has_failed(), has_succeeded()
            environment: class, need following defined functions:
                update(), get_grid(), get_shape(), display()
            start: tuple, starting position for agent
       returns: none
       initializes class
    """

    def __init__(self, agent, environment, start, alpha=0.9, gamma=0.9, epsilon=1):
        self.memorygrid = 255 * np.ones(environment.get_shape())
        self.agent = agent
        self.environment = environment
        self.grid_shape = environment.get_shape()
        self.moves = 0
        self.attempts = 0
        self.wins = 0
        self.start = start
        # TODO: check what typical values are for alpha and gamma
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    """ params: none
        returns: none
        runs q-learning algorithm, updates agent location and q-states, displays environment and relevant data
    """

    # << description >>
    # measurement has 3 valid types: "attempts", "successes", "wins"
    def learn(self, limit, save_plots=False):
        history = np.zeros(shape=(limit+1, 3))
        while self.moves < limit:
            action = 0
            curr_location = self.agent.get_location()
            curr_state = self.agent.get_state(curr_location, self.environment)
            if random.uniform(0, 1.0) > self.epsilon:  # exploitation
                action = np.argmax(curr_state)
            else:  # choose random action
                action = random.randint(0, len(self.agent.get_actions()) - 1)
            if self.agent.can_move(action, self.environment):
                grid = self.environment.get_grid()
                next_location = self.agent.move(action)
                next_state = self.agent.get_state(next_location, self.environment)
                next_projection = max(next_state)
                next_reward = grid[next_location[0]][next_location[1]]
                current_key = self.agent.get_state_key(curr_location, self.environment)
                q = curr_state[action] + self.alpha * (next_reward + (self.gamma * next_projection) -
                                                       curr_state[action])
                self.agent.update_state(current_key, action, q)
                if (save_plots):
                    print("Moves: {}".format(self.moves))
                    fig, ax = plt.subplots(1,3)
                    self.display(axis=ax)
                self.memorygrid[curr_location[0]][curr_location[1]] *= 0.9
                self.agent.update_location(next_location)
                self.moves += 1
                if self.epsilon > 0.1: self.epsilon -= 0.01
                if self.agent.has_failed(self.environment):
                    self.agent.update_location(self.start)
                    self.attempts += 1
                    #history[self.attempts] = [self.moves, self.attempts, self.wins]
                    #self.moves = 0
                if self.agent.has_succeeded(self.environment):
                    self.wins += 1
                    self.attempts += 1
                    #self.moves = 0
                    self.agent.update_location(self.start)
                    #history[self.attempts] = [self.moves, self.attempts, self.wins]
                history[self.moves] = [self.moves, self.attempts, self.wins]
                self.environment.update()
                # plt.show()
        plt.close()
        return history

    """ params: axis
        returns: none
        displays environment grid, memory grid, state, and other relevant visual information
    """

    def display(self, axis):
        plt.suptitle("Alpha = {:.2f}; Gamma = {:.2f}; Epsilon = {:.2f}\nWins: {} Attempt: {} Moves: {}".format(self.alpha, self.gamma, self.epsilon, self.wins, self.attempts, self.moves))
        axis[0].imshow(self.environment.display(self.agent.get_location(), self.agent.get_value()))
        axis[0].set_title("Environment")
        axis[1].imshow(
            self.agent.display_state(self.agent.get_state_key(self.agent.get_location(), self.environment)))
        axis[1].set_title("'Optimal' Next State")
        axis[2].imshow(self.memorygrid)
        axis[2].set_title("Heat Map")
        plt.savefig(fname=str(self.moves))
        plt.close()
        # plt.pause(0.001)

    def reset_params(self, new_alpha=None, new_gamma=None):
        self.moves = 0
        self.attempts = 0
        self.wins = 0
        self.epsilon = 1
        if new_alpha:
            self.alpha = new_alpha
        if new_gamma:
            self.gamma = new_gamma


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
    
    # returns True if agent's location is same as car's location on environment grid; False otherwise
    def has_failed(self, environment):
        return environment.get_grid()[self.location[0]][self.location[1]] == DEATH
        
    # returns True if agent has made it to top row of environment grid; False otherwise
    def has_succeeded(self, environment):
        return self.location[0] == 0


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
ENVIRONMENT_WIDTH = 5
ENVIRONMENT_HEIGHT = 5
START_X = int(ENVIRONMENT_WIDTH / 2)
START_Y = ENVIRONMENT_HEIGHT - 1
IMG_SAVE_DIR = "./Frogger Examples/"

def plot_histories(history_list, hyperparam_list, hyperparam_str, control_value, fig_name):
    colors = ['red','orange','yellow', 'green','blue','cyan', 'purple','black','brown', 'pink', 'lime', 'magenta']
    fig, ax = plt.subplots(1, 2)
    if hyperparam_str == "alpha":
        title = "Gamma = {:.2f}".format(control_value)
    else:
        title = "Alpha = {:.2f}".format(control_value)
    plt.suptitle(title)
    ax[0].set_title("Successes vs Moves")
    ax[1].set_title("Ratio of Successes to Attempts vs Moves")
    ax[0].set_xlabel("Moves")
    ax[1].set_xlabel("Moves")
    ax[0].set_ylabel("Successes")
    ax[1].set_ylabel("Successes:Attempts")
    for i in range(len(history_list)):
        label = "{} = {}".format(hyperparam_str, hyperparam_list[i])
        color = colors[i%len(colors)]
        history = history_list[i]
        ax[0].plot(history[:,0], history[:,2], label=label, color=color)
        ax[1].plot(history[:,0], history[:,2]/history[:,1], label=label, color=color)
    plt.legend()
    plt.savefig(fname=fig_name,format='jpg')
    plt.show()

if __name__ == "__main__":
    assert len(sys.argv) >= 4 and (sys.argv[3] == 'single' or sys.argv[3] == 'sweep_alpha' or sys.argv[3] == 'sweep_gamma')

    alpha = float(sys.argv[1])
    gamma = float(sys.argv[2])

    os.chdir(IMG_SAVE_DIR)

    start = (START_Y, START_X)
    shape = (ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT)

    frogger_q = QLearn(agent=Frogger_Agent(actions=['up', 'down', 'left', 'right'], location=start),
        environment=Frogger_Environment(shape=shape),start=start, alpha=alpha, gamma=gamma)

    if sys.argv[3] == "single":
        moves = 100
        if len(sys.argv) > 4:
            moves = int(sys.argv[4])
        plt.gray()
        frogger_q.learn(moves, save_plots=True)
    else:
        hyperparam_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        history_list = []
        fig_name = "histories"
        if len(sys.argv) > 5:
            fig_name = sys.argv[5]
        if sys.argv[3] == "sweep_alpha":
            hyperparam_str = "alpha"
        else:
            hyperparam_str = "gamma"
        for i, hyperparam in enumerate(hyperparam_list):
            print("{} = {:.2f}".format(hyperparam_str, hyperparam_list[i]))
            if (hyperparam_str == "alpha"):
                frogger_q.reset_params(new_alpha=hyperparam)
            else:
                frogger_q.reset_params(new_gamma=hyperparam)
            history_list.append(frogger_q.learn(50000))
        plot_histories(history_list, hyperparam_list, hyperparam_str, control_value, "gamma=point1")