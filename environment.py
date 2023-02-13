# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import random
import time

import numpy as np
from itertools import product

import gym
from gym import spaces
import matplotlib.pyplot as plt

###############################################################################
################################ Global variables #############################
###############################################################################

# Default parameters for the environment configuration
S = 5 # number of states per dimension
timeOut = 50

# Parameters associated with stochasticity
stochasticRewards = False

std_color = 0
trap_color = 1
wall_color = 2
target_color = 3

###############################################################################
########################### Class StochasticGridWorld #########################
###############################################################################

class StochasticGridWorld_2D(gym.Env):
    """
    Implementing a simple RL environment consisting of a 2D grid world
    where the agent has to reach a fixed objective while avoiding a trap
    and several walls with potentially stochastic transitions and rewards.
    
    Input:     - state_space: RL environment state space.
               - action_space: RL environment action space.
               - playerPosition: Position of the player (x, y).
               - trapPosition: Position of the trap (x, y).
               - targetPosition: Position of the target (x, y).
               - timeElapsed: Time elapsed.
               - state: RL state or observation.
               - reward: RL reward signal.
               - done: RL termination signal (episode).
               - info: Additional RL information.
                                
    METHODS: - __init__: Initialization of the RL environment.
             - reset: Resetting of the RL environment.
             - step: Update the RL environment according to the agent's action.
             - render: Render graphically the current state of the RL environment.
    """

    def __init__(self, s=S, type='STD'):
        """
        GOAL: Perform the initialization of the RL environment.
        
        INPUTS: - size: Size of the square grid world.
        
        OUTPUTS: /
        """

        super(StochasticGridWorld_2D, self).__init__()

        # Initialize the random function with a fixed random seed
        random.seed(time.time())

        # coloring code
        self.std_color = std_color
        self.trap_color = trap_color
        self.wall_color = wall_color
        self.target_color = target_color

        # REWARDS
        self.R_max = 1
        self.R_min = -1

        # Definition of the observation/state and action spaces
        self.size = s
        # self.state_space = list(product(np.arange(0, S), np.arange(0, S)))
        state_space = np.arange(0, self.size*self.size, dtype=int)
        # spaces.Box(low=np.array([0, 0, self.std_color]), high=np.array([size-1, size-1, self.target_color]), dtype=np.uint8)
        self.action_space = np.arange(0, 4, dtype=int)
        return_space = np.arange(self.R_min, self.R_max + 0.09, 0.1)

        self.return_space =  [round(r,1) if np.abs(r) > 0.01 else 0.0 for r in return_space]
        ################### Initialization of the traps and target positions ############################
        l = self.size - 1

        # self.trapPosition =  [[1, 0], [2, 0], [3, 0]]
        if type == 'ICED':
        # self.trapPosition =  [[i+1, 0] for i in range(self.size - 1)]
            self.trapPosition =  [[i+1, 0] for i in range(self.size - 2)]
            self.targetPosition = [[l,  0]]
        elif type == 'DOUBLE':
        # self.trapPosition =  [[i+1, 0] for i in range(self.size - 1)]
            self.trapPosition =  [[i, 1] for i in range(self.size)]
            self.targetPosition = [[i, l] for i in range(self.size)]
        elif type == 'DOUBLE_POS':
        # self.trapPosition =  [[i+1, 0] for i in range(self.size - 1)]
            self.trapPosition =  []
            self.targetPosition = [[i, j] for i in range(self.size) for j in [1, l]]
        else:
            self.trapPosition =  [[i+1, 0] for i in range(self.size - 1)]
            self.targetPosition = [[l,  l]]

        trapPosition = [self.get_state(t) for t in self.trapPosition]
        targetPosition = [self.get_state(r) for r in self.targetPosition]
        self.wallPosition =  []
        wallPosition = [self.get_state(w) for w in self.wallPosition]
        
        self.state_space = list()
        for s in state_space:
            if (s not in trapPosition and s not in wallPosition and s not in targetPosition):
             self.state_space.append(s)
        self.sample_space, self.d_tuple = self.compute_samplespace(self.state_space, self.action_space, self.return_space)
        #################################################################################################

        # Initialization of the player position
        self.agentPosition = [random.randint(0,self.size-1), random.randint(0,self.size-1)]
        # while self.agentPosition == self.targetPosition or self.agentPosition in self.trapPosition or self.agentPosition in self.wallPosition:
        while self.agentPosition in self.targetPosition or self.agentPosition in self.trapPosition or self.agentPosition in self.wallPosition:
            self.agentPosition = [random.randint(0,self.size-1), random.randint(0,self.size-1)]

        # Initialization of the time elapsed
        self.timeElapsed = 0

        # Initialization of the RL variables
        self.state = self.get_state(self.agentPosition)
        # self.state = np.array([self.agentPosition[0], self.agentPosition[1]])
        self.reward = 0.
        self.done = 0
        self.info = {}
        self.true_dist = None

    def reset(self):
        """
        GOAL: Perform a reset of the RL environment.
        
        INPUTS: /
        
        OUTPUTS: - state: RL state or observation.
        """

        # Reset of the player position and time elapsed
        # int(random.random() * (self.size-1))
        self.agentPosition = [random.randint(0,self.size-1), random.randint(0,self.size-1)]
        while self.agentPosition in self.targetPosition or self.agentPosition in self.trapPosition or self.agentPosition in self.wallPosition:
            self.agentPosition = [random.randint(0,self.size-1), random.randint(0,self.size-1)]

        # Initialization of the time elapsed
        self.timeElapsed = 0

        # Reset of the RL variables
        self.state = self.get_state(self.agentPosition)
        # np.array([self.agentPosition[0], self.agentPosition[1]])
        self.reward = 0.
        self.done = 0
        self.info = {}

        return self.state


    def step(self, action):
        """
        GOAL: Update the RL environment according to the agent's action.
        
        INPUTS: - action: RL action outputted by the agent.
        
        OUTPUTS: - state: RL state or observation.
                 - reward: RL reward signal.
                 - done: RL termination signal.
                 - info: Additional RL information.
        """

        # Stochasticity associated with the next move of the agent
        moveRange = 1
        agentPosition_temp = self.agentPosition.copy()
        # Go right
        if action == 0:
            agentPosition_temp[0] = min(self.agentPosition[0]+moveRange, self.size-1)
        # Go down
        elif action == 1:
            agentPosition_temp[1] = max(self.agentPosition[1]-moveRange, 0)
        # Go left
        elif action == 2:
            agentPosition_temp[0] = max(self.agentPosition[0]-moveRange, 0)
        # Go up
        elif action == 3:
            agentPosition_temp[1] = min(self.agentPosition[1]+moveRange, self.size-1)
        # Invalid action
        else:
            print("Error: invalid action...")

        # Incrementation of the time elapsed
        self.timeElapsed += 1
        
        # Assign the appropriate RL reward

        # if agentPosition_temp == self.targetPosition:
        if agentPosition_temp in self.targetPosition:
            if stochasticRewards:
                self.reward = np.random.normal(loc=self.R_max, scale=0.1)
            else:
                self.reward = self.R_max
            self.done = 1
            self.agentPosition = agentPosition_temp

        elif agentPosition_temp in self.trapPosition:
            if stochasticRewards:
                self.reward = np.random.normal(loc=self.R_min, scale=0.1)
            else:
                self.reward = self.R_min
            self.done = 1
            self.agentPosition = agentPosition_temp
        elif agentPosition_temp in self.wallPosition:
            if stochasticRewards:
                self.reward = np.random.normal(loc=0.0, scale=0.1)
            else:
                self.reward = self.R_min
            self.done = 0
        else:
            if stochasticRewards:
                self.reward = np.random.normal(loc=0.0, scale=0.1)
            else:
                self.reward = 0.0
            self.done = 0
            self.agentPosition = agentPosition_temp


        # Check if the time elapsed reaches the time limit
        if self.timeElapsed >= timeOut:
            self.done = 1

        # Update of the RL state
        # self.state = np.array([self.agentPosition[0], self.agentPosition[1]])
        self.state = self.get_state(self.agentPosition)
        # Return of the RL variables
        return self.state, self.reward, self.done, self.info


    def setState(self, state):
        """
        GOAL: Reset the RL environment and set a specific initial state.
        
        INPUTS: - state: Information about the state to set.
        
        OUTPUTS: - state: RL state of the environment.
        """

        # Reset of the environment
        self.reset()

        # Set the initial state as specified
        self.timeElapsed = 0
        self.agentPosition = [state[0], state[1]]
        self.state = self.get_state(self.agentPosition)

        return self.state

    def get_state(self, agentPosition):
        ind = agentPosition[1]*(self.size) + agentPosition[0]
        # state = np.zeros((1, S*S))
        # state[ind] = 1
        return ind

    def compute_samplespace(self, state_space, action_space, return_space):

        sample_space = list(product(state_space, return_space))
        s_dimension = len(state_space)
        # a_dimension = len(action_space)
        r_dimension = len(return_space)
        d_tuple = (s_dimension, r_dimension)
        return sample_space, d_tuple


class RandomPolicy():

    def __init__(self, number_of_actions):
        self.n_actions = number_of_actions
    
    def processState(self, state):
        return state

    def chooseAction(self, state=None, value = None):
        return random.randint(0, self.n_actions - 1)

class UpperPolicy():

    def __init__(self, number_of_actions):
        self.n_actions = number_of_actions
    
    def processState(self, state):
        return state

    def chooseAction(self, state=None, value = None):
        return 3
