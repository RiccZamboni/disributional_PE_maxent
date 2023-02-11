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

###############################################################################
################################ Global variables #############################
###############################################################################

# Default parameters for the environment configuration
S = 3 # number of states per dimension
timeOut = 50

# Parameters associated with stochasticity
stochasticRewards = True

std_color = 0
trap_color = 1
wall_color = 2
target_color = 3

###############################################################################
########################### Class StochasticGridWorld #########################
###############################################################################

class StochasticGridWorld_3D(gym.Env):
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

    def __init__(self, size=S):
        """
        GOAL: Perform the initialization of the RL environment.
        
        INPUTS: - size: Size of the square grid world.
        
        OUTPUTS: /
        """

        super(StochasticGridWorld_3D, self).__init__()

        # Initialize the random function with a fixed random seed
        random.seed(time.time())

        # coloring code
        self.std_color = std_color
        self.trap_color = trap_color
        self.wall_color = wall_color
        self.target_color = target_color

        # REWARDS
        self.R_max = 1.
        self.R_min = -1.

        # Definition of the observation/state and action spaces
        self.state_space = list(product(np.arange(0,3), np.arange(0,3), np.arange(0,3)))
        # spaces.Box(low=np.array([0, 0, self.std_color]), high=np.array([size-1, size-1, self.target_color]), dtype=np.uint8)
        self.action_space = np.array([0, 1, 2, 3])
        # self.return_space = np.array([self.R_min, 0., self.R_max])
        self.return_space = np.arange(self.R_min, self.R_max, 0.01)
        self.size = size

        ################### Initialization of the traps and target positions ############################
        self.trapPosition =  [int(self.size/2), int(self.size/2), self.trap_color]
        self.wallPosition =  [int(self.size/2) + 1, int(self.size/2) + 1,  self.wall_color]
        self.targetPosition = [self.size-1,  self.size-1,   self.target_color]
        #################################################################################################

        # Initialization of the player position
        self.agentPosition = [int(random.random() * (self.size-1)), int(random.random() * (self.size-1))]
        while self.agentPosition == self.targetPosition[0:1] or self.agentPosition == self.trapPosition[0:1] or self.agentPosition == self.wallPosition[0:1]:
            self.agentPosition = [int(random.random() * (self.size-1)), int(random.random() * (self.size-1))]

        # Initialization of the time elapsed
        self.timeElapsed = 0

        # Initialization of the RL variables
        self.state = np.array([self.agentPosition[0], self.agentPosition[1],  self.std_color])
        self.reward = 0.
        self.done = 0
        self.info = {}

    def reset(self):
        """
        GOAL: Perform a reset of the RL environment.
        
        INPUTS: /
        
        OUTPUTS: - state: RL state or observation.
        """

        # Reset of the player position and time elapsed
        self.agentPosition = [int(random.random() * (self.size-1)), int(random.random() * (self.size-1))]
        while self.agentPosition == self.targetPosition[0:1] or self.agentPosition == self.trapPosition[0:1] or self.agentPosition == self.wallPosition[0:1]:
            self.agentPosition = [int(random.random() * (self.size-1)), int(random.random() * (self.size-1))]

        # Initialization of the time elapsed
        self.timeElapsed = 0

        # Reset of the RL variables
        self.state = np.array([self.agentPosition[0], self.agentPosition[1],  self.std_color])
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
        agentPosition_temp = self.agentPosition
        
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

        if agentPosition_temp == self.targetPosition:
            if stochasticRewards:
                self.reward = round(np.random.normal(loc=self.R_max, scale=0.1),2)
            else:
                self.reward = self.R_max
            self.done = 1
            color = self.target_color
            self.agentPosition = agentPosition_temp

        elif agentPosition_temp == self.trapPosition:
            if stochasticRewards:
                self.reward = round(np.random.normal(loc=self.R_min, scale=0.1),2)
            else:
                self.reward = self.R_min
            self.done = 0
            color = self.trap_color
            self.agentPosition = agentPosition_temp
        elif agentPosition_temp == self.wallPosition:
            if stochasticRewards:
                self.reward = round(np.random.normal(loc=self.R_min, scale=0.1),2)
            else:
                self.reward = self.R_min
            self.done = 0
            color = self.std_color
            # self.agentPosition = self.agentPosition
        else:
            if stochasticRewards:
                self.reward = round(np.random.normal(loc=0.0, scale=0.1),2)
            else:
                self.reward = 0.0
            self.done = 0
            self.agentPosition = agentPosition_temp
            color = self.std_color


        # Check if the time elapsed reaches the time limit
        if self.timeElapsed >= timeOut:
            self.done = 1

        # Update of the RL state
        self.state = np.array([self.agentPosition[0], self.agentPosition[1], color])

        # Return of the RL variables
        return self.state, self.reward, self.done, self.info

    
    def render(self, mode='human'):
        """
        GOAL: Render graphically the current state of the RL environment.
        
        INPUTS: /
        
        OUTPUTS: /
        """

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, self.size+1, 1))
        ax.set_yticks(np.arange(0, self.size+1, 1))
        ax.set(xlim=(0, self.size), ylim=(0, self.size))
        plt.scatter(self.agentPosition[0]+0.5, self.agentPosition[1]+0.5, s=100, color='blue')
        plt.scatter(self.targetPosition[0]+0.5, self.targetPosition[1]+0.5, s=100, color='green')
        plt.scatter(self.trapPosition[0]+0.5, self.trapPosition[1]+0.5, s=100, color='red')
        plt.scatter(self.wallPosition[0]+0.5, self.wallPosition[1]+0.5, s=100, color='black')
        plt.grid()
        text = ''.join(['Time elapsed: ', str(self.timeElapsed)])
        plt.text(0, self.size+0.2, text, fontsize=12)
        plt.show()
        #plt.savefig("Figures/Distributions/StochasticGridWorldState.pdf", format="pdf")


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
        self.state = np.array([self.agentPosition[0], self.agentPosition[1], self.std_color])

        return self.state


class RandomPolicy():

    def __init__(self, number_of_actions):
        self.n_actions = number_of_actions
    
    def processState(self, state):
        return state

    def chooseAction(self, state=None, value = None):
        return random.randint(0, self.n_actions)
