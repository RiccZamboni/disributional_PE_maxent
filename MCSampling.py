# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import itertools
import numpy  as np


###############################################################################
################################ Global variables #############################
###############################################################################




###############################################################################
####################### Class MonteCarloSampling #########################
###############################################################################

class MonteCarloSampling():
    def __init__(self, env, gamma, trajSamples):

        # Initialization of important variables
        self.env = env
        self.gamma = gamma
        self.trajSamples = trajSamples
    
    def generate_trajectories(self, policy, n_traj = None):
        n_traj = self.trajSamples if n_traj is None else n_traj
        # Initialization of the memory storing the MC samples
        trajectories = []

        # Generation of the MC samples
        for _ in range(n_traj):

            trajectory = []
            # Initialization of some variables
            # expectedReturn = 0
            step = 0

            # Reset of the environment and initialization to the desired state
            state = self.env.reset()
            action = policy.chooseAction(state)
            # state = self.environment.setState(initialState)

            # Execution of the action specified
            nextState, reward, done, _ = self.env.step(action)

            # Update of the trajectory
            trajectory +=[(state, action, reward)]
            step += 1

            # Loop until episode termination
            while done == 0:

                # Execute the next ation according to the policy selected
                state = policy.processState(nextState)
                action = policy.chooseAction(state)
                nextState, reward, done, _ = self.env.step(action)

                # Update of the expected return
                # expectedReturn += (reward * (self.gamma**step))
                trajectory +=[(state, action, reward)]
                step += 1
            
            # Add the MC sample to the memory
            trajectories.append(trajectory)
        # Output the MC samples collected
        return trajectories


    def generate_return_distribution_from_trajectories(self, trajectories):
        z_samples = list()
        for tr in range(self.trajSamples):
            traj_i = trajectories[tr]
            G = 0
            for t in reversed(range(0, len(trajectories[tr]))):
                s_t, a_t, r_t = traj_i[t]
                G += (r_t*(self.gamma**(t)))
            # z_samples.append((s_t, a_t, round(G,1)))
            z_samples.append((s_t, round(G,1)))
        return z_samples

    def estimate_distribution_DKW(self, z_samples):
        eta_MC = np.zeros((self.env.d_tuple))
        counters = np.zeros((self.env.d_tuple[0]))
        for z in z_samples:
            state = [idx for idx in range(len(self.env.state_space))  if self.env.state_space[idx] == z[0]]
            g_ind = [idx for idx in range(len(self.env.return_space))  if self.env.return_space[idx] == z[1]]
            counters[state] +=1
            eta_MC[state, g_ind] +=1
        for s in range(self.env.d_tuple[0]):
            eta_MC[s][:] = eta_MC[s][:]/counters[s]
        return eta_MC


        

        
