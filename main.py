# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################
#%%
import argparse
import importlib
import time
import random
import json

from environment import StochasticGridWorld_2D as StochasticGridWorld_2D
from environment import RandomPolicy as RandomPolicy

from Utils import plot_solution, compute_kl
from UCF import UCF as UCF
from MCSampling import MonteCarloSampling as MonteCarloSampling


###############################################################################
################################ Global variables #############################
###############################################################################


###############################################################################
##################################### MAIN ####################################
###############################################################################

if(__name__ == '__main__'):

    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-splits", default=2, type=int, help="number of splits")
    parser.add_argument("-environment_size", default=4, type=int, help="environment size")
    parser.add_argument("-search", default=True, type=bool, help="search type")

    args = parser.parse_args()

    # Checking of the parameters validity
    environment = 'StochasticGridWorld_2D'

    # Name of the file for saving the RL policy learned
    fileName = 'SavedModels/'+ '_' + environment + '{}'.format(time.time())
    
    # Initialization of the RL environment
    size = args.environment_size
    env = StochasticGridWorld_2D(size)

    policy_rand = RandomPolicy(len(env.action_space))

    # Initialization of the DRL algorithm
    sampler = MonteCarloSampling(env,  gamma = 0.98,  trajSamples= 10000)
    trajetories = sampler.generate_trajectories(policy_rand)
    z_samples = sampler.generate_return_distribution_from_trajectories(trajetories)
    # eta_MC = sampler.estimate_distribution_DKW(z_samples)
    Agent = UCF(env, policy = policy_rand, samples = z_samples, gamma = 0.98, trajSamples= 10000)

    # Training of the RL agent
    n_splits = args.splits
    if args.search:
          search_dict = Agent.search(hyper_c=1, splits=n_splits)
          with open("search_dict_{}_{}.json".format(env.size, n_splits), "w") as outfile: 
               json.dump(search_dict, outfile)
    greedy_search_dict, eta_hat = Agent.greedy_search(hyper_c=1000, splits=n_splits)
    with open("greedy_search_dict_{}_{}.json".format(env.size, n_splits), "w") as outfile:
         json.dump(greedy_search_dict, outfile)
    # kl = compute_kl(env, eta_MC, eta_hat)
    pass
    # eta_star = node_star.eta_hat_plot
    #%%
    # Agent.plotExpectedPerformance(node_star)

    # Testing of the RL agent
    # Agent.testing(env, verbose=True, rendering=False)
# %%
