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
from datetime import datetime

from environment import StochasticGridWorld_2D as StochasticGridWorld_2D
from environment import RandomPolicy as RandomPolicy, UpperPolicy

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
     parser.add_argument("-samples", default=1000, type=int, help="number of samples")
     parser.add_argument("-environment_size", default=4, type=int, help="environment size")
     parser.add_argument("-environment_type", default='DOUBLE', type=str, help="environment type")
     parser.add_argument("-full_search", default=False, type=bool, help="search type")
     parser.add_argument("-vectorized", default=True, type=str, help="vectorized features")
     args = parser.parse_args()

     # Training of the RL agent
     now = datetime.now()
     # dd/mm/YY H:M:S
     dt_string = now.strftime("%d_%H_%M")

     # Initialization of the RL environment
     size = args.environment_size
     type_env = args.environment_type
     vectorized = args.vectorized
     n_splits = args.splits
     samples_n = args.samples

     env = StochasticGridWorld_2D(size, type_env)
     # policy = UpperPolicy(len(env.action_space))
     policy = RandomPolicy(len(env.action_space))

     # Initialization of the DRL algorithm
     
     sampler = MonteCarloSampling(env,  gamma = 0.98,  trajSamples= samples_n)
     trajetories = sampler.generate_trajectories(policy)
     z_samples = sampler.generate_return_distribution_from_trajectories(trajetories, samples_n)
     eta_MC = sampler.estimate_distribution_DKW(z_samples)
     env.true_dist = eta_MC

     print("#######")
     print("Factor init")
     print("Type {}".format(type_env))
     print("Samples {}".format(samples_n))
     print("Size {}".format(size))
     print("Splits {}".format(n_splits))
     print("#######")

     if args.full_search:
          Agent_full_search = UCF(env, policy = policy, samples = z_samples, vectorized = vectorized, gamma = 0.98, trajSamples= samples_n)
          print("#######")
          print("Full Search")
          print("#######")
          search_dict = Agent_full_search.search(hyper_c=1, splits=n_splits)
          with open("full_search_dict_{}_{}_{}_{}.json".format(env.size, type_env, samples_n, dt_string), "w") as outfile:
                    json.dump(search_dict, outfile)
     else:
          z_samples = sampler.generate_return_distribution_from_trajectories(trajetories, samples_n)
          
          Agent = UCF(env, policy = policy, samples = z_samples, 
          vectorized = vectorized, gamma = 0.98, trajSamples= samples_n)

          print("#######")
          print("Factor init")
          print("Type {}".format(type_env))
          print("Samples {}".format(samples_n))
          print("Size {}".format(size))
          print("Splits {}".format(n_splits))
          print("#######")

          greedy_search_dict = dict()
          greedy_search_dict['INFO'] = {
               "Type" : str(type_env),
               "Samples" : str(samples_n),
               "Size" : str(size),
               "Splits" : str(n_splits)
          }

          greedy_search_dict, eta_hat, eta_hat_full, eta_hat_opt = Agent.greedy_search(greedy_search_dict, hyper_c=10, splits=n_splits)


          with open("greedy_search_dict_{}_{}_{}_{}.json".format(env.size, type_env, samples_n, dt_string), "w") as outfile:
               json.dump(greedy_search_dict, outfile)