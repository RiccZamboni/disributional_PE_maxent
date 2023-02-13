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

from Utils import plot_solution, compute_kl, compare_values, return_percentages, plot
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
    parser.add_argument("-samples", default=10000, type=int, help="number of samples")
    parser.add_argument("-environment_size", default=2, type=int, help="environment size")
    parser.add_argument("-environment_type", default='ICED', type=str, help="environment type")
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
    env = StochasticGridWorld_2D(size, type_env)

    # policy = UpperPolicy(len(env.action_space))
    policy = RandomPolicy(len(env.action_space))

    # Initialization of the DRL algorithm
    samples_n = args.samples
    sampler = MonteCarloSampling(env,  gamma = 0.98,  trajSamples= samples_n)
    trajetories = sampler.generate_trajectories(policy)
    z_samples = sampler.generate_return_distribution_from_trajectories(trajetories, samples_n)
    eta_MC = sampler.estimate_distribution_DKW(z_samples)
    env.true_dist = eta_MC

    print("#######")
    print("Factor init")
    print("Type {}".format(type_env))
    print("Size {}".format(size))
    print("Splits {}".format(n_splits))
    print("#######")

    greedy_search_dict = dict()
    greedy_search_dict['INFO'] = {
     "Type" : str(type_env),
     "Samples_TOT" : str(samples_n),
     "Size" : str(size),
     "Splits" : str(n_splits)
    }

    kl_est  = list()
    kl_full = list()
    RMSE_est = list()
    RMSE_full = list()
    resolution = 10
    percentages_n = return_percentages(samples_n, resolution)
    for n in percentages_n:
      greedy_search_dict[str(n)] = dict()
      n_samples = sampler.generate_return_distribution_from_trajectories(trajetories, n)
      
      Agent = UCF(env, policy = policy, samples = n_samples, 
      vectorized = vectorized, gamma = 0.98, trajSamples= n)
      greedy_search_dict[str(n)], eta_hat, eta_hat_full = Agent.greedy_search(greedy_search_dict[str(n)], hyper_c=1000, splits=n_splits)
      kl_est.append(compute_kl(env, env.true_dist, eta_hat))
      kl_full.append(compute_kl(env, env.true_dist, eta_hat_full))
      est, full = compare_values(env, eta_hat, eta_hat_full, env.true_dist)
      RMSE_est.append(est)
      RMSE_full.append(full)
      pass

    plot(RMSE_est, RMSE_full, percentages_n, 'RMSE', "greedy_search_dict_{}_{}_{}_{}.json".format(env.size, type_env, samples_n, dt_string))
    plot(kl_est, kl_est, percentages_n, 'KL-divergence', "greedy_search_dict_{}_{}_{}_{}.json".format(env.size, type_env, samples_n, dt_string))

    # with open("results/greedy_search_dict_{}_{}_{}_{}.json".format(env.size, type_env, samples_n, dt_string), "w") as outfile:
    #      json.dump(greedy_search_dict, outfile)
    pass