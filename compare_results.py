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
import numpy as np

from environment import StochasticGridWorld_2D, StochasticGridWorld_rect
from environment import RandomPolicy as RandomPolicy, UpperPolicy

from Utils import plot_solution, compute_kl, compare_expected_values, return_percentages, plot
from UCF import UCF as UCF, DummyNode, FNode
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
    parser.add_argument("-splits", default=2, type=int, help="number of splits K")
    parser.add_argument("-samples", default=500, type=int, help="number of samples (trajectories)")
    parser.add_argument("-environment_size", default=8, type=int, help="environment size")
    parser.add_argument("-environment_type", default='RECT', type=str, help="environment type (see environment.py)")
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
    env = StochasticGridWorld_rect(size, type_env)

    policy = UpperPolicy(len(env.action_space))
    # policy = RandomPolicy(len(env.action_space))

    # Initialization of the DRL algorithm
    samples_n = args.samples
    sampler = MonteCarloSampling(env,  gamma = 0.98,  trajSamples= samples_n)

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
    iter_tot = 10
    hyper = 1000
    resolution = 10
    percentages_n = return_percentages(samples_n, resolution)
    percentages_h = return_percentages(hyper, resolution)
    for hyper in percentages_h:
      kl_est  = np.zeros((iter_tot,resolution))
      kl_full = np.zeros((iter_tot,resolution))
      RMSE_est = np.zeros((iter_tot,resolution))
      RMSE_full = np.zeros((iter_tot,resolution))
      MSE_est = np.zeros((iter_tot,resolution))
      MSE_full = np.zeros((iter_tot,resolution))
      output = dict()
      output['N'] = str(percentages_n)
      output['hyper'] = str(percentages_n)
      for i in range(iter_tot):
        trajetories = sampler.generate_trajectories(policy)
        z_samples = sampler.generate_return_distribution_from_trajectories(trajetories, samples_n)
        eta_MC = sampler.estimate_distribution_DKW(z_samples)
        env.true_dist = eta_MC

        for n_id, n in enumerate(percentages_n):
          greedy_search_dict[str(n)] = dict()
          n_samples = sampler.generate_return_distribution_from_trajectories(trajetories, n)
    
          Agent = UCF(env, policy = policy, samples = n_samples, 
          vectorized = vectorized, gamma = 0.98, trajSamples= n)
          greedy_search_dict[str(n)], eta_hat, eta_hat_full, _ = Agent.greedy_search(greedy_search_dict[str(n)], hyper_c=hyper, splits=n_splits)
          kl_est[i, n_id] = compute_kl(env, env.true_dist, eta_hat)
          kl_full[i, n_id] = compute_kl(env, env.true_dist, eta_hat_full)
          est, full = compare_expected_values(env, env.true_dist, eta_hat, eta_hat_full)
          MSE_est[i, n_id] = est
          MSE_full[i, n_id] = full
          RMSE_est[i, n_id] = np.sqrt(est)
          RMSE_full[i, n_id] = np.sqrt(full)
          pass
    
      output['kl_est_m'], output['kl_est_std']  = str(np.mean(kl_est, axis=0)), str(np.std(kl_est, axis=0))
      output['kl_full_m'], output['kl_full_std']  = str(np.mean(kl_full, axis=0)), str(np.std(kl_full, axis=0))
      output['MSE_est_m'], output['MSE_est_std']  = str(np.mean(MSE_est, axis=0)), str(np.std(MSE_est, axis=0))
      output['MSE_full_m'], output['MSE_full_std']  = str(np.mean(MSE_full, axis=0)), str(np.std(MSE_full, axis=0))
      output['RMSE_est_m'], output['RMSE_est_std']  = str(np.mean(RMSE_est, axis=0)), str(np.std(RMSE_est, axis=0))
      output['RMSE_full_m'], output['RMSE_full_std']  = str(np.mean(RMSE_full, axis=0)), str(np.std(RMSE_full, axis=0))
      with open("results/TABLE_s{}_t{}_n{}_h{}_d{}.json".format(env.size, type_env, samples_n, hyper, dt_string), "w") as outfile:
          json.dump(output, outfile)
    pass