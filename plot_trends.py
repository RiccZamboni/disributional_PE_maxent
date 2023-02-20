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
import pandas as pd

from environment import StochasticGridWorld_2D, StochasticGridWorld_rect
from environment import RandomPolicy as RandomPolicy, UpperPolicy, AlmostUpperPolicy

from Utils import generate_plot, return_percentages, generate_full_plot
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
    parser.add_argument("-samples", default=50, type=int, help="number of samples")
    parser.add_argument("-environment_size", default=9, type=int, help="environment size")
    parser.add_argument("-environment_type", default='DOUBLE', type=str, help="environment type")
    parser.add_argument("-full_search", default=False, type=bool, help="search type")
    parser.add_argument("-vectorized", default=True, type=str, help="vectorized features")
    parser.add_argument("-output_type", default='B_tot', type=str, help="type of output") # B_tot
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
    output_type = args.output_type
    # env = StochasticGridWorld_2D(size, type_env)
    env = StochasticGridWorld_rect(size, type_env)

    # policy = UpperPolicy(len(env.action_space))
    # policy = RandomPolicy(len(env.action_space))
    policy = AlmostUpperPolicy(len(env.action_space))

    # Initialization of the DRL algorithm
    samples_n = args.samples
    sampler = MonteCarloSampling(env,  gamma = 0.98,  trajSamples= samples_n)

    print("#######")
    print("Factor init")
    print("Type {}".format(type_env))
    print("Size {}".format(size))
    print("Splits {}".format(n_splits))
    print("Output {}".format(output_type))
    print("#######")

    greedy_search_dict = dict()
    greedy_search_dict['INFO'] = {
     "Type" : str(type_env),
     "Samples_TOT" : str(samples_n),
     "Size" : str(size),
     "Splits" : str(n_splits),
     "Output":str(output_type)
    }
    iter_tot = 10
    hyper = 300
    resolution = 4
    percentages_h = return_percentages(hyper, resolution)
    # percentages_n = np.round(np.logspace(0.1**(1/10), 1000**(1/10), resolution))
    # percentages_n = return_percentages(samples_n, resolution)
    df = pd.DataFrame()
    kl_opt  = np.zeros((iter_tot*resolution,1))
    kl_full = np.zeros((iter_tot*resolution,1))
    b_opt  = np.zeros((iter_tot*resolution,1))
    b_full = np.zeros((iter_tot*resolution,1))
    kl_list = list()
    b_list = list()

    for h_idx, h in enumerate(percentages_h):

      for i in range(iter_tot):
        trajetories = sampler.generate_trajectories(policy)
        z_samples = sampler.generate_return_distribution_from_trajectories(trajetories, samples_n)
        eta_MC = sampler.estimate_distribution_DKW(z_samples)
        env.true_dist = eta_MC
        greedy_search_dict[str(i)] = dict()
        n_samples = sampler.generate_return_distribution_from_trajectories(trajetories, samples_n)
        
        Agent = UCF(env, policy = policy, samples = n_samples, vectorized = vectorized,  gamma = 0.98, trajSamples= samples_n)
        #greedy_search_dict[str(i)], out_hat, out_full, out_opt = Agent.compare_distributions(greedy_search_dict[str(i)], env.true_dist,hyper_c=h, splits=n_splits)
        greedy_search_dict[str(i)], kl_hat, b_hat, kl_f, b_f, kl_o, b_o = Agent.compare_distributions(greedy_search_dict[str(i)], env.true_dist,hyper_c=h, splits=n_splits)
        kl_list.append(kl_hat[:-1])
        b_list.append(b_hat[:-1])
        kl_full[iter_tot*h_idx + i] = kl_f
        kl_opt[iter_tot*h_idx + i] = kl_o
        b_full[iter_tot*h_idx + i] = b_f
        b_opt[iter_tot*h_idx + i] = b_o
        
        # out_hat_pd = pd.DataFrame({'{}_hat_{}'.format(output_type, str(iter_tot*h_idx + i)):out_hat[:-1]})
        #out_full_pd = pd.DataFrame({'{}_full_{}'.format(output_type,str(iter_tot*h_idx + i)):output_full[iter_tot*h_idx + i]*np.ones(len(out_hat[:-1]))})
        #out_opt_pd = pd.DataFrame({'{}_opt_{}'.format(output_type,str(iter_tot*h_idx + i)):output_opt[iter_tot*h_idx + i]*np.ones(len(out_hat[:-1]))})
        #df = pd.concat([df, out_hat_pd, out_full_pd, out_opt_pd], axis=1)
      #df.to_csv("results/DF_s{}_o{}_t{}_n{}_h{}_d{}.csv".format(env.size, output_type, type_env, samples_n, hyper, dt_string)) 
      # generate_plot(kl_list, kl_full, kl_opt, env.size, type_env, samples_n, dt_string, hyper)
      # with open("resultgreedy_search_dict_s{}_t{}_n{}_h{}_d{}.json".format(env.size, type_env, samples_n, hyper, dt_string), "w") as outfile:
      #      json.dump(greedy_search_dict, outfile)
    plotting_feedback = generate_full_plot(b_list, b_full, b_opt, env.size, type_env, samples_n, dt_string, percentages_h, iter_tot, 'B_tot')
    generate_full_plot(kl_list, kl_full, kl_opt, env.size, type_env, samples_n, dt_string, percentages_h, iter_tot, 'KL')
    with open("results/plotting_feedback_s{}_o{}_t{}_n{}_h{}_d{}.json".format(env.size, output_type, type_env, samples_n, hyper, dt_string), "w") as outfile:
          json.dump(plotting_feedback, outfile)
    pass