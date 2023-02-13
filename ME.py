# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import random
import time
import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy
from itertools import product
import maxentropy
import cvxpy as cp
import pandas as pd

from MCSampling import MonteCarloSampling as MonteCarloSampling
from FCGenerator import FeatureGenerator as FeatureGenerator

###############################################################################
################################ Global variables #############################
###############################################################################

g = 0.9
numberOfSamples_std = 10000
iter_std = 100
TrajectoryLenght_std = 10000

###############################################################################
########################### Class MaximumEntropy #########################
###############################################################################

class ME():

    def __init__(self, environment, vectorized):
        
        # Initialization of important variables
        self.environment = environment
        self.delta = 0.98
        self.samplespace = environment.sample_space
        self.d_tuple = environment.d_tuple
        self.vectorized = vectorized
        self.div = 0

    def maxent_step(self, feature_class, exp_features):
        model = maxentropy.MinDivergenceModel(features = feature_class.feature_vector, samplespace = self.samplespace, vectorized=self.vectorized)
        model.verbose = False
        # Fit the model
        model.fit(exp_features)
        eta_hat = self.extract_dist(model.probdist())
        lambda_hat = model.params
        log_part = model.log_norm_constant()
        expectations = model.expectations()
        return eta_hat, lambda_hat, log_part, expectations

    def extract_dist(self, distribution):
        eta = distribution.reshape(self.d_tuple)
        return eta




