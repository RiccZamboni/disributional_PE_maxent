
from numpy.linalg import norm
import numpy as np
import collections
from itertools import product
from tqdm import tqdm

from MCSampling import MonteCarloSampling as MonteCarloSampling
from FCGenerator import FeatureGenerator as FeatureGenerator
from ME import ME as ME

from Utils import compute_kl, compute_entropy

###############################################################################
################################ Global variables #############################
###############################################################################

g = 0.99
numberOfSamples_std = 10000
iter_std = 100
TrajectoryLenght_std = 10000


class DummyNode():

    def __init__(self, hyper_c, splits):
        self.parent = None
        self.B = np.infty
        self.hyper_c = hyper_c
        self.splits = splits

    def evaluate(self):
        self.B = np.infty


class FNode():

        """ Node class for UCF algorithm """
        def __init__(self, optimizer, features, samples, parent):
            self.optimizer = optimizer
            self.features = features
            self.samples = samples
            self.hyper_c = parent.hyper_c
            self.splits = parent.splits
            self.N = len(samples)

            self.exp_features = self.features.apply_functions(samples)
            self.exp_counters = self.features.apply_functions(samples, counter=True)
            eta_hat, lambda_hat, log_partition, expectations = self.optimizer.maxent_step(feature_class = self.features, exp_features = self.exp_features)
            self.eta_hat = eta_hat
            self.lambda_hat = lambda_hat
            self.log_partition = log_partition
            self.expectations = expectations
            self.sets = self.features.sets
            self.kl = compute_kl(self.optimizer.environment, self.optimizer.environment.true_dist, eta_hat)
            self.h_0 = compute_entropy(self.optimizer.environment, self.optimizer.environment.true_dist)
            self.delta = 0.01
            self.Phi = self.optimizer.environment.R_max
            
            self.parent = parent
            self.children = list()
            self.L = None
            self.B = None
            self.B_tot = None
            self.done = False

        def getL(self):
            # test = np.inner(self.lambda_hat, self.exp_features)
            self.L  = self.hyper_c*(self.log_partition - np.inner(self.lambda_hat, self.exp_features))
            self.l = self.get_log_l()

        def getB(self):
            self.norm = self.get_norm(self.lambda_hat)
            self.complexity_loc = np.sum(np.sqrt(self.exp_counters))/self.N
            # self.B_i =  10*norm(self.lambda_hat, 1)*np.sqrt(self.exp_f*len(samples))/self.N - np.dot(self.lambda_hat, exp_f)
            self.B =  self.norm*(self.Phi*self.complexity_loc + self.Phi*np.sqrt(np.log(1/self.delta)/(2*self.N)))

        def getN(self, samples):
            self.N = len(samples)
            
        def evaluate(self):
            self.getB()
            self.getL()
            self.bound_full = - self.h_0 + self.L + self.B
            self.B_tot = self.B + self.L

        def get_norm(self, lambda_hat):
            # lambda_norm = 10*norm(lambda_hat,  np.inf)
            lambda_norm = 10*norm(lambda_hat,  np.inf)
            # il massimo lambda per Rmax = 1 e S_alpha = 1 è 2.39 MA non cambia comunque il trend
            # lambda_norm = 10*2.39
            return lambda_norm

        def greedy_search(self, step, greedy=False):
            not_expanded = True
            node_star = self
            self.expand(self.features, self.i_star) if greedy else self.expand(self.features)

            for leaf_down in self.children:
                leaf_down.evaluate()
                if leaf_down.B_tot <= node_star.B_tot:
                    not_expanded = False
                    node_star = leaf_down
                else:
                    pass
            self.done = not_expanded
            step += 1
            return node_star, not_expanded, step


        def expand(self, features, i_star=None):
            next_features, done = FeatureGenerator.expand_features(features, i_star, self.splits)
            self.done = done
            if not self.done:
                for f in next_features:
                    self.children.append(FNode(self.optimizer, f, self.samples, parent = self))


        def contract(self, features):
            next_features = FeatureGenerator.contract_features(features)
            self.children_up = [FNode(f, self.samples, parent = self) if f is not None else DummyNode() for f in next_features]

        def search(self, greedy=False):
            self.expand(self.features, self.i_star) if greedy else self.expand(self.features)
            if self.done is not True:
                for leaf_down in self.children:
                    leaf_down.evaluate()
                    leaf_down.search()
            else:
                pass

        def extract(self, out):
            out['B_tot'] = str(self.B_tot)
            out['L'] = str(self.L)
            out['B'] = str(self.B)
            out['c'] = str(self.complexity_loc)
            out['kl'] = str(self.kl)
            out['sets'] = str(self.sets)
            out['children'] = dict()
            i = 0
            if not self.done:
                list_star = list()
                for c in self.children:
                    if c.B_tot <= self.B_tot:
                        list_star.append(str(i))
                    out['children'][str(i)] = dict()
                    out['children'][str(i)] = c.extract(out['children'][str(i)])
                    i += 1
                out['star'] = list_star
            return out

        def get_log_l(self):
            eta = self.eta_hat
            l = 0
            for z in self.samples:
                state = [idx for idx in range(len(self.optimizer.environment.state_space))  if self.optimizer.environment.state_space[idx] == z[0]]
                g_ind = [idx for idx in range(len(self.optimizer.environment.return_space))  if self.optimizer.environment.return_space[idx] == z[1]]
                l -= np.log(eta[state, g_ind])
            l = l/self.N
            return l


class UCF():

    def __init__(self, env, policy, samples, vectorized,  gamma = g, trajSamples= numberOfSamples_std):
        # Initialization of mportant variables
        self.environment = env
        self.policy = policy
        self.gamma = gamma
        self.delta = 0.98
        self.vectorized = vectorized
        self.trajSamples = trajSamples
        self.feature_generator = FeatureGenerator(self.environment, self.vectorized)
        self.features = self.feature_generator.features
        self.features_full = self.feature_generator.features_full
        self.optimizer = ME(self.environment, self.vectorized)
        self.samples = samples
        

    def search(self, hyper_c, splits):
        dummy = DummyNode(hyper_c, splits)
        node = FNode(self.optimizer, self.features, self.samples, parent = dummy)
        node.evaluate()
        node.search()
        search_result = dict()
        search_result = node.extract(search_result)
        return search_result


    def greedy_search(self, greedy_search_result, hyper_c, splits):
        done = False
        greedy = False
        dummy = DummyNode(hyper_c, splits)
        node_s = FNode(self.optimizer, self.features, self.samples, parent = dummy)
        node_full = FNode(self.optimizer, self.features_full, self.samples, parent = dummy)
        node_s.evaluate()
        step = 0
        while not done and step < 10:
             node_s, done, step = node_s.greedy_search(step, greedy)
             print("#######")
             print("Factor done")
             print("Set {}".format(node_s.sets))
             print("L {}".format(node_s.L))
             print("B {}".format(node_s.B))
             print("KL {}".format(node_s.kl))
             print("#######")
             greedy_search_result[str(step)] = node_s.extract(dict())
        return greedy_search_result, node_s.eta_hat, node_full.eta_hat



