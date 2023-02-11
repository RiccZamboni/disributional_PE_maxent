
from numpy.linalg import norm
import numpy as np
import collections
from itertools import product

from MCSampling import MonteCarloSampling as MonteCarloSampling
from FCGenerator import FeatureGenerator as FeatureGenerator
from ME import ME as ME

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
            
            self.exp_features = self.features.apply_functions_non_vectorized(samples)
            self.exp_counters = self.features.apply_functions_non_vectorized(samples, counter=True)
            eta_hat, lambda_hat, log_partition, expectations = self.optimizer.maxent_step(feature_class = self.features, exp_features = self.exp_features)
            self.eta_hat = eta_hat
            self.lambda_hat = lambda_hat
            self.log_partition = log_partition
            self.expectations = expectations
            self.sets = self.features.sets
            
            self.delta = 0.98
            self.Phi = self.optimizer.environment.R_max
            self.N = len(samples)
            self.parent = parent
            self.children = list()
            self.L = None
            self.B = None
            self.B_tot = None
            self.done = False

        def getL(self):
            self.L  = self.hyper_c*(self.log_partition - np.inner(self.lambda_hat, self.exp_features))


        def getB(self):
            self.norm = self.get_norm(self.lambda_hat)
            self.complexity_loc = np.inner(np.sqrt(self.exp_counters), np.ones(len(self.exp_counters)))
            # self.B_i =  10*norm(self.lambda_hat, 1)*np.sqrt(self.exp_f*len(samples))/self.N - np.dot(self.lambda_hat, exp_f)
            self.B =  self.norm*(self.Phi*self.complexity_loc + self.Phi*np.sqrt(np.log(1/self.delta)/(2*self.N)))


        def getN(self, samples):
            self.N = len(samples)
            

        def evaluate(self):
            self.getB()
            self.getL()
            self.B_tot = self.B + self.L
            self.i_star = self.get_f_star()

        def get_f_star(self):
            # return np.argmin(self.B_vect)
            return None

        def get_norm(self, lambda_hat):
            lambda_norm = 8*norm(lambda_hat,  np.inf)
            return lambda_norm

        def greedy_search(self, greedy=False):
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
            return node_star, not_expanded


        def expand(self, features, i_star=None):
            next_features, done = FeatureGenerator.expand_features(features, i_star, self.splits)
            self.done = done
            if self.done is not True:
                for f in next_features:
                    # if f is not None:
                    self.children.append(FNode(self.optimizer, f, self.samples, parent = self))
                    # else:
                    #     pass


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
            out['sets'] = str(self.sets)
            out['children'] = dict()
            i = 0
            if not self.done:
                for c in self.children:
                    out['children'][str(i)] = dict()
                    out['children'][str(i)] = c.extract(out['children'][str(i)])
                    i += 1
            return out

        def next_leaf_depr(self):
            not_expanded = True
            not_contracted = True
            current = self
            current.expand(current.features)
            for leaf_down in current.children_down:
                not_expanded = False if leaf_down.evaluate() <= current.B else True
                current = leaf_down if leaf_down.evaluate() <= current.B else current
            if not_expanded:
                current.contract(current.features)
                for leaf_up in current.children_up:
                    not_contracted = False if leaf_up.evaluate() <= current.B else True
                    current = leaf_up if leaf_up.evaluate() <= current.B else current
            done  = not_expanded and not_contracted
            return current, done


class UCF():

    def __init__(self, env, policy, samples,  gamma = g, trajSamples= numberOfSamples_std):
        # Initialization of mportant variables
        self.environment = env
        self.policy = policy
        self.gamma = gamma
        self.delta = 0.98
        self.trajSamples = trajSamples
        self.feature_generator = FeatureGenerator(self.environment)
        self.features = self.feature_generator.features
        self.optimizer = ME(self.environment)
        self.samples = samples
        

    def search(self, hyper_c, splits):
        dummy = DummyNode(hyper_c, splits)
        node = FNode(self.optimizer, self.features, self.samples, parent = dummy)
        node.evaluate()
        node.search()
        search_result = dict()
        search_result = node.extract(search_result)
        return search_result


    def greedy_search(self, hyper_c, splits):
        done = False
        greedy = False
        dummy = DummyNode(hyper_c, splits)
        node_s = FNode(self.optimizer, self.features, self.samples, parent = dummy)
        node_s.evaluate()
        while not done:
             node_s, done = node_s.greedy_search(greedy)
        greedy_search_result = dict()
        greedy_search_result = node_s.extract(greedy_search_result)
        return greedy_search_result, node_s.eta_hat



