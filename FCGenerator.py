# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################
import numpy as np
import itertools
from Utils import algorithm_u, subsets_k, partition

###############################################################################
################################ Global variables #############################
###############################################################################


###############################################################################
####################### Class FeatureGenerator #########################
###############################################################################



class FeatureGenerator(object):
    
    def __init__(self, env):
        # Initialization of important variables
        self.env = env
        self.features =  feature_class(self.env, [self.env.state_space])

    def expand_features(self, i_star = None, splits = 2):
        next_features = list()
        done = True
        set_star = self.sets[i_star] if i_star else self.sets
        for set_id, set in enumerate(set_star):
            set_expanded, empty = split_set_and_add(i_star, set, set_star, splits) if i_star else split_set_and_add(set_id, set, set_star, splits)
            if not empty:
                for set_exp in set_expanded:
                    next_features.append(feature_class(self.env, set_exp))
                done = False
        return next_features, done

    def generate_feature_plus(self, feature_class):
        next

    def generate_feature_minus(self, feature_class):
        next


def split_set_and_add(set_id, set, sets, splits):
    threshold = len(set)//splits if hasattr(set, '__len__') else 0
    empty = False
    if threshold >=1:
        out = list()
        for s in partition(set):
        #algorithm_u(set, splits):
            sets_add = sets
            if all(len(el) >= threshold for el in s):
                sets_add[set_id:set_id+len(s)-1] = [s_i for s_i in s] 
                out.append(sets_add)
            else:
                next
    else:
        out = None
        empty = True
    return out, empty


class feature_class():

    def __init__(self, env, sets):
        self.env = env
        self.s_dim = len(self.env.state_space)
        self.a_dim = len(self.env.action_space)
        self.r_dim = len(self.env.return_space)
        self.sets = sets
        feature_vector = list()
        if hasattr(sets, '__len__'):
            for set in sets:
                f = self.feature_from_set(set)
                feature_vector.append(f)
        else:
            set = [sets]
            f = self.feature_from_set(set)
            feature_vector.append(f)
        self.feature_vector = feature_vector


    def apply_functions_non_vectorized(self, samples, counter = False):
        output = list()
        for feature_element in self.feature_vector:
            output.append(sum([feature_element(sample, counter) for sample in samples])/len(samples))
        return output

    def feature_from_set(self, set):

        def f(input, counter=False):
            # input: state, action, return
            state = input[0]
            g  = input[1] if not counter else 1
            if hasattr(set, "__len__"):
                output = g if state in set else 0.
            else:
                output = g if state == set else 0.
            return output
        f.__name__ = str(set)
        return f

    def feature_from_set_matrix(self, set):
        f_matrix = np.zeros((self.s_dim, self.a_dim, self.r_dim))
        for state in set:
            f_matrix[state][:][:] = 1

        def f(input):
            # input: state, action, return
            output = f_matrix*input
            return output
        f.__name__ = str(set)
        return f

