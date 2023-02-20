# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################
import numpy as np
import itertools
from Utils import algorithm_u, subsets_k, partition, get_partitions

###############################################################################
################################ Global variables #############################
###############################################################################


###############################################################################
####################### Class FeatureGenerator #########################
###############################################################################



class FeatureGenerator(object):
    
    def __init__(self, env, vector):
        # Initialization of important variables
        self.env = env
        self.vectorized = vector
        self.features =  feature_class(self.env, self.vectorized, [self.env.state_space])
        self.features_full =  feature_class(self.env, self.vectorized, [[s] for s in self.env.state_space])
        set_double = [[s for s in range(h*self.env.high, h*self.env.high + self.env.size)]  for h in range(self.env.size)]
        self.features_opt =  feature_class(self.env, self.vectorized, set_double)


    def expand_features(self, i_star = None, splits = 2):
        next_features = list()
        done = True
        set_star = self.sets[i_star] if i_star else self.sets
        for set_id, set in enumerate(set_star):
            set_expanded, empty = split_set_and_add(i_star, set, set_star, splits) if i_star else split_set_and_add(set_id, set, set_star, splits)
            if not empty:
                for set_exp in set_expanded:
                    next_features.append(feature_class(self.env, self.vectorized, set_exp))
                done = False
        return next_features, done


def split_set_and_add(set_id, set, sets, splits):
    threshold = len(set)//splits if hasattr(set, '__len__') else 0
    empty = False
    if threshold >=1:
        out = list()
        for s in get_partitions(set, threshold)[1:]:
        # for s in algorithm_u(set, splits):
            # if all(len(el) >= threshold for el in s):
            sets_add = sets.copy()
            sets_add[set_id:set_id+len(s)-1] = [s_i for s_i in s]
            out.append(sets_add)
            # else:
            #     next
    else:
        out = None
        empty = True
    return out, empty


class feature_class():

    def __init__(self, env, vectorized , sets):
        self.env = env
        self.s_dim = len(self.env.state_space)
        self.a_dim = len(self.env.action_space)
        self.r_dim = len(self.env.return_space)
        self.vectorized = vectorized
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


    def apply_functions(self, samples, counter = False):
        output = list()
        if not self.vectorized:
            for feature_element in self.feature_vector:
                if counter:
                    output.append(sum([feature_element(sample, counter) for sample in samples]))
                else:
                    output.append(sum([feature_element(sample, counter) for sample in samples])/len(samples))
        else:
            for feature_element in self.feature_vector:
                if counter:
                    output.append(sum(feature_element(samples, counter)))
                else:
                    output.append(sum(feature_element(samples, counter))/len(samples))
                    
        return output


    def feature_from_set(self, set):
        if not self.vectorized:
            def f(input, counter=False):
                # input: state, action, return
                state = input[0]
                g  = input[1] if not counter else 1
                if hasattr(set, "__len__"):
                    output = g if state in set else 0.
                else:
                    output = g if state == set else 0.
                return output
        else:
            def f(input, counter=False):
                output = np.zeros(len(input))
                for idx, el in enumerate(input):
                    # input: state, action, return
                    state = el[0]
                    g  = el[1] if not counter else 1
                    if hasattr(set, "__len__"):
                        output[idx] = g if state in set else 0.
                    else:
                        output[idx] = g if state == set else 0.
                return output
        f.__name__ = str(set)
        return f

