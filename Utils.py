# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import itertools
import numpy  as np
import matplotlib.pyplot as plt
from itertools import combinations, islice
from scipy.special import rel_entr
from scipy.stats import entropy
from math import comb
import matplotlib.ticker


###############################################################################
################################ Global variables #############################
###############################################################################




###############################################################################
####################### Utilities #########################
###############################################################################


def plot_solution(distribution, dims):
    for s, a in zip(range(dims[0]), range(dims[1])):
        x = distribution[s, a, :]
        counts, bins = np.histogram(x)
        plt.xlabel('Return')
        plt.ylabel('Probability')
        plt.title('Histogram of Z({},{})'.format(s,a))
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()

def subsets_k(collection, k): yield from partition_k(collection, k, k)

def partition_k(collection, min, k):
  if len(collection) == 1:
    yield [ collection ]
    return

  first = collection[0]
  for smaller in partition_k(collection[1:], min - 1, k):
    if len(smaller) > k: continue
    # insert `first` in each of the subpartition's subsets
    if len(smaller) >= min:
      for n, subset in enumerate(smaller):
        yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
    # put `first` in its own subset 
    if len(smaller) < k: yield [ [ first ] ] + smaller

def compute_kl(env, eta_0, eta_hat):
    eta_true = np.ravel(eta_0)
    eta_est = np.ravel(eta_hat)
    kl = sum(rel_entr(eta_true, eta_est))
    return kl

def compute_entropy(env, eta_0):
    eta = np.ravel(eta_0)
    entr = entropy(eta)
    return entr

def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller


def algorithm_u(ns, m):
    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)

def get_partitions(iterable, minl=2):
    s = set(iterable)
    return [list(s)]+\
           [[list(c), list(s.difference(c))]
            for r in range(minl, len(s)//2+1)
            for c in ( combinations(s, r) if len(s)//2 != r else
             islice(combinations(s, r), comb(len(s),r)//2))
           ]

def compare_values(env, eta_hat, eta_hat_full, eta_true):
    RMSE_hat = 0
    RMSE_full = 0

    for s in range(len(env.state_space)):
        v_hat = np.dot(eta_hat[s][:], env.return_space)
        v_hat_full = np.dot(eta_hat_full[s][:], env.return_space)

        v_true = np.sum(eta_true[s][:])
        RMSE_hat += (v_true - v_hat)**2
        RMSE_full += (v_true - v_hat_full)**2
    RMSE_full = np.sqrt(RMSE_full)
    RMSE_hat = np.sqrt(RMSE_hat)
    return RMSE_hat, RMSE_full

def plot(v_est, v_full, samples, title, full_name):
    fig, ax = plt.subplots()

    font1 = {'family':'serif','color':'black','size':15}
    font2 = {'family':'serif','color':'black','size':20}

    v_max = max(max(v_est), max(v_full))
    v_min = min(min(v_est), min(v_full))
    ax.plot(samples, v_est, '--', label= 'Factorized Representation', linewidth=2.)
    ax.plot(samples, v_full, '-.',  label= 'Full Representation', linewidth=2.)
    plt.xlabel('N', fontdict = font1)
    plt.ylabel('Normalized {}'.format(title), fontdict = font1)
    plt.title('{} trend with number of samples'.format(title), fontdict = font2)
    ax.grid()
    plt.legend(loc='upper right')
    ax.set_yticklabels(ax.get_yticks())

    fig.savefig("images/PLOT_{}_{}.png".format(title, full_name))
    


def return_percentages(N, slices):
    list_n = [int(N*portion) for portion in np.linspace(0.01, 1, slices)]
    return list_n