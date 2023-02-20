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
import pandas as pd
import matplotlib.ticker
import math


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

def compute_kl_per_state(env, eta_0, eta_hat):
    # kl = 0
    kl = np.inf
    # kl = list()
    for s in range(len(env.state_space)):
        kl = min(sum(rel_entr(eta_0[s, :], eta_hat[s, :])), kl)
        # kl += sum(rel_entr(eta_0[s, :], eta_hat[s, :]))
        # kl.append(sum(rel_entr(eta_0[s, :], eta_hat[s, :])))
    # kl = np.sum(kl)/len(kl)
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

def compare_expected_values(env, eta_true, eta_hat, eta_hat_full):
    MSE_hat = 0
    MSE_full = 0

    for s in range(len(env.state_space)):
        v_hat = np.dot(eta_hat[s][:], env.return_space)
        v_hat_full = np.dot(eta_hat_full[s][:], env.return_space)
        v_true = np.dot(eta_true[s][:], env.return_space)

        MSE_hat += (v_true - v_hat)**2
        MSE_full += (v_true - v_hat_full)**2
    return MSE_hat, MSE_full

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


def generate_plot(kl_list, kl_full, kl_opt, size, type_env, samples_n, dt_string, hyper):
    fig, ax = plt.subplots()
    font1 = {'family':'serif','color':'black','size':12}
    font2 = {'family':'serif','color':'black','size':15}

    t_max = max([len(kl_list[i]) for i in range(len(kl_list))])
    n = range(t_max)

    kl_full_m, kl_full_std = np.mean(kl_full, axis=0)*np.ones(t_max), np.std(kl_full, axis=0)*np.ones(t_max)
    kl_opt_m, kl_opt_std = np.mean(kl_opt, axis=0)*np.ones(t_max), np.std(kl_opt, axis=0)*np.ones(t_max)

    plt.fill_between(n, kl_full_m-kl_full_std, kl_full_m+kl_full_std, alpha=0.2, label='Fully Factorized Solution')
    # plt.plot(n, kl_full_m, linewidth = 2.,alpha=0.7, label='Fully Factorized Solution')
    plt.fill_between(n, kl_opt_m-kl_opt_std, kl_opt_m+kl_opt_std, alpha=0.2, label='Oracle Solution')
    # plt.plot(n, kl_opt_m, linewidth = 2.,alpha=0.7, label='Oracle Solution')
    out = pd.DataFrame()
    for i in range(len(kl_list)):
        out = pd.concat([out, pd.DataFrame({'{}'.format(str(i)): kl_list[i]})], axis=1)
    out = out.to_numpy()
    mean = np.nanmean(out, axis=1)
    std = np.nanstd(out, axis=1)
    std.resize(len(mean))
    n = range(len(mean))
    # plt.plot(n, mean, 'k-.', linewidth = 2., label='Estimated Solution' )
    plt.fill_between(n, mean - std, mean + std, label='Estimated Solution')
    plt.ylabel('KL', fontdict = font1)
    plt.xlabel('Factorization Steps', fontdict = font1)
    plt.title('KL trend with number of factorizations', fontdict = font2)
    ax.grid()
    plt.legend()
    fig.savefig("images/COMPARE_RECT_s{}_t{}_n{}_h{}_d{}.png".format(size, type_env, samples_n, hyper, dt_string))


def generate_full_plot(kl_list, kl_full, kl_opt, size, type_env, samples_n, dt_string, hyper_list, resolution, key):
    output = dict()
    fig, ax = plt.subplots(len(hyper_list), sharex=True)

    t_max = max([len(kl_list[i]) for i in range(len(kl_list))])
    n = range(t_max)
    out = pd.DataFrame()

    kl_full_m, kl_full_std = np.mean(kl_full, axis=0)*np.ones(t_max), np.std(kl_full, axis=0)*np.ones(t_max)
    kl_opt_m, kl_opt_std = np.mean(kl_opt, axis=0)*np.ones(t_max), np.std(kl_opt, axis=0)*np.ones(t_max)

    for i in range(len(hyper_list)):
        ax[i].plot(n, kl_full_m, alpha=0.7)
        ax[i].fill_between(n, kl_full_m-kl_full_std, kl_full_m+kl_full_std, alpha=0.1,  linewidth = 2.5)
        # plt.plot(n, kl_full_m, linewidth = 2.,alpha=0.7, label='Fully Factorized Solution')
        ax[i].plot(n, kl_opt_m, alpha=0.7)
        ax[i].fill_between(n, kl_opt_m-kl_opt_std, kl_opt_m+kl_opt_std, alpha=0.1,  linewidth = 2.5)
        # plt.plot(n, kl_opt_m, linewidth = 2.,alpha=0.7, label='Oracle Solution')

        out = pd.DataFrame()
        c = np.zeros(t_max)
        for k in range(i*resolution, resolution*i+resolution):
            c[len(kl_list[k])-1] += 1
            out = pd.concat([out, pd.DataFrame({'{}'.format(str(k)): kl_list[k]})], axis=1)
        out = out.to_numpy()
        
        mean = np.nanmean(out, axis=1)
        std = np.nanstd(out, axis=1)
        std.resize(len(mean))
        n_single = range(len(mean))
        if len(mean) < 2:
            t1 = n_single[0]
            t2 = mean[0]
            ax[i].scatter([t1], [t2], c='black')
            ax[i].errorbar([t1], [t2], yerr=std[0], c='black')
            # ax[i].fill_between(n_single, mean - std, mean + std, 'k', alpha=0.01)
        else:
            ax[i].plot(n_single, mean, 'k-',  alpha=0.8)
            # ax[i].plot(n_single, mean, 'ko', linewidth = .5, alpha=0.5)
            ax[i].plot(n_single, mean-std, 'k-.',  alpha=0.5)
            ax[i].plot(n_single, mean+std, 'k-.',  alpha=0.5)
            if key == 'KL':
                y_max = max(max(mean[1:]+2*std[1:]), max(kl_opt_m+kl_opt_std/2), max(kl_full_m+kl_full_std/2))
            else:
                ax[i].plot(n_single, mean, 'ko', linewidth = .5)
                y_max = max(max(mean[:]+2*std[:]), max(kl_opt_m+2*kl_opt_std), max(kl_full_m+2*kl_full_std))
            y_min = min(min(mean-2*std), min(kl_full_m-kl_full_std/2), min(kl_opt_m-kl_opt_std/2))
            ax[i].set_ylim([ y_min, y_max])
            # ax[i].set_xlim([1, t_max])
            # ax[i].fill_between(n_single, mean - std, mean + std, 'k', alpha=0.1)
        ax[i].grid()
        ax[i].set_ylabel('{}'.format(key),  fontsize= 14)
        ax[i].set_title(r'$\beta$= {}'.format(round(hyper_list[i])), loc='right')

        output['beta {}'.format(hyper_list[i])] = str(c/sum(c))
        
    plt.xlabel('Factorization Steps',  fontsize= 16)
    new_list = range(0, t_max)
    plt.xticks(new_list)
    # plt.title('KL trend with factorization steps')
    fig.tight_layout()
    # plt.legend()
    fig.savefig("res_images/FULL_COMPARE_HYPER_RECT_s{}_o{}_t{}_n{}_d{}.png".format(size, key, type_env, samples_n, dt_string))
    return output

def return_percentages(N, slices):
    list_n = [round(N*portion, 2) for portion in np.linspace(0.01, 1.5, slices)]
    return list_n