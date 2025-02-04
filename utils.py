import math
import numpy as np


def log_stirling_approximation(n):
    """
    Stirling's approximation for the logarithm of the factorial
    """
    if n == 0:
        return 0
    if n < 30:
        return math.log(math.factorial(n))
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)


def log_binomial_coefficient(n, k):
    """
    Logarithm of the binomial coefficient using Stirling's approximation
    """
    return (log_stirling_approximation(n) -
            log_stirling_approximation(k) -
            log_stirling_approximation(n - k))


def log_prob_bin(k, n, r):
    """
    Logarithm of P(x = k), if X ~ Bin(n, r)
    """
    return log_binomial_coefficient(n, k) + k * math.log(max(r, 1e-10)) + (n - k) * math.log(max(1 - r, 1e-10))


def bin_cum(n, k, r, step=1):
    """
    Logarithm of P(x <= k), if X ~ Bin(n, r)
    """
    prob_cum = 0
    for i in range(int(k / step) + 1):
        prob_cum += math.exp(log_prob_bin(i * step, n, r))
    return prob_cum


def bin_b(n, k, delta):
    """
    Estimation of sup(r : P(x <= k) >= delta), if X ~ Bin(m, r)
    """
    r_sup, r_inf, r = 1, 0, 0.5
    for j in range(20):
        pro = bin_cum(n, k, r)
        if pro >= delta:
            r_inf = r
        else:
            r_sup = r
        r = (r_sup + r_inf) / 2
    return r

def multinomial(lst):
    res, i = 1, 1
    for a in lst:
        for j in range(1,a+1):
            res *= i
            res //= j
            i += 1
    return res

def sup_bin(n, k, delta, step=1):
    """
    Estimation of sup(r : P(x <= k) >= delta), if X ~ Bin(m, r)
    """
    gamma_sup, gamma_inf, gamma = 1, 0, 0.5
    for j in range(20):
        pro = bin_cum(n, k, gamma, step)
        if pro >= delta:
            gamma_inf = gamma
        else:
            gamma_sup = gamma
        gamma = (gamma_sup + gamma_inf) / 2
    return gamma

def inf_bin(k, m, delta):
    """
    Estimation of inf(r : P(x <= k) >= 1 - delta), if X ~ Bin(m, r)
    """
    gamma_sup, gamma_inf, gamma = 1, 0, 0.5
    for j in range(20):
        pro = bin_cum(k, m, gamma)
        if 1 - pro >= delta:
            gamma_sup = gamma
        else:
            gamma_inf = gamma
        gamma = (gamma_sup + gamma_inf) / 2
    return gamma

def multinomial(lst, n):
    assert np.sum(lst) == n
    res = log_stirling_approximation(n)
    for a in lst:
        res -= log_stirling_approximation(a)
    return math.exp(res)

def optimal_test_bound(m, d):
    return 1 - math.exp(-math.log(1 / d) / m)