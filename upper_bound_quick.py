from scipy.optimize import minimize, LinearConstraint
from time import time
from utils import log_stirling_approximation, sup_bin, optimal_test_bound, multinomial
from exact_values import compute_test_set_bound

import numpy as np
import itertools
import math


def K(n, l, weights):
    """
    Efficiently compute all combinations that:
    1. Sum to n
    2. Have weighted sum <= l

    Parameters:
    - n: total sum constraint
    - l: maximum weighted sum
    - weights: weight for each position

    Returns list of valid combinations
    """

    def recursive_build(curr_n, curr_l, pos, curr_comb):
        # Base cases
        if pos == len(weights):
            if curr_n == 0 and curr_l <= l:
                results.append(curr_comb[:])
            return
        if curr_n < 0 or curr_l > l:
            return

        # Maximum possible value at this position
        max_val = min(curr_n, int(l / weights[pos]) if weights[pos] > 0 else curr_n)

        # Try each possible value at this position
        for i in range(max_val + 1):
            curr_comb[pos] = i
            recursive_build(curr_n - i,
                            curr_l + i * weights[pos],
                            pos + 1,
                            curr_comb)

    results = []
    recursive_build(n, 0, 0, [0] * len(weights))
    return results

def K_jac(combinations, i):
    elements = []
    for element in combinations:
        if element[i] > 0:
            elements.append(element)
    return elements

def F(p, n, combinations):
    m = len(p)
    tot = 0
    for elements in combinations:
        curr = math.log10(multinomial(elements, n))
        for j in range(m):
            curr += elements[j] * math.log10(np.abs(p[j] + 1e-20))
        curr = 10 ** curr
        tot += curr
    return tot

def gradient_F(p, n, combinations):
    """
    Compute the gradient of F with respect to each parameter.

    Parameters:
    - p: Current probability vector
    - n: Total number of samples
    - combinations: Possible combinations satisfying risk constraint
    - delta: Risk threshold

    Returns:
    - Gradient vector representing sensitivity of F to each parameter
    """
    # Ensure probabilities sum to 1 and are positive
    p = np.maximum(p, 1e-10)
    p = p / np.sum(p)
    m = len(p)

    # Initialize gradient vector
    gradient = np.zeros(m)

    # Iterate through each parameter
    for i in range(m):
        # Compute sensitivity for this parameter
        param_sensitivity = 0

        # Iterate through all valid combinations
        for k in combinations[i]:
            # Compute probability of this combination
            try:
                # Log-probability to avoid numerical issues
                log_prob = (
                        log_stirling_approximation(n) -  # Multinomial coefficient
                        sum(log_stirling_approximation(k_j) for k_j in k)  # Individual coefficients
                )

                # Add log probabilities of each parameter
                for j in range(m):
                    # Careful handling of zero probabilities
                    log_prob += k[j] * (np.log(p[j]) if p[j] > 0 else -np.inf)

                # Probability of the combination
                combination_prob = np.exp(log_prob)

                # Derivative of log-probability with respect to parameter i
                # This is k[i]/p[i] when p[i] > 0
                if p[i] > 0:
                    log_prob_derivative = k[i] / p[i] if k[i] > 0 else 0
                else:
                    log_prob_derivative = 0

                # Accumulate gradient component
                param_sensitivity += combination_prob * log_prob_derivative

            except Exception as e:
                # Log any unexpected computational errors
                print(f"Error processing combination {k}: {e}")

        gradient[i] = param_sensitivity

    return gradient

def ineq_constraint(p, weights, alpha, tol):
    return np.dot(p[:-1], weights) + tol * np.dot(weights, weights) * p[-1] - alpha

def obj(p):
    """objective function, to be solved."""
    return -p[-1]

def bound_optimization(n, l, a, b, x_weights, delta, tol):
    print(f"** Lauching bound computation with the following parameters: n = {n}, l = {l}, a = {round(a, 6)}, b = {b}, "
          f"x_weights = {x_weights}, delta = {delta}, tol = {tol} **")
    # Sanity checks
    assert l >= a
    assert x_weights[0] > a
    assert b > x_weights[-1]
    assert 0 <= a < x_weights[0]

    # These are the weights used to compute F, while x_weights are used to compute E
    x_weights.append(b)
    F_weights = np.array(x_weights[:-1].copy()) + 1e-5
    F_weights = np.insert(F_weights, 0, a)
    m = len(F_weights)

    # Constraints parameters
    lower = np.zeros(m + 1)
    upper = np.ones(m + 1)
    upper[-1] = np.inf
    lower_cum = np.zeros(m + 1)
    for i in range(m - 1):
        lower_cum[i] = 1 - round(sup_bin(n, min(l // F_weights[i + 1], n), delta), 6)
    lower_cum[-2], lower_cum[-1] = lower_cum[-3], 1
    curr_upper_bound = compute_test_set_bound(n, l % b, l, b, delta)[0]

    # Constraints are defined
    constraints = [LinearConstraint(np.tri(m + 1), lower_cum, upper),  #  (see Prop.X) to prevent too small values for F
                {'type': 'ineq', 'fun': ineq_constraint, 'args': (x_weights, curr_upper_bound, 0)}]  # Keeps track of
                                                                                                     #  the best bound
    #  Since we are looking for Chebychev center, we need a bounded box to search in. Those constraints are the box
    for i in range(m):
        a = np.zeros(m)
        a[i] = 1
        constraints.append({'type': 'ineq', 'fun': ineq_constraint, 'args': (a.copy(), 0, -1)})
    a = np.ones(m)
    constraints.append({'type': 'ineq', 'fun': ineq_constraint, 'args': (a.copy(), -1, -1)})

    t_init = time()
    # All the possible loss occurrences (< l) are computed
    print("Computing combinations...")
    combinations = K(n, l, F_weights)
    jac_combinations = []
    for i in range(m):
        # To accelerate gradient computation, for each dimension index i of p, only the occurences with k_i > 0 are kept
        jac_combinations.append(K_jac(combinations, i))
    print(f"\n(Took {round(time() - t_init, 2)} sec. to compute combinations.)\n")

    t_init = time()
    upper_bounds = [curr_upper_bound.copy()]  # Keep track of the bounds seen
    last_called = 0
    while True:
        initial_guess = np.zeros(m + 1)  # initial guess can be anything
        initial_guess[0], initial_guess[-1] = 0.5, 0.5
        result = minimize(obj, initial_guess, constraints=constraints)  # Computes Chebychev center of current region
        if F(result.x[:-1], n, combinations) > delta:
            curr_upper_bound = np.dot(result.x[:-1], x_weights)
            if curr_upper_bound - last_called > 0.005:
                last_called = curr_upper_bound.copy()
                print(f"Current bound value: {round(curr_upper_bound, 6)}...")
            # If F > delta, bound computed is valid; updates const. #3 so that each new center must have better bound
            constraints[1] = {'type': 'ineq', 'fun': ineq_constraint, 'args': (x_weights, curr_upper_bound.copy(), -1)}
        else:
            jac = gradient_F(result.x[:-1], n, jac_combinations)
            if np.sqrt(np.dot(jac, jac)) < 1e-10:
                jac /= np.sqrt(np.dot(jac, jac))
            bias = np.dot(jac, result.x[:-1])
            # If F < delta, bound computed is not valid; add constraint so that new center must have better F than that
            constraints.append({'type': 'ineq', 'fun': ineq_constraint, 'args': (jac.copy(), bias.copy(), -1)})

        upper_bounds.append(curr_upper_bound)
        if len(upper_bounds) > 150:
            if upper_bounds[-1] - upper_bounds[-150] <= tol:
                break
    assert upper_bounds[-1] > upper_bounds[0], "Something went wrong :("
    print(f"\nFinal upper bound: {round(upper_bounds[-1], 6)} (took {round(time() - t_init, 2)} sec. to compute, once combinations were computed).")
    return upper_bounds[-1]

n = 10
l = 0
delta = 0.05
a = 0#optimal_test_bound(1000, delta)
b = 1
x_weights = [0.2, 0.6, 0.8]
tol = 1e-5

bound_optimization(n, l, a, b, x_weights, delta, tol)