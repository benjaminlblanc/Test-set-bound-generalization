from scipy.optimize import minimize, LinearConstraint
from time import time
from utils import log_stirling_approximation, sup_bin, optimal_test_bound, multinomial

import numpy as np
import itertools
import math


def K(n, l, weights):
    possible_values = []
    for j in range(len(weights)):
        if j == 0:
            possible_values.append(list(range(n + 1)))
        elif l == weights[j]:
            possible_values.append(list(range(2)))
        else:
            val = min(math.floor(l / (weights[j] + 1e-10)), n)
            possible_values.append(list(range(val + 1)))

    elements = []
    for element in itertools.product(*possible_values):
        if np.sum(np.array(element) * np.array(weights)) <= l and np.sum(element) == n:
            elements.append(element)
    return elements

def K_jac(combinations, i):
    elements = []
    for element in combinations:
        if element[i] > 0:
            elements.append(element)
    return elements

def F(p, n, combinations, m):
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
    assert b == x_weights[-1]
    assert a < x_weights[0]

    # These are the weights used to compute F, while x_weights are used to compute E
    F_weights = np.array(x_weights[:-1].copy()) + 1e-5
    F_weights = np.insert(F_weights, 0, a)
    m = len(F_weights)

    # Constraints parameters
    lower = np.zeros(m + 1)
    upper = np.ones(m + 1)
    upper[-1] = np.inf
    lower_cum = np.zeros(m + 1)
    for i in range(m - 1):
        lower_cum[i] = 1 - round(sup_bin(n, min(l // F_weights[i + 1], n), delta), 8)
    lower_cum[-2], lower_cum[-1] = 0, 1
    upper_bound = 0

    # Constraints are defined
    constraints = [LinearConstraint(np.eye(m + 1), lower, upper),  # Each probability is between 0 and 1
                   LinearConstraint(np.tri(m + 1), lower_cum, upper),  # Each cumulative probability is upper bounded
                                                                       #  (see Prop.X) to prevent too small values for F
                   {'type': 'ineq', 'fun': ineq_constraint, 'args': (x_weights, upper_bound, -1)}]  # Keeps track of the
                                                                                                    #  best bound seen
    t_init = time()
    # All the possible loss occurrences (< l) are computed
    combinations = K(n, l, F_weights)
    jac_combinations = []
    for i in range(m):
        # To accelerate gradient computation, for each dimension index i of p, only the occurences with k_i > 0 are kept
        jac_combinations.append(K_jac(combinations, i))
    print(f"\n(Took {round(time() - t_init, 2)} sec. to compute combinations.)\n")

    t_init = time()
    upper_bounds = []  # Keep track of the bounds seen
    while True:
        initial_guess = np.zeros(m + 1) + 1 / m  # initial guess can be anything
        result = minimize(obj, initial_guess, constraints=constraints)  # Computes Chebychev center of current region
        if F(result.x[:-1], n, combinations, m) > delta:
            alpha = np.dot(result.x[:-1], x_weights)
            print(f"Current bound value: {round(alpha, 6)}...")
            # If F > delta, bound computed is valid; updates const. #3 so that each new center must have better bound
            constraints[2] = {'type': 'ineq', 'fun': ineq_constraint, 'args': (x_weights, alpha, -1)}
        else:
            jac = gradient_F(result.x[:-1], n, jac_combinations)
            if np.sqrt(np.dot(jac, jac)) < 1e-10:
                jac /= np.sqrt(np.dot(jac, jac))
            bias = np.dot(jac, result.x[:-1])
            # If F < delta, bound computed is not valid; add constraint so that new center must have better F than that
            constraints.append({'type': 'ineq', 'fun': ineq_constraint, 'args': (jac.copy(), bias.copy(), -1)})

        upper_bounds.append(upper_bound)
        if len(upper_bounds) > 100:
            if upper_bounds[-1] - upper_bounds[-100] <= tol:
                break
    print(f"\nFinal upper bound: {round(upper_bounds[-1], 6)} (took {round(time() - t_init, 2)} sec. to compute, once combinations were computed).")
    return upper_bounds[-1]

n = 10
l = 2
delta = 0.05
a = optimal_test_bound(1000, delta)
b = 1
x_weights = [0.2, 0.6, 0.8, 1]
tol = 1e-5

bound_optimization(n, l, a, b, x_weights, delta, tol)