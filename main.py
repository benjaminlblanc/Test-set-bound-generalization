import itertools

from jupyter_server.utils import run_sync_in_loop

from utils import *
import numpy as np
from time import time

class Func_repartition_tri:
    def __init__(self, n, weights, tot_loss):
        self.n = n
        self.weights = weights
        self.tot_loss = tot_loss
        possible_values = []
        terms = []
        for i in range(len(weights)):
            val = min(math.floor(tot_loss / (weights[i] - 1e-10)), n)
            if weights[i] == 0:
                possible_values.append(list(range(n + 1)))
            elif tot_loss == weights[i]:
                possible_values.append(list(range(2)))
            else:
                possible_values.append(list(range(val + 1)))
        for elements in itertools.product(*possible_values):
            if np.sum(np.array(elements) * np.array(weights)) <= tot_loss and np.sum(elements) == n:
                terms.append(elements)
        self.terms = terms

    def compute_prob(self, p):
        prob = 0
        for elements in self.terms:
            curr = math.log10(multinomial(elements))
            for j in range(len(self.weights)):
                curr += elements[j] * math.log10(np.abs(p[j] + 1e-20))
            curr = 10 ** curr
            prob += curr
        return prob

def tri_b_round(F_tri, r, curr_bound, confidence, weights, direction):
        if direction == '1':
            val_1, val_2 = 1, 2
        else:
            val_1, val_2 = 2, 1
        delta_lower, delta_upper = 0, min(1 - r[val_1] + 1e-5, r[val_2] - 1e-5)
        best_r, best_bound = r.copy(), curr_bound.copy()
        for i in range(15):
            delta = (delta_upper + delta_lower) / 2
            r[val_1] += delta
            r[val_2] -= delta
            new_r, success = tri_cum(F_tri, r.copy(), confidence, direction)
            new_bound = np.sum(new_r * weights)
            if success:
                delta_upper = delta
                if best_bound <= new_bound:
                    best_r = new_r.copy()
                    best_bound = new_bound.copy()
            else:
                delta_lower = delta
            r[val_1] -= delta
            r[val_2] += delta
        return best_r, best_bound

def tri_b(n, total_loss, confidence, weights):
    r = np.array([0.99, 0.005, 0.005])
    curr_bound = np.sum(r * weights)
    F_tri = Func_repartition_tri(n, weights, total_loss)
    best_bound = 0
    while True:
        r, curr_bound = tri_b_round(F_tri, r, curr_bound, confidence, weights, direction='1')
        r, curr_bound = tri_b_round(F_tri, r, curr_bound, confidence, weights, direction='2')
        if curr_bound > best_bound:
            best_bound = curr_bound
        else:
            break
    return r

def tri_cum(F_tri, r, confidence, direction):
    if direction == '1':
        val_1, val_2 = 2, 0
    else:
        val_1, val_2 = 0, 2

    delta_lower, delta_upper = 0, min(1 - r[val_1] + 1e-5, r[val_2] - 1e-5)
    best_r = r.copy()
    success = False
    for i in range(20):
        delta = (delta_upper + delta_lower) / 2
        r[val_1] += delta
        r[val_2] -= delta
        curr_prob = F_tri.compute_prob(r)
        if confidence <= curr_prob:
            best_r = r.copy()
            success = True
            if direction == '1':
                delta_lower = delta
            else:
                delta_upper = delta
        else:
            if direction == '1':
                delta_upper = delta
            else:
                delta_lower = delta
        r[val_1] -= delta
        r[val_2] += delta
    return best_r, success

def compute_test_set_bound(n, mod, total_loss, loss_upper_bound, confidence):
    """
    Method of Centers algorithm to find the tightest bound

    Parameters:
    -----------
    level : float
        Type of bound to be computed depending on knowing the loss classes ('known_classes'), knowing the loss upper
            bound ('known_upper_bound') or not knowing anything ('unknown').

    Returns:
    --------
    float: Tightest bound
    """
    #mod = (total_loss % loss_upper_bound) / (n - total_loss // loss_upper_bound)
    if mod != 0:
        weights = np.array([0, mod, loss_upper_bound])
        print(weights)
        p = tri_b(n, total_loss, confidence, weights)
        return np.sum(p * weights), p
    else:
        p = bin_b(n, int(total_loss // loss_upper_bound), confidence)
        return np.sum(p * loss_upper_bound), p

# Example usage
def main():
    n = 10  # Number of sample point
    total_loss = 1  # Number of errors in each category
    loss_upper_bound = 1  # Type of bound to compute
    delta = 0.05  # 95% confidence level
    for j in [total_loss / n, total_loss % loss_upper_bound]:
        # Initialize and compute bound
        init_time = time()
        test_set_bound, p = compute_test_set_bound(n, j, total_loss, loss_upper_bound, delta)
        print(test_set_bound, p)
    print(f"Test Set Bound: {np.round(test_set_bound, 6)}; test average loss: {round(total_loss / n, 4)} (took {round(time()-init_time, 2)} sec. to compute).")
if __name__ == "__main__":
    main()