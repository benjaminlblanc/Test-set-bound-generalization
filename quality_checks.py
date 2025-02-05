from upper_bound_quick import bound_optimization
from upper_bound_long import opt

n = 10
a = 0
b = 1
delta = 0.05
tol = 1e-5

for l in [1, 2, 5, 9]:
    for vals in [[0.1, 0.9], [0.2, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8]]:
        print(f"Computing with l = {l} and vals = {vals}.")
        x_weights = vals.copy()
        x_weights.append(1)
        bnd_1 = bound_optimization(n, l, a, b, x_weights, delta, tol)
        bnd_2 = opt(n, l, b, delta, vals, upper=True)
        assert bnd_2 - bnd_1 < 0.01, "Something's wrong..."