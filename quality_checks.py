from upper_bound_quick import bound_optimization
from upper_bound_long import opt

n = 10
a = 0
b = 1
delta = 0.05
tol = 1e-5

for l in [0, 0.1, 1, 2, 2.5, 5, 8, 9.5]:
    for vals in [[0.1, 0.9], [0.2, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8], [0.1, 0.4, 0.5, 0.6, 0.95]]:
        print(f"Computing quality checks with l = {l} and vals = {vals}.")
        bnd_1 = bound_optimization(n, l, a, b, vals, delta, tol)
        bnd_2, delta_constraint = opt(n, l, b, delta, vals, upper=True)
        assert not (delta_constraint > 0 and bnd_2 - bnd_1 > 0.01), "Something's wrong..."