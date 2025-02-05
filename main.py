from upper_bound_quick import bound_optimization

n = 10
l = 0
delta = 0.05
a = 0#optimal_test_bound(1000, delta)
b = 1
x_weights = [0.2, 0.6, 0.8, 1]
tol = 1e-5

bound_optimization(n, l, a, b, x_weights, delta, tol)