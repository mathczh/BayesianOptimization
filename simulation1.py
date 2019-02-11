from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from math import cos
def black_box_function(x, y):#, x1, x2):
    return -x ** 2 - (y**2 - 1) ** 2 + 1 + 100 * x + 200 * y - 2 * x * y# + 3 * x1 * x + x1 * cos(x2) + x1 * y# with strong trend

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={'x': (-10, 10), 'y': (-15, 15)},# 'x1': (-2,2), 'x2': (-3,3)},
    verbose=2,
    random_state=1,
)

utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

next_point_to_probe = optimizer.suggest(utility)
print("Next point to probe is:", next_point_to_probe)

target = black_box_function(**next_point_to_probe)
print("Found the target value to be:", target)

optimizer.maximize(
    init_points=3,
    n_iter=1000,
)
