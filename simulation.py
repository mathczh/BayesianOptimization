from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
#from math import cos, sin
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def black_box_function(x):
    return -x ** 2 + 40 * x + 10 * np.sin(x) #- (y**2 - 1) ** 2 + 1 + 2 * x + 2 * y - 2 * x * y #+ 3 * x1 * x + x1 * cos(x2) + x1 * y# with strong trend

#x = np.linspace(-20, 20, 10000)#.reshape(-1, 1)
#y = black_box_function(x)
#plt.plot(x,y)
#plt.show()

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={'x': (-20, 20)},# 'y': (-3, 3)},#'x1': (-2,2), 'x2': (-3,3)},
#    verbose=2,
    random_state=27,
)

utility = UtilityFunction(kind="ei", kappa=5, xi=0.0)

next_point_to_probe = optimizer.suggest(utility)
print("Next point to probe is:", next_point_to_probe)

target = black_box_function(**next_point_to_probe)
print("Found the target value to be:", target)

optimizer.maximize(
    init_points=2,
    n_iter=100,
)



def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size':30}
    )
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((-20, 20))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((-20, 20))
    acq.set_ylim((0, np.max(utility) + 50))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

#plot_gp(optimizer, x, y)


#optimizer.maximize(init_points=0, n_iter=1, kappa=5)
#plot_gp(optimizer, x, y)


