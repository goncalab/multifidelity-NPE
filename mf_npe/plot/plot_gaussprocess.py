# from bayes_opt import BayesianOptimization
# from bayes_opt import acquisition
# import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib import gridspec

# import numpy as np
# import matplotlib.pyplot as plt
# from bayes_opt.util import UtilityFunction

# def plot_bo_2d(optimizer, pbounds, grid_points=80, acq_kind="ucb", kappa=2.576, xi=0.0):
#     """
#     Visualize a 2D Bayesian Optimization run:
#       1) GP posterior mean over (mu, sigma)
#       2) GP posterior std (uncertainty)
#       3) Acquisition landscape for the chosen utility
#       4) Observed points + current suggestion
    
#     Args
#     ----
#     optimizer : bayes_opt.BayesianOptimization
#         A fitted optimizer (after calling .maximize() at least once).
#     pbounds : dict
#         Like {'mu': (a, b), 'sigma': (c, d)}.
#     grid_points : int
#         Resolution per axis for the meshgrid.
#     acq_kind : str
#         'ucb', 'ei', or 'poi' (as in the library).
#     kappa : float
#         UCB parameter.
#     xi : float
#         EI/POI exploration parameter.
#     """
#     # ----- Build a grid over the 2D parameter space
#     mu_lin    = np.linspace(pbounds['mu'][0],    pbounds['mu'][1],    grid_points)
#     sigma_lin = np.linspace(pbounds['sigma'][0], pbounds['sigma'][1], grid_points)
#     MU, SIGMA = np.meshgrid(mu_lin, sigma_lin)
#     X = np.c_[MU.ravel(), SIGMA.ravel()]

#     # ----- GP posterior (mean and std) on the grid
#     # The optimizer keeps its sklearn GP in optimizer._gp
#     y_mean, y_std = optimizer._gp.predict(X, return_std=True)
#     Z_mean = y_mean.reshape(MU.shape)
#     Z_std  = y_std.reshape(MU.shape)

#     # ----- Acquisition on the grid
#     util = UtilityFunction(kind=acq_kind, kappa=kappa, xi=xi)
#     # y_max: best observed so far (the lib maximizes the target)
#     y_max = optimizer.max['target'] if optimizer.max is not None else np.max(y_mean)
#     A = util.utility(X, optimizer._gp, y_max).reshape(MU.shape)

#     # ----- Observed points and targets
#     # optimizer.res is a list of {'target': float, 'params': {...}}
#     obs = np.array([[r['params']['mu'], r['params']['sigma']] for r in optimizer.res])
#     targets = np.array([r['target'] for r in optimizer.res])

#     # Current suggestion from the chosen acquisition:
#     suggestion = optimizer.suggest(util)
#     sug_mu, sug_sigma = suggestion['mu'], suggestion['sigma']

#     # ----- Plot GP mean
#     plt.figure(figsize=(6, 5))
#     plt.contourf(MU, SIGMA, Z_mean, levels=30)
#     if len(obs):
#         plt.scatter(obs[:, 0], obs[:, 1], s=35)
#     plt.scatter([sug_mu], [sug_sigma], marker="*", s=200)
#     plt.xlabel("mu")
#     plt.ylabel("sigma")
#     plt.title("GP posterior mean (objective)")

#     # ----- Plot GP std (uncertainty)
#     plt.figure(figsize=(6, 5))
#     plt.contourf(MU, SIGMA, Z_std, levels=30)
#     if len(obs):
#         plt.scatter(obs[:, 0], obs[:, 1], s=35)
#     plt.scatter([sug_mu], [sug_sigma], marker="*", s=200)
#     plt.xlabel("mu")
#     plt.ylabel("sigma")
#     plt.title("GP posterior std (uncertainty)")

#     # ----- Plot acquisition
#     plt.figure(figsize=(6, 5))
#     plt.contourf(MU, SIGMA, A, levels=30)
#     if len(obs):
#         plt.scatter(obs[:, 0], obs[:, 1], s=35)
#     plt.scatter([sug_mu], [sug_sigma], marker="*", s=200)
#     plt.xlabel("mu")
#     plt.ylabel("sigma")
#     plt.title(f"Acquisition: {acq_kind.upper()} (kappa={kappa}, xi={xi})")
#     plt.show()

# # def target(x):
# #     return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

# def posterior(optimizer, grid):
#     mu, sigma = optimizer._gp.predict(grid, return_std=True)
#     return mu, sigma

# def plot_gp(optimizer, x, y):
#     acquisition_function_ = optimizer.acquisition_function
#     fig = plt.figure(figsize=(16, 10))
#     steps = len(optimizer.space)
#     fig.suptitle(
#         'Gaussian Process and Utility Function After {} Steps'.format(steps),
#         fontsize=30
#     )

#     gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
#     axis = plt.subplot(gs[0])
#     acq = plt.subplot(gs[1])

#     x_obs = np.array([[res["params"]["mu"]] for res in optimizer.res])
#     y_obs = np.array([res["target"] for res in optimizer.res])

#     acquisition_function_._fit_gp(optimizer._gp, optimizer._space)
#     mu, sigma = posterior(optimizer, x)

#     axis.plot(x, y, linewidth=3, label='Target')
#     axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
#     axis.plot(x, mu, '--', color='k', label='Prediction')

#     axis.fill(np.concatenate([x, x[::-1]]),
#               np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
#         alpha=.6, fc='c', ec='None', label='95% confidence interval')

#     axis.set_xlim((-2, 10))
#     axis.set_ylim((None, None))
#     axis.set_ylabel('f(x)', fontdict={'size':20})
#     axis.set_xlabel('x', fontdict={'size':20})

#     utility = -1 * acquisition_function_._get_acq(gp=optimizer._gp)(x)
#     x = x.flatten()

#     acq.plot(x, utility, label='Utility Function', color='purple')
#     acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
#              label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
#     acq.set_xlim((-2, 10))
#     #acq.set_ylim((0, np.max(utility) + 0.5))
#     acq.set_ylabel('Utility', fontdict={'size':20})
#     acq.set_xlabel('x', fontdict={'size':20})

#     axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
#     acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
#     # Save figure
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.savefig(f'bo_gp_{steps}_steps.png')
    
    
#     return fig, fig.axes