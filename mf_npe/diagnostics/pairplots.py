

import os
import matplotlib
import torch
matplotlib.use("Agg")  # non-interactive; no windows, fewer semaphores
from matplotlib import pyplot as plt
from sbi import analysis as analysis
from sbi.analysis import conditional_corrcoeff, conditional_pairplot

class PairPlot():
    """
    Class to create pairplots for posterior samples.
    """

    def __init__(self, true_xen, task_setup):
        self.true_xen = true_xen
        self.task_setup = task_setup
        
        self.hf_simulator = task_setup.hf_simulator
        self.lf_simulator = task_setup.lf_simulator
        self.config_data = task_setup.config_data
        self.main_path = task_setup.main_path
        
        self.theta_dim = task_setup.theta_dim
        
        # Number of posterior samples to generate
        self.n_samples_to_generate = 1000


    def _plot_pairplot(self, posterior_samples, true_theta, type_estimator, n_train_sims, i): 
                                              
        fig, axes = analysis.pairplot(
                posterior_samples,
                limits=self.hf_simulator.parameter_ranges(self.theta_dim),
                ticks=self.hf_simulator.parameter_ranges(self.theta_dim),
                figsize=(5, 5),
                points=true_theta,
                points_offdiag={"markersize": 6},
                points_colors="r",
                title=f"method: {type_estimator} (n_sims: {n_train_sims})",
                labels=[rf"$\theta_{d}$" for d in range(self.theta_dim)],
        )
        
        # Save the plot
        path_paiplots = f'{self.main_path}/pairplots/'
        if not os.path.exists(path_paiplots):
            os.makedirs(path_paiplots)
            
        plt.savefig(f"{path_paiplots}/{type_estimator}_{n_train_sims}_{i}.svg")
        plt.savefig(f"{path_paiplots}/{type_estimator}_{n_train_sims}_{i}.pdf")

        plt.close(fig)
        


    def _plot_pairplot_lf_hf(self, posterior_samples, lf_posterior_samples, true_theta, type_estimator, n_train_sims):                                
        fig, axes = analysis.pairplot(
                [posterior_samples, lf_posterior_samples],
                limits=self.hf_simulator.parameter_ranges(self.theta_dim),
                ticks=self.hf_simulator.parameter_ranges(self.theta_dim),
                figsize=(5, 5),
                points=true_theta,
                points_offdiag={"markersize": 6},
                points_colors="r",
                title=f"method: {type_estimator} (n_sims: {n_train_sims})",
                labels=[rf"$\theta_{d}$" for d in range(self.theta_dim)],
        )
        
        # Save the plot
        path_pairplots = f'{self.main_path}/pairplots/'
        if not os.path.exists(path_pairplots):
            os.makedirs(path_pairplots)
            
        fig.savefig(f"{path_pairplots}/lf_hf_{type_estimator}_{n_train_sims}_{self.config_data['type_lf']}.svg")
        fig.savefig(f"{path_pairplots}/lf_hf_{type_estimator}_{n_train_sims}_{self.config_data['type_lf']}.pdf")


        
    def _plot_pairplot_with_true_posterior(self, posterior_samples, true_posterior_samples, true_theta, type_estimator, n_train_sims, name='true_comparison', simulator_name='simulator'):                                        
        if getattr(self.hf_simulator, "prior_ranges", None) is not None:
            limits = self.hf_simulator.parameter_ranges(self.theta_dim)
        else:
            limits = None
        
        fig, axes = analysis.pairplot(
                [posterior_samples, true_posterior_samples],
                **({"limits": limits} if limits is not None else {}),
                #limits=torch.tensor([[0, 1]] * self.theta_dim),  # for lotka-volterra [0, 2] * self.theta_dim,
                #ticks=self.hf_simulator.parameter_ranges(self.theta_dim),
                #upper=['hist', 'contour'],
                upper=['contour', 'contour'],
                upper_kwargs = {
                    "bw_method": "scott",
                    "bins": 50,
                    "levels": [0.68, 0.95, 0.99],
                    "percentile": True,
                    "mpl_kwargs": {
                        "colors": ['#FFA15A']}},
                figsize=(5, 5),
                points=true_theta,
                points_offdiag={"markersize": 6},
                points_colors="r",
                # title=f"method: {type_estimator} (n_sims: {n_train_sims})",
                title=f"{simulator_name})",
                labels=[rf"$\theta_{d}$" for d in range(self.theta_dim)],
        )
        
        path_pairplots = f'{self.main_path}/pairplots/'
        if not os.path.exists(path_pairplots):
            os.makedirs(path_pairplots)
            
        fig.savefig(f"{path_pairplots}/{name}_{type_estimator}_{n_train_sims}.svg")
        fig.savefig(f"{path_pairplots}/{name}_{type_estimator}_{n_train_sims}.pdf")


    def _plot_conditional_pairplot(self, posterior, true_x):
        print("single sample", posterior.sample((1,), x=true_x))
        print("shape sample", posterior.sample((1,), x=true_x).shape)
        
        posterior.set_default_x(true_x)
        
        # Plot slices through posterior, i.e. conditionals.
        _ = conditional_pairplot(
            density=posterior,
            condition=posterior.sample((1,)),
        )       
        if self.show_plots:
            plt.show()     