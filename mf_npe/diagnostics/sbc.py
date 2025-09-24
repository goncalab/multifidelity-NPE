import os
import torch
from mf_npe.utils.utils import dump_pickle, summary_statistics_wrapper
from sbi.analysis.plot import sbc_rank_plot
from sbi.diagnostics.sbc import _run_sbc

import os
from sbi import utils as utils
import torch
from sbi.analysis.plot import sbc_rank_plot
from sbi.diagnostics.sbc import _run_sbc
from mf_npe.utils.utils import dump_pickle
from sbi import analysis as analysis
from mf_npe.utils.utils import summary_statistics_wrapper


class SimulationBasedCalibration():
    def __init__(self, true_xen, task_setup):
        self.true_xen = true_xen
        self.task_setup = task_setup
        
        self.hf_simulator = task_setup.hf_simulator
        self.lf_simulator = task_setup.lf_simulator
        self.config_data = task_setup.config_data
        self.main_path = task_setup.main_path
        
        # Number of posterior samples to generate
        self.n_samples_to_generate = 1000


    def run_sbc(self, posterior, type_estimator, n_simulations):
        # Don't put it too low, since num_bins is default /20
        num_sbc_samples = 100 # 1000 #200
        num_posterior_samples = self.n_samples_to_generate        
        xs, thetas, add_ons = summary_statistics_wrapper(num_sbc_samples, self.hf_simulator, self.config_data, self.main_path)         
        # Create posterior samples
        posterior_samples = []
        for i in range(num_posterior_samples): # typically 1000
            density_estimator = posterior.__dict__['posterior_estimator']
            posterior_sample = [density_estimator.sample((1,), xs[j])[0] for j in range(len(xs))]
            posterior_samples.append(torch.stack(posterior_sample))
        posterior_samples = torch.stack(posterior_samples)

        # Noting happens with posterior here, so we can just pass our logit. Since the code only uses it for VIPosterior
        ranks = _run_sbc(thetas, xs, posterior_samples, posterior, "marginals", show_progress_bar=False)
        
        # check_stats = check_sbc(ranks, thetas, dap_samples, num_posterior_samples=num_posterior_samples)
        # print(f"kolmogorov-smirnov p-values: {check_stats['ks_pvals'].numpy()}")
        # print(f"c2st accuracies: {check_stats['c2st_ranks'].numpy()}")
        # print(f"c2st accuracies (dap): {check_stats['c2st_dap'].numpy()}")
        
        sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="hist",
            num_bins=None,  # by passing None we use a heuristic for the number of bins.
        )
        
        rank_dic = {
            'ranks': ranks,
            'num_posterior_samples': num_posterior_samples
        }
        
        # Save a pickle with ranks
        save_dir = f"{self.main_path}/sbc"
        name = f"sbc_{type_estimator}_{n_simulations}.p"
        dump_pickle(save_dir, name, rank_dic)
        
        fig, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="cdf")
        
        plot_dir = f"{save_dir}/plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Save the CDF plot
        print(f"Saving SBC CDF plot to {plot_dir}")
        fig.savefig(f"{plot_dir}/sbc_cdf_{n_simulations}_{self.config_data['type_lf']}.svg", format="svg")
        fig.savefig(f"{plot_dir}/sbc_cdf_{n_simulations}_{self.config_data['type_lf']}.pdf", format="pdf")
        

        

    def run_sbc_lf_hf(self, posterior, lf_posterior, type_estimator, n_simulations):
        # Don't put it too low, since num_bins is default /20
        num_sbc_samples = 1000 # 1000 #200
        num_posterior_samples = self.n_samples_to_generate
        
        # "True data for sbc"
        xs, thetas, add_ons = summary_statistics_wrapper(num_sbc_samples, self.hf_simulator, self.config_data, self.main_path)   
        # lf_xs, lf_thetas, lf_add_ons = summary_statistics_wrapper(num_sbc_samples, self.lf_simulator, self.task)     
        
        # Create posterior samples hf model
        posterior_samples = []
        for i in range(num_posterior_samples): # typically 1000
            density_estimator = posterior.__dict__['posterior_estimator']
            posterior_sample = [density_estimator.sample((1,), xs[j])[0] for j in range(len(xs))]
            posterior_samples.append(torch.stack(posterior_sample))
        posterior_samples = torch.stack(posterior_samples)


        # Create posterior samples lf model
        lf_posterior_samples = []
        for i in range(num_posterior_samples): # typically 1000
            lf_density_estimator = lf_posterior.__dict__['posterior_estimator']
            # SBC must be evaluated on true_xs, since we want to see how uncalibrated it is to compared to hf data
            lf_posterior_sample = [lf_density_estimator.sample((1,), xs[j])[0] for j in range(len(xs))]
            lf_posterior_samples.append(torch.stack(lf_posterior_sample))
        lf_posterior_samples = torch.stack(lf_posterior_samples)

        # Noting happens with posterior here, so we can just pass our logit. Since the code only uses it for VIPosterior
        ranks = _run_sbc(thetas, xs, posterior_samples, posterior, "marginals", show_progress_bar=False)
        # lf ranks computed with respect to true x and true theta
        lf_ranks = _run_sbc(thetas, xs, lf_posterior_samples, lf_posterior, "marginals", show_progress_bar=False)
        
        
        # Save a pickle with ranks
        save_dir = f"{self.main_path}/sbc"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plot_dir = f"{save_dir}/plots"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # check_stats = check_sbc(ranks, thetas, dap_samples, num_posterior_samples=num_posterior_samples)
        # print(f"kolmogorov-smirnov p-values: {check_stats['ks_pvals'].numpy()}")
        # print(f"c2st accuracies: {check_stats['c2st_ranks'].numpy()}")
        # print(f"c2st accuracies (dap): {check_stats['c2st_dap'].numpy()}")
        
        # Create a single figure and axes
        # fig, ax = plt.subplots(figsize=(6, 4))  # You can adjust figsize
        
        fig, ax =  sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="hist",
            num_bins=None,  # by passing None we use a heuristic for the number of bins.
            ranks_labels=["HF"],
            colors=["#0000A6"], # blue
        )
        
        sbc_rank_plot(
            ranks=lf_ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="hist",
            ranks_labels=["LF"],
            colors=["#7AAE41"], #green
            ax=ax,  # reuse the returned ax array
            fig=fig
        )
        

        fig.savefig(f"{plot_dir}/sbc_lf_hf_histogram_{n_simulations}_{self.config_data['type_lf']}.svg", format="svg")
        fig.savefig(f"{plot_dir}/sbc_lf_hf_histogram_{n_simulations}_{self.config_data['type_lf']}.pdf", format="pdf")
        
        num_params = ranks.shape[1]
        
        fig_cdf, ax_cdf = sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="cdf",
            num_bins=None,  # by passing None we use a heuristic for the number of bins.
            ranks_labels=["HF"] * num_params,
            colors=['#0000A6', '#7E7ED2'], # Blue
        )
        
        sbc_rank_plot(
            ranks=lf_ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="cdf",
            ranks_labels=["LF"] * num_params,
            colors=['#7AAE41', '#B6E880'], # Green
            ax=ax_cdf,
            fig=fig_cdf,
        )
        
        # Add legend (in case the function doesn’t do it)
        
        fig_cdf.savefig(f"{plot_dir}/sbc_lf_hf_cdf_{n_simulations}_{self.config_data['type_lf']}.svg", format="svg")
        fig_cdf.savefig(f"{plot_dir}/sbc_lf_hf_cdf_{n_simulations}_{self.config_data['type_lf']}.pdf", format="pdf")


        rank_dic = {
            'hf_ranks': ranks,
            'lf_ranks': lf_ranks,
            'num_posterior_samples': num_posterior_samples
        }
        
        name = f"sbc_{type_estimator}_{n_simulations}.p"
        dump_pickle(save_dir, name, rank_dic)
        
        