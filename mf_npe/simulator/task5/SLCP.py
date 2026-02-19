# import mf_npe.task_setup as task_setup
import os
import pickle
from mf_npe.simulator.Prior import Prior
import torch
import numpy as np
from torch.distributions.normal import Normal
from sbi import utils as utils
from pyro import distributions as pdist
import pyro

# from sbibm.tasks.task import Task
# from sbibm.tasks.simulator import Simulator


class SLCP(Prior):
    
    """
    Simple Likelihood Complex Posterior simulator, as in:

    - Vandegar et al. (2020): "Neural Empirical Bayes Neural Empirical Bayes: Source Distribution Estimation and its Applications to Simulation-Based Inference"
    - Lueckmann et al. (2021): "Benchmarking Simulation-Based Inference"
    """
    
    def __init__(self, config_data, distractors=False):
        super().__init__()
        if distractors:
            self.x_dim = config_data['x_dim_hf']
        else:
            self.x_dim = config_data['x_dim_lf']
            
        self.theta_dim = config_data['theta_dim']
        self.distractors = distractors # distractor is the low fidelity variant 
        
        self.config_data = config_data
        self.prior_ranges = {k: config_data['all_prior_ranges'][k] for k in list(config_data['all_prior_ranges'].keys())[:config_data['theta_dim']]}
        self.num_data = 4
    
    def printName(self):
        if self.distractors:
            return "SLCP with distractors"
        else:
            return "SLCP"

    
    
    def prior(self):
        # lf and hf prior are the same for this task
        lf_prior, _ = super().get_prior(self.prior_ranges, self.config_data)
        return lf_prior
    
    def parameter_ranges(self, theta_dim):
        param_ranges = super().parameter_ranges(self.prior_ranges)
        return [param_ranges[f'range_theta{i+1}'] for i in range(theta_dim)]


    
    def summary_statistics(self, n_simulations, prior):
        """ Summary statistics of gaussian samples are just the samples themselves in the dimensionality of x."""
        
        thetas = prior.sample((n_simulations,))
        simulations = self.simulator(thetas)
        return simulations, thetas, {}

    def simulator(self, thetas):
        """ SLCP from SBIBM, https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/slcp/task.py"""
        
        num_samples = thetas.shape[0]
        
        m = torch.stack(
                (thetas[:, [0]].squeeze(), thetas[:, [1]].squeeze())
            ).T
        if m.dim() == 1:
            m.unsqueeze_(0)
        
        s1 = thetas[:, [2]].squeeze() ** 2
        s2 = thetas[:, [3]].squeeze() ** 2
        
        rho = torch.nn.Tanh()(thetas[:, [4]]).squeeze()
        
        S = torch.empty((num_samples, 2, 2))
        S[:, 0, 0] = s1 ** 2
        S[:, 0, 1] = rho * s1 * s2
        S[:, 1, 0] = rho * s1 * s2
        S[:, 1, 1] = s2 ** 2
        
        # Add eps to diagonal to ensure PSD
        eps = 0.000001
        S[:, 0, 0] += eps
        S[:, 1, 1] += eps
        
        data_dist = pdist.MultivariateNormal(
                m.unsqueeze(1).float(), S.unsqueeze(1).float()
                ).expand(
                    (
                        num_samples,
                        self.num_data,
                    )
                )
                
        data = pyro.sample("data", data_dist).reshape((num_samples, 8))
            
        if not self.distractors:
            return data
            
        else:
            this_dir = os.path.dirname(__file__)
            gmm_path = os.path.join(this_dir, "gmm.torch")
            permutation_idx_path = os.path.join(this_dir, "permutation_idx.torch")
            
            gmm = torch.load(gmm_path)
            permutation_idx = torch.load(permutation_idx_path)

            noise = gmm.sample((num_samples,)).type(data.dtype)
            data_and_noise = torch.cat([data, noise], dim=1)

            
            print("data_and_noise", data_and_noise.shape, "permutation_idx", permutation_idx.shape)

            return data_and_noise[:, permutation_idx]

    
        
    def get_reference_posterior_samples(self, n_true_observation_samples, true_xen, path_to_pickles) -> torch.Tensor:
        """Get samples from the true posterior distribution for the SLCP task.
        """


        # Load true_xen file from data
        n_true_xen = true_xen.shape[0]
        
        pickled_simulations = f"true_xen_{n_true_xen}.p"
        open_pickles_simulations = open(f"{path_to_pickles}/true_data/{pickled_simulations}", "rb")
        loaded_simulations = pickle.load(open_pickles_simulations)
        
        true_add_ons = loaded_simulations['true_add_ons']
        
        if 'reference_posterior_samples' not in true_add_ons:
            raise ValueError(f"reference_posterior_samples not found in true_add_ons for SLCP. Make sure that 1) the true_xen from the original OSF directory are used, 2) the path to the true posterior is correct.")
        
        true_posterior_samples = true_add_ons['reference_posterior_samples']
        
        # Transpose true_posterior_samples to have shape (n_true_xen, n_true_observation_samples, theta_dim)
        true_posterior_samples = true_posterior_samples
        
        # subsample n_true_observation_samples from the reference_posterior_samples
        if n_true_observation_samples > true_posterior_samples.shape[1]:
            raise ValueError("n_true_observation_samples is larger than the number of reference posterior samples.")

        subsampled_reference_posterior_samples = true_posterior_samples[:, :n_true_observation_samples]
        
        return subsampled_reference_posterior_samples

