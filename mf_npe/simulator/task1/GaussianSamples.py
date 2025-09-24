# import mf_npe.task_setup as task_setup
from mf_npe.simulator.Prior import Prior
import torch
import numpy as np
from torch.distributions.normal import Normal
from sbi import utils as utils


class GaussianSamples(Prior):
    """The low fidelity simulator are Gaussian i.i.d samples."""
    
    def __init__(self, config_data):
        super().__init__()
        self.x_dim = config_data['x_dim_lf']
        self.subsample_rate = config_data['subsample_rate']
        self.theta_dim = config_data['theta_dim']
        
        self.sigma = 0.3 # Fixed sigma if theta_dim = 1
        self.config_data = config_data
        self.prior_ranges = {k: config_data['all_prior_ranges'][k] for k in list(config_data['all_prior_ranges'].keys())[:config_data['theta_dim']]}

    
    def printName(self):
        return "Gaussian Samples"
    
    
    def prior(self):
        lf_prior, _ = super().get_prior(self.prior_ranges, self.config_data)
        return lf_prior
    
    def parameter_ranges(self, theta_dim):
        param_ranges = super().parameter_ranges(self.prior_ranges)
        return [param_ranges[f'range_theta{i+1}'] for i in range(theta_dim)]

    def _GS_timestep(self, theta) -> torch.Tensor:
        """ 
        The parameters for the low fidelity model are the mean and standard deviation of the Gaussian distribution.
        """
        mu = theta[0]
        
        if self.theta_dim == 1:
            sigma = self.sigma # Fixed sigma
        else:
            sigma = theta[1]
            
        dist = Normal(mu, sigma)
        x_t = dist.sample()  # An alternative would be a random walk: x_t = x_prev + self._diffusion(sigma) * self._dW(self.dt) (but easier i.i.d sampling to proove the point)
        return x_t 
    
    
    def integrator(self, theta):
        """Generate a trace of Gaussian i.i.d. samples with forward Euler.
        """
        trace = torch.zeros(self.x_dim)
        # Sampling from a gaussian is not conditional
        for delta_t in range(0, self.x_dim): 
            trace[delta_t] = self._GS_timestep(theta)
            
        return trace
        
        
    def traces_simulator(self, n_simulations, prior):
        """
        This simulator works for conditional and unconditional time series data,
        such as the Ornstein-Uhlenbeck process and the Gaussian distribution over time.
        """        
        simulations = torch.zeros(n_simulations, self.x_dim)
        thetas = prior.sample((n_simulations,))
        
        print("Generating simulations ...")

        for i in range(n_simulations):
            simulations[i] = self.integrator(thetas[i]).flatten()
        
        return simulations, thetas
    
    
    def summary_statistics(self, n_simulations, prior):
        """ Summary statistics of gaussian samples are just the samples themselves in the dimensionality of x."""
        # Do not subsample        
        simulations, thetas = self.traces_simulator(n_simulations, prior)
        subsampled_simulations = simulations

        return subsampled_simulations, thetas, dict(full_trace=simulations) 
    
    def simulator(self, thetas):   
        # Do not subsample
        simulations = torch.stack([self.integrator(theta).flatten() for theta in thetas])

        return simulations
    
        
    def true_log_likelihood(self, theta, true_x) -> torch.Tensor:
        """The true log likelihood of the Gaussian samples is the sum of the log probabilities of the samples.
        """
        mu = theta[:, 0]
        sigma = theta[:, 1] if self.theta_dim in [2] else self.sigma
                                
        loglik = 0 # Will be overwritten from the 1st sample
        for j in range(len(true_x)):
            log_prob = Normal(mu, sigma).log_prob(true_x[j])
            loglik = loglik + log_prob
        
        loglik = torch.tensor(loglik, requires_grad=True) 
        
        return loglik