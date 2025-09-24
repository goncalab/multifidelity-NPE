from mf_npe.simulator.Prior import Prior
import torch
import numpy as np
from torch.distributions.normal import Normal
from sbi import utils as utils

class OUprocessTshift(Prior):
    def __init__(self, config_data, gamma, mu_offset):
        """Ornstein-Uhlenbeck process is a high-fidelity model """
        
        super().__init__()
        self.length_total_trace = config_data['length_total_trace'] # config_data['x_dim'] # true_x is a subsample. so eg. if 1000 then random 100 points for fitting
        # dt should be resolved
        self.dt = 0.1 #float(self.tn - self.t0) / self.length_total_trace # rule of thumb: dt should be around 100 times faster than 1/gamma
        self.length_total_trace = config_data['length_total_trace']
        # self.theta_dim = config_data['theta_dim']
        self.x_dim = config_data['x_dim_lf']
        self.subsample_rate = config_data['subsample_rate']
        self.logscale = config_data['logspace']
        self.first_n_samples = config_data['first_n_samples']
        
        self.gamma = gamma
        self.mu_offset = mu_offset
        self.theta_dim = config_data['theta_dim']
        self.shift_theta = 2 # Shift the mean
        
        self.prior_ranges = {k: config_data['all_prior_ranges'][k] for k in list(config_data['all_prior_ranges'].keys())[:config_data['theta_dim']]}
        
        self.config_data = config_data
    
    def printName(self):
        return "OU process"
        
    def prior(self):
        _, hf_prior = super().get_prior(self.prior_ranges, self.config_data)
        return hf_prior       
    
    
    def parameter_ranges(self, theta_dim):
        param_ranges = super().parameter_ranges(self.prior_ranges)
        return [param_ranges[f'range_theta{i+1}'] for i in range(theta_dim)]
        
    def xt0_dist(self, xto, mu):
        '''
        Define the distribution of the initial condition.
        The mean_xt is {true) mu + a certain value, since it should be adjusted
        to the true theta.
        '''
        sigma_xto = 1 
        rel_init_value = mu + xto
        xto_dist = Normal(rel_init_value, sigma_xto)
        
        return xto_dist
    
    def _dW(self, delta_t: float) -> float:
        """Sample a random number at each call."""
        return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))
        
    def _drift(self, x: float, mu: float, gamma: float) -> float:
        """Drift term of the Ornstein-Uhlenbeck process."""
        return gamma * (mu - x)
        
    def _diffusion(self, sigma:float) -> float:
        """Diffusion term of the Ornstein-Uhlenbeck process."""
        return sigma
    
    def _OU_timestep(self, theta, x_prev) -> torch.Tensor:
        """Run the Ornstein-Uhlenbeck process.
        
            Each simulation is the likelihood under specific theta configuration
            Runs the OU process for a given theta
        """
        mu = theta[0] + self.shift_theta # Shift the mean by 2
        sigma = theta[1]
        
        if self.theta_dim == 2:
            gamma = self.gamma # 1/gamma = tau, the time constant
        elif self.theta_dim == 3 or self.theta_dim == 4:
            gamma = theta[2] # The steeper (smaller gamma), the more similar the HF model is
       
        x_t = x_prev + self._drift(x_prev, mu, gamma) * self.dt + self._diffusion(sigma) * self._dW(self.dt)
        x_t = x_t.clone().detach()
        
        return x_t     
    
    
    def integrator(self, theta):        
        
        # print("samples theta", theta)
        
        # length of the trace is equal to the dimension of x
        trace = torch.zeros(self.length_total_trace)
        
        if self.theta_dim == 2 or self.theta_dim == 3:
            mu_offset = self.mu_offset + self.shift_theta # shift mu_offset
        if self.theta_dim == 4:
            mu_offset = theta[3] + self.shift_theta  # shift mu_offset
            
        mu = theta[0]
        xto_dist = self.xt0_dist(mu_offset, mu)
        xto = xto_dist.sample() 
        trace[0] = xto
        
        for delta_t in range(1, self.length_total_trace):           
            trace[delta_t] = self._OU_timestep(theta, trace[delta_t-1])
            
        return trace 

    
    def traces_simulator(self, n_simulations, prior):
        """
        This simulator works for conditional and unconditional time series data,
        such as the Ornstein-Uhlenbeck process and the Gaussian distribution over time.
        """        
        thetas = prior.sample((n_simulations,))
        print("Generating simulations ...")
        
        simulations = torch.stack([self.integrator(theta).flatten() for theta in thetas])

        return simulations, thetas
    
    
    def generate_idx(self):
        # Should exponentially with n of x_dimensions
        # Create subsamples with the idx
        if self.logscale:
            max_logspace = np.log10(self.length_total_trace-1)        
            idx_floats = np.logspace(0, max_logspace, 10, endpoint = True)         
            idx = np.round(idx_floats, 1).astype(int)
            idx[0] = 0 
        else:
            # Linear method
            idx = np.arange(0, self.first_n_samples, self.subsample_rate)
        return idx
    
    
    def summary_statistics(self, n_simulations, prior):
        """ Summary statistics are the x subsamples from the first n simulations."""
        
        simulations, thetas = self.traces_simulator(n_simulations, prior)
    
        idx = self.generate_idx()
        subsampled_simulations = torch.stack([trace[idx] for trace in simulations])

        return subsampled_simulations, thetas, dict(full_trace=simulations) 
    
    
    def simulator(self, thetas):        
        simulations = torch.stack([self.integrator(theta).flatten() for theta in thetas])
        
        idx = self.generate_idx()
        subsampled_simulations = torch.stack([trace[idx] for trace in simulations])
        
        return subsampled_simulations

    # Log-Likelihood Function Over Parameters
    def true_log_likelihood(self, thetas, true_x) -> torch.Tensor:
        """ Analytical solution for likelihood: p(x|theta) = p(x|mu, sigma, gamma)

            Source: A Multiresolution Method for Parameter Estimation of Diffusion Processes
                    DOI:10.1080/01621459.2012.720899
                    Supeng Kou, Benjamin P. Olding

            Expectes multiple thetas (e.g. 1000) and a fixed x,
            goes into rejection sampling function that requires a tensor
        """
        mu = thetas[:, 0]
        sigma = thetas[:, 1]
    
        gamma = thetas[:, 2] if self.theta_dim in [3, 4] else self.gamma
        mu_offset = thetas[:, 3] if self.theta_dim == 4 else self.mu_offset
        
        loglik = torch.zeros(len(mu))
        
        idx = self.generate_idx()
        idx[0] = 0 # We dont want dt to be 0, otherwise we cannot compute true likelihood
        xto_dist = self.xt0_dist(mu_offset, mu)
        xto = true_x[0]

        xto_log_prob = xto_dist.log_prob(xto) # Because log(1) = 0. # xto_dist.log_prob(x0)
        logsum = xto_log_prob
        
        for j in range(1, self.x_dim):  # length_total_trace
            #print(idx)
            #print(idx[j]-idx[j-1])
            g = (1 - torch.exp(-2 * gamma * self.dt*(idx[j]-idx[j-1]))) / gamma # self.dt * sample_rate
            
            x_prev = true_x[j-1]
            x_curr = true_x[j]

            term = 1 / (torch.sqrt(torch.pi * g) * sigma)
            exp1 = -(1 / ( g * sigma**2))
            exp2 = (mu - x_curr) - torch.sqrt(1 - gamma * g) * (mu - x_prev)
            exp = torch.exp(exp1 * exp2**2)

            logsum = logsum + torch.log(term * exp)

        loglik = logsum

        return loglik