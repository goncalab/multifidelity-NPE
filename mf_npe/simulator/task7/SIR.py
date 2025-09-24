import os

import pickle
from turtle import pd
from typing import Callable, List, Optional

import numpy as np
from mf_npe.simulator.Prior import Prior
import torch
from torch.distributions.normal import Normal
from sbi import utils as utils
from torchdiffeq import odeint
from torch.distributions import Binomial


import torch
torch.set_default_dtype(torch.float32)


class SIR(Prior):
    """SIR epidemic model

        Inference is performed for two parameters:
        - Contact rate, beta
        - Mean recovery rate, gamma, (in 1/days)

        Args:
            N: Total population
            I0: Initial number of infected individuals
            R0: Initial number of recovered individuals
            days: Number of days
            saveat: When to save during solving
            summary: Summaries to use

        References:
            [1]: https://jrmihalj.github.io/estimating-transmission-by-fitting-mechanistic-models-in-Stan/
        """
    
    def __init__(self, config_data):
        super().__init__()

        self.x_dim = config_data['x_dim_hf']
        self.theta_dim = config_data['theta_dim']
        
        self.N: float = 1000000.0
        self.I0: float = 1.0
        self.R0: float = 0.0
        self.S0 = self.N - self.I0 - self.R0
        self.days: float = 160.0
        self.saveat: float = 1.0
        self.total_count: int = 1000
              
        self.t = torch.arange(0, self.days + 1e-12, self.saveat)  # shape (T,)
        self.num_steps = self.t.numel()
        
        self.number_observed_groups = 3  # S, I, R
        
        self.subsample_stride = 17
        self.sub_idx = torch.arange(0, self.num_steps, self.subsample_stride)  # len=10
        assert self.sub_idx.numel() == 10, "Expected exactly 10 subsampled points."
        

        # Initial number of susceptible individuals
        self.u0 = torch.tensor([self.S0, self.I0, self.R0])
        self.prior_ranges = None #{k: config_data['all_prior_ranges'][k] for k in list(config_data['all_prior_ranges'].keys())[:config_data['theta_dim']]}
        self.device = 'cpu' #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    
    def printName(self):
        return "SIR"

    
    def prior(self):        
        loc = torch.tensor([torch.log(torch.tensor(0.4)), torch.log(torch.tensor(0.125))])
        scale = torch.tensor([0.5, 0.2])
        
        # LogNormal in PyTorch parameterized by mean=loc and std=scale of underlying Normal
        return torch.distributions.Independent(
            torch.distributions.LogNormal(loc=loc, scale=scale), 1
        )

       
    def generate_idx(self):
        # Should exponentially with n of x_dimensions
        # Create subsamples with the idx
        idx = np.arange(0, self.number_observed_groups*self.num_steps, self.subsample_rate) # Full 100 timesteps trace: TODO check again
        return idx

    def parameter_ranges(self, theta_dim):
        
        if getattr(self, "prior_ranges", None) is None:
            return [None] * theta_dim  # or just return []
    
        param_ranges = super().parameter_ranges(self.prior_ranges)
        return [param_ranges[f'range_theta{i+1}'] for i in range(theta_dim)]


    
    def summary_statistics(self, n_simulations, prior):
        """ Summary statistics are subsamples of the ODE solution. from 100D to 20D"""
                
        if self.t.shape[0] != self.num_steps:
            raise ValueError("t shape does not match num_steps")
                
        thetas = prior.sample((n_simulations,)) # 
        sims = self.simulator(thetas)
        
        # Subsample I population
        data = self.simulator_wrapper(thetas)

        # Optional debugging plot overlaying full I vs subsampled counts (disabled unless debug)
        try:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            for i in range(2):
                for j in range(5):
                    idx = i * 5 + j
                    if idx < n_simulations:
                        t_full = self.t.cpu().numpy()
                        I_full = sims[idx, 1].detach().cpu().numpy()
                        t_sub = self.t[self.sub_idx].cpu().numpy()
                        y_sub = data[idx].detach().cpu().numpy() * (self.N / self.total_count)
                        axs[i, j].plot(t_full, I_full, color="gray", alpha=0.6)
                        axs[i, j].plot(t_sub, y_sub, marker="o", linestyle="none")
                        # axs[i, j].set_title(f"theta: {thetas[idx].detach().cpu().numpy()}")
            plt.tight_layout()
            plt.savefig("./mf_npe/simulator/task7/images/subsampled_sir.png", dpi=300)
            plt.savefig("./mf_npe/simulator/task7/images/subsampled_sir.pdf", dpi=300)
            plt.close()
        except Exception:
            pass

        
        print("data dtype", data.dtype)

        return data, thetas, dict(full_trace=sims)
        
    def simulator_wrapper(self, thetas):
        sims = self.simulator(thetas)
        
        # Subsample I population
        I_sub = sims[:, 1, self.sub_idx]  # (B, 10)
        probs = (I_sub / self.N).clamp(0.0, 1.0)

        # SBIBM: sample Binomial counts with total_count trials
        data = Binomial(total_count=self.total_count, probs=probs).sample()  # (B,10)
        data = data.to(dtype=torch.get_default_dtype())
        
        return data
        

    def sir_ode(self, t, y):
        vals, par = y
        
        print("ODE Parameters shape:", par.shape)
        print("ODE Values shape:", vals.shape)
                
        #S,I,R = u
        #b,g = p # parameters
        b = par[:, 0]
        g = par[:, 1]
        S = vals[:, 0]
        I = vals[:, 1]
        R = vals[:, 2]
        
        du_1 = -b * S * I / self.N
        du_2 = b * S * I / self.N - g * I
        du_3 = g * I

        return torch.stack([du_1, du_2, du_3], dim=1), torch.zeros_like(par)


    def simulator(self, parameters):
        """ SIR FROM SBIBM
        """
        
        batch_size, _num_par = parameters.shape
        # TODO: if smt is not working: n of prey and predators is 2, so i've put initial = to prey-preditors. but not sure if that's correct    

        initial = self.u0.expand(batch_size, 3).to(self.device) 

        try:
            sol, _param = odeint(
                self.sir_ode,
                (initial, parameters),
                self.t.to(self.device),
                method="dopri5",
            )
        except:
            print("Warning: Solving with fixed stepsize!")
            sol, _param = odeint(
                self.sir_ode,
                (initial, parameters),
                self.t.to(self.device),
                method="rk4",
                options={"step_size": 1e-2},
            )

        
        vals_over_time = sol.permute(1, 2, 0)  # (B, 2, T)

        # Replace non-finite with small positives (avoid log issues later)
        vals_over_time = torch.where(
            torch.isfinite(vals_over_time),
            vals_over_time,
            torch.zeros_like(vals_over_time),
        ).clamp_(min=1e-10)
        
        
        # Plot SIR simulations
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                if idx < batch_size:
                    axs[i, j].plot(self.t.numpy(), vals_over_time[idx].reshape(self.number_observed_groups, self.num_steps).cpu().numpy().T)
                    axs[i, j].set_xlabel("Time")
                    axs[i, j].set_ylabel("Population")
        plt.tight_layout()
        plt.savefig('./mf_npe/simulator/task7/images/simulated_sir.png')
        plt.savefig('./mf_npe/simulator/task7/images/simulated_sir.pdf')
        plt.close()
        
        return vals_over_time



    def get_reference_posterior_samples(self, n_true_observation_samples, true_xen, path_to_pickles) -> torch.Tensor:
        """Get samples from the true posterior distribution for the SIR task.
        """

        # Load true_xen file from data
        n_true_xen = true_xen.shape[0]
        
        pickled_simulations = f"true_xen_{n_true_xen}.p"
        open_pickles_simulations = open(f"{path_to_pickles}/true_data/{pickled_simulations}", "rb")
        loaded_simulations = pickle.load(open_pickles_simulations)
        
        true_add_ons = loaded_simulations['true_add_ons'] # There are the true posterior samples
        true_posterior_samples = true_add_ons['reference_posterior_samples']
        
        print("loaded_simulations keys", loaded_simulations.keys())
        
        print(" true_add_ons", loaded_simulations['true_add_ons'])
        print("true xen", loaded_simulations['true_xen'])
        print("true thetas", loaded_simulations['true_thetas'])
        

        # subsample n_true_observation_samples from the reference_posterior_samples
        if n_true_observation_samples > true_posterior_samples.shape[1]:
            raise ValueError("n_true_observation_samples is larger than the number of reference posterior samples.")

        subsampled_reference_posterior_samples = true_posterior_samples[:, :n_true_observation_samples]
        
        print("subsampled_reference_posterior_samples", subsampled_reference_posterior_samples.shape)
        
        return subsampled_reference_posterior_samples




