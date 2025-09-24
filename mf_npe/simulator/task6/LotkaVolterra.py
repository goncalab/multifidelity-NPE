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

import math
import torch
torch.set_default_dtype(torch.float64)
from pyro.infer.mcmc import MCMC, NUTS


from pyro.distributions import ConditionalDistribution

class LotkaVolterra(Prior):
    
    def __init__(self, config_data, days: float = 20.0, saveat: float = 0.1, summary: Optional[str] = "subsample"):
        super().__init__()

        self.x_dim = config_data['x_dim_hf']
        self.theta_dim = config_data['theta_dim']
        
        #self.num_steps = 50
        self.n_of_prey_preditors = 2
        
        
        self.saveat: float = 0.1 # Timestep
        self.days: float = 20.0   
        self.t = torch.arange(0.0, self.days + 1e-12, self.saveat) # size: 201
        self.num_steps = self.t.numel() # 201
        self.u0 = torch.tensor([30.0, 1.0]) # Initial position
        

        self.subsample_rate = 21 # Like in benchmarking: 201 points, and 10D summary (x2 for preditor/prey) config_data['subsample_rate']
        self.config_data = config_data
        # self.prior_ranges = {k: config_data['all_prior_ranges'][k] for k in list(config_data['all_prior_ranges'].keys())[:config_data['theta_dim']]}
        self.device = 'cpu' #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    
    def printName(self):
        return "Lotka-Volterra"
    
    
    def prior(self):
        # Match sbibm Lotka-Volterra prior from SBIBM
        mu_p1 = -0.125
        mu_p2 = -3.0
        sigma_p = 0.5
        loc = torch.tensor([mu_p1, mu_p2, mu_p1, mu_p2])
        scale = torch.tensor([sigma_p, sigma_p, sigma_p, sigma_p])
        # LogNormal in PyTorch parameterized by mean=loc and std=scale of underlying Normal
        return torch.distributions.Independent(
            torch.distributions.LogNormal(loc=loc, scale=scale), 1
        ) # The 1 is the same as the toevent(1) in pyro


    def generate_idx(self):
        # Should exponentially with n of x_dimensions
        # Create subsamples with the idx
        idx = np.arange(0, self.n_of_prey_preditors*self.num_steps, self.subsample_rate) # Full 100 timesteps trace
        return idx

    def parameter_ranges(self, theta_dim):
        param_ranges = super().parameter_ranges(self.prior_ranges)
        return [param_ranges[f'range_theta{i+1}'] for i in range(theta_dim)]

    
    def summary_statistics(self, n_simulations, prior):
        """ Summary statistics are subsamples of the ODE solution. from 100D to 20D"""
                
        if self.t.shape[0] != self.num_steps:
            raise ValueError("t shape does not match num_steps")
        
        #base = torch.tensor([0.6859157,0.10761319,0.88789904,0.116794825]) 
        #thetas =  base.unsqueeze(0).repeat(n_simulations, 1) 
        # print("thetas", thetas)
        
        thetas = prior.sample((n_simulations,)) # 
        # print("prior samples", prior.sample((n_simulations,)))
        # raise ValueError("Stop here")
        
        simulations = self.simulator(thetas)
        
        tidx = torch.arange(0, simulations.shape[-1], self.subsample_rate, device=simulations.device)  # length 10
        subsampled_simulations = simulations[:, :, tidx]  # (B, 2, 10)

        # 4) SBIBM observation noise: LogNormal(log(values), 0.1)
        # base = subs.clamp(1e-10, 1e4)
        # noisy = torch.distributions.LogNormal(
        #     loc=torch.log(base),
        #     scale=torch.tensor(0.1, device=base.device, dtype=base.dtype)
        # ).rsample()  # (B, 2, 10)

        # 5) flatten species-major: [x(t_0..t_9), y(t_0..t_9)]
        subsampled_simulations = subsampled_simulations.reshape(n_simulations, -1)  # (B, 20
        
        print("subsampled simulations", subsampled_simulations)
        
        # Plot subsampled simulations ontop of simulations
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                if idx < n_simulations:
                    t_full = self.t.cpu().numpy()               # (T,)
                    Y_full = simulations[idx].detach().cpu().numpy().T   # (T, n_species)

                    t_sub = self.t[tidx].detach().cpu().numpy()          # (M,)
                    Y_sub = (subsampled_simulations[idx]
                            .detach().cpu().numpy()
                            .reshape(self.n_of_prey_preditors, -1)      # (n_species, M)
                            .T)                                         # (M, n_species)

                    axs[i, j].plot(t_full, Y_full, color='gray', alpha=0.5)
                    axs[i, j].plot(t_sub, Y_sub, linestyle='none', marker='o')  # points only
                    axs[i, j].set_title(f"theta: {thetas[idx].detach().cpu().numpy()}")
                    axs[i, j].set_xlabel("Time")
                    axs[i, j].set_ylabel("Population")
        plt.tight_layout()
        plt.savefig('subsampled_lotka_volterra.png')
        plt.close()
        
        # PRINT SUBSAMPLED SIMUALTIONS
        print("SIMULATION FILE", subsampled_simulations)
        
        
        # Plot true observations
        path = f"./mf_npe/simulator/task6/files/num_observation_1"
        
        
        # Open csv file data observation.csv and plot it
        csv_file = f"{path}/observation.csv"
        
        # From 1D to 2D reshape
        
        
        
        import pandas as pd
        import matplotlib.pyplot as plt

        
        # Read CSV into numpy
        observation = pd.read_csv(csv_file).to_numpy().flatten()
        
        
        print("OBSERVATION FILE", observation)

        # Reshape: (n_species, num_steps) → then transpose → (num_steps, n_species)
        observation = observation.reshape(self.n_of_prey_preditors, -1).T   # shape (M, n_species)

        # Sanity check: must match number of subsampled timepoints
        assert observation.shape[0] == len(self.t[tidx]), \
            f"Timepoints ({len(self.t[tidx])}) and observations ({observation.shape[0]}) do not match!"

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.t[tidx].cpu().numpy(), observation, marker='o')   # works: x = (M,), y = (M, n_species)
        plt.title("True Observations from CSV")
        plt.xlabel("Time")
        plt.ylabel("Population")
        plt.savefig(f"{path}/true_observations.png")
        plt.close()


        return subsampled_simulations, thetas, dict(full_trace=simulations)


    def lotka_volterra_ode(self, t, y):
        vals, par = y
        
        print("ODE Parameters shape:", par.shape)
        print("ODE Values shape:", vals.shape)
        
        alpha = par[:, 0]
        beta = par[:, 1]
        gamma = par[:, 2]
        delta = par[:, 3]
        x_val = vals[:, 0] # prey
        y_val = vals[:, 1] # predator
        dx_val = alpha * x_val - beta * x_val * y_val
        dy_val = - gamma * y_val + delta * x_val * y_val 

        
        return torch.stack([dx_val, dy_val], dim=1), torch.zeros_like(par)
        

    def simulator(self, parameters):
        """ LotkaVolterra from Gloeckler et al. (2023) - Adversarial Robustness of Amortized Bayesian Inference and
            Sourcerer https://github.com/mackelab/sourcerer/blob
            and SBIBM
            
            ------------------------------------------------------------------
            function f(du,u,p,t)
                x, y = u
                alpha, beta, gamma, delta = p
                du[1] = alpha * x - beta * x * y
                du[2] = -gamma * y + delta * x * y
            end

        """
        
        batch_size, _num_par = parameters.shape
        initial = self.u0.expand(batch_size, 2).to(self.device) #initial * torch.tensor([30.0, 1.0], device=self.device)

        try:
            sol, _param = odeint(
                self.lotka_volterra_ode,
                (initial, parameters),
                self.t.to(self.device),
                method="dopri5",
            )
        except:
            print("Warning: Solving with fixed stepsize!")
            sol, _param = odeint(
                self.lotka_volterra_ode,
                (initial, parameters),
                self.t.to(self.device),
                method="rk4",
                options={"step_size": 1e-2},
            )

        # (batch_size, 2, 50)
        # logged = sol.permute(1, 2, 0)
        # # (batch_size, 100)
        # logged = logged.reshape(batch_size, -1)
        

        # mask = torch.isfinite(logged)
        # logged[~mask] = 0.0
        
        
        vals_over_time = sol.permute(1, 2, 0)  # (B, 2, T)

        # Replace non-finite with small positives (avoid log issues later)
        vals_over_time = torch.where(
            torch.isfinite(vals_over_time),
            vals_over_time,
            torch.zeros_like(vals_over_time),
        ).clamp_(min=1e-10)
        
        
        # Plot lotka-volterra simulations
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                if idx < batch_size:
                    axs[i, j].plot(self.t.numpy(), vals_over_time[idx].reshape(self.n_of_prey_preditors, self.num_steps).cpu().numpy().T)
                    axs[i, j].set_title(f"theta: {parameters[idx].cpu().numpy()}")
                    axs[i, j].set_xlabel("Time")
                    axs[i, j].set_ylabel("Population")
        plt.tight_layout()
        plt.savefig('simulated_lotka_volterra.png')
        plt.close()
        

        return vals_over_time

        

    def get_reference_posterior_samples(self, n_true_observation_samples, true_xen, path_to_pickles) -> torch.Tensor:
        """Get samples from the true posterior distribution for the SLCP task.
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




