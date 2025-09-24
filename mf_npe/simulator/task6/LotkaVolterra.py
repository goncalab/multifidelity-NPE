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
        self.n_of_prey_preditors = 2 # TODO: refine this in code, is not changing as expected atm
        
        
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

    
    # def prior(self):
    #     # TODO: check this prior
    #     prior_mean: float = 0.0
    #     prior_scale: float = 0.5
        
    #     prior = torch.distributions.Independent(
    #         torch.distributions.Normal(
    #             torch.ones(4) * prior_mean, torch.ones(4) * prior_scale
    #         ),
    #         1,
    #     )

    #     return prior


    def generate_idx(self):
        # Should exponentially with n of x_dimensions
        # Create subsamples with the idx
        idx = np.arange(0, self.n_of_prey_preditors*self.num_steps, self.subsample_rate) # Full 100 timesteps trace: TODO check again
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
        # TODO: if smt is not working: n of prey and predators is 2, so i've put initial = to prey-preditors. but not sure if that's correct
        # initial = torch.ones(batch_size, self.n_of_prey_preditors, device=self.device)
    
        # 30.0, 1.0
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

        

    # def get_reference_posterior_samples(
    #     self,
    #     n_true_observation_samples: int,
    #     true_xen,
    #     path_to_pickles: Optional[str] = None,
    #     *,
    #     num_warmup: int = 2_000,
    #     num_chains: int = 1,
    #     max_tree_depth: int = 10,
    #     use_cache: bool = True,
    # ) -> torch.Tensor:
    #     """
    #     Draw samples from the posterior p(theta | x_obs) for Lotka–Volterra.

    #     Observation model matches SBIBM:
    #         - Simulate full trajectory with self.simulator(...)
    #         - Subsample every `self.subsample_rate` steps (2 species × 10 points = 20D)
    #         - Independent LogNormal noise on each component with scale=0.1

    #     Args:
    #         n_true_observation_samples: Number of posterior samples to return.
    #         true_xen: Observed summary statistics (Tensor, numpy array, or list).
    #                 Accepts shape (20,), (2, 10), or (10, 2). Will be flattened to (20,).
    #         path_to_pickles: Directory for caching. If provided and use_cache=True,
    #                         results are cached per (observation hash, n_samples).
    #         num_warmup: NUTS warmup steps.
    #         num_chains: MCMC chains (use >1 if you want R-hat diagnostics).
    #         max_tree_depth: NUTS max tree depth.
    #         use_cache: Whether to read/write cache in path_to_pickles.

    #     Returns:
    #         torch.Tensor of shape (n_true_observation_samples, 4)
    #     """
    #     import os
    #     import pickle
    #     import hashlib
    #     import numpy as np
    #     import pyro
    #     import pyro.distributions as pdist
    #     from pyro.infer.mcmc import MCMC, NUTS

    #     # ---- normalize observed data to a contiguous torch tensor of length 20
    #     if isinstance(true_xen, torch.Tensor):
    #         y = true_xen.detach().clone()
    #     else:
    #         y = torch.as_tensor(true_xen, dtype=torch.get_default_dtype())

    #     if y.ndim == 2:
    #         y = y.reshape(-1)
    #     if y.ndim != 1:
    #         raise ValueError(f"Expected observation to be 1D or 2D, got shape {tuple(y.shape)}")

    #     # For default settings (days=20, saveat=0.1, subsample_rate=21): 2 species × 10 points = 20-D
    #     expected_dim = 2 * (self.t.numel() // self.subsample_rate + (1 if (self.t.numel() % self.subsample_rate) else 0))
    #     # In SBIBM they take exactly 10 points; with 201 timepoints and stride 21 we indeed get 10.
    #     # To be robust, clamp to 20 for your current setup:
    #     expected_dim = 20
    #     if y.numel() != expected_dim:
    #         raise ValueError(f"Expected observation of length {expected_dim}, got {y.numel()}")

    #     y = y.to(dtype=torch.get_default_dtype(), device=self.device).contiguous()

    #     # ---- caching
    #     cache_path = None
    #     if path_to_pickles is not None:
    #         os.makedirs(path_to_pickles, exist_ok=True)
    #         h = hashlib.sha1(y.detach().cpu().numpy().tobytes()).hexdigest()
    #         cache_path = os.path.join(
    #             path_to_pickles,
    #             f"lotka_volterra_refpost_{h}_N{n_true_observation_samples}.pkl",
    #         )
    #         if use_cache and os.path.exists(cache_path):
    #             with open(cache_path, "rb") as f:
    #                 cached = pickle.load(f)
    #             # return as torch tensor, ensure dtype/device consistent
    #             return torch.as_tensor(cached, dtype=torch.get_default_dtype(), device=self.device)

    #     # ---- Pyro model: prior -> simulator -> subsample -> LogNormal(obs | log(sim), 0.1)
    #     subsample_idx = torch.arange(0, self.t.numel(), self.subsample_rate, device=self.device)

    #     def model(y_obs=None):
    #         theta = pyro.sample("parameters", self.prior())  # support: positive reals via LogNormal
    #         sim = self.simulator(theta.reshape(1, -1))       # (1, 2, T)
    #         # pick subsamples -> flatten to (20,)
    #         us = sim[:, :, subsample_idx].reshape(-1)        # (2 * M,)
    #         base = us.clamp(1e-10, 1e4)

    #         pyro.sample(
    #             "obs",
    #             pdist.LogNormal(loc=torch.log(base), scale=base.new_tensor(0.1)).to_event(1),
    #             obs=y_obs,
    #         )
    #         return theta

    #     # ---- run NUTS
    #     pyro.clear_param_store()
    #     kernel = NUTS(model, max_tree_depth=max_tree_depth, adapt_step_size=True)
    #     mcmc = MCMC(
    #         kernel,
    #         num_samples=n_true_observation_samples,
    #         warmup_steps=num_warmup,
    #         num_chains=num_chains,
    #         mp_context="spawn" if (num_chains > 1) else None,
    #     )
    #     mcmc.run(y_obs=y)

    #     posterior = mcmc.get_samples()["parameters"]  # shape: (n_samples, 4), constrained space (positive)
    #     posterior = posterior.to(self.device)

    #     # ---- optional: save cache
    #     if cache_path is not None and use_cache:
    #         with open(cache_path, "wb") as f:
    #             pickle.dump(posterior.detach().cpu().numpy(), f)

    #     return posterior


    
   
    # def get_reference_posterior_samples(self, n_true_observation_samples, true_xen, path_to_pickles) -> torch.Tensor:
    #     """Get samples from the true posterior distribution for the SLCP task.
    #     """
        
    #     # TODO
    #     pass
    
        

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




