
#%%
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import jaxley as jx
from jaxley.channels import Na, K, Leak, HH

from mf_npe.simulator.task2.Neuron import Neuron
from mf_npe.simulator.Prior import Prior
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive; no windows, fewer semaphores
import matplotlib.pyplot as plt
import mf_npe.config.plot as plot_config


class MultiCompartmentalNeuron(Neuron, Prior):
    def __init__(self, config_data, n_compartments):
        self.dt = config_data['dt']
        self.t_max = config_data['t_max'] #self.dt * config_data['length_total_trace'] 
        
        # params for stimulus
        self.t = np.arange(0, self.t_max+self.dt, self.dt) #* dt
        self.t_on = 10.0
        self.duration = 100.0 # mS
        self.t_off = self.t_on + self.duration # duration - t_on 
        self.i_amp = 0.055 # Bij vorige test 0.0055
        
        self.n_compartments = n_compartments
        
        self.config_data = config_data
        self.prior_ranges = {k: config_data['all_prior_ranges'][k] for k in list(config_data['all_prior_ranges'].keys())[:config_data['theta_dim']]}
        
    def printName(self):
        return f"{self.n_compartments}-Multicompartmental"
    
    def prior(self):
        lf_prior, _ = super().get_prior(self.prior_ranges, self.config_data)
        return lf_prior
    
    def parameter_ranges(self, theta_dim):
        param_ranges = super().parameter_ranges(self.prior_ranges)
        return [param_ranges[f'range_theta{i+1}'] for i in range(theta_dim)]
    
        
    def _jaxley_neuron(self):
        ''' Create a Hodgkin huxley cell with jaxley. The number of compartments can be adjusted in the call of the class,
            For a multicompartmental model, we used 100 compartments, and for a singlecompartmental neuron model 1 compartment.'''

        # Main difference with single compartmental neuron
        n_compartments = self.n_compartments
        
        comp = jx.Compartment()
        # 100 segments for multi compartment, 100 for multicompartment
        branch = jx.Branch(comp, ncomp=n_compartments)       
        cell = jx.Cell(branch, parents=[-1])

        # Add HH channels.
        cell.insert(HH())
        
        # visualize the cell
        cell.compute_xyz()  # Only needed for visualization.
        # fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        # _ = cell.vis(ax=ax)
        cell.vis()
        plt.axis("off")
        plt.title("Retina bipolar cell")
        
        if plot_config.show_plots:
            plt.show()
        
        # Default stepcurrent, afterwards we add noise
        I_curr = jx.step_current(i_delay=self.t_on, i_dur=self.duration, i_amp=self.i_amp, delta_t=self.dt, t_max=self.t_max)  
        cell.delete_stimuli()
        # Stimulate first branch at beginning of compartment, espeically important for multicompartment, to not miss
        cell.branch(0).loc(0.0).stimulate(I_curr)
        
        # Add recordings
        cell.delete_recordings()
        cell.branch(0).loc(0.0).record()
        # show dictionary of cell elements
        print(f"created 1 neuron with {n_compartments}", cell.nodes)
        
        return cell, I_curr
    

    
    def get_noise_params(self, n_simulations):
        noise_fact = 0.01 # 0.01 # If the noise factor is 0.1, it cannot draw any samples (takes too much time) # was 0.01, put higher to make sure posterior inference is alright
        noise_params = np.random.normal(0, noise_fact, (n_simulations, 1, int(self.t_max // self.dt) + 2))
        return noise_params
    

    def summary_statistics(self, n_simulations, prior, integrator):
        x_clean = torch.tensor([])
        theta_clean = torch.tensor([])
        full_trace_clean = torch.tensor([])
        
        while len(x_clean) < n_simulations:            
            n_resamples = n_simulations - len(x_clean)        
            print(f"{n_resamples} datasamples left to resample")
            
            theta = prior.sample((n_resamples,))
            thetas_jax = jnp.asarray(theta)
            noise_params = self.get_noise_params(n_resamples)
            simulations = super().simulate_jit(thetas_jax, noise_params, integrator)
            full_trace = torch.from_numpy(np.asarray(simulations[:, 0]).copy())
            x = torch.stack([self.calculate_summary_statistics(full_trace[i], self.t_on, self.t_off, self.t, self.dt) for i in range(n_resamples)])

            # Apply mask to filter out invalid samples (NaNs, Inf, and invalid data)
            mask = super().mask_invalid_samples(x)
            
            # Concatenate the new samples
            x_clean = torch.cat((x_clean, x[mask]), dim=0)
            theta_clean = torch.cat((theta_clean, theta[mask]), dim=0)
            full_trace_clean = torch.cat((full_trace_clean, full_trace[mask]), dim=0)
            
                        
        if torch.isnan(x_clean).any() or torch.isinf(x_clean).any() or torch.sum(x_clean) == 0:
            raise ValueError("x_clean has NaNs, infinite values, or zeros")
        
        
        # Injection current
        _, I_curr = self._jaxley_neuron()

        return x_clean, theta_clean, dict(full_trace=full_trace_clean, inj_current=I_curr) 
    
     
    def simulator(self, theta, integrator, 
                  allow_resampling_invalid_samples=False, 
                  active_learning_list=None):
            # This function is used in the density estimator to sample from it again. 
            # So there might be new samples that are not valid, and we have to sample again.
            print(f"number of thetas: {theta.shape[0]}")
            n_thetas = theta.shape[0]
            
            
            if allow_resampling_invalid_samples:
                x_clean = torch.tensor([])
                theta_clean = torch.tensor([])
                full_trace_clean = torch.tensor([])
                
                counter = 0
        
                while len(x_clean) < n_thetas:       
                    n_resamples = n_thetas - len(x_clean)        
                    print(f"{n_resamples} datasamples left to resample")
                                        
                    # First round:
                    if counter == 0:
                        theta = theta
                    elif counter > 0:
                        if active_learning_list is None:
                            theta = self.prior().sample((n_resamples,))
                        else:
                            # Get the number of thetas that have to be resampled from the list
                            theta = active_learning_list[:n_resamples]
                                            
                    thetas_jax = jnp.asarray(theta)
                    noise_params = self.get_noise_params(n_resamples)
                    simulations = super().simulate_jit(thetas_jax, noise_params, integrator)
                    full_trace = torch.from_numpy(np.asarray(simulations[:, 0]).copy())
                    x = torch.stack([self.calculate_summary_statistics(full_trace[i], self.t_on, self.t_off, self.t, self.dt) for i in range(n_resamples)])

                    # Apply mask to filter out invalid samples (NaNs, Inf, and invalid data)
                    mask = super().mask_invalid_samples(x)
                    
                    # Concatenate the new samples
                    x_clean = torch.cat((x_clean, x[mask]), dim=0)
                    theta_clean = torch.cat((theta_clean, theta[mask]), dim=0)
                    full_trace_clean = torch.cat((full_trace_clean, full_trace[mask]), dim=0)
                    
                    
                    # Truncated active learning list by n_resamples
                    if active_learning_list is not None:
                        active_learning_list = active_learning_list[n_resamples:]
                    counter += 1
                        
            else:
                x_clean = torch.tensor([])
                theta_clean = torch.tensor([])
                full_trace_clean = torch.tensor([])
                
                # Just give the thetas that are not NaNs back.
                thetas_jax = jnp.asarray(theta)
                noise_params = self.get_noise_params(n_thetas)
                simulations = super().simulate_jit(thetas_jax, noise_params, integrator)
                full_trace = torch.from_numpy(np.asarray(simulations[:, 0]).copy())
                x = torch.stack([self.calculate_summary_statistics(full_trace[i], self.t_on, self.t_off, self.t, self.dt) for i in range(n_thetas)])

                mask = super().mask_invalid_samples(x)
                
                # Concatenate the new samples
                x_clean = torch.cat((x_clean, x[mask]), dim=0)
                theta_clean = torch.cat((theta_clean, theta[mask]), dim=0)
                full_trace_clean = torch.cat((full_trace_clean, full_trace[mask]), dim=0)
            
   
                
            _, I_curr = self._jaxley_neuron()
            
            return x_clean, theta_clean, dict(full_trace=full_trace_clean, inj_current=I_curr) 

 
    
    def _mask_valid_thetas(self, thetas):
        # The ratio of gK is 5-10% relative to gNa
        mask = (0.05 * thetas[:, 0] <= thetas[:, 1]) & (thetas[:, 1] <= 0.10 * thetas[:, 0])
        
        return mask
        
        
    def generate_true_pairs(self, n_true_xen, integrator):
        prior = self.prior()
        valid_thetas = []

        # Generate valid thetas
        while len(valid_thetas) < n_true_xen:
            n_resamples = n_true_xen - len(valid_thetas)
            print(f"there have to be {n_resamples} samples generated")
            thetas = prior.sample((n_resamples,))
            theta_mask = self._mask_valid_thetas(thetas)
            
            valid_thetas.append(thetas[theta_mask])

        valid_thetas = np.vstack(valid_thetas)[:n_true_xen]
        print("valid_thetas: ", valid_thetas)
        
        if len(valid_thetas) != n_true_xen:
            raise ValueError("the number of valid thetas is not equal to the requested number of true data")
        
        
        # Generate true data with the simulator given the valid thetas
        true_xen = []
        true_thetas = []
        while len(true_xen) < n_true_xen:
            noise_params = self.get_noise_params(n_thetas)
            thetas_jax = jnp.asarray(valid_thetas)
            simulations = super().simulate_jit(thetas_jax, noise_params, integrator)
            n_thetas = valid_thetas.shape[0]  # should be equal to the true xen

            full_trace = torch.from_numpy(np.asarray(simulations[:, 0]).copy())
            summary_stats = torch.stack([self.calculate_summary_statistics(full_trace[i], self.t_on, self.t_off, self.t, self.dt) for i in range(n_thetas)])

            x_mask = super().mask_valid_samples(summary_stats)
            
            true_xen.append(summary_stats[x_mask])
        # Filter now summary statistics that are not valid, and sample valid 
        
        
        return true_xen, valid_thetas

# %%
