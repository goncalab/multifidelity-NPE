

from jax import jit, vmap
import numpy as np
from scipy import stats as spstats
import jax.numpy as jnp
import torch
import multiprocessing as mp


def simulate_batch(args):
    thetas, noise_params, integrator = args
    jitted = jit(integrator)
    vmapped = vmap(jitted, in_axes=(0, 0)) # Happens atm on CPU (on mac)
    
    # print("jax devices", jax.devices())
    # print("default", jax.default_backend())
    
    return vmapped(thetas, noise_params)

class Neuron():
    def __init__(self):
        pass
    
    def noisy_step_current(
        self, # Just added
        i_delay: float,
        i_dur: float,
        i_amp: float,
        delta_t: float,
        t_max: float,
        i_offset: float = 0.0,
        noise = None,
        ):
        """
        Return step current in unit nA.

        Unlike the `datapoint_to_step()` method, this takes a single value for the amplitude
        and returns a single step current. The output of this function can be passed to
        `.stimulate()`, but not to `integrate(..., currents=)`.
        
        This function is a modification from Jaxley, a differentiable neuroscience simulator. Jaxley is
        licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
        """
        
        print("noise: ", noise)
        
        dt = delta_t
        window_start = int(i_delay / dt)
        window_end = int((i_delay + i_dur) / dt)
        time_steps = int(t_max // dt) + 2

        # Create a vector with small noise        
        current = jnp.zeros((time_steps,)) + i_offset + noise
        i_amp_noise = i_amp + noise[window_start:window_end]
        
        return current.at[window_start:window_end].set(i_amp_noise)   
    
    
    # def mask_invalid_samples(self, summary_stats):
    #         mask = ~torch.isnan(summary_stats).any(dim=1) & ~torch.isinf(summary_stats).any(dim=1)
    #         n_spikes, mu_rp, std_rp, mean_v = summary_stats[:, 0], summary_stats[:, 1], summary_stats[:, 2], summary_stats[:, 3]
            
    #         return mask 
    
    
    def calculate_summary_statistics(self, x, t_on, t_off, t, dt):
        """Calculate summary statistics

        Parameters
        ----------
        x : output of the simulator

        Returns
        -------
        np.array, summary statistics:
        
        1. number of spikes, 
        2. the mean resting potential, 
        3. the standard deviation of the resting potential,
        
        the first four voltage moments: 
        4. mean, 
        5. standard deviation, 
        6. skewness 
        7. kurtosis.
        """
                
        n_mom = 4
        #n_summary = 7
        #n_summary = np.minimum(n_summary, n_mom + 3)

        # initialise array of spike counts
        v = np.array(x)  # x["data']

        # put everything to -10 that is below -10 or has negative slope
        ind = np.where(v < -10)
        v[ind] = -10
        ind = np.where(np.diff(v) < 0)
        v[ind] = -10

        # remaining negative slopes are at spike peaks
        ind = np.where(np.diff(v) < 0)
        
        spike_times = np.array(t)[ind]
        spike_times_stim = spike_times[(spike_times > t_on) & (spike_times < t_off)]
    
        # number of spikes
        if spike_times_stim.shape[0] > 0:        
            spike_times_stim = spike_times_stim[
                np.append(1, np.diff(spike_times_stim)) > 0.5
            ]
        
        # convert torch to numpy
        x = x.numpy()
        
        rest_pot = np.mean(x[t < t_on])
        rest_pot_std = np.std(x[int(0.9 * t_on / dt) : int(t_on / dt)])

        # moments
        std_pw = np.power(
            np.std(x[(t > t_on) & (t < t_off)]), np.linspace(3, n_mom, n_mom - 2)
        )
        std_pw = np.concatenate((np.ones(1), std_pw))
        moments = (
            spstats.moment(
                x[(t > t_on) & (t < t_off)], np.linspace(2, n_mom, n_mom - 1)
            )
            / std_pw
        )
        
        # concatenation of summary statistics
        sum_stats_vec = np.concatenate((
            np.array([float(spike_times_stim.shape[0])]), #  np.array([rest_pot]), # 
            np.array([
                rest_pot,
                rest_pot_std,
                np.mean(x[(t > t_on) & (t < t_off)]),
            ]),
            #np.array([moments[1], moments[2]]) #moments[0], moments, moments, # rest_pot, 
        ))
        
        # sum_stats_vec = sum_stats_vec #[0:n_summary]
    
        return torch.tensor(sum_stats_vec).float()
    
    
    # Version without parallelization
    def simulate_jit(self, thetas, noise_params, integrator):
        # Fast for-loops with jit compilation.
        jitted_simulate = jit(integrator)
        # voltages = [jitted_simulate(params, noise) for params, noise in zip(thetas, noise_params)] # Not needed

        # Using vmap for parallelization.
        vmapped_simulate = vmap(jitted_simulate, in_axes=(0,0))
        voltages = vmapped_simulate(thetas, noise_params)
        
        return voltages
    
    # Version with parallelization
    def simulate_jit_multiprocess(self, thetas, noise_params, integrator, num_workers=15): # mp.cpu_count()-1
        # raise ValueError("It should not simulate with multiprocessing")
        print("num_workers: ", num_workers)
        chunks = np.array_split(np.arange(len(thetas)), num_workers)
        
        # print("chunks: ", chunks)
        
        args = [
            (thetas[chunk], noise_params[chunk], integrator)
            for chunk in chunks
        ]

        # Parallelize the simulation using multiprocessing.
        mp_context = mp.get_context("spawn") # a bit slower than fork, but safer
        with mp_context.Pool(processes=num_workers) as pool:
            voltages = pool.map(simulate_batch, args)
        
        return jnp.concatenate(voltages)