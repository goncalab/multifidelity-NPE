import jax.numpy as jnp
import jaxley as jx

class Integrator:
    def __init__(self, lf_cell, config_data):
        self.lf_cell = lf_cell
        self.config_data = config_data

    def __call__(self, theta, noise):
        return simulate_neuron(theta, noise, self.lf_cell, self.config_data)
    

def noisy_step_current(
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
        
        dt = delta_t
        window_start = int(i_delay / dt)
        window_end = int((i_delay + i_dur) / dt)
        time_steps = int(t_max // dt) + 2

        # Create a vector with small noise        
        current = jnp.zeros((time_steps,)) + i_offset + noise
        i_amp_noise = i_amp + noise[window_start:window_end]
        
        return current.at[window_start:window_end].set(i_amp_noise)   



# Only for multicompartmental neuron, where I did not readout from data
def simulate_neuron(params, noise_params, cell, config_data):
    param_state = None
    param_state = cell.data_set("HH_gNa", params[0], param_state)
    param_state = cell.data_set("HH_gK", params[1], param_state)

    current = noisy_step_current(
        i_delay=10.0,
        i_dur=10.0,
        i_amp=0.55,
        delta_t=config_data['dt'],
        t_max=config_data['t_max'],
        i_offset=0.0,
        noise=noise_params
    )
    data_stimuli = None
    data_stimuli = cell.branch(0).comp(0).data_stimulate(current, data_stimuli)

    return jx.integrate(cell, param_state=param_state, data_stimuli=data_stimuli, delta_t=config_data['dt'], t_max=config_data['t_max'])
