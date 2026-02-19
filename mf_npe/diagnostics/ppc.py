import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
import mf_npe.config.plot as plot_config
from mf_npe.diagnostics.histogram import plot_xen_histogram
from mf_npe.plot.plot_traces import plot_CompNeuron_xen
from mf_npe.simulator.task2.simulation_func import simulate_neuron

class PPC():
    """
    Class to create pairplots for posterior samples.
    """

    def __init__(self, true_xen, task_setup):
        self.true_xen = true_xen
        self.task_setup = task_setup
        
        self.hf_simulator = task_setup.hf_simulator
        self.lf_simulator = task_setup.lf_simulator
        self.config_data = task_setup.config_data
        self.main_path = task_setup.main_path
        
        self.theta_dim = task_setup.theta_dim
        self.task = task_setup.config_data['task']
        
        # Number of posterior samples to generate
        self.n_samples_to_generate = 1000

    def plot_ppc_timeseries(self, x_pp, config_data, true_x, title, main_path):
        n_to_plot = 10
        df = pd.DataFrame(x_pp[:n_to_plot])
        df.index = [f'{i+1}' for i in range(n_to_plot)]
                    
        type_x = f'{title} x(t)'
        fig = px.line(df.T)
                    
        fig.add_trace(
            go.Scatter(
                x=np.arange(config_data['x_dim_hf']),
                y=true_x,
                mode="lines",
                line=go.scatter.Line(color="black", dash='dash'),
                showlegend=False)
        )
                
        fig.update_layout(
            # xaxis=dict(
            #     rangeslider=dict(
            #         visible=True)),
            template="simple_white",
            title=f"{type_x}", 
            width=plot_config.width_plots,
            height=plot_config.height_plots,
            font_color=plot_config.axis_color,
            xaxis_title="t",
            yaxis_title="X(t)",
            # limits on y-axis
            yaxis_range=[0, 10], 
            legend_title="samples")
        
        first_xo = round(true_x[0].item(), 2)
        
        path_ppc = f"{main_path}/ppc"
        if not os.path.exists(path_ppc):
            os.makedirs(path_ppc)   
        fig.write_image(f"{path_ppc}/{type_x}_xo{first_xo}_{config_data['type_lf']}.svg")
        fig.write_html(f"{path_ppc}/{type_x}_xo{first_xo}_{config_data['type_lf']}.html")
        
        if plot_config.show_plots:
            fig.show()
        

    def _posterior_predictive_check(self, posterior_samples, x_o, full_trace_true, inference_method, n_train_sims, i):
        if self.task == 'task1' or self.task == 'task6':
            x_pp = self.hf_simulator.simulator(abs(posterior_samples))         
            print("x_pp", x_pp.shape)   
            self.plot_ppc_timeseries(x_pp, self.config_data, x_o, f'PPC {inference_method}, n_sims {n_train_sims}', self.main_path)
        elif self.task == 'task2':
            print("posterior_samples", posterior_samples.shape)
            posterior_samples = posterior_samples[:20]
            print("posterior_samples trimm", posterior_samples.shape)
            
            cell, _ = self.hf_simulator._jaxley_neuron()
            x_pp, theta_clean, add_ons = self.hf_simulator.simulator(posterior_samples, 
                                                lambda params, noise_params: simulate_neuron(params, noise_params, cell, self.config_data),
                                                allow_resampling_invalid_samples=False)
            full_traces = add_ons['full_trace']
            I_inj = add_ons['inj_current']
            
            plot_xen_histogram(x_pp, 'PPC task 2', x_o)
            plot_CompNeuron_xen(x_o, I_inj, full_traces, self.config_data['dt'], f'PPC {inference_method}, n_sims: {n_train_sims}',
                                inference_method, n_train_sims, i, self.main_path, true_x_trace=full_trace_true)
        elif self.task == 'task3':
            path_simulations = ''
            
            # IMPORTANT
            # Check if you load the correct true xen (true file is saved in paper folder) mf_npe/exports/paper/SpikingNetwork/true_xen_1000.p from 10/1 at 12:02
            # Otherwise it will not work!!!
            
            # Load the posterior samples from the cluster
            with open(path_simulations, 'rb') as f:
                    posterior_samples = np.load(f)
                    print("loaded a", posterior_samples)
                    print("loaded a shape", posterior_samples.shape)
            
            x_pp = self.hf_simulator.simulator(posterior_samples, x_o)
        
        elif self.task == 'task4':
            x_pp = []
        else:
            raise ValueError(f"Unknown task: {self.task}. Cannot perform posterior predictive check.")
            
        return x_pp  # Returning empty lists for lf and prior as they are not used in task1 and task3
