from mf_npe.plot.plot_traces import plot_CompNeuron_xen, plot_OU_xen

def plot_true_data_switch(task, true_xen, true_thetas, true_add_ons, lf_simulator, config_data, main_path):
    if task == 'task1':
        lf_xen = lf_simulator.simulator(true_thetas) # Just for plotting
        true_trace = true_add_ons['full_trace']
        plot_OU_xen(true_xen, lf_xen, true_trace, config_data, main_path)
        
    elif task == 'task2':
        true_trace = true_add_ons['full_trace']
        I_curr = true_add_ons['inj_current']
        plot_CompNeuron_xen(true_xen, I_curr, true_trace, 
                            config_data['dt'], 'True xen and current','true_x', '-', i='', path_to_save=main_path)
        
        ### Just for plotting: generate LF data based on the true_thetas: Commented because simulator takes some time
        #lf_cell, _ = task_setup.lf_simulator._jaxley_neuron()
        #integrator_fn = Integrator(lf_cell, self.config_data)
        # lf_xen, lf_thetas, true_add_ons = task_setup.lf_simulator.simulator(true_thetas, 
        #                                                     integrator_fn, # param and noise param lambda inside
        #                                                     allow_resampling_invalid_samples=True)
        # lf_trace = true_add_ons['full_trace']
        # lf_I_curr = true_add_ons['inj_current']
        # plot_CompNeuron_xen(lf_xen, lf_I_curr, lf_trace, 
        #                     task_setup.config_data['dt'], 'lf x and current', 'lf_x', '-', i='')
        
    elif task == 'task3':
        pass
    else:
        raise ValueError(f"Unknown task: {task}. Cannot plot true data.")