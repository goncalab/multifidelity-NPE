# # add current date and time to the simulation name
# from datetime import datetime
# from mf_npe.config.plot import width_plots, height_plots, font_size, title_size, gridwidth, axis_color, show_plots
# from mf_npe.utils.task_setup import load_task_setup, process_device


# # Choose which task you want to run
# sim_name = 'OUprocess'
# # sim_name = 'L5PC'
# # sim_name = 'SynapticPlasticity'
# # sim_name = 'GaussianBlob'

# # settings plots
# width_plots, height_plots, font_size, title_size, gridwidth, axis_color, show_plots = width_plots, height_plots, font_size, title_size, gridwidth, axis_color, show_plots

# # Run for high-data mode over only 2 network inits.
# batch_lf_sims = [10**3]   #  (e.g., 10**3, 10**4, 10**5)
# batch_hf_sims = [100]     # For mf-abc: Put this on 1000 to have a good estimate of the z-score (will not be used to generate samples)
# n_network_initializations = 1

# batch_mf_sims = [[lf, hf] for lf in batch_lf_sims for hf in batch_hf_sims]

# config_model = dict(
#     max_num_epochs=2**31 - 1, # high number since we have early stopping
#     batch_size = 200, # increasing the batch size will speed up the training, but the model will be less accurate
#     learning_rate= 5e-4, # Learning rate for Adam optimizer
#     type_estimator='npe', # we always compute the posterior (npe), and do not evaluated likelihood or ratio methods (e.g., NLE, NRE)
#     device = process_device(),
#     validation_fraction = 0.1, # Fraction of the data to use for validation
#     patience=20, # The number of epochs to wait for improvement on the validation set before terminating training.
#     n_transforms = 5, 
#     n_bins=8,
#     n_hidden_features = 50,
#     clip_max_norm = 5.0, # value to which to clip total gradient norm to prevent exploding gradients. Use None for no clipping
    
#     # Choose between logit transforming or z_scoring thetas, not both
#     # logit_transform_theta_net = True, # for training in unbound space: Then we do not have that much leakage in posterior
#     # z_score_theta = False, 
#     # z_score_x = True,
#     logit_transform_theta_net = True, # for training in unbound space: Then we do not have that much leakage in posterior
#     z_score_theta = False, 
#     z_score_x = True,
#     # For active learning
#     active_learning_pct=0.8,
#     n_rounds_AL = 5, # From 1 to 5 
#     n_theta_samples = 1000, #250,
#     n_ensemble_members = 5, 
#     )


# # setting if running through main
# if sim_name == 'OUprocess':
#     theta_dim = 2 # Number of trainable parameters for HF model
#     n_true_xen = 2
# if sim_name == 'L5PC':
#     theta_dim = 2
#     n_true_xen = 100
# if sim_name == 'SynapticPlasticity':
#     theta_dim = 24
#     n_true_xen = 300_000
# if sim_name == 'GaussianBlob':
#     theta_dim = 3
#     n_true_xen = 30 
# if sim_name == 'SLCP':
#     theta_dim = 5
#     n_true_xen = 10 
# if sim_name == 'LotkaVolterra':
#     theta_dim = 4
#     n_true_xen = 10 

# config_data, lf_simulator, hf_simulator = load_task_setup(sim_name, theta_dim=theta_dim, n_true_xen=n_true_xen) 


# main_path = f"./../data/{sim_name}/{config_data['theta_dim']}_dimensions"
# CURR_TIME = datetime.now().strftime("%Y-%m-%d %Hh%M")





