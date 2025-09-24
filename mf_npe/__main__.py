
# #%%
# import pandas as pd
# import mf_npe.task_setup as task_setup
# from mf_npe.one_experiment import run_evaluation, run_one_experiment
# import os

# # must be before importing jax
# if task_setup.config_model["device"] == "cpu":
#     os.environ["JAX_PLATFORM_NAME"] = "cpu" # For cluster, because cpu is faster atm
# elif task_setup.config_model["device"] == "gpu":
#     os.environ["JAX_PLATFORM_NAME"] = "gpu"
    
# from mf_npe.plot.method_performance import plot_methods_performance_paper
# from sbi import utils as utils # pip3 lis to check what the local path is of sbi
# from sbi import analysis as analysis
# import plotly
# from IPython.display import display, HTML
# from mf_npe.plot.plot_switch import plot_true_data_switch
# from mf_npe.utils.calculate_error import mean_confidence_interval
# from mf_npe.utils.utils import generate_train_data, generate_true_data, load_train_data_over_batch_sizes, set_global_seed
# import plotly.io as pio

# if task_setup.show_plots:
#     plotly.offline.init_notebook_mode()
#     display(HTML(
#         '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
#     ))
# else:
#     pio.renderers.default = 'svg'

# def run_multiple_experiments():
#     train_model = True
    
#     #options: 'npe', 'mf_npe', 'sbi_npe', 'mf_tsnpe', 'a_mf_tsnpe', 'mf_abc'
#     models_to_run = ['mf_npe']  
#     add_info = f"HF({task_setup.config_data['type_lf']}) - HF"   # HF* - HF    # HF+ - HF

#     # Define how many initializations you want to run. 
#     n_network_initializations = task_setup.n_network_initializations 
#     df_all_seeds =  pd.DataFrame()
    

#     if train_model:
#         # Generate/load array of dictionaries with settings from task_setup.py
#         load_true_data = True
#         load_train_data = True
#         true_xen, true_thetas, true_add_ons = generate_true_data(b_load_true_data=load_true_data,
#                                                                 path_to_pickles=task_setup.main_path, 
#                                                                 n_true_xen=task_setup.n_true_xen,
#                                                                 hf_simulator=task_setup.hf_simulator,
#                                                                 config_data=task_setup.config_data)
#         if load_true_data:
#             print("True data has been LOADED.")
#         else:
#             print("True data has been GENERATED.")

#         if load_train_data:
#             hf_data = load_train_data_over_batch_sizes('hf', task_setup.batch_hf_sims, task_setup.main_path, task_setup.config_data)
#             lf_data = load_train_data_over_batch_sizes('lf', task_setup.batch_lf_sims, task_setup.main_path, task_setup.config_data) # generate_train_data_over_batch_sizes(task_setup.config_data, task_setup.lf_simulator, task_setup.batch_lf_sims, 'lf', task_setup.main_path) # 
#         else:
#             lf_data, hf_data = generate_train_data(load_train_data=True, path_to_pickles=task_setup.main_path, batch_lf_sims=task_setup.batch_lf_sims, batch_hf_sims=task_setup.batch_hf_sims, lf_simulator=task_setup.lf_simulator, hf_simulator=task_setup.hf_simulator, config_data=task_setup.config_data)

#         if load_true_data:
#             print("Train data has been LOADED.")
#         else:
#             print("Train data has been GENERATED.")
#         # Plot data dependent on which task is chosen
#         plot_true_data_switch(task_setup.config_data["task"], true_xen, true_thetas, true_add_ons,
#                               task_setup.lf_simulator, task_setup.config_data, task_setup.main_path)
        
#         for net_init in range(n_network_initializations):   
#             seed = 42 # 'None' if you dont want to seed
#             key = set_global_seed(seed) # JAX key
#             b_load_model = False
            
#             # Run one experiment
#             train_data = run_one_experiment(seed, models_to_run, 
#                                              lf_data, hf_data, 
#                                              true_xen, true_thetas, true_add_ons, 
#                                              task_setup.batch_lf_sims, 
#                                              task_setup.batch_hf_sims,
#                                              task_setup.sim_name,
#                                              task_setup.config_model,
#                                              task_setup.main_path,
#                                              net_init, b_load_model)
#             # Evaluate the model
#             df_one_seed  = run_evaluation(train_data['task_setup'], 
#                                             true_xen, true_thetas, true_add_ons,
#                                             train_data['n_lf_samples'], train_data['n_hf_samples'], train_data['n_mf_samples'],
#                                             train_data['all_methods'], 
#                                             net_init, 
#                                             train_data['num_hifi'], 
#                                             hf_data)
            
#             df_all_seeds = pd.concat([df_all_seeds, df_one_seed], ignore_index=True)
#             df_all_seeds['round'] = net_init # Add the round to the dataframe as a column
            
#             # Save df_all_seed in a csv file
#             path_dataframe = f'{task_setup.main_path}/raw_evaluation'
#             if not os.path.exists(path_dataframe):
#                 os.makedirs(path_dataframe)
            
            
#             # Metric, num_net_inits, methods, num-samples
#             df_all_seeds.to_pickle(f"{path_dataframe}/{task_setup.config_data['evaluation_metric']}_net{net_init}_{models_to_run}_{task_setup.batch_lf_sims}_{task_setup.batch_lf_sims}_{task_setup.CURR_TIME}.pkl")
#             # Load the pickle file
#             unpickled_df = pd.read_pickle(f"{path_dataframe}/{task_setup.config_data['evaluation_metric']}_net{net_init}_{models_to_run}_{task_setup.batch_lf_sims}_{task_setup.batch_lf_sims}_{task_setup.CURR_TIME}.pkl")  

#             # group pickle over the number of network initializations by calculating the mean and confidence interval
#             grouped_df = unpickled_df.groupby(['n_lf_simulations', 'n_hf_simulations', 'fidelity'])['raw_data'].apply(mean_confidence_interval).reset_index()
#             plot_methods_performance_paper(grouped_df, f"{task_setup.sim_name} {add_info} ({task_setup.config_data['theta_dim']} dims, avg over {net_init+1} networks)", task_setup.batch_lf_sims, unpickled_df['evaluation_metric'][0], task_setup) 
#     else:
#         # Load pickle file of c2st results
#         path = f'{task_setup.main_path}/raw_evaluation'
#         unpickled_df = pd.read_pickle(f"{path}/raw_c2st_results_OUprocess_2025-01-06 12h18.pkl")  

#         # group pickle over the number of network initializations by calculating the mean and confidence interval
#         grouped_df = unpickled_df.groupby(['n_lf_simulations', 'n_hf_simulations', 'fidelity'])['raw_data'].apply(mean_confidence_interval).reset_index()
        
#         plot_methods_performance_paper(grouped_df, f"{task_setup.sim_name} ({task_setup.config_data['theta_dim']} dims)", task_setup.batch_lf_sims, unpickled_df['evaluation_metric'][0], task_setup) 

# if __name__ == "__main__":
#     run_multiple_experiments()


#  #%%
