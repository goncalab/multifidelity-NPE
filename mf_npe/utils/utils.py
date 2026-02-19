import os
import time
from typing import Optional

import jax
import pandas as pd
from mf_npe.plot.plot_traces import plot_true_OU_traces
from sbi import analysis as analysis
from mf_npe.simulator.task1.GaussianSamples import GaussianSamples
from mf_npe.simulator.task1.OUprocess import OUprocess
from mf_npe.simulator.task2.simulation_func import simulate_neuron
import numpy as np
import pickle
import torch
import matplotlib
from pathlib import Path
from sklearn.linear_model import LinearRegression
import plotly.express as px
import mf_npe.config.plot as plot_setup

import logging
import random


def set_global_seed(seed:Optional[int]=None):
    if seed is None:
        # print("No seed provided. Randomness will not be deterministic.")
        return None

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # JAX
    key = jax.random.PRNGKey(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    return key  # return JAX key to pass into your code



def dump_pickle(path, name_pickle, variable):
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, name_pickle)
    with open(file_path, "wb") as f:
        pickle.dump(variable, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Verify save
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise IOError(f"Pickle file {file_path} was not saved correctly.")
    
    return file_path


def summary_statistics_wrapper(n_simulations, simulator, config_data, main_path):
    
    # Compute how long it takes to generate the samples
    start_time = time.time()
    task = config_data['task']
    
    # Generate the samples
    if task == 'task2':
        cell, I_curr = simulator._jaxley_neuron()
        
        xen, thetas, add_ons = simulator.summary_statistics(
            n_simulations,
            lambda params, noise_params: simulate_neuron(params, noise_params, cell, config_data)
        )
    else:
        # Note: for task3, we don't need the prior because the simulations are loaded.
        xen, thetas, add_ons = simulator.summary_statistics(n_simulations, simulator.prior())    
        
    end_time = time.time()
    # Print how long it took to generate the samples
    total_time = end_time - start_time
    time_for_one_simulation = total_time / n_simulations
    print(f"TOT SIM time: {total_time} sec")
    print(f"ONE SIM: 1/{n_simulations} simulations:", time_for_one_simulation, "sec")
    
    path = main_path + "/time"
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    # Save in a text file how much it took to generate the samples
    with open(f"{path}/simulation_time.txt", "a") as f:
        f.write(f"Simulator: {simulator}\n")
        f.write(f"TOT SIM time: {total_time} sec for {n_simulations} simulations \n")
        f.write(f"ONE SIM time: 1/{n_simulations} simulations: {time_for_one_simulation} sec\n")
        f.write("\n")
        f.write("############################\n")
        
    
    return xen, thetas, add_ons

#################################### GENERATE TRUE DATA #############################################

def generate_true_data(simulate_true_data, 
                       path_to_pickles, 
                       n_true_xen,
                       hf_simulator,
                       config_data):
    """Main class for generating low and high fidelity true data (observations)

    Returns:
        arrray: Array of dictionaries of logit transformed data for numerical stability.
    """
    
    if simulate_true_data:
        n_simulations = n_true_xen
        
        # Generate true data
        true_xen, true_thetas, true_add_ons = summary_statistics_wrapper(n_simulations, hf_simulator, config_data, path_to_pickles)
        
        # Save true data
        n_to_simulate = n_true_xen
        print(f"Simulating {n_to_simulate} true data to save")
        
        simulations = {'true_xen': true_xen, 
                       'true_thetas': true_thetas, 
                       'true_add_ons': true_add_ons}   
        
        pickle_simulations = f"true_xen_{n_to_simulate}.p"
        path = f'{path_to_pickles}/true_data'

        dump_pickle(path=path, name_pickle=pickle_simulations, variable=simulations)
        
    else:
        # Load true data
        pickled_simulations = f"true_xen_{n_true_xen}.p"
        open_pickles_simulations = open(f"{path_to_pickles}/true_data/{pickled_simulations}", "rb")
        loaded_simulations = pickle.load(open_pickles_simulations)
            
        true_xen = loaded_simulations['true_xen']
        true_thetas = loaded_simulations['true_thetas']
        true_add_ons = loaded_simulations['true_add_ons']
        
    return true_xen, true_thetas, true_add_ons



########################## TRAIN DATA FUNCTIONS ##########################################

def generate_train_data(simulate_train_data, 
                        path_to_pickles,
                        batch_lf_sims,
                        batch_hf_sims,
                        hf_simulator,
                        lf_simulator,
                        config_data,
                        ):
    # Generate train data    
    if simulate_train_data:       
        print(f"Generating hf data")
        hf_data = generate_train_data_over_batch_sizes(config_data, hf_simulator, batch_hf_sims, 'hf', path_to_pickles) 
        
        # if lf_data is a dictionary of simulators (multiple fidelities)
        if isinstance(lf_simulator, dict):
            lf_data = {}
            for fidelity, sim in lf_simulator.items():
                print(f"Generating {fidelity} data")
                lf_data[fidelity] = generate_train_data_over_batch_sizes(config_data, sim, batch_lf_sims, fidelity, path_to_pickles)
        else:   
            lf_data = generate_train_data_over_batch_sizes(config_data, lf_simulator, batch_lf_sims, 'lf', path_to_pickles) 
        print("Training data is generated!")
    else:
        hf_data = load_train_data_over_batch_sizes('hf', batch_hf_sims, path_to_pickles, config_data)
        
        # if lf_data is a dictionary of simulators (multiple fidelities)
        if isinstance(lf_simulator, dict):
            lf_data = {}
            for fidelity, sim in lf_simulator.items():
                print(f"Loading {fidelity} data")
                lf_data[fidelity] = load_train_data_over_batch_sizes(fidelity, batch_lf_sims, path_to_pickles, config_data)
        else:
            lf_data = load_train_data_over_batch_sizes('lf', batch_lf_sims, path_to_pickles, config_data)
        print("Training data is loaded!")
    return lf_data, hf_data


def load_train_data_over_batch_sizes(fidelity, n_samples_sizes, path_to_pickles, config_data):
    loaded_simulations = {}
    for i, n_samples in enumerate(n_samples_sizes):
        if fidelity == 'lf':
            if config_data['type_lf'] == 'x_inv':
                print("loading x_inv simulations")
                pickle_simulations = f"{fidelity}_x_inv_simulations_{n_samples}.p"
            elif config_data['type_lf'] == 't_shift':
                print("loading t_shift simulations")
                pickle_simulations = f"{fidelity}_t_shift_simulations_{n_samples}.p"
            elif config_data['type_lf'] == 'hf':
                print("loading hf simulations")
                pickle_simulations = f"{fidelity}_hf_simulations_{n_samples}.p"
            elif config_data['type_lf'] == 'gs':
                print("loading gs simulations")
                pickle_simulations = f"{fidelity}_simulations_{n_samples}.p"
            elif config_data['type_lf'] == 'noise':
                pickle_simulations = f"{fidelity}_noise_{config_data['noise']}_simulations_{n_samples}.p"
            elif config_data['type_lf'] == 'noise_inv':
                pickle_simulations = f"{fidelity}_noise_inv_{config_data['noise']}_simulations_{n_samples}.p"
            else:
                pickle_simulations = f"{fidelity}_simulations_{n_samples}.p"
        else:
            pickle_simulations = f"{fidelity}_simulations_{n_samples}.p"
        open_pickles_simulations = open(f"{path_to_pickles}/train_data/{pickle_simulations}", "rb")
        loaded_simulations[i] = pickle.load(open_pickles_simulations)
    
    return loaded_simulations


def generate_train_data_over_batch_sizes(config_data, simulator, n_samples_sizes, fidelity, path_to_pickles):
    generated_simulations = {}
    for i, n_samples in enumerate(n_samples_sizes):
        # generate the samples
        x, theta, add_ons = summary_statistics_wrapper(n_samples, simulator, config_data, path_to_pickles)
                            
        # Check if the number of generated samples is the same as the requested
        n_samples_generated = x.shape[0]
        if(n_samples_generated != n_samples):
            # Program must stop, otherwise false results in summary data plots.
            raise Warning("number of samples of batch size is not the same as the once that were generated")
        
        
        # Save simulations
        simulations = {'fidelity': fidelity,
                       'n_samples': n_samples_generated, 
                       'x': x, 
                       'theta': theta}
        
        generated_simulations[i] = simulations
        
        # Save data in different pickle files
        if config_data['type_lf'] == 'x_inv':
            pickle_simulations = f"{fidelity}_x_inv_simulations_{n_samples}.p"
        elif config_data['type_lf'] == 't_shift':
            pickle_simulations = f"{fidelity}_t_shift_simulations_{n_samples}.p"
        elif config_data['type_lf'] == 'hf':
            pickle_simulations = f"{fidelity}_hf_simulations_{n_samples}.p"
        elif config_data['type_lf'] == 'gs':
            pickle_simulations = f"{fidelity}_simulations_{n_samples}.p"
        elif config_data['type_lf'] == 'noise':
            pickle_simulations = f"{fidelity}_noise_{config_data['noise']}_simulations_{n_samples}.p"
        elif config_data['type_lf'] == 'noise_inv':
            pickle_simulations = f"{fidelity}_noise_inv_{config_data['noise']}_simulations_{n_samples}.p"
        else:
            pickle_simulations = f"{fidelity}_simulations_{n_samples}.p"
        
        path = f'{path_to_pickles}/train_data/'
        dump_pickle(path=path, name_pickle=pickle_simulations, variable=simulations)
    
    return generated_simulations


############################################################################################

  
def matplotlib_settings():
   logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)
   matplotlib.rcParams['axes.edgecolor'] = '#2A3F5F'
   matplotlib.rcParams['text.color'] = '#2A3F5F'
   matplotlib.rcParams['axes.titlecolor'] = '#2A3F5F'
   matplotlib.rcParams['xtick.color'] = '#2A3F5F'
   matplotlib.rcParams['ytick.color'] = '#2A3F5F'
   matplotlib.rcParams['axes.labelcolor'] = '#2A3F5F'
   path = Path(matplotlib.get_data_path(), "fonts/ttf/OpenSans-Regular.ttf")
   matplotlib.rcParams['font.family'] = f'{path}'
   matplotlib.get_cachedir()
        


def true_posterior_comparison(diff_evaluation, main_path, config_data, CURR_TIME):
    '''
       Compare how close the analytical solutions of the hf posterior and lf posterior are for the OU process
    '''
   
    gamma = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    mu_offset = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    # mu_offset = torch.tensor([0.0, 1.0])
    # gamma = torch.tensor([0.5, 1.0])
    
    theta_dims = 2 # experiment 1, 4 for experiment 2
    n_true_xen = 50 #30
    n_sims = 10**3 # FDONT DO ANTYHTING WITH IT ATM ix the number of simulations. Final test: 10**4 doen
    
    df_all_variables = pd.DataFrame()

    # Create a for loop with mu_offset and gamma 
    for i in range(len(gamma)):
        for j in range(len(mu_offset)):            
            lf_simulator = GaussianSamples(config_data, theta_dim=theta_dims) 
            hf_simulator = OUprocess(config_data, gamma[i], mu_offset[j], theta_dim=theta_dims) 
            lf_prior = lf_simulator.prior()
            hf_prior = hf_simulator.prior()
            
            true_xen, true_thetas, true_add_ons = hf_simulator.summary_statistics(n_true_xen, hf_prior)
            lf_xen = lf_simulator.simulator(true_thetas)
            
            true_full_trace = true_add_ons['full_trace'] 
            
            # Plot the summary stats for different values of gamma and mu_offset
            plot_true_OU_traces(true_full_trace, true_xen, lf_xen, gamma[i], mu_offset[j], main_path)
            
            # Evaluate C2ST on true hf and lf samples
            true_posterior_samples = diff_evaluation.get_true_posterior_samples(true_xen, hf_prior, hf_simulator)
            true_lf_posterior_samples = diff_evaluation.get_true_posterior_samples(true_xen, lf_prior, lf_simulator) # evaluate lf on true_xen!
            
            for k, xto in enumerate(true_xen):
                _ = analysis.pairplot([true_posterior_samples[k], true_lf_posterior_samples[k]], 
                                  limits=hf_simulator.parameter_ranges(4),
                                  upper=['contour', 'contour'], #lower=[None, None],
                                  figsize=(5, 5), points=[torch.tensor(true_thetas[k])], 
                                  labels=['$\mu$', '$\sigma$', '$\gamma$', '$\mu_\mathrm{offset}$'],
                                  title=f"Analytical posteriors, $x_o$:{k}",
                                  filename=f"analytical_posteriors_{j}_x_o{k}",
                                  samples_colors=["#0000A6","#636EFA"],
                                  # points_colors=["#636EFA"]
                                  )
            
            # c2st evaluation over 30 true_xen in a dataframe for a particular gamma and mu_offset
            analytical_df_c2st = diff_evaluation.evaluate_c2st(true_xen, true_lf_posterior_samples, true_posterior_samples, n_sims, inference_method='true_comparison', gamma=gamma[i], mu_offset=mu_offset[j])
            
            # Append results: what was gamma, what was xto?
            df_all_variables = pd.concat([df_all_variables, analytical_df_c2st], ignore_index=True)
            
    # Save dataframe in csv file
    path = f'{main_path}/analytical_posterior/'
    if not os.path.exists(path):
        os.makedirs(path)
        
    X = df_all_variables[['gamma', 'mu_offset']].values
    z = df_all_variables['mean'] # mean in the table is the mean c2st value
    # Fit linear model to get a linear expression of the relation between gamma and mu_offset
    model = LinearRegression()
    model.fit(X, z)

    # Coefficients
    a, b = model.coef_
    c = model.intercept_
    print(f"Equation: z = {a:.2f} * x + {b:.2f} * y + {c:.2f}")
    
    df_all_variables.to_csv(f'{path}/compare_analytical_solutions_OU_{CURR_TIME}.csv', mode='a', index=False)
    
    # plot gamma-mu_offset
    df = df_all_variables
    fig = px.scatter(df, x="gamma", y="mu_offset", color="mean")
    fig.update_layout(width=500, height=500)
    if plot_setup.show_plots:
        fig.show()

    fig.write_image(f"{path}/analytical_{CURR_TIME}.svg")
    fig.write_html(f"{path}/analytical_{CURR_TIME}.html")
    



def generate_true_observations_from_sbibm(simulator_task):
    '''
    Generates true observations and true parameters for the SLCP or Lotka-Volterra tasks from SBIBM.
    Saves them to the data folder.
    
    input:
    simulator_task: 'SLCP' or 'LotkaVolterra'

    '''
    
    # Note: lotka-volterra task has 40_000 reference samples, SLCP has 50_000 reference samples. We will have to randomly sample from the reference samples.

    # generate tru_xen from true_observations file
    true_xen = []  # store tensors here
    n_to_simulate = 10 # number of true observations


    if simulator_task == 'SLCP':
        theta_dim = 5
        task_number = 5
    elif simulator_task == 'LotkaVolterra':
        theta_dim = 4
        task_number = 6
    elif simulator_task == 'SIR':
        theta_dim = 2
        task_number = 7
    else:
        raise ValueError(f"Simulator task {simulator_task} is not supported for generating true data.")

    for i in range(1, n_to_simulate + 1):
        # Load CSV into a DataFrame
        df = pd.read_csv(f'./mf_npe/simulator/task{task_number}/files/num_observation_{i}/observation.csv')
        
        # Convert DataFrame to a 1D torch tensor of floats
        tensor_obs = torch.tensor(df.values.flatten(), dtype=torch.float32)
        true_xen.append(tensor_obs)

    # Stack all tensors along a new dimension
    true_xen = torch.stack(true_xen)

    print(true_xen.shape)
    print(true_xen)
    
    true_thetas = []  # store tensors here

    for i in range(1, n_to_simulate + 1):
        # Load CSV into a DataFrame
        df = pd.read_csv(f'./mf_npe/simulator/task{task_number}/files/num_observation_{i}/true_parameters.csv')
        
        # Convert DataFrame to a 1D torch tensor of floats
        tensor_obs = torch.tensor(df.values.flatten(), dtype=torch.float32)
        true_thetas.append(tensor_obs)

    # Stack all tensors along a new dimension
    true_thetas = torch.stack(true_thetas)

    print(true_thetas.shape)
    print(true_thetas)


    # Add ons for these tasks = true_reference_posterior_samples
    reference_posterior_samples = []

    for i in range(1, n_to_simulate + 1):
        # Load CSV into a DataFrame
        df = pd.read_csv(f'./mf_npe/simulator/task{task_number}/files/num_observation_{i}/reference_posterior_samples.csv.bz2')

        # Convert DataFrame to a 1D torch tensor of floats
        tensor_obs = torch.tensor(df.values.flatten(), dtype=torch.float32)
        reference_posterior_samples.append(tensor_obs)

    # stack all reference posterior samples
    reference_posterior_samples = torch.stack(reference_posterior_samples)
    print(reference_posterior_samples.shape) # has 50_000 samples in total

    true_add_ons = dict(reference_posterior_samples=reference_posterior_samples)

    
    path_to_pickles =f'./data/{simulator_task}/{theta_dim}_dimensions'

    simulations = {'true_xen': true_xen, 
                        'true_thetas': true_thetas, 
                        'true_add_ons': true_add_ons}   
    pickle_simulations = f"true_xen_{n_to_simulate}.p"
    path = f'{path_to_pickles}/true_data'


    def dump_pickle(path, name_pickle, variable):
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, name_pickle)
        with open(file_path, "wb") as f:
            pickle.dump(variable, f, protocol=pickle.HIGHEST_PROTOCOL)


    dump_pickle(path=path, name_pickle=pickle_simulations, variable=simulations)
    
    
    
def get_n_samples(simulations):
        n_lf_samples = []
        n_hf_samples = []
        n_mf_samples = []

        n_hf_samples = [simulations['hf'][i]['n_samples'] for i in range(len(simulations['hf']))]
        n_lf_samples = [simulations['lf'][i]['n_samples'] for i in range(len(simulations['lf']))]

        for i in range(len(simulations['lf'])):
            for j in range(len(simulations['hf'])):
                n_mf_samples.append([simulations['lf'][i]['n_samples'], simulations['hf'][j]['n_samples']])


        return n_lf_samples, n_hf_samples, n_mf_samples