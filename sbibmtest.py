#%%
import pandas as pd
import torch
import os
import pickle

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
        tensor_obs = torch.tensor(df.values, dtype=torch.float32)
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

    

generate_true_observations_from_sbibm('SIR') # 'SLCP' or 'LotkaVolterra'


# %%
