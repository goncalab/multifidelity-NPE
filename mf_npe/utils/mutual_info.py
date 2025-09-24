#%%
import numpy as np
import pandas as pd
import torch
from entropy_estimators import continuous
import bmi.estimators as MINE # https://github.com/cbg-ethz/bmi
import pickle

### Create a new conda environment
### conda create -n mi python=3.10
### conda activate mi
### pip install torch entropy-estimators benchmark-mi
###

path_lf = '../../data/OUprocess/2_dimensions/train_data/lf_simulations_10000.p'
path_hf = '../../data/OUprocess/2_dimensions/train_data/hf_simulations_10000.p'

# Must be simulations form the

# Load x_lf and x_hf from pickle files
# pickle_in = open(path_lf, "rb")
# x_lf = pickle.load(pickle_in)
# pickle_in.close()

pickle_in = open(path_hf, "rb")
x_hf = pickle.load(pickle_in)
pickle_in.close()

# print("x_lf", x_lf)
# print("x_hf", x_hf)
# # # Detach to numpy
# X = x_lf['x'].detach().cpu().numpy()
Y = x_hf['x'].detach().cpu().numpy()

# generate LF simulations given the dictionalry
theta_hf = x_hf['theta'].detach().cpu().numpy()

# x_lf = lf.simulator(torch.tensor(theta_hf))

# Set k (number of neighbors for estimation)
k = 5

mi_ksg_estimate = continuous.get_mi(X, Y, k=k)
print(f"Estimated MI I(x_lf; x_hf): {mi_ksg_estimate:.3f} nats")


# estimated MI using MINE
MI_estimator = MI_estimator = MINE.MINEEstimator(standardize=True) # bmi.estimators.InfoNCEEstimator() # 
MI_value = MI_estimator.estimate(X, Y)  # this should return a scalar
print(f"Estimate by MINE: {MI_value:.2f}")

# Estimate entropy
entropy_xhf_estimate = continuous.get_h(Y, k=k)
print(f"Estimated Entropy H(x_hf): {entropy_xhf_estimate:.3f} nats")

# Ratio of MI to Entropy
uncertainty_coefficient = mi_ksg_estimate / entropy_xhf_estimate
print(f"Uncertainty Coefficient KSG: {uncertainty_coefficient:.3f}")

# Ratio MINE to Entropy
uncertainty_coefficient_MINE = MI_value / entropy_xhf_estimate
print(f"Uncertainty Coefficient MINE: {uncertainty_coefficient_MINE:.3f}")

# Save MI in a dataframe


#%%


# Create a dataframe
results_df = pd.DataFrame({
    "noise": noise.item(),
    "entropy": entropy_xhf_estimate,
    "MI_MINE": MI_value,
    "MI_KSG": mi_ksg_estimate,
    "UC_MINE": uncertainty_coefficient_MINE,
    "UC_KSG": uncertainty_coefficient,
}, index=[0])

# Define the filename dynamically
filename = f"MI_dataframe_noise={round(noise.item(), 2)}_{lf_simulator_name}.pkl"
save_dir = f'{task_setup.main_path}/MI'

# Save the dataframe to a pickle file    
dump_pickle(save_dir, filename, results_df)




# from mf_npe.evaluation import dump_pickle
# from mf_npe.simulator.task1.OUprocess import OUprocess
# from mf_npe.simulator.task1.GaussianSamples import GaussianSamples
# from mf_npe.simulator.task1.OUprocessNoise import OUprocessNoise
# from mf_npe.simulator.task1.OUprocessXinverseNoise import OUprocessXinverseNoise
# from mf_npe.simulator.task2.L5PC import L5PC
# from mf_npe.simulator.task2.simulation_func import simulate_neuron
# from mf_npe.utils.task_setup import load_task_setup


# simulator_task = 'OUprocess' #'OUprocess' #'L5PC' #'SynapticPlasticity' #'OUprocess' #'SLCP' #'LotkaVolterra' #'SIR' #'GaussianBlob' #
# theta_dim = 2
# n_true_xen = 1
# config_data, lf_simulator, hf_simulator = load_task_setup(simulator_task, theta_dim=theta_dim, n_true_xen=n_true_xen) 

# # Generate xen samples

# # Generate theta samples
# n_samples = 10000
# theta = hf_simulator.prior().sample((n_samples,))

# x_lf = lf_simulator.simulator(theta)  # (n_samples, x_dim)
# x_hf = hf_simulator.simulator(theta)  # (n_samples, x_dim)

# print("x_lf", x_lf.shape)
# print("x_hf", x_hf.shape)

# Load x_lf and x_hf from pickle files
#path = '../../data/SynapticPlasticity/24_dimensions/models/train_mf_npe_LF100000_HF100000_Ninits1_seed1-1.pkl'


# #%%
# #test
# theta = torch.tensor([[2, 0.01]])

# x_lf = lf_simulator.simulator(theta)  # (n_samples, x_dim)
# x_hf = hf_simulator.simulator(theta) 

# print("x_lf", x_lf)
# print("x_hf", x_hf)
# # %%

# # NCE
# # MI_estimator = bmi.estimators.InfoNCEEstimator()
# # MI_value = MI_estimator.estimate(X.reshape(-1, 1), Y.reshape(-1, 1))  # this should return a scalar
# # print(f"Estimate by NCE: {MI_value:.2f}")

# # Estimate by NCE x_inv: 0.09
# # Estimate by NCE x: 0.65, 0.68
# # Estimate by NCE x_inv: 0.10, 0.09
# # Estimate by NCE x: 0.65, 0.66


# MI_estimator = bmi.estimators.NWJEstimator()
# MI_value = MI_estimator.estimate(X.reshape(-1, 1), Y.reshape(-1, 1))  # this should return a scalar
# print(f"Estimate by histogram: {MI_value:.2f}")



# #%%


# def euclidean_pairwise(samples, ord=2):
#     return torch.cdist(
#         samples.unsqueeze(0).double(),
#         samples.unsqueeze(0).double(),
#         p=ord,
#     )[0]
    
# # NOTE: This works only if the box "fits" the torus size.
# # The default box is -1 to 1, so the default torus size is 2
# def n_torus_pairwise(samples, torus_size=2.0, ord=2):
#     diff = torch.cdist(
#         samples.transpose(0, 1).unsqueeze(-1).double(),
#         samples.transpose(0, 1).unsqueeze(-1).double(),
#     )
#     min_diff = torch.min(diff, torus_size - diff)
#     return torch.linalg.vector_norm(min_diff, ord=ord, dim=0)

    
# def kozachenko_leonenko_estimator(samples, eps=1e-100, hack=1e10, on_torus=False):
#     num_samples, dim = samples.shape
#     assert num_samples <= 20000  # to not blow up memory
#     if on_torus:
#         pairdist = n_torus_pairwise(samples)
#     else:
#         pairdist = euclidean_pairwise(samples)
#     nn_dist = torch.min(
#         pairdist + torch.eye(len(samples), device=samples.device) * hack, dim=1
#     ).values

#     nn_dist = torch.clamp(nn_dist, min=eps)
#     return (
#         dim * torch.mean(torch.log(nn_dist))
#         + np.log(((np.pi ** (dim / 2.0)) / gamma(1.0 + (dim / 2.0))))
#         - digamma(1)  # nearest neighbor
#         + digamma(num_samples)
#     )
    
# H_x = kozachenko_leonenko_estimator(x_lf)
# H_y = kozachenko_leonenko_estimator(x_hf)
# H_xy = kozachenko_leonenko_estimator(torch.cat((x_lf, x_hf), dim=1))

# MI_kozachenko_leonenko = H_x + H_y - H_xy
# print("H_y", H_y)
# print(f"Estimated MI I(x_lf; x_hf) using Kozachenko-Leonenko: {MI_kozachenko_leonenko:.3f} nats")


# %%




#### TASK 1
# if lf_simulator_name == 'OUprocess_x_inv':
#     lf_simulator =  OUprocessXinverseNoise(task_setup.config_data, task_setup.gamma, task_setup.mu_offset, noise) 
# elif lf_simulator_name == 'OUprocess':
#     lf_simulator =  OUprocessNoise(task_setup.config_data, task_setup.gamma, task_setup.mu_offset, noise) # task_setup.lf_simulator
# hf_simulator = task_setup.hf_simulator


# lf_simulator = L5PC(task_setup.config_data, n_compartments=1)
# hf_simulator = L5PC(task_setup.config_data, n_compartments=8) 




#### TASK 2
# cell, _ = hf_simulator._jaxley_neuron()
# x_hf, theta_clean, add_ons = hf_simulator.simulator(theta, 
#                                     lambda params, noise_params: simulate_neuron(params, noise_params, cell, task_setup.config_data),
#                                     allow_resampling_invalid_samples=False)
# x_lf, theta_clean_lf, add_ons_lf = lf_simulator.simulator(theta, 
#                                     lambda params, noise_params: simulate_neuron(params, noise_params, cell, task_setup.config_data),
#                                     allow_resampling_invalid_samples=False)


# # If one is bigger than the other, remove the rows where theta_clean is different from theta_clean_lf
# mask = np.all(theta_clean == theta_clean_lf, axis=1)  # Axis 1 to compare row-wise

# xf_hf = x_hf[mask]
# x_lf = x_lf[mask]



# Compute c2st also on exactlyt he same data!



# Simulate: Adjust the noise level in task_setup
#x_lf = lf_simulator.simulator(theta)  # (n_samples, x_dim)

# Flip the x dimensions: 
# print("before flip", x_lf)
# x_lf = x_lf.flip(dims=[1])
# print("after flip", x_lf)

# x_hf = hf_simulator.simulator(theta)  # (n_samples, x_dim)

# print first 5 lf simulations

# noise = torch.tensor([10])
# lf_simulator_name = 'L5PC' #'OUprocess_x_inv' # OUprocess
