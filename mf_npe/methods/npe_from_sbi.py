
from mf_npe.utils.utils import dump_pickle
from sbi.inference import NPE
from sbi.utils.user_input_checks import process_prior
from sbi.neural_nets import posterior_nn
import torch.nn as nn
import os
import matplotlib.pyplot as plt

def train_npe_with_sbi(self, curr_hf_data):
    # Estimate the high fidelity posteriors with SBI

    # Check prior, return PyTorch prior.
    prior, num_parameters, prior_returns_numpy = process_prior(self.hf_prior)
    
    # Define the density estimator where thetas are not z-scored (since data is logit-transformed) and x is z-scored
    posterior_settings = posterior_nn(model="zuko_nsf", 
                                        z_score_theta=self.z_score_theta,
                                        z_score_x=self.z_score_x, #can also be structured, but independent corresponds to my mf-npe case.
                                        hidden_features=self.config_model['n_hidden_features'], 
                                        num_transforms=self.config_model['n_transforms'], 
                                        num_bins=self.config_model['n_bins'], 
                                        embedding_net=nn.Identity())       

    inference = NPE(prior=prior, density_estimator=posterior_settings) 

    theta, x = curr_hf_data['theta'], curr_hf_data['x']

    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()

    posterior = inference.build_posterior(density_estimator) #

    print("inference summary", inference.summary)
    # Plot the training and validation log probs
    plt.plot(inference.summary['training_loss'], label='training')
    plt.plot(inference.summary['validation_loss'], label='validation')

    # Make directory if does not exist yet
    path_plot = self.main_path + "/loss/"
    if not os.path.exists(path_plot):
        os.makedirs(path_plot)

    plt.savefig(f"{path_plot}/SBI loss_{self.config_data['sim_name']}_{curr_hf_data['n_samples']}.png")
    plt.legend()
    plt.close()
        
    return posterior