import torch
from mf_npe.simulator.Prior import Prior
from mf_npe.fsbi.utils import load_and_merge, apply_n_conditions
import matplotlib.pyplot as plt
import torch
import numpy as np
import mf_npe.config.plot as plot_config

class PolynomialNetwork(Prior):
    def __init__(self, config_data):
        self.config_data = config_data
        self.prior_ranges = {k: config_data['all_prior_ranges'][k] for k in list(config_data['all_prior_ranges'].keys())[:config_data['theta_dim']]}
    
    def printName(self):
        return "Polynomial Search Space"
    
    def prior(self):
        lf_prior, _ = super().get_prior(self.prior_ranges, self.config_data)
        return lf_prior
    
    
    def parameter_ranges(self, theta_dim):
        param_ranges = super().parameter_ranges(self.prior_ranges)
        return [param_ranges[f'range_theta{i+1}'] for i in range(theta_dim)]
    
    
    def import_data(self):
        # directory where all the rules simulated in the paper are stored, with all metrics computed.
        data_path = "./mf_npe/simulator/task3/data_synapsesbi/" #task_setup.data_path 

        # name: bg: stability task (background inputs), IF: integrate and fire neuron model in Auryn,EE EI IE II recurrent synapses plastic, 6pPol: polynomial parmeterization with 6 params. 
        dataset_aux = load_and_merge(data_path, ("bg_IF_EEEIIEII_6pPol_all.npy",)) # filter out the ones that are bad (as in tutorial), from posteriors, priors, .... from which he selected the 20min pool.
        
        cond_r = ("rate", 1, 50)
        cond_ri = ("rate_i", 1, 50)
        
        # Condition dataset only on the rates
        condition = apply_n_conditions(dataset_aux, (cond_r,cond_ri))

        dataset = dataset_aux[condition]
        print(str(np.sum(condition)) + "/" + str(len(condition)), "samples kept for training")
        
        return dataset
    
    
    def summary_statistics(self, n_simulations, simulator_prior=None, plot=True):
        # x: train only on rate
        full_dataset = self.import_data()
        
        # Take a random permutation of the full dataset: otherwise the train and validation set will be the same
        dataset = full_dataset[np.random.permutation(len(full_dataset))][:n_simulations]
        
        # Summary statistics
        all_exc_rates = [data['rate'] for data in dataset]
        all_inh_rates = [data['rate_i'] for data in dataset]
        
        rates_exc = torch.tensor(np.array(all_exc_rates), dtype=torch.float32).unsqueeze(1)
        rates_inh = torch.tensor(np.array(all_inh_rates), dtype=torch.float32).unsqueeze(1)
        x = torch.cat((rates_exc, rates_inh), dim=1)        
        
        # EE_EIrules = [data[1][:12] for data in dataset] # 12 dimensional
        allRules = [data[1][:-1] for data in dataset] # 24 dimensional

        theta = allRules # EE_EIrules
        theta = torch.tensor(np.array(theta), dtype=torch.float32)  # Convert to pytorch objects
        
        # TODO: Plot all theta: Should be in another function. Otherwise quite confusing.
        tpreEE = theta[:, 0]
        tpostEE = theta[:, 1]
        alphaEE = theta[:, 2]
        betaEE = theta[:, 3]
        gammaEE = theta[:, 4]
        kappaEE = theta[:, 5]
        
        tpreEI = theta[:, 6]
        tpostEI = theta[:, 7]
        alphaEI = theta[:, 8]
        betaEI = theta[:, 9]
        gammaEI = theta[:, 10]
        kappaEI = theta[:, 11]
        
        lambdaEE = kappaEE * tpostEE + gammaEE * tpreEE
        lambdaEI = kappaEI * tpostEI + gammaEI * tpreEI
                
        add_ons = dict(dataset=dataset)
        
        if plot == True:    
            # Plot excitatory rules
            self.plot_scatter(alphaEE, betaEE, lambdaEE, rates_exc, "$r_{\mathrm{exc}}$")
            # Plot inhibitory rules
            self.plot_scatter(alphaEI, betaEI, lambdaEI, rates_inh, "$r_{\mathrm{inh}}$") 
            
        print("x in Polynomial", x.shape)
        print("theta in polynomial", theta.shape)
        
        
        return x, theta, add_ons
    
    def plot_scatter(self, alpha, beta, lambda_rule, rates, type_rule):
        # EE and EI rules
        plt.figure()
        plt.scatter((-alpha - beta).numpy(), lambda_rule.numpy(), c=rates.numpy(), alpha=0.5, cmap='viridis')
        plt.xlabel('(-alphaEE - betaEE)')
        plt.ylabel('lambdaEE')
        plt.title(f'Polynomial Scatter plot of {type_rule}')
        plt.colorbar(label='rates_exc')
        if plot_config.show_plots:
            plt.show()
        
    
    def simulator(self, theta, true_x, allow_resampling_invalid_samples=False):        
 
        print("true_x", true_x)
        true_exc_rate = true_x[:, 0]
        true_inh_rate = true_x[:, 1]
        
        rates_exc = torch.tensor(np.array(true_exc_rate), dtype=torch.float32).unsqueeze(1)
        rates_inh = torch.tensor(np.array(true_inh_rate), dtype=torch.float32).unsqueeze(1)
        x = torch.cat((rates_exc, rates_inh), dim=1)        
        
        theta = torch.tensor(np.array(theta), dtype=torch.float32)  # Convert to pytorch objects
        
        # TODO: Plot all theta: Should be in another function. Otherwise quite confusing.
        tpreEE = theta[:, 0]
        tpostEE = theta[:, 1]
        alphaEE = theta[:, 2]
        betaEE = theta[:, 3]
        gammaEE = theta[:, 4]
        kappaEE = theta[:, 5]
        
        tpreEI = theta[:, 6]
        tpostEI = theta[:, 7]
        alphaEI = theta[:, 8]
        betaEI = theta[:, 9]
        gammaEI = theta[:, 10]
        kappaEI = theta[:, 11]
        
        lambdaEE = kappaEE * tpostEE + gammaEE * tpreEE
        lambdaEI = kappaEI * tpostEI + gammaEI * tpreEI
                
        
        # Important: use same rates_exc at the ones I used to simulate the data!!!
        # Plot excitatory rules
        self.plot_scatter(alphaEE, betaEE, lambdaEE, rates_exc, "PPC $r_{\mathrm{exc}}$")
        # Plot inhibitory rules
        self.plot_scatter(alphaEI, betaEI, lambdaEI, rates_inh, "PPC $r_{\mathrm{inh}}$") 
                
        print("x in Polynomial", x.shape)
        print("theta in polynomial", theta.shape)
        
        return x, theta, None
        
    
