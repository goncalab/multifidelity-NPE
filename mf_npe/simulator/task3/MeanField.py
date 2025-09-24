import torch
from mf_npe.simulator.Prior import Prior
import matplotlib.pyplot as plt
import torch
from mf_npe.simulator.task3.PolynomialNetwork import PolynomialNetwork
import mf_npe.config.plot as plot_config

class MeanField(Prior):
    
    def __init__(self, config_data):
        self.config_data = config_data
        self.prior_ranges = {k: config_data['all_prior_ranges'][k] for k in list(config_data['all_prior_ranges'].keys())[:config_data['theta_dim']]}
        
    def printName(self):
        return "Mean Field"
    
    def prior(self):
        lf_prior, _ = super().get_prior(self.prior_ranges, self.config_data)
        return lf_prior
    
    def parameter_ranges(self, theta_dim):
        param_ranges = super().parameter_ranges(self.prior_ranges)
        return [param_ranges[f'range_theta{i+1}'] for i in range(theta_dim)]
        
    def summary_statistics(self, n_simulations, simulator_prior=None):
        """
        Generate summary statistics for the given number of simulations. Trained on only 12 statistics.
        Args:
            n_simulations (int): The number of simulations to run.
        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): A tensor of shape (N, 2) where N is the number of valid simulations,
                  containing the rates for excitatory and inhibitory neurons.
                - theta_matrix (torch.Tensor): A tensor of shape (N, len(theta_keys)) containing the valid
                  sampled parameters.
                - add_ons (dict): A dictionary containing additional information, specifically:
                    - 'dataset' (torch.Tensor): A tensor containing the rates for excitatory neurons.
        Notes:
            - The function samples parameters from the prior distribution and computes the rates for excitatory
              and inhibitory neurons.
            - It filters out invalid simulations based on certain conditions.
            - It plots scatter plots for the valid simulations.
            - we dont need simulator prior here. 
        """
        
        max_n_samples = 10**8
        
        # NOTE: simulator prior is for consistense with other simulators.
        
        
        # use thetas from high fidelity model to generate the prior: otherwise unfair comparison to simulations from the files.
        polynom = PolynomialNetwork(self.config_data)
        x, theta_s, add_ons = polynom.summary_statistics(max_n_samples, plot=False) # Load as much as possible

        
        #prior = self.prior()
        #theta_s = prior.sample((max_n_samples,))
        
        # Last 12 keys are not used to generated the summary statistics
        theta_keys = ['tpreEE', 'tpostEE', 'alphaEE', 'betaEE', 'gammaEE', 'kappaEE', 
                      'tpreEI', 'tpostEI', 'alphaEI', 'betaEI', 'gammaEI', 'kappaEI',
                      'tpreIE', 'tpostIE', 'alphaIE', 'betaIE', 'gammaIE', 'kappaIE', 
                      'tpreII', 'tpostII', 'alphaII', 'betaII', 'gammaII', 'kappaII']
        
        theta = {key: theta_s[:, i] for i, key in enumerate(theta_keys)}
        
        lambdaEE = theta['kappaEE'] * theta['tpostEE'] + theta['gammaEE'] * theta['tpreEE']
        lambdaEI = theta['kappaEI'] * theta['tpostEI'] + theta['gammaEI'] * theta['tpreEI']
        
        # # Valid masks for EE and EI: must be laying in 3rd quadrant
        # valid_mask_EE = (theta['alphaEE'] + theta['betaEE'] > 0) & (lambdaEE < 0)
        # valid_mask_EI = (theta['alphaEI'] + theta['betaEI'] > 0) & (lambdaEI < 0)
        
        # # Combine masks
        # valid_mask = valid_mask_EE & valid_mask_EI
        
        # theta = {key: value[valid_mask] for key, value in theta.items()}
        # lambdaEE, lambdaEI = lambdaEE[valid_mask], lambdaEI[valid_mask]
                
        rates_exc = (-theta['alphaEE'] - theta['betaEE']) / lambdaEE
        rates_inh = (-theta['alphaEI'] * rates_exc) / (theta['betaEI'] + lambdaEI * rates_exc)
        
        x = torch.cat((rates_exc.unsqueeze(1), rates_inh.unsqueeze(1)), dim=1).float()
        
        mask = ~torch.isnan(x).any(dim=1) & ~torch.isinf(x).any(dim=1) & \
               (x[:, 0] > 1) & (x[:, 0] < 50) & (x[:, 1] > 1) & (x[:, 1] < 50)
        
        x, theta = x[mask], {key: value[mask] for key, value in theta.items()}
        lambdaEE, lambdaEI = lambdaEE[mask], lambdaEI[mask]
        
        # Return now the number of samples that we are interested in
        add_ons = dict(dataset=rates_exc)
        
        x = x[:n_simulations, :]
        theta = {key: value[:n_simulations] for key, value in theta.items()}
        add_ons = {key: value[:n_simulations] for key, value in add_ons.items()}
        lambdaEE = lambdaEE[:n_simulations]
        lambdaEI = lambdaEI[:n_simulations]

        # plot exc and inh with right number of simulations
        self._plot_scatter((-theta['alphaEE'] - theta['betaEE']).numpy(), lambdaEE.numpy(), x[:, 0].numpy(), '(-alphaEE - betaEE)', 'lambdaEE', 'r_exc')
        self._plot_scatter((-theta['alphaEI'] - theta['betaEI']).numpy(), lambdaEI.numpy(), x[:, 1].numpy(), '(-alphaEI - betaEI)', 'lambdaEI', 'r_inh')

        theta_matrix = torch.stack([theta[key] for key in theta_keys], dim=1)
        
        # TODO: clamp zero values: log(0)=NaN in log transform. "divison by zero" in theta_matrix
        # mask = (theta_matrix == 0).any(dim=1)
        # x = x[~mask]
        # theta_matrix = theta_matrix[~mask]
        
        # Check for NaNs and Infs and division by zero
        assert not torch.isnan(x).any(), 'NaNs in x'
        assert not torch.isnan(theta_matrix).any(), 'NaNs in theta_matrix'
        assert not torch.isinf(x).any(), 'Infs in x'
        assert not torch.isinf(theta_matrix).any(), 'Infs in theta_matrix'
        assert not (x == 0).any(), 'Division by zero in x'
        assert not (theta_matrix == 0).any(), 'Division by zero in theta_matrix'
        
        
        print("x in mean field", x.shape)
        print("theta in mean field", theta_matrix.shape)
        
        # Give all 24 thetas, not only 12
        return x, theta_matrix, add_ons
    
    def _plot_scatter(self, x, y, c, xlabel, ylabel, title):
        plt.figure()
        plt.scatter(x, y, c=c, alpha=0.5, cmap='viridis')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'Mean field Scatter plot of {title}')
        plt.colorbar(label=title)
        
        if plot_config.show_plots:
            plt.show()
    
    
    def simulator(self, theta):
        '''
              Not used yet, needed for posterior predictive checks
            theta:[τpre EE, τpost EE, αEE, βEE, γEE, κEE]
            TODO: Add a bit of gaussian noise to the equation, otherwise inference wont work
        '''
        # mean field simulator
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
        
        # TODO: Make the simulator stochastic
        rates_exc = (- alphaEE - betaEE) / lambdaEE
        rates_inh = (-alphaEI * rates_exc) / (betaEI + lambdaEI * rates_exc) # Rate inh is dependent on rate_exc
        
        x = torch.cat((rates_exc, rates_inh), dim=1)
        print("x in mean field", x)
    
        plt.figure()
        plt.scatter((-alphaEE - betaEE).numpy(), lambdaEE.numpy(), c=rates_exc.numpy(), alpha=0.5, cmap='viridis')
        plt.xlabel('(-alphaEE - betaEE)')
        plt.ylabel('lambdaEE')
        plt.title('Polynomial Scatter plot of r_exc')
        plt.colorbar(label='rates_exc')
        plt.show()
        
        plt.figure()
        plt.scatter((-alphaEI - betaEI).numpy(), lambdaEI.numpy(), c=rates_inh.numpy(), alpha=0.5, cmap='viridis')
        plt.xlabel('(-alphaEI - betaEI)')
        plt.ylabel('lambdaEI')
        plt.title('Polynomial Scatter plot of r_inh')
        plt.colorbar(label='rates_inh')
        plt.show()
        
        return x