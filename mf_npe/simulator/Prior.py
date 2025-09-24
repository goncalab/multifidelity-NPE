
import torch
from sbi.utils import BoxUniform # pip3 lis to check what the local path is of sbi
from sbi.utils.user_input_checks import process_prior

class Prior:
    """The prior class for the simulator."""
    
    def __init__(self):
        pass
        
    def get_prior(self, param_ranges, config_data):
        """ 
        The prior is a uniform distribution over the parameters.
        Args:
            param_ranges (dict): The parameter ranges for the prior.
            e.g. 
            param_ranges = {
             "g_Na": (0.5, 80.0),
             "g_K": (1e-4, 15.0)
            }
        """
        min_values = torch.tensor([param_ranges[key][0] for key in param_ranges])
        max_values = torch.tensor([param_ranges[key][1] for key in param_ranges])
        
        theta_dim = config_data['theta_dim']
        
        prior = BoxUniform(low=min_values[:theta_dim], high=max_values[:theta_dim])
        prior, *_ = process_prior(prior)
        prior, num_parameters, prior_returns_numpy = process_prior(prior)
        
        lf_prior = prior
        hf_prior = prior
        
        return lf_prior, hf_prior
    
    
    def mask_invalid_samples(self, summary_stats):
            mask = ~torch.isnan(summary_stats).any(dim=1) & ~torch.isinf(summary_stats).any(dim=1)
            
            return mask 
    
    def parameter_ranges(self, param_ranges):
        "These are the parameter ranges for the HF model. The function is used for plotting."
        return {f'range_theta{i+1}': list(param_ranges[key]) for i, key in enumerate(param_ranges)}