import numpy as np
import torch 
from mf_npe.simulator.Prior import Prior


class GaussianBlobTshift(Prior):
    """ Simulator for the Gaussian blob (image) task."""
    
    def __init__(self, config_data, theta_dim, x_shift, y_shift, gamma_shift):
        super().__init__()
        self.x_dim_lf = config_data['x_dim_lf']
        self.theta_dim = theta_dim
        self.prior_ranges = {k: config_data['all_prior_ranges'][k] for k in list(config_data['all_prior_ranges'].keys())[:config_data['theta_dim']]}

    def printName(self):
        return "Gaussian Blob (shifted)"
    
    def prior(self):
        lf_prior, _ = super().get_prior(self.prior_ranges, self.config_data)
        return lf_prior

    def parameter_ranges(self, theta_dim):
        param_ranges = super().parameter_ranges(self.prior_ranges)
        return [param_ranges[f'range_theta{i+1}'] for i in range(theta_dim)]

    def simulator(self, thetas: torch.Tensor, shiftx=1, shifty=1) -> torch.Tensor:
        """
        SBI-compatible simulator: maps parameters theta = [xoff, yoff, gamma]
        to a flattened grayscale image.
        """
        images = []
        for t in thetas.numpy():
            xoff, yoff, gamma = t
            img = _simulate_blob(xoff + shiftx, yoff + shifty, 
                                gamma, image_size=np.sqrt(self.x_dim_lf), sigma=2.0,
                                trials=20, random_state=None)
            
            # Normalize to [0, 1] and convert to torch
            im = torch.tensor(img.astype(np.float32) / 255.0).flatten()
            images.append(im)

        # Stack images into a tensor
        images = torch.stack(images)
        return images
    
    def summary_statistics(self, n_simulations, prior):
        """ Summary statistics are just the samples themselves in the dimensionality of x."""
        thetas = prior.sample((n_simulations,))
        simulations = self.simulator(thetas)
        return simulations, thetas, {}

# Underlying simulator for blob task
def _simulate_blob(xoff, yoff, gamma,
                image_size=32, sigma=2.0,
                trials=255, random_state=None):
    ## NOTE: increasing sigma and decreasing trials makes the task harder    
    # Set up the random number generator
    rng = np.random.default_rng(random_state)
    coords = np.arange(image_size)
    xx, yy = np.meshgrid(coords, coords, indexing='xy')

    # compute the distance from the center of blob
    r2 = (xx - xoff)**2 + (yy - yoff)**2

    # probability prob with exponential decay from center of blob
    p = 0.9 - 0.8 * (np.exp(-0.5 * (r2 / (sigma**2))**gamma))

    # optionally, we remove center of with a differnce of exponentials
    #p = 0.9 - 0.8 * (np.exp(-0.5 * (r2 / (sigma**2))**gamma) - np.exp(-0.5 * (r2 / (sigma *.95**2))**gamma))
    #p = np.clip(p, 0.0, 1.0)

    # sample pixle values from binomial distribution
    image = rng.binomial(n=trials, p=p)
    return image.astype(np.uint8)

