from typing import Optional
import numpy as np
import torch 
from mf_npe.simulator.Prior import Prior
from torch.distributions.normal import Normal
from sbi.neural_nets import posterior_nn
# Plot the averaged image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sbi.utils import BoxUniform

from mf_npe.utils import utils

class GaussianBlobLowResolution(Prior):
    """ Simulator for the Gaussian blob (image) task."""
    
    def __init__(self, config_data):
        super().__init__()
        
        self.config_data = config_data
        self.x_dim_lf = config_data['x_dim_lf']
        self.x_dim_hf = config_data['x_dim_hf']
        self.prior_ranges = {k: config_data['all_prior_ranges'][k] for k in list(config_data['all_prior_ranges'].keys())[:config_data['theta_dim']]}
        self.option = config_data['decrease_resolution_option']
        
        self.lf_img_width = int(np.sqrt(self.x_dim_lf))
        self.hf_img_width = int(np.sqrt(self.x_dim_hf))

    def printName(self):
        return "Gaussian Blob Low resolution"
    
    def prior(self):
        lf_prior, _ = super().get_prior(self.prior_ranges, self.config_data)
        
        # size one axis
        
        # truncate prior to be within the image boundaries
        lf_prior = BoxUniform(low=torch.tensor([0, 0, 0.5]), high=torch.tensor([self.lf_img_width - 2, self.lf_img_width - 2, 5.0]))


        return lf_prior

    def parameter_ranges(self, theta_dim):
        param_ranges = super().parameter_ranges(self.prior_ranges)
        return [param_ranges[f'range_theta{i+1}'] for i in range(theta_dim)]

    def simulator(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        SBI-compatible simulator: maps parameters theta = [xoff, yoff, gamma]
        to a flattened grayscale image.
        """
        images = []
        for t in thetas.numpy():
            xoff, yoff, gamma = t
            img = _simulate_blob(xoff, yoff, gamma,
                                image_size=self.lf_img_width, sigma=2.0, # if 16*16, then image_size=16
                                trials=20, random_state=None)
            # Normalize to [0, 1] and convert to torch
            im = torch.tensor(img.astype(np.float32) / 255.0).flatten()
            images.append(im)

        # Stack images into a tensor
        images = torch.stack(images)
        
        # Upscale simulations to the size of high-fidelity (e.g. 64x64 or 256x256)
        images = F.interpolate(images.view(-1, 1, self.lf_img_width, self.lf_img_width), size=(self.hf_img_width, self.hf_img_width), mode='bilinear', align_corners=False)

        # Add noise to images, so that std is similar to high-fidelity
        noise = torch.randn_like(images) * 0.001
        images = images + noise
        images = torch.clamp(images, 0.0, 1.0)

        return images
    
    
    def summary_statistics(self, n_simulations, prior):
        """ Summary statistics are just the samples themselves in the dimensionality of x."""
        thetas = prior.sample((n_simulations,))
        
        # Use theta's up to image size (so 32)
        low_res_simulations = self.simulator(thetas)
        
        
        #low_res_simulations = F.interpolate(simulations.view(-1, 1, self.lf_img_width, self.lf_img_width), size=(self.hf_img_width, self.hf_img_width), mode='bilinear', align_corners=False)

        # Plot the simulations
        self.plot_simulations(thetas, low_res_simulations, './mf_npe/simulator/task4/images')

        print("low_res_simulations shape", low_res_simulations.shape)  # e.g. torch.Size([100, 4096])

        return low_res_simulations, thetas, {}
    
    def plot_simulations(self, thetas: torch.Tensor, simulations: torch.Tensor, path):
        # number of sims to show (max 5)
        n_to_plot = min(5, simulations.shape[0])
        fig, axs = plt.subplots(1, n_to_plot, figsize=(3 * n_to_plot, 3))

        # ensure axs is iterable even if n_to_plot == 1
        axs = np.atleast_1d(axs)

        for i, ax in enumerate(axs):
            high_res = simulations[i].reshape(
                int(np.sqrt(self.x_dim_hf)), int(np.sqrt(self.x_dim_hf))
            )
            ax.imshow(high_res, cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.show()


        # save plot
        plt.savefig(f"{path}/low_res_images.png", dpi=300)
        plt.savefig(f"{path}/low_res_images.pdf")
        plt.close()
        
            

# Underlying simulator for blob task
def _simulate_blob(xoff, yoff, gamma,
                image_size=32, sigma=2.0,
                trials=255, random_state=None):
    ## NOTE: increasing sigma and decreasing trials makes the task harder    
    # Set up the random number generator
    rng = np.random.default_rng(random_state)
    coords = np.arange(image_size)
    xx, yy = np.meshgrid(coords, coords, indexing='xy')

    # perturb the center of the blob (xoff, yoff) so that the posterior does not converge (by adding noise)
    xoff  = xoff + rng.normal(0, 0.5)
    yoff  = yoff + rng.normal(0, 0.5)

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

