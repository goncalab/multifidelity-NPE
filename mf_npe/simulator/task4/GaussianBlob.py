import numpy as np
import torch 
from mf_npe.simulator.Prior import Prior
from torch.distributions import Binomial
import matplotlib.pyplot as plt
class GaussianBlob(Prior):
    """ Simulator for the Gaussian blob (image) task."""
    
    def __init__(self, config_data):
        super().__init__()
        
        self.config_data = config_data
        self.x_dim_hf = config_data['x_dim_hf']
        self.x_dim_out = config_data['x_dim_out']
        self.prior_ranges = {k: config_data['all_prior_ranges'][k] for k in list(config_data['all_prior_ranges'].keys())[:config_data['theta_dim']]}
        self.gamma_jitter_std = 0.1
        self.gamma_jitter_mode="logmul"  # "add" or "logmul"

    def printName(self):
        return "Gaussian Blob"
    
    def prior(self):
        lf_prior, _ = super().get_prior(self.prior_ranges, self.config_data)
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
                                image_size=np.sqrt(self.x_dim_hf), sigma=12.0,
                                trials=255, random_state=None)
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
        
        self.plot_simulations(thetas, simulations, './mf_npe/simulator/task4/images')
        
        return simulations, thetas, {}
    
    
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
        plt.savefig(f"{path}/high_res_images.png", dpi=300)
        plt.savefig(f"{path}/high_res_images.pdf")
        plt.close()


    def true_log_likelihood(self, thetas, image,
                            sigma=12.0, trials=255, mc_samples=100, device="cpu"):
        """
        Compute log p(image | xoff, yoff, gamma) under the Binomial blob model.

        image      : torch.Tensor (H, W) or flattened (H*W,), observed counts (int) or normalized [0,1].
        xoff, yoff : floats or 0-dim tensors
        gamma      : float or 0-dim tensor
        sigma      : blob spread
        trials     : binomial total_count, i.e., max pixel value
        jit_std    : std of Gaussian jitter on blob center
        mc_samples : number of Monte Carlo samples for marginalizing jitter
        """
        
        xoff = thetas[:, 0]
        yoff = thetas[:, 1]
        gamma = thetas[:, 2]
        
        # print("thetas shape", thetas.shape) # thetas shape torch.Size([100, 3])
        # print("image shape", image.shape) # image shape torch.Size([4096])
        
        # ensure torch scalars
        xoff = torch.as_tensor(xoff, device=device, dtype=torch.float32)
        yoff = torch.as_tensor(yoff, device=device, dtype=torch.float32)
        gamma = torch.as_tensor(gamma, device=device, dtype=torch.float32)

        # Image size
        side = int(np.sqrt(self.x_dim_hf))
        
        coords = torch.arange(side, dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid(coords, coords, indexing="xy")
                
        ll_list = []
        for _ in range(mc_samples):
            rng = np.random.default_rng(None)
            xoff  = xoff + rng.normal(0, 0.5)
            yoff  = yoff + rng.normal(0, 0.5)
            r2 = (xx - xoff.view(-1,1,1))**2 + (yy - yoff.view(-1,1,1))**2
            
            p = 0.9 - 0.8 * (torch.exp(-0.5 * (r2 / (sigma**2))**gamma.view(-1,1,1)))
            p = p.clamp(1e-6, 1 - 1e-6)

            # Flatten the probabilities to match the image shape for the binomial distribution
            p = p.view(-1, side*side)  

            ll = Binomial(total_count=trials, probs=p).log_prob(image).sum()
            ll_list.append(ll)

        if mc_samples == 1:
            return ll_list[0]
        else:
            ll_stack = torch.stack(ll_list)  # (mc_samples,)
            m = ll_stack.max()
            # Log likelihood with log-sum-exp trick
            log_likelihood = m + torch.log(torch.mean(torch.exp(ll_stack - m)))

            return log_likelihood

            
    # def true_log_likelihood(self, thetas, image,
    #                         sigma=2.0, trials=255,
    #                         Kxy=10, Kg=3,
    #                         xy_jitter_std=0.5,
    #                         gamma_jitter_std=0.10,
    #                         gamma_jitter_mode="logmul",   # "add" or "logmul"
    #                         device="cpu", crn_seed=123,
    #                         image_is_normalized=True):
    #     """
    #     Returns per-theta log p(image | theta) marginalizing over:
    #     - x,y jitter ~ N(0, xy_jitter_std^2) with Kxy samples
    #     - gamma jitter: additive N(0, gamma_jitter_std^2) or log-multiplicative
    #     Deterministic via CRN (fixed seed).
    #     Vectorized across Kg and Kxy; uses closed-form binomial log pmf.
    #     """

    #     # ----- inputs -----
    #     thetas = torch.as_tensor(thetas, device=device, dtype=torch.float32)
    #     B = thetas.shape[0]
    #     x0, y0, gamma0 = thetas[:, 0], thetas[:, 1], thetas[:, 2]

    #     side = int(np.sqrt(self.x_dim_hf))
    #     P = side * side

    #     image = torch.as_tensor(image, device=device, dtype=torch.float32)
    #     if image_is_normalized:
    #         image_counts = (image.clamp(0, 1) * trials).round().clamp_(0, trials)
    #     else:
    #         image_counts = image
    #     # [P]
    #     image_counts = image_counts.view(P)

    #     # ----- grid (cache these in __init__ in real code) -----
    #     coords = torch.arange(side, device=device, dtype=torch.float32)
    #     xx, yy = torch.meshgrid(coords, coords, indexing="xy")  # [S,S]
    #     # broadcast shapes will add [Kg,Kxy,B] leading dims below
    #     xx = xx.view(1, 1, 1, side, side)
    #     yy = yy.view(1, 1, 1, side, side)

    #     # ----- CRN: jitter samples (one shot) -----
    #     g_xy = torch.Generator(device=device).manual_seed(crn_seed)
    #     # [Kg,Kxy,B]
    #     eps_x = xy_jitter_std * torch.randn(Kg, Kxy, B, device=device, generator=g_xy)
    #     eps_y = xy_jitter_std * torch.randn(Kg, Kxy, B, device=device, generator=g_xy)

    #     if gamma_jitter_std > 0:
    #         g_g  = torch.Generator(device=device).manual_seed(crn_seed + 999)
    #         # [Kg]
    #         eps_g = gamma_jitter_std * torch.randn(Kg, device=device, generator=g_g)
    #         if gamma_jitter_mode == "add":
    #             gamma = (gamma0.view(1, 1, B) + eps_g.view(Kg, 1, 1)).clamp(min=1e-3)  # [Kg,1,B]
    #         else:  # "logmul"
    #             gamma = (gamma0.view(1, 1, B) * torch.exp(eps_g).view(Kg, 1, 1)).clamp(min=1e-3)
    #     else:
    #         Kg = 1
    #         gamma = gamma0.view(1, 1, B)  # [1,1,B]

    #     # ----- centers with xy jitter -----
    #     # [Kg,Kxy,B,1,1]
    #     cx = x0.view(1, 1, B, 1, 1) + eps_x.view(Kg, Kxy, B, 1, 1)
    #     cy = y0.view(1, 1, B, 1, 1) + eps_y.view(Kg, Kxy, B, 1, 1)

    #     # ----- squared distances for all jitters -----
    #     # [Kg,Kxy,B,S,S]
    #     r2 = (xx - cx)**2 + (yy - cy)**2

    #     # ----- compute probabilities for all jitters -----
    #     # gamma: [Kg,1,B] -> [Kg,Kxy,B,1,1]
    #     gamma_b = gamma.view(Kg, 1, B, 1, 1)
    #     # p: [Kg,Kxy,B,S,S] -> [Kg,Kxy,B,P]
    #     p = 0.9 - 0.8 * torch.exp(-0.5 * (r2 / (sigma**2))**gamma_b)
    #     p = p.clamp_(1e-6, 1 - 1e-6).view(Kg, Kxy, B, P)

    #     # ----- closed-form binomial log pmf -----
    #     # k: [P] -> [1,1,1,P] for broadcast
    #     k = image_counts.view(1, 1, 1, P)
    #     n = float(trials)
    #     # const per-pixel (depends only on k)
    #     # [P] -> [1,1,1,P]
    #     const = (torch.lgamma(torch.tensor(n + 1.0, device=device)) 
    #             - torch.lgamma(k + 1.0)
    #             - torch.lgamma(torch.tensor(n, device=device) - k + 1.0))
    #     # lp: [Kg,Kxy,B,P]
    #     lp = const + k * torch.log(p) + (n - k) * torch.log1p(-p)
    #     # sum over pixels -> [Kg,Kxy,B]
    #     ll = lp.sum(dim=3)

    #     # ----- log-mean-exp over Kg and Kxy -----
    #     # logmeanexp over dims (0,1)
    #     m = ll.amax(dim=(0,1), keepdim=True)                       # [1,1,B]
    #     lse = m + torch.log(torch.mean(torch.exp(ll - m), dim=(0,1), keepdim=True))  # [1,1,B]
    #     ll_final = lse.view(B)  # [B]

    #     return ll_final

    # def true_log_likelihood(self, thetas, image,
    #                         sigma=2.0, trials=255, mc_samples=128, device="cpu"): # 128
    #     """
    #         Compute log p(image | xoff, yoff, gamma) under the Binomial blob model.

    #         image      : torch.Tensor (H, W) or flattened (H*W,), observed counts (int) or normalized [0,1].
    #         thetas     : torch.Tensor (B, 3) where each row is [xoff, yoff, gamma]
    #         trials     : binomial total_count, i.e., max pixel value
    #         mc_samples : number of Monte Carlo samples for marginalizing jitter
    #     """
        
    #     # thetas: [B, 3], image: [P] normalized in [0,1]
    #     image = torch.as_tensor(image, device=device, dtype=torch.float32)
    #     image_counts = (image * trials).round().clamp(0, trials)  # back to counts

    #     # reshape to [1, P] for broadcasting
    #     side = int(np.sqrt(self.x_dim_hf))
    #     image_counts = image_counts.view(1, side*side)

    #     xoff0, yoff0, gamma = [thetas[:, i].to(device, torch.float32) for i in range(3)]

    #     coords = torch.arange(side, device=device, dtype=torch.float32)
    #     xx, yy = torch.meshgrid(coords, coords, indexing="xy")

    #     xx = xx.unsqueeze(0)  # [1,S,S]
    #     yy = yy.unsqueeze(0)
        
    #     crn_seed=123 
        
    #     g_xy = torch.Generator(device=device).manual_seed(crn_seed)
    #     # jitters for each MC sample and each θ (vectorized):
    #     # eps_x, eps_y: [mc_samples, B]
    #     eps_x = 0.5 * torch.randn(mc_samples, xoff0.shape[0], device=device, generator=g_xy)
    #     eps_y = 0.5 * torch.randn(mc_samples, yoff0.shape[0], device=device, generator=g_xy)

    #     gamma = gamma.view(-1, 1, 1)
        
        
    #     g_g  = torch.Generator(device=device).manual_seed(crn_seed + 999)
    #     if self.gamma_jitter_std > 0:
    #         eps_g = self.gamma_jitter_std * torch.randn(Kg, device=device, generator=g_g)  # [Kg]
    #             # gamma_tilde[k,b] = gamma0[b] * exp(eps_g[k])
    #     else:
    #         Kg = 1
    #         eps_g = torch.zeros(1, device=device)
        
    #     ll_list = []
    #     for k in range(mc_samples):
    #         # jitter fresh each time
    #         xoff = xoff0 + eps_x[k]
    #         yoff = yoff0 + eps_y[k]

    #         r2 = (xx - xoff.view(-1,1,1))**2 + (yy - yoff.view(-1,1,1))**2
    #         p = 0.9 - 0.8 * torch.exp(-0.5 * (r2 / (sigma**2))**gamma) # .view(-1,1,1)
    #         p = p.clamp(1e-6, 1-1e-6).view(p.shape[0], -1)  # [B,P]

    #         dist = torch.distributions.Binomial(total_count=trials, probs=p)
    #         lp = dist.log_prob(image_counts)   # [B,P]
    #         ll = lp.sum(dim=1)                 # [B]
    #         ll_list.append(ll)

    #     ll_stack = torch.stack(ll_list)  # [M,B]    
    #     m = ll_stack.max(dim=0).values
    #     return m + torch.log(torch.mean(torch.exp(ll_stack - m), dim=0))  # [B]




# Underlying simulator for blob task
def _simulate_blob(xoff, yoff, gamma,
                image_size=32, sigma=8.0, # was 2 when 32x32
                trials=255, random_state=None):
    ## NOTE: increasing sigma and decreasing trials makes the task harder    
    # Set up the random number generator
    rng = np.random.default_rng(random_state)
    coords = np.arange(image_size)
    xx, yy = np.meshgrid(coords, coords, indexing='xy')

    # perturb the center of the blob (xoff, yoff) so that the posterior does not converge (by adding noise)
    xoff  = xoff + rng.normal(0, 0.5)
    yoff  = yoff + rng.normal(0, 0.5)
    
    gamma_jitter_std = 0.0

    
    if gamma_jitter_std > 0:
        # log-multiplicative jitter that makes sure it stays positive
        gamma_tilde = float(gamma * np.exp(gamma_jitter_std * rng.normal()))
    else:
        gamma_tilde = float(gamma)

    # compute the distance from the center of blob
    r2 = (xx - xoff)**2 + (yy - yoff)**2

    # probability prob with exponential decay from center of blob
    p = 0.9 - 0.8 * (np.exp(-0.5 * (r2 / (sigma**2))**gamma_tilde))
    p  = np.clip(p, 1e-6, 1-1e-6)

    # optionally, we remove center of with a differnce of exponentials
    #p = 0.9 - 0.8 * (np.exp(-0.5 * (r2 / (sigma**2))**gamma) - np.exp(-0.5 * (r2 / (sigma *.95**2))**gamma))
    #p = np.clip(p, 0.0, 1.0)

    # sample pixle values from binomial distribution
    image = rng.binomial(n=trials, p=p)
    return image.astype(np.uint8)

