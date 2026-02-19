from mf_npe.utils.mf_abc import MFABC, make_mfabc_cloud
import torch
from sbi.analysis import pairplot

def run_mf_abc(self, lf_data, true_thetas):
    posterior_samples_all_batches = []
    posterior_weights_all_batches = []
    num_hifi_total = []
    
    # loop over n_lf simulations, hf simulations are actively sampled
    for k, n_simulations in enumerate(self.batch_lf_sims):  
        curr_dataset = lf_data[k]
        
        # Make sure to put a large enough HF dataset
        x_t, theta_t = curr_dataset['x'], curr_dataset['theta']
        
        # Z-score stats for the data: Make a data-dependent approximation (like in SBI)
        # To then dynamically transform the generated given this mean/std
        mean_x, std_x = get_stats(self, x_t)
        #mean_theta, std_theta = self.get_stats(theta_t) # transform true thetas with the same mean and std
        # x_z, theta_z = z_score(self, x_t, mean_x, std_x), self.z_score(theta_t, mean_theta, std_theta)
        
        posterior_samples_k = []
        weights_k = []
        
        for j, xo in enumerate(self.obs):     
            xo_z = z_score(self, xo, mean_x, std_x)

            mfabc = MFABC(
                parameter_sampler=theta_t, #self.parameter_sampler, # plug in our data
                lofi=lambda theta: lofi(self, theta, xo_z, mean_x, std_x), # pass in the observed data
                hifi=lambda theta, pass_lo: hifi(self, theta, pass_lo, xo_z, mean_x, std_x) 
            )
                            
            # 30% HF
            # eps 1: looser, eps 2: accept simulations that are within ~1 SD
            epsilons = (1.0, 1.0) # (eps_lofi, eps_hifi)

            print(f"[batch {k}, obs {j}] Estimated epsilons: {epsilons}")
            etas = (0.9, 0.3)
            cloud = make_mfabc_cloud(mfabc, theta_t, epsilons, etas, N=n_simulations)
            
            for i, particle in enumerate(cloud):
                print(f"Particle {i}: eta={particle.eta:.2f}, weight={particle.w:.2f}, distances={particle.p.dist}, cost={particle.p.cost}")

            # Print how many samples have weight > 0
            print(f"Number of particles with weight > 0: {sum(1 for p in cloud if p.w > 0)}")

            params = torch.stack([p.p.theta for p in cloud])  # (N, D)
            weights = torch.tensor([p.w for p in cloud], dtype=torch.float32)
            weights = torch.clamp(weights, min=0.0)
            weights = weights / torch.sum(weights)
            
            print("weights before importance sampling", weights.shape)
            
            # importance resampling 
            # use only a subset of reweighted samples (= how much we want to generate)
            # Try multinomial sampling, with fallback in case of error
            try:
                idxs_post = torch.multinomial(weights, num_samples=self.n_samples_to_generate, replacement=True)
            except RuntimeError as e:
                print(f"Multinomial sampling failed: {e}. Trying again.")
                weights = torch.ones(len(cloud), dtype=torch.float32)
                weights = weights / torch.sum(weights)
                idxs_post = torch.multinomial(weights, num_samples=self.n_samples_to_generate, replacement=True)
            
                            
            post_samples = params[idxs_post]
            posterior_samples_k.append(post_samples)

            # Optional plot for first few obs in first batch
            if getattr(self.hf_simulator, "prior_ranges", None) is not None:
                limits = self.hf_simulator.parameter_ranges(self.theta_dim)
            else:
                limits = None

            # Optional plot for first few obs in first batch
            if k == 0 and j < 3:
                _ = pairplot(
                    post_samples.numpy(),
                    **({"limits": limits} if limits is not None else {}),
                    figsize=(6, 6),
                    labels=[rf"$\theta_{i+1}$" for i in range(post_samples.shape[1])],
                    bins=30,
                    points=true_thetas[j].numpy(),
                    title=f"method: MF-abc (n_sims: {n_simulations})",
                )
                
        # Stack obs-wise: (n_obs, n_samples, n_params)
        posterior_samples_k = torch.stack(posterior_samples_k)
        posterior_samples_all_batches.append(posterior_samples_k)
        
        # Count number of high-fidelity simulations used
        num_hifi = sum(1 for p in cloud if len(p.p.dist) == 2)
        print(f"Number of hf simulations: {num_hifi}")
        
        num_hifi_total.append(num_hifi)
        
    # print the size/shape of posterior_samples_all_batches
    print(f"Size of posterior_samples_all_batches: {posterior_samples_all_batches[0].shape}")
    
    return posterior_samples_all_batches, None, num_hifi_total


def get_stats(self, x):
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    return mean, std

def z_score(self, x, mean, std):
    z =(x - mean) / std #(abs(x)-abs(mean))/abs(std)
    
    return z

def z_inv(self, z, mean, std):
    x = z * std + mean
    return x


# Distance to observed data
def distance(self, sim, obs):
    # Root mean squared error
    rmse = torch.sqrt(torch.mean((sim - obs) ** 2))
    return rmse


def lofi(self, theta, xo, mean, std):
    # put theta in right shape
    theta = theta.unsqueeze(0)
    if self.task == 'task1' or self.task == 'task4' or self.task == 'task5':
        # Does not work currently if lf_simulator is a dict
        x_lf = self.lf_simulator.simulator(theta) 
    elif self.task == 'task7':
        x_lf = self.lf_simulator.simulator_wrapper(theta)
    else:
        ValueError("Task not implemented yet for mf-abc")
    # z-score x_lf     
    x_lf = z_score(self, x_lf, mean, std)

    return distance(self, x_lf, xo), x_lf


def hifi(self, theta, pass_lo, xo, mean, std): 
    theta = theta.unsqueeze(0)   
    if self.task == 'task1'or self.task == 'task4' or self.task == 'task5' or self.task == 'task6':
        x_hf = self.hf_simulator.simulator(theta) 
    elif self.task == 'task7':
        x_hf = self.hf_simulator.simulator_wrapper(theta)
    else:
        ValueError("Task not implemented yet for mf-abc")
    # z-score x_hf    
    x_hf = z_score(self, x_hf, mean, std)
    
    return distance(self, x_hf, xo) 

def parameter_sampler(self):
    dist = self.hf_prior

    return dist.sample()