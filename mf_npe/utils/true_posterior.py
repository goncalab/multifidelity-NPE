# import torch 
# from torch import Tensor

# from rbi.utils.mcmc_kernels import AdaptiveMultivariateGaussianKernel,LearnableIndependentKernel,KernelScheduler

# from pyro.distributions import ConditionalDistribution
# from rbi.utils.mcmc import MCMC 
# from rbi.utils.distributions import SIRDistribution, MCMCDistribution



# class LotkaVolterraPosterior(ConditionalDistribution):
#     def __init__(self, prior, potential_fn) -> None:
#         super().__init__()
#         self.prior = prior
#         self.potential_fn = potential_fn

#         k1 = AdaptiveMultivariateGaussianKernel()
#         k2 = LearnableIndependentKernel()

#         self.k = KernelScheduler([k1,k2, k1,k2], [0, 50, 100, 150])
        
        
# class LotkaVolterraPosterior(ConditionalDistribution):
#     def __init__(self, prior, potential_fn) -> None:
#         super().__init__()
#         self.prior = prior
#         self.potential_fn = potential_fn

#         k1 = AdaptiveMultivariateGaussianKernel()
#         k2 = LearnableIndependentKernel()

#         self.k = KernelScheduler([k1,k2, k1,k2], [0, 50, 100, 150])


#     def condition(self, context:Tensor):

#         proposal = SIRDistribution(self.prior, self.potential_fn, context=context, K= 10)
#         mcmc = MCMC(self.k , self.potential_fn, proposal, context=context ,thinning=10, warmup_steps=150, num_chains=500, device=context.device)

#         return MCMCDistribution(mcmc)
    
    


# def get_potential_fn(self, device: str = "cpu") -> Callable:
#         """Return a potential function i.e. the unormalized posterior distirbution.

#         Args:
#             device (str, optional): Device. Defaults to "cpu".

#         Returns:
#             Callable: Function that gets parameter and data and computes the log posterior potential.
#         """
#         likelihood = self.get_loglikelihood_fn(device=device)
#         prior = self.get_prior(device=device)

#         def potential_fn(x, theta):
#             x = x.to(device)
#             theta = theta.to(device)
#             likelihood_fn = likelihood(theta)
#             l = likelihood_fn.log_prob(x) + prior.log_prob(theta)

#             return l.squeeze()

#         return potential_fn


# LotkaVolterraPosterior(self.get_prior(device), self.get_potential_fn(device))

# # def run_true_posterior_samples(task: InferenceTask, x: Tensor, num_samples_top:int = 10000, num_samples_rand:int = 1000, top_xs=10, rand_xs:int=100, device:str= "cpu"):

# #     try:
# #         posterior = task.get_true_posterior(device)
# #     except:
# #         return {}, {}
    
# #     x_selected = x[:top_xs]

# #     p_x = posterior.condition(x_selected.to(device))

# #     print("Sampling")
# #     samples  =p_x.sample((num_samples_top,)).cpu()
# #     del p_x 

# #     samples_dict_top = {}
# #     for i in range(top_xs):
# #         samples_dict_top[i] = samples[:, i, :]

# #     index = torch.randperm(x.shape[0])[:rand_xs]
# #     x_rand = x[index]

# #     print("Sampling rand")
# #     iters = rand_xs // 50 + 1
# #     samples_r = []
# #     for i in range(iters):
# #         if i*50 < rand_xs:
# #             p_x_rand = posterior.condition(x_rand[i * 50: (i+1)*50].to(device))

# #             samples_rand  =p_x_rand.sample((num_samples_rand,)).cpu()
# #             samples_r.append(samples_rand)

# #             del p_x_rand 

# #     samples_r = torch.concat(samples_r, 1)

# #     samples_dict_rand = {}
# #     for i in range(rand_xs):
# #         samples_dict_rand[int(index[i])] = samples_r[:, i, :]

# #     return samples_dict_top, samples_dict_rand