from pyro.distributions import ConditionalDistribution
from rbi.utils.distributions import SIRDistribution, MCMCDistribution
from rbi.utils.mcmc_kernels import AdaptiveMultivariateGaussianKernel,LearnableIndependentKernel,KernelScheduler
from rbi.utils.mcmc import MCMC 


from torch import Tensor


class LotkaVolterraPosterior(ConditionalDistribution):
    def __init__(self, prior, potential_fn) -> None:
        super().__init__()
        self.prior = prior
        self.potential_fn = potential_fn

        k1 = AdaptiveMultivariateGaussianKernel()
        k2 = LearnableIndependentKernel()

        self.k = KernelScheduler([k1,k2, k1,k2], [0, 50, 100, 150])


    def condition(self, context:Tensor):
        proposal = SIRDistribution(self.prior, self.potential_fn, context=context, K= 10)
        mcmc = MCMC(self.k , self.potential_fn, proposal, context=context ,thinning=10, warmup_steps=150, num_chains=500, device=context.device)

        return MCMCDistribution(mcmc)
    
