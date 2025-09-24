# Multifidelity ABC with PyTorch
# Based on: Multifidelity Approximate Bayesian Computation with Sequential Monte Carlo Parameter Sampling
# Thomas P. Prescott, Ruth E. Baker (https://arxiv.org/abs/2001.06256)

import torch
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union
import random

# -------------------------------
# Data Structures
# -------------------------------

# @dataclass
# class Parameters:
#     values: torch.Tensor
    

@dataclass
class Particle:
    theta: torch.Tensor  # parameters
    dist: Tuple[float, ...]  # distances (lofi, hifi)
    cost: Tuple[float, ...]  # costs (lofi, hifi)

@dataclass
class MFABCParticle:
    p: Particle
    eta: float
    w: float

Cloud = Union[List[Particle], List[MFABCParticle]]

# -------------------------------
# Utility Functions
# -------------------------------

def cost(p, i=None):
    if isinstance(p, MFABCParticle):
        p = p.p
    if i is None:
        return sum(p.cost)
    return p.cost[i] if i < len(p.cost) else 0.0

def accept(p: Particle, epsilon: float) -> bool:
    return p.dist[1] <= epsilon

# -------------------------------
# MFABC Core Logic
# -------------------------------

class MFABC:
    def __init__(self, parameter_sampler: Callable, lofi: Callable, hifi: Callable):
        self.parameter_sampler = parameter_sampler
        self.lofi = lofi
        self.hifi = hifi

def simulate_mfabc_particle(mfabc: MFABC, theta_t: torch.Tensor, epsilons: Tuple[float, float], etas: Tuple[float, float]) -> Tuple[MFABCParticle, float]:
    theta = theta_t
    d_lo, pass_lo = mfabc.lofi(theta)
    c_lo = 1.0  # Placeholder cost
    
    if d_lo < epsilons[0]:
        eta, w = etas[0], 1.0
    else:
        eta, w = etas[1], 0.0

    if random.random() < eta:
        d_hi = mfabc.hifi(theta, pass_lo)
        c_hi = 1.0  # Placeholder cost
        close = (d_lo < epsilons[0], d_hi < epsilons[1])
        if close[0] != close[1]:
            #print("Original W:", w)
            #print("Close:", close)
            w += (float(close[1]) - float(close[0])) / eta
            #w = float(close[0]) + (float(close[1]) - float(close[0])) / eta
            #print("New W:", w)
        p = Particle(theta, (d_lo, d_hi), (c_lo, c_hi))
    else:
        p = Particle(theta, (d_lo,), (c_lo,))

    return MFABCParticle(p, eta, w), sum(p.cost)

# -------------------------------
# Cloud Generation
# -------------------------------

def make_mfabc_cloud(mfabc: MFABC, theta_t: torch.Tensor, epsilons: Tuple[float, float], etas: Tuple[float, float], N: int = None, budget: float = None) -> List[MFABCParticle]:
    cloud = []
    run_cost = 0.0

    if N is not None:
        for i in range(N):
            p, _ = simulate_mfabc_particle(mfabc, theta_t[i], epsilons, etas)
            cloud.append(p)
    # elif budget is not None:
    #     while run_cost < budget:
    #         p, c = simulate_mfabc_particle(mfabc, _, epsilons, etas)
    #         run_cost += c
    #         if run_cost < budget:
    #             cloud.append(p)
    else:
        raise ValueError("Either N or budget must be specified.")

    return cloud



# # Multifidelity ABC with PyTorch
# # Based on: Multifidelity Approximate Bayesian Computation with Sequential Monte Carlo Parameter Sampling
# # Thomas P. Prescott, Ruth E. Baker (https://arxiv.org/abs/2001.06256)

# import torch
# from dataclasses import dataclass
# from typing import Callable, List, Tuple, Union
# import random

# # -------------------------------
# # Data Structures
# # -------------------------------

# # @dataclass
# # class Parameters:
# #     values: torch.Tensor
    

# @dataclass
# class Particle:
#     theta: torch.Tensor  # parameters
#     dist: Tuple[float, ...]  # distances (lofi, hifi)
#     cost: Tuple[float, ...]  # costs (lofi, hifi)

# @dataclass
# class MFABCParticle:
#     p: Particle
#     eta: float
#     w: float

# Cloud = Union[List[Particle], List[MFABCParticle]]

# # -------------------------------
# # Utility Functions
# # -------------------------------

# def cost(p, i=None):
#     if isinstance(p, MFABCParticle):
#         p = p.p
#     if i is None:
#         return sum(p.cost)
#     return p.cost[i] if i < len(p.cost) else 0.0

# def accept(p: Particle, epsilon: float) -> bool:
#     return p.dist[1] <= epsilon

# # -------------------------------
# # MFABC Core Logic
# # -------------------------------

# class MFABC:
#     def __init__(self, parameter_sampler: Callable, lofi: Callable, hifi: Callable):
#         self.parameter_sampler = parameter_sampler
#         self.lofi = lofi
#         self.hifi = hifi

# def simulate_mfabc_particle(mfabc: MFABC, theta_t: torch.Tensor, epsilons: Tuple[float, float], etas: Tuple[float, float]) -> Tuple[MFABCParticle, float]:
#     theta = theta_t
#     d_lo, pass_lo = mfabc.lofi(theta)
#     c_lo = 1.0  # Placeholder cost
    
#     if d_lo < epsilons[0]:
#         eta, w = etas[0], 1.0
#     else:
#         eta, w = etas[1], 0.0

#     if random.random() < eta:
#         d_hi = mfabc.hifi(theta, pass_lo)
#         c_hi = 1.0  # Placeholder cost
#         close = (d_lo < epsilons[0], d_hi < epsilons[1])
#         if close[0] != close[1]:
#             #print("Original W:", w)
#             #print("Close:", close)
#             w += (float(close[1]) - float(close[0])) / eta
#             #w = float(close[0]) + (float(close[1]) - float(close[0])) / eta
#             #print("New W:", w)
#         p = Particle(theta, (d_lo, d_hi), (c_lo, c_hi))
#     else:
#         p = Particle(theta, (d_lo,), (c_lo,))

#     return MFABCParticle(p, eta, w), sum(p.cost)

# # -------------------------------
# # Cloud Generation
# # -------------------------------

# def make_mfabc_cloud(mfabc: MFABC, theta_t: torch.Tensor, epsilons: Tuple[float, float], etas: Tuple[float, float], N: int = None, budget: float = None) -> List[MFABCParticle]:
#     cloud = []
#     run_cost = 0.0

#     if N is not None:
#         for i in range(N):
#             p, _ = simulate_mfabc_particle(mfabc, theta_t[i], epsilons, etas)
#             cloud.append(p)
#     # elif budget is not None:
#     #     while run_cost < budget:
#     #         p, c = simulate_mfabc_particle(mfabc, _, epsilons, etas)
#     #         run_cost += c
#     #         if run_cost < budget:
#     #             cloud.append(p)
#     else:
#         raise ValueError("Either N or budget must be specified.")

#     return cloud
