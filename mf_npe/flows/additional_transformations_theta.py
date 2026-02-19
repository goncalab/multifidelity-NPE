import torch
import torch.nn as nn
from torch import Tensor
import zuko 
from zuko.transforms import AffineTransform

from torch.distributions import (
    AffineTransform,
)
from zuko.flows import (
     UnconditionalDistribution,
 )
from zuko.distributions import DiagNormal


def z_standardization_parameters(
    batch_t: Tensor,
    structured_dims: bool = False,
    min_std: float = 1e-14,
) -> tuple[Tensor, Tensor]:
    """Computes mean and standard deviation for z-scoring

    Args:
        batch_t: Batched tensor from which mean and std deviation (across
            first dimension) are computed.
        structured_dim: Whether data dimensions are structured (e.g., time-series,
            images), which requires computing mean and std per sample first before
            aggregating over samples for a single standardization mean and std for the
            batch, or independent (default), which z-scores dimensions independently.
        min_std:  Minimum value of the standard deviation to use when z-scoring to
            avoid division by zero.
        backend: Whether to use nflows or zuko backend

    Returns:
        Mean and standard deviation for z-scoring
    """

    if structured_dims:
        # Structured data so compute a single mean over all dimensions
        # equivalent to taking mean over per-sample mean, i.e.,
        # `torch.mean(torch.mean(.., dim=1))`.
        t_mean = torch.mean(batch_t)
        # Compute std per-sample first.
        sample_std = torch.std(batch_t, dim=1)
        sample_std[sample_std < min_std] = min_std
        # Average over all samples for batch std.
        t_std = torch.mean(sample_std)
    else:
        t_mean = torch.mean(batch_t, dim=0)
        t_std = torch.std(batch_t, dim=0)
        t_std[t_std < min_std] = min_std

    # Return mean and std for z-scoring.
    return t_mean, t_std



def standardizing_transform_theta(
    batch_t: Tensor,
    structured_dims: bool = False,
    min_std: float = 1e-14,
) -> zuko.flows.UnconditionalTransform:
    """Builds standardizing transform for Zuko flows

    Args:
        batch_t: Batched tensor from which mean and std deviation (across
            first dimension) are computed.
        structured_dim: Whether data dimensions are structured (e.g., time-series,
            images), which requires computing mean and std per sample first before
            aggregating over samples for a single standardization mean and std for the
            batch, or independent (default), which z-scores dimensions independently.
        min_std:  Minimum value of the standard deviation to use when z-scoring to
            avoid division by zero.

    Returns:
        Affine transform for z-scoring
    """
    t_mean, t_std = z_standardization_parameters(batch_t, structured_dims, min_std)
    return zuko.flows.UnconditionalTransform(
        AffineTransform,
        loc=-t_mean / t_std,
        scale=1 / t_std,
        buffer=True,
    )


def create_base(features):
     return UnconditionalDistribution(
         DiagNormal,
         torch.full([features], -3.0),
         torch.full([features], +3.0),
         buffer=True,
     )

class CallableTransform:
    """Wraps a PyTorch Transform to be used in Zuko UnconditionalTransform."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self):
        return self.transform
    

def biject_transform_theta(
    transform,
    eps: float = 1e-6
) -> zuko.flows.UnconditionalTransform:
    """
    Builds logit-transforming transform for Zuko flows on a bounded interval.

    Args:
        prior: A PyTorch distribution with a defined support.
        eps: Small constant to avoid numerical issues at 0 and 1.

    Returns:
        Logit transformation for the given range.
    """
    
    # Get the bijective transform for the given distribution support
    return zuko.flows.UnconditionalTransform(
        CallableTransform(transform),
        buffer=True,
    )