#%%
import jax
import torch
import torch.nn as nn
from torch import Tensor
import zuko 
import logging
from zuko.transforms import AffineTransform
from typing import Union
from zuko.flows import Flow

from sbi.neural_nets.estimators.zuko_flow import ZukoFlow
from zuko.lazy import (
    Flow,
)

from torch.distributions import (
    AffineTransform,
)

from sbi.utils.sbiutils import mcmc_transform

# https://github.com/sbi-dev/sbi/blob/main/sbi/utils/sbiutils
def standardizing_net(
    batch_t: Tensor,
    structured_dims: bool = False,
    min_std: float = 1e-7,
) -> nn.Module:
    """Builds standardizing network

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
        Neural network module for z-scoring
    """


    if structured_dims:
        # Structured data so compute a single mean over all dimensions
        # equivalent to taking mean over per-sample mean, i.e.,
        # `torch.mean(torch.mean(.., dim=1))`.
        t_mean = torch.mean(batch_t)
    else:
        # Compute per-dimension (independent) mean.
        t_mean = torch.mean(batch_t, dim=0)

    if len(batch_t > 1):
        if structured_dims:
            # Compute std per-sample first.
            sample_std = torch.std(batch_t, dim=1)
            sample_std[sample_std < min_std] = min_std
            # Average over all samples for batch std.
            t_std = torch.mean(sample_std)
        else:
            t_std = torch.std(batch_t, dim=0)
            t_std[t_std < min_std] = min_std
    else:
        t_std = torch.ones(1)
        logging.warning(
            """Using a one-dimensional batch will instantiate a Standardize transform
            with (mean, std) parameters which are not representative of the data. We
            allow this behavior because you might be loading a pre-trained net.
            If this is not the case, please be sure to use a larger batch."""
        )
    
    # print("batch_t shape:", batch_t)
    # print("t_mean:", t_mean)
    # print("t_std:", t_std)
    
    # If nans in batch_t, throw error
    nan_in_data = torch.isnan(batch_t).any()
    assert not nan_in_data, """Input data contains NaNs. In case you are encoding
                            missing trials with NaNs, consider setting
                            z_score_x='none' to disable z-scoring."""

    nan_in_stats = torch.logical_or(torch.isnan(t_mean).any(), torch.isnan(t_std).any())
    assert not nan_in_stats, """Training data mean or std for standardizing net must not
                            contain NaNs. In case you are encoding missing trials with
                            NaNs, consider setting z_score_x='none' to disable
                            z-scoring."""

    return Standardize(t_mean, t_std)


def standardizing_transform_zuko(
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
    t_mean, t_std = z_standardization(batch_t, structured_dims, min_std)
    return zuko.flows.UnconditionalTransform(
        AffineTransform,
        loc=-t_mean / t_std,
        scale=1 / t_std,
        buffer=True,
    )



class CallableTransform:
    """Wraps a PyTorch Transform to be used in Zuko UnconditionalTransform."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self):
        return self.transform
    

def biject_transform_zuko(
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


class Standardize(nn.Module):
    def __init__(self, mean: Union[Tensor, float], std: Union[Tensor, float]):
        super(Standardize, self).__init__()
        mean, std = map(torch.as_tensor, (mean, std))
        self.mean = mean
        self.std = std
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, tensor):
        return (tensor - self._mean) / self._std


def z_standardization(
    batch_t: Tensor,
    structured_dims: bool = False,
    min_std: float = 1e-14,
):
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


import torch
import zuko
from zuko.flows import (
     Flow,
     UnconditionalDistribution,
 )
from zuko.distributions import DiagNormal

def create_base(features):
     return UnconditionalDistribution(
         DiagNormal,
         torch.full([features], -3.0),
         torch.full([features], +3.0),
         buffer=True,
     )


def build_zuko_flow(
        batch_theta: Tensor,
        batch_x: Tensor,
        embedding_net: nn.Module,
        z_score_theta: bool,        
        z_score_x: bool,
        logit_transform_theta: bool, # transformation by adjusting the network weights
        hidden_features: int,
        num_transforms: int,
        num_bins:int,
        nf_type: str = "NSF",
        base_model = None,
        additional_hf_transform = None,
        device = "cpu",
        prior = None,
):
    theta_numel = batch_theta.shape[-1]
    # x_numel = batch_x.shape[-1]
    embedded_x_numel = embedding_net(batch_x).shape[-1]
    
    print("theta_numel:", theta_numel)
    print("x_numel:", batch_x.shape)
    

    
    print("embedded_x_numel:", embedded_x_numel)
    
    # # SMALL TEST 1 TRANFORM     
    if nf_type == "NSF":
        flow_built = zuko.flows.NSF(features=theta_numel, 
                                    context=embedded_x_numel, 
                                    bins=num_bins, 
                                    transforms=num_transforms, 
                                    hidden_features=[hidden_features]*num_transforms)
        
    elif nf_type == "NSF_PRETRAIN":        
        #num_transforms_lf = num_transforms #- 1 # We want to replace the last layer with the hf model
        
        # print("features: ", theta_numel)
        # print("context: ", embedded_x_numel)
        # print("num_bins: ", num_bins)
        # print("num_transforms: ", num_transforms)
        # print("hidden_features: ", hidden_features)

        # orders = [
        #     [0, 1, 2],
        #     [1, 0, 2],
        #     [0, 1, 2],
        #     [1, 0, 2],
        #     [0, 1, 2]
        # ]
        
        # flow_built = zuko.flows.NSF(features=theta_numel, 
        #                             context=embedded_x_numel, 
        #                             bins=num_bins, 
        #                             transforms=num_transforms, 
        #                             hidden_features=[hidden_features]*num_transforms,
        #                             order=orders)
        
        # print("lf flow", flow_built)
        

        # WORKING CODE
        flow_built = zuko.flows.NSF(features=theta_numel, 
                                    context=embedded_x_numel, 
                                    bins=num_bins, 
                                    transforms=num_transforms, 
                                    hidden_features=[hidden_features]*num_transforms)  #zuko.flows.NSF(features=theta_numel,context=embedded_x_numel,hidden_features=[hidden_features]*num_transforms_lf,transforms=num_transforms_lf)
    elif nf_type == "NSF_FINETUNE":
        
        # flow_built = Flow(
        #     transform=[
        #         base_model.net.transform.transforms[0],
        #         base_model.net.transform.transforms[1],
        #         base_model.net.transform.transforms[2],
        #         base_model.net.transform.transforms[3],
        #         base_model.net.transform.transforms[4], # just all transforms it had
        #         additional_hf_transform, # No affine layer needed because we have a z-score layer in neural_net_hf
        #     ],
        #     base=base_model.net.base, 
        # )
        
        if logit_transform_theta or z_score_theta: # So first dimension is a transformation
            flow_built = Flow(
                # Ignore the first dimension, which is trained in an unconstrained space or z-transformed.
                transform=[
                    base_model.net.transform.transforms[1],
                    base_model.net.transform.transforms[2], ##### FOR 1 LAYER TEST
                    base_model.net.transform.transforms[3],
                    base_model.net.transform.transforms[4],
                    base_model.net.transform.transforms[5], # just all transforms it had
                    # additional_hf_transform, # No affine layer needed because we have a z-score layer in neural_net_hf
                ],
                base=base_model.net.base, 
            ) 
        else:
            flow_built = Flow(
                transform=[
                    base_model.net.transform.transforms[0],
                    base_model.net.transform.transforms[1],  ##### FOR 1 LAYER TEST
                    base_model.net.transform.transforms[2],
                    base_model.net.transform.transforms[3],
                    base_model.net.transform.transforms[4], # just all transforms it had
                    # additional_hf_transform, # No affine layer needed because we have a z-score layer in neural_net_hf
                ],
                base=base_model.net.base, 
            )
            
    else:
        raise NotImplementedError(f"Normalizing flow type {nf_type} not implemented.")


    transforms = flow_built.transform.transforms
    
    
    if z_score_theta:
        transforms = (
            standardizing_transform_zuko(batch_theta,structured_dims=False),
            *transforms,
        )
        
    if logit_transform_theta:        
        transform = mcmc_transform(prior)
        
        transforms = (
            biject_transform_zuko(transform),
            *transforms,
        )
        
                        
    if z_score_x:
        # Prepend standardizing transform to y-embedding.
        # For img: structured_dims = True, else False
        # TODO: put z_score back
        embedding_net = nn.Sequential(
            standardizing_net(batch_x, structured_dims = False), embedding_net
        )
        

    # Combine transforms.
    neural_net = zuko.flows.Flow(transforms, flow_built.base).to(device)    
    flow = ZukoFlow(neural_net, embedding_net, batch_theta[0].shape, batch_x[0].shape)
    
    return flow


def logit_theta(theta: Tensor, lower_bound: Tensor, upper_bound: Tensor) -> Tensor:
    """
    Return the logit() of an input.

    The `logit` maps the interval `[lower_bound, upper_bound]` to an unbounded space.

    Args:
        theta: Input to be transformed.
        lower_bound: Lower bound of the transformation.
        upper_bound: Upper bound of the transformation.

    Returns: theta_t that is unbounded.
    """
    
    range_ = upper_bound - lower_bound
    theta_01 = (theta - lower_bound) / range_

    return torch.log(theta_01 / (1 - theta_01))


def sigmoidal_theta(theta: Tensor, lower_bound: Tensor, upper_bound: Tensor):
    """
    Return the sigmoidal() of an input.

    The `sigmoidal` maps the unbounded space to the interval `[lower_bound, upper_bound]`.

    Args:
        theta: Input to be transformed.
        lower_bound: Lower bound of the transformation.
        upper_bound: Upper bound of the transformation.

    Returns: theta_t that is bounded.
    """
    range_ = upper_bound - lower_bound
    return lower_bound + range_ * torch.sigmoid(theta)


# %%
