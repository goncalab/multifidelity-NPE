from mf_npe.flows.additional_transformations_theta import biject_transform_theta, standardizing_transform_theta
from mf_npe.flows.additional_transformations_x import standardizing_net_x
import torch
import torch.nn as nn
from torch import Tensor
import zuko 
from zuko.flows import Flow
from sbi.neural_nets.estimators.zuko_flow import ZukoFlow
from sbi.utils.sbiutils import mcmc_transform

def build_zuko_flow(
        batch_theta: Tensor,
        batch_x: Tensor,
        embedding_net: nn.Module,
        z_score_theta: bool,        
        z_score_x: bool,
        logit_transform_theta: bool, # transformation by adjusting the network weights
        hidden_features: int,
        num_transforms: int, # how many transformations to use in the flow (e.g., 5), which is equivalent to the number of coupling layers in NSF
        num_bins:int, # number of bins for NSF
        nf_type: str = "NSF", # Neural Spline Flow (NSF) or NSF_PRETRAIN or NSF_FINETUNE
        base_model = None,
        device = "cpu",
        prior = None,
) -> ZukoFlow:
    '''
    This function builds a ZukoFlow given user-specific parameters (e.g., whether the data should be z-scored or not). 
    It can be used for both pretraining and finetuning, depending on the nf_type argument.
    '''
    theta_numel = batch_theta.shape[-1]
    embedded_x_numel = embedding_net(batch_x).shape[-1]
        
    if nf_type == "NSF" or nf_type == "NSF_PRETRAIN":
        flow_built = zuko.flows.NSF(features=theta_numel, 
                                    context=embedded_x_numel, 
                                    bins=num_bins, 
                                    transforms=num_transforms, 
                                    hidden_features=[hidden_features]*num_transforms)
    elif nf_type == "NSF_FINETUNE":
        # Check if base model is provided for finetuning
        if base_model is None:
            raise ValueError("base_model must be provided for nf_type='NSF_FINETUNE'.")
        
        # Check if the first transformation should be ignored during finetuning in case we trained in an unbounded space (e.g., with logit transform)
        start_idx = 1 if (logit_transform_theta or z_score_theta) else 0
        end_idx = start_idx + num_transforms
        
        # Safety check to ensure that the base model has enough transforms to build the flow
        transforms_list = base_model.net.transform.transforms
        if end_idx > len(transforms_list):
            raise ValueError(
                f"Requested transforms[{start_idx}:{end_idx}] but only "
                f"{len(transforms_list)} transforms exist in base_model."
            )
        
        # Build flow by copying the transformations from the base model
        flow_built = Flow(
            transform=list(transforms_list[start_idx:end_idx]),
            base=base_model.net.base,
        )
    else:
        raise NotImplementedError(f"Normalizing flow type {nf_type} not implemented.")

    transforms = flow_built.transform.transforms
    
    if z_score_theta:
        transforms = (
            standardizing_transform_theta(batch_theta,structured_dims=False),
            *transforms,
        )
        
    if logit_transform_theta:        
        transform = mcmc_transform(prior)
        
        transforms = (
            biject_transform_theta(transform),
            *transforms,
        )
                        
    if z_score_x:
        # Prepend standardizing transform to y-embedding.
        # For img: structured_dims = True, else False
        embedding_net = nn.Sequential(
            standardizing_net_x(batch_x, structured_dims = False), embedding_net
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


def sigmoidal_theta(theta: Tensor, lower_bound: Tensor, upper_bound: Tensor) -> Tensor:
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