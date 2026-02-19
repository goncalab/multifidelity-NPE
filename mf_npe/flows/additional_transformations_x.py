import torch
import torch.nn as nn
from torch import Tensor
import logging
from typing import Union

# https://github.com/sbi-dev/sbi/blob/main/sbi/utils/sbiutils
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

def standardizing_net_x(
    batch_t: Tensor,
    structured_dims: bool = False,
    min_std: float = 1e-7,
) -> nn.Module:
    """Building block for standardizing the x (data)

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