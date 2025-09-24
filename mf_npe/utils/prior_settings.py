import os
from typing import Any, Optional, Union
import torch
from torch import Tensor
from torch.distributions import Independent
from torch.distributions.normal import Normal

def check_device(device: str) -> None:
    """Check whether the device is valid.

    Args:
        device: target torch device
    """
    try:
        torch.randn(1, device=device)
    except (RuntimeError, AssertionError) as exc:
        raise RuntimeError(
            f"""Could not instantiate torch.randn(1, device={device}). Make sure
             the device is set up properly and that you are passing the
             corresponding device string. It should be something like 'cuda',
             'cuda:0', or 'mps'. Error message: {exc}."""
        ) from exc

def process_device(device: str) -> str:
    """Set and return the default device to cpu or gpu (cuda, mps).

    Args:
        device: target torch device
    Returns:
        device: processed string, e.g., "cuda" is mapped to "cuda:0".
    """

    if device == "cpu":
        return "cpu"
    else:
        # If user just passes 'gpu', search for CUDA or MPS.
        if device == "gpu":
            # check whether either pytorch cuda or mps is available
            if torch.cuda.is_available():
                current_gpu_index = torch.cuda.current_device()
                device = f"cuda:{current_gpu_index}"
                check_device(device)
                torch.cuda.set_device(device)
            elif torch.backends.mps.is_available():
                device = "mps:0"
                # MPS support is not implemented for a number of operations.
                # use CPU as fallback.
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                # MPS framework does not support double precision.
                torch.set_default_dtype(torch.float32)
                check_device(device)
            else:
                raise RuntimeError(
                    "Neither CUDA nor MPS is available. "
                    "Please make sure to install a version of PyTorch that supports "
                    "CUDA or MPS."
                )
        # Else, check whether the custom device is valid.
        else:
            check_device(device)

        return device

class BoxNormal(Independent):
    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        reinterpreted_batch_ndims: int = 1,
        device: Optional[str] = None,
    ):
        """Multidimensional uniform distribution defined on a box.

        A `Normal` distribution initialized with e.g. a parameter vector loc or scale of
         length 3 will result in a /batch/ dimension of length 3. A log_prob evaluation
         will then output three numbers, one for each of the independent Normals in
         the batch. Instead, a `BoxNormal` initialized in the same way has three
         /event/ dimensions, and returns a scalar log_prob corresponding to whether
         the evaluated point is in the range defined by loc and scale.

        Refer to torch.distributions.Normal and torch.distributions.Independent for
         further documentation.

        Args:
            loc: mean range.
            scale: standard deviation.
            reinterpreted_batch_ndims (int): the number of batch dims to
                reinterpret as event dims.
            device: device of the prior, inferred from low arg, defaults to "cpu",
                should match the training device when used in SBI.
        """

        # Type checks.
        assert isinstance(loc, Tensor) and isinstance(
            scale, Tensor
        ), f"low and high must be tensors but are {type(loc)} and {type(scale)}."
        if not loc.device == scale.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at least"
                f"two devices, {loc.device} and {scale.device}."
            )

        # Device handling
        device = loc.device.type if device is None else device
        device = process_device(device)

        super().__init__(
            Normal(
                loc=torch.as_tensor(
                    loc, dtype=torch.float32, device=torch.device(device)
                ),
                scale=torch.as_tensor(
                    scale, dtype=torch.float32, device=torch.device(device)
                ),
                validate_args=False,
            ),
            reinterpreted_batch_ndims,
        )


def ensure_theta_batched(theta: Tensor) -> Tensor:
    r"""
    Return parameter set theta that has a batch dimension, i.e. has shape
     (1, shape_of_single_theta)

     Args:
         theta: parameters $\theta$, of shape (n) or (1,n)
     Returns:
         Batched parameter set $\theta$
    """

    # => ensure theta has shape (1, dim_parameter)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    return theta