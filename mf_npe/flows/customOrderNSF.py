#%%
import torch
import torch.nn as nn
from zuko.flows import Flow
from zuko.flows.autoregressive import MaskedAutoregressiveTransform


class CustomOrderNSF(nn.Module):
    def __init__(
        self,
        features: int,
        context: int,
        orders: list[list[int]],
        bins: int = 8,
        hidden_features: int = 64,
        # num_blocks: int = 2,
        activation: nn.Module = nn.ReLU(),
        #univariate: type = SplineCouplingTransform,
        shapes: list[torch.Size] = None,
    ):
        super().__init__()
        self.features = features
        self.context = context

        if shapes is None:
            # default spline shape (e.g., 8-bin spline: widths & heights)
            shapes = [(bins,), (bins,)]

        transforms = []

        for order in orders:
            transform = MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.tensor(order),
                passes=None,  # fully autoregressive
                # univariate=univariate,
                shapes=shapes,
                hidden_features=hidden_features,
                # num_blocks=num_blocks,
                activation=activation,
            )
            transforms.append(transform)

        self.flow = Flow(transforms)

    def log_prob(self, x, context=None):
        return self.flow.log_prob(x, context=context)

    def sample(self, num_samples, context=None):
        return self.flow.sample(num_samples, context=context)

    def forward(self, x, context=None):
        return self.flow(x, context=context)
    
    
orders = [
    [0, 1, 2],
    [1, 0, 2],
    [0, 2, 1],
    [2, 0, 1]
]

model = CustomOrderNSF(
    features=3,
    context=4,
    orders=orders,
    bins=8,
    hidden_features=64,
    # num_blocks=2,
)

x = torch.randn(32, 3)
c = torch.randn(32, 4)

log_p = model.log_prob(x, context=c)
samples = model.sample(32, context=c)


#%%