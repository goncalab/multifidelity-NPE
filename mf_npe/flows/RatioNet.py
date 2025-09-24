import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

class RatioNet(nn.Module, ):
    def __init__(self, dim_theta_x):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_theta_x, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64,  1)             # logit
        )
    def forward(self, theta_x):
        return self.net(theta_x).squeeze(-1)

# class RatioNet(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)  # outputs logit
#         )

#     def forward(self, x):
#         return self.net(x)