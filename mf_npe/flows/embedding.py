# ### PSEUDOCODE

# # Create an embedding network that takes the low-fidelity data and outputs an embedding that is as close as 
# # possible to the high-fidelity data.

# ####

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class EmbeddingNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(EmbeddingNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
#     def train_embed(self, lf_loader, hf_loader, epochs=100, lr=0.001):
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#         criterion = nn.MSELoss()
        
#         self.train()
#         for epoch in range(epochs):
#             for (lf_batch), (hf_batch) in zip(lf_loader, hf_loader):
#                 optimizer.zero_grad()
#                 output = self(lf_batch)
#                 loss = criterion(output, hf_batch)
#                 loss.backward()
#                 optimizer.step()

#             if epoch % 10 == 0:
#                 print(f"Epoch {epoch}, Loss: {loss.item()}")
                
                
#     def evaluate_embed(self, lf_loader):
#         self.eval()
#         outputs = []
#         with torch.no_grad():
#             for lf_batch, _ in lf_loader:
#                 outputs.append(self(lf_batch))
#         return torch.cat(outputs, dim=0)
        
        
