# Assume you have: 
# - E_LF (encoder for LF data)
# - E_HF (encoder for HF data)
# - posterior_net (SBI posterior)
# - lf_loader, hf_loader (DataLoaders)

# Stage 1: LF pretraining
# pretrain_on_LF(E_LF, posterior_net, lf_loader)

# # Stage 2: Adversarial alignment (WGAN-GP)
# adversarial_align(E_LF, E_HF, lf_loader, hf_loader, latent_dim=64)

# # Stage 3: Fine-tune on HF
# fine_tune_on_HF(E_LF, posterior_net, hf_loader)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# ====== Discriminator for WGAN-GP ======
class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, z):
        return self.net(z)

# ====== Gradient penalty (WGAN-GP) ======
def gradient_penalty(D, real, fake):
    alpha = torch.rand(real.size(0), 1, device=device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)
    grads = grad(outputs=d_interpolates, inputs=interpolates,
                 grad_outputs=torch.ones_like(d_interpolates),
                 create_graph=True, retain_graph=True)[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

# ====== Example SBI loss wrapper (placeholder) ======
def sbi_loss_fn(posterior_net, theta, features):
    """Negative log-likelihood loss for SBI"""
    return -posterior_net.log_prob(theta, features).mean()

# ====== Stage 1: Pretrain on LF ======
def pretrain_on_LF(E_LF, posterior_net, lf_loader, epochs=50, lr=1e-3):
    optimizer = optim.Adam(list(E_LF.parameters()) + list(posterior_net.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for x_lf, theta in lf_loader:
            x_lf, theta = x_lf.to(device), theta.to(device)
            z_lf = E_LF(x_lf)
            loss = sbi_loss_fn(posterior_net, theta, z_lf)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
        print(f"[Stage 1] Epoch {epoch+1}: Loss = {total_loss/len(lf_loader):.4f}")

# ====== Stage 2: Adversarial Alignment (WGAN-GP) ======
def adversarial_align(E_LF, E_HF, lf_loader, hf_loader, latent_dim=64, epochs=20,
                      lr_LF=1e-4, lr_D=1e-4, lambda_gp=10, n_critic=5):
    D = Discriminator(latent_dim).to(device)
    opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.9))
    opt_LF = optim.Adam(E_LF.parameters(), lr=lr_LF)
    
    # Freeze HF encoder
    E_HF.eval()
    for p in E_HF.parameters():
        p.requires_grad = False
    
    hf_iter = iter(hf_loader)
    
    for epoch in range(epochs):
        for i, (x_lf, _) in enumerate(lf_loader):
            try:
                x_hf, _ = next(hf_iter)
            except StopIteration:
                hf_iter = iter(hf_loader)
                x_hf, _ = next(hf_iter)
            
            x_lf, x_hf = x_lf.to(device), x_hf.to(device)
            
            z_lf = E_LF(x_lf).detach()
            with torch.no_grad():
                z_hf = E_HF(x_hf)
            
            # ----- Update Discriminator -----
            for _ in range(n_critic):
                d_loss = D(z_lf).mean() - D(z_hf).mean()
                gp = gradient_penalty(D, z_hf, z_lf)
                loss_D = d_loss + lambda_gp * gp
                
                opt_D.zero_grad(); loss_D.backward(); opt_D.step()
            
            # ----- Update LF encoder (Generator) -----
            z_lf = E_LF(x_lf)
            loss_adv = -D(z_lf).mean()  # fool discriminator
            
            opt_LF.zero_grad(); loss_adv.backward(); opt_LF.step()
        
        print(f"[Stage 2] Epoch {epoch+1}: Adv Loss = {loss_adv.item():.4f}")

# ====== Stage 3: Fine-tune on HF ======
def fine_tune_on_HF(E_LF, posterior_net, hf_loader, epochs=30, lr=1e-4, lambda_adv=0.01, D=None):
    optimizer = optim.Adam(list(E_LF.parameters()) + list(posterior_net.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    for epoch in range(epochs):
        total_loss = 0.0
        for x_hf, theta in hf_loader:
            x_hf, theta = x_hf.to(device), theta.to(device)
            z_hf = E_LF(x_hf)
            
            # SBI loss
            loss_sbi = sbi_loss_fn(posterior_net, theta, z_hf)
            
            # Optional small adversarial regularization (if discriminator is kept)
            loss_adv = 0.0
            if D is not None:
                loss_adv = -D(z_hf).mean()
            
            loss = loss_sbi + lambda_adv * loss_adv
            # loss = loss_sbi
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        
        scheduler.step(total_loss)
        print(f"[Stage 3] Epoch {epoch+1}: Loss = {total_loss/len(hf_loader):.4f}")


# # Stage 1: LF pretraining
# train_sbi_on_LF(E_LF, posterior_net, lf_dataloader)

# # Stage 2: adversarial alignment
# HF_encoder.eval()
# for p in HF_encoder.parameters():
#     p.requires_grad = False

# for epoch in range(align_epochs):
#     for x_lf, x_hf in dataloader:
#         z_lf = E_LF(x_lf)
#         with torch.no_grad():
#             z_hf = HF_encoder(x_hf)

#         # Discriminator step
#         loss_D = BCE(discriminator(z_hf), 1) + BCE(discriminator(z_lf.detach()), 0)
#         opt_D.zero_grad(); loss_D.backward(); opt_D.step()

#         # LF encoder step
#         loss_adv = BCE(discriminator(z_lf), 1)
#         opt_LF.zero_grad(); loss_adv.backward(); opt_LF.step()

# # Stage 3: fine-tune on HF
# train_sbi_on_HF(E_LF, posterior_net, hf_dataloader, finetune=True)


# #######
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sbi.inference import SNPE, prepare_for_sbi
# from sbi.utils import BoxUniform

# # Discriminator
# class Discriminator(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(latent_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
#     def forward(self, z):
#         return self.net(z)

# latent_dim = 64
# discriminator = Discriminator(latent_dim).to(device)
# optim_D = optim.Adam(discriminator.parameters(), lr=1e-4)
# bce = nn.BCELoss()

# lambda_adv = 0.1  # weight for adversarial loss

# # HF encoder (frozen)
# HF_encoder.eval()
# for p in HF_encoder.parameters():
#     p.requires_grad = False

# # LF encoder (trainable)
# LF_encoder.train()
# optim_LF = optim.Adam(LF_encoder.parameters(), lr=1e-4)

# # Set up SBI posterior estimator
# prior = BoxUniform(low=torch.tensor([-2.0, -2.0]), high=torch.tensor([2.0, 2.0]))
# inference = SNPE(prior=prior)

# for x_lf, x_hf, theta in dataloader:

#     # ----- Forward pass -----
#     z_lf = LF_encoder(x_lf)
#     with torch.no_grad():
#         z_hf = HF_encoder(x_hf)

#     # ----- Discriminator step -----
#     pred_hf = discriminator(z_hf.detach())
#     pred_lf = discriminator(z_lf.detach())

#     loss_D = bce(pred_hf, torch.ones_like(pred_hf)) + \
#              bce(pred_lf, torch.zeros_like(pred_lf))

#     optim_D.zero_grad()
#     loss_D.backward()
#     optim_D.step()

#     # ----- LF encoder + SBI loss -----
#     # SBI: log posterior loss (standard SNPE)
#     x_aligned = LF_encoder(x_lf)  # new latent for LF data
#     loss_sbi = inference._loss(theta, x_aligned)  # internal loss (negative log-likelihood)

#     # Adversarial loss: fool discriminator
#     pred_lf = discriminator(x_aligned)
#     loss_adv = bce(pred_lf, torch.ones_like(pred_lf))

#     loss_total = loss_sbi + lambda_adv * loss_adv

#     optim_LF.zero_grad()
#     loss_total.backward()
#     optim_LF.step()

#     # Update SNPE estimator parameters (posterior network)
#     inference._optimizer.step()

# # import torch
# # import torch.nn as nn
# # import torch.optim as optim

# # # Discriminator for adverserial loss
# # class Discriminator(nn.Module):
# #     def __init__(self, latent_dim):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(latent_dim, 128),
# #             nn.ReLU(),
# #             nn.Linear(128, 1),
# #             nn.Sigmoid()
# #         )
# #     def forward(self, z):
# #         return self.net(z)

# # # Setup
# # latent_dim = 64
# # discriminator = Discriminator(latent_dim).to(device)

# # optim_D = optim.Adam(discriminator.parameters(), lr=1e-4)
# # optim_LF = optim.Adam(LF_encoder.parameters(), lr=1e-4)

# # bce = nn.BCELoss()

# # # Training loop
# # for x_lf, x_hf in dataloader:
# #     z_lf = LF_encoder(x_lf)
# #     with torch.no_grad():
# #         z_hf = HF_encoder(x_hf)

# #     # ===== Train Discriminator =====
# #     pred_hf = discriminator(z_hf.detach())
# #     pred_lf = discriminator(z_lf.detach())

# #     loss_D = bce(pred_hf, torch.ones_like(pred_hf)) + \
# #              bce(pred_lf, torch.zeros_like(pred_lf))

# #     optim_D.zero_grad()
# #     loss_D.backward()
# #     optim_D.step()

# #     # ===== Train LF encoder (Generator) =====
# #     z_lf = LF_encoder(x_lf)
# #     pred_lf = discriminator(z_lf)

# #     loss_LF = bce(pred_lf, torch.ones_like(pred_lf))

# #     optim_LF.zero_grad()
# #     loss_LF.backward()
# #     optim_LF.step()
