# %%
from typing import Optional
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
import copy
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function


def create_train_val_dataloaders(
    theta,
    x,
    validation_fraction, # for me it was 0.1
    domain_labels=None,  # Optional, for domain adaptation
    batch_size=200, # train batch size
):

    num_examples = theta.shape[0]
    prior_masks = torch.ones_like(theta) 
    # domain_labels for adversarial loss: # 0 for source domain, 1 for target domain for pretraining
    if domain_labels is None:
        dataset = data.TensorDataset(theta, x, prior_masks)
    else:
        dataset = data.TensorDataset(theta, x, prior_masks, domain_labels) #prior_masks
        # Check if shapes are compatible
        if domain_labels.shape[0] != num_examples:
            raise ValueError("Domain labels must have the same number of examples as theta and x.")
    
    num_training_examples = int((1 - validation_fraction) * num_examples)
    num_validation_examples = num_examples - num_training_examples

    permuted_indices = torch.randperm(num_examples)
    
    train_indices, val_indices = (
        permuted_indices[:num_training_examples],
        permuted_indices[num_training_examples:],
    )
    
    train_loader = data.DataLoader(dataset, 
                                batch_size=min(batch_size, num_training_examples), 
                                drop_last=True,
                                sampler=SubsetRandomSampler(train_indices.tolist()))
    val_loader = data.DataLoader(dataset,
                                batch_size=min(batch_size, num_validation_examples),
                                shuffle=False,
                                drop_last=True,
                                sampler=SubsetRandomSampler(val_indices.tolist()))
        
    return train_loader, val_loader


def plot_loss_func(training_loss, validation_loss, title):
    import matplotlib
    matplotlib.use("Agg")  # non-interactive; no windows, fewer semaphores
    import matplotlib.pyplot as plt

    plt.plot(training_loss, label="Training loss")
    plt.plot(validation_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Negative log likelihood")
    plt.title(title)
    plt.legend()
    plt.show()
    

def fit_conditional_normalizing_flow(
    network,
    optimizer,
    training_dataset,
    validation_dataset,
    x_embedder=None,
    early_stopping_patience=20, # Like in SBI
    nb_epochs=2**31 - 1, # Like in SBI
    print_every=10,
    clip_max_norm: Optional[float] = 5.0,
    plot_loss=False, 
    type_flow='HF',
    seed=False,
):
    training_loss = []
    validation_loss = []
    best_validation_loss = float("inf")
    _converged = False
    
    epoch = 0
    epoch_validation_loss = float("inf")
    
    while epoch <= nb_epochs and not _converged: 
        
        network.train()
        train_loss_sum = 0
        for batch in training_dataset:
            optimizer.zero_grad()
            
            # Get batches on current device.
            theta_batch, x_batch, masks_batch = (
                batch[0].to("cpu"),
                batch[1].to("cpu"),
                batch[2].to("cpu"),
            )
                        
            # if x_encoder is not None:
            #     x_batch = x_encoder(x_batch)
            # else:
            #     x_batch = x_batch

            # sum over losses 
            losses = - network.log_prob(theta_batch.unsqueeze(0), x_batch)[0]
            loss = torch.mean(losses)
            train_loss_sum += losses.sum().item()
                        
            loss.backward()
            # Clipping to avoid exploding gradients
            if clip_max_norm is not None:
                clip_grad_norm_(
                    network.parameters(), # this is the self._classifeir in the SBI package
                    max_norm=clip_max_norm,
                )
            optimizer.step()
        
        epoch += 1
        
        train_loss_average = train_loss_sum / (
            len(training_dataset) * training_dataset.batch_size  # type: ignore
        )

        epoch_training_loss = train_loss_average #np.mean(train_loss_average)
        training_loss.append(epoch_training_loss)
        
        
        # Calculate validation performance
        network.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch in validation_dataset:
                theta_batch, x_batch, masks_batch = (
                    batch[0].to("cpu"),
                    batch[1].to("cpu"),
                    batch[2].to("cpu"),
                )
                
                val_losses = - network.log_prob(theta_batch.unsqueeze(0), x_batch)[0]
                val_loss_sum += val_losses.sum().item()

        # take the mean over all validation samples
        epoch_validation_loss = val_loss_sum / (
            len(validation_dataset) * validation_dataset.batch_size  # type: ignore
        )

        validation_loss.append(epoch_validation_loss)

        if epoch % print_every == 0:
            print(
                f"Epoch: {epoch}, Train Loss: {epoch_training_loss}, Val Loss: {epoch_validation_loss}", end='\r'
            )

        if epoch == 0 or epoch_validation_loss < best_validation_loss:
            best_validation_loss = epoch_validation_loss
            # Save the best model so far in a variable
            best_model = copy.deepcopy(network) # .state_dict()
            #best_model = torch.save(network.state_dict(), 'best_flow.pth') #save('best_autoencoder.pth')
            _epochs_since_last_improvement = 0
        else:
            _epochs_since_last_improvement += 1
        
        # If no validation improvement over many epochs, stop training.
        if _epochs_since_last_improvement > early_stopping_patience - 1:
            _converged = True
            print() # New line
            print(
                f"Early stopping after {epoch} epochs. "
                f"Best validation loss: {best_validation_loss:.4f})"
            )
            
    if plot_loss:
        plot_loss_func(training_loss, validation_loss, type_flow)
    
    # Avoid keeping the gradients in the resulting network, which can
    # cause memory leakage when benchmarking.
    network.zero_grad(set_to_none=True)
    
    # print("training loss", training_loss)
    # print("validation loss", validation_loss)
    
    # Return the best model
    #neural_net.load_state_dict(self._best_model_state_dict)

    return best_model



    

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)
    

#Intermediate, learnable function of x, like described in https://arxiv.org/pdf/1702.05464: a feature extractor that is learnable
class XEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)



def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the CORAL loss between source and target feature matrices.
    
    Args:
        source (Tensor): source data features (n_s x d)
        target (Tensor): target data features (n_t x d)
    
    Returns:
        Tensor: scalar CORAL loss
    """
    d = source.size(1)  # feature dimension

    # Compute mean-centered features
    source_mean = torch.mean(source, dim=0, keepdim=True)
    target_mean = torch.mean(target, dim=0, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    # Covariance matrices (unbiased estimator)
    n_s = source.size(0)
    n_t = target.size(0)
    cov_source = (source_centered.T @ source_centered) / (n_s - 1)
    cov_target = (target_centered.T @ target_centered) / (n_t - 1)

    # Frobenius norm of the difference
    loss = torch.norm(cov_source - cov_target, p='fro') ** 2
    loss = loss / (4 * d * d)
    
    return loss



# def compute_mmd(x, y, sigma=1.0):
    
#     # Check if trainable features are provided raise value error
#     # if not x.requires_grad or not y.requires_grad:
#     #     raise ValueError("Both x and y must have requires_grad=True for MMD computation. Enable xEncoder in pipeline.py (see self.xEncoder) to have requires_grad=True.")

#     def rbf_kernel(a, b, sigma):
#         norm = lambda x: (x ** 2).sum(dim=1, keepdim=True)
#         a_sq = norm(a)
#         b_sq = norm(b)
#         dist_sq = a_sq - 2 * a @ b.T + b_sq.T
#         return torch.exp(-dist_sq / (2 * sigma ** 2))

#     K_xx = rbf_kernel(x, x, sigma)
#     K_yy = rbf_kernel(y, y, sigma)
#     K_xy = rbf_kernel(x, y, sigma)

#     return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

def _rbf_kernel(a, b, sigma2):
    # a: [m, d], b: [n, d]; returns [m, n]
    # sigma2 is the *variance* (sigma^2) for numerical stability
    a2 = (a**2).sum(dim=1, keepdim=True)         # [m, 1]
    b2 = (b**2).sum(dim=1, keepdim=True).T       # [1, n]
    dist2 = a2 - 2 * (a @ b.T) + b2              # [m, n]
    return torch.exp(-dist2 / (2.0 * sigma2))

def compute_mmd(x, y, sigmas=(0.5, 1.0, 2.0), chunk_size=None):
    m, n = x.size(0), y.size(0)
    if m < 2 or n < 2:
        return compute_mmd_biased(x, y, sigmas=sigmas, remove_diagonal=True)

    mmd2 = torch.zeros((), device=x.device, dtype=x.dtype)  # tensor accumulator
    for sigma in sigmas:
        sigma2 = float(sigma) ** 2

        K_xx = _rbf_kernel(x, x, sigma2)
        K_yy = _rbf_kernel(y, y, sigma2)

        sum_xx = (K_xx.sum() - K_xx.diagonal().sum())
        sum_yy = (K_yy.sum() - K_yy.diagonal().sum())

        if chunk_size is None:
            sum_xy = _rbf_kernel(x, y, sigma2).sum()
        else:
            sum_xy = torch.zeros((), device=x.device, dtype=x.dtype)
            for i in range(0, m, chunk_size):
                xi = x[i:i+chunk_size]
                sum_xy = sum_xy + _rbf_kernel(xi, y, sigma2).sum()

        term = (sum_xx / (m * (m - 1))) + (sum_yy / (n * (n - 1))) - (2.0 * sum_xy / (m * n))
        mmd2 = mmd2 + term

    return mmd2 / len(sigmas)


def compute_mmd_biased(x, y, sigmas=(0.5, 1.0, 2.0), remove_diagonal=True):
    """
    Biased MMD^2 (means over all pairs). Optionally drop diagonals in K_xx, K_yy.
    More stable for tiny batches but (slightly) biased.
    """
    m, n = x.size(0), y.size(0)
    mmd2 = 0.0
    for sigma in sigmas:
        sigma2 = float(sigma) ** 2
        K_xx = _rbf_kernel(x, x, sigma2)
        K_yy = _rbf_kernel(y, y, sigma2)
        K_xy = _rbf_kernel(x, y, sigma2)

        if remove_diagonal:
            K_xx = K_xx - torch.diag_embed(torch.diagonal(K_xx))
            K_yy = K_yy - torch.diag_embed(torch.diagonal(K_yy))
            # renormalize means after removing diagonals
            mmd2 += (K_xx.sum() / (m * (m - 1) if m > 1 else m)) \
                  + (K_yy.sum() / (n * (n - 1) if n > 1 else n)) \
                  - 2.0 * (K_xy.mean())
        else:
            mmd2 += K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()

    return mmd2 / len(sigmas)


def fit_pretrained_conditional_normalizing_flow(
    network,
    optimizer,
    training_dataset,
    validation_dataset,
    x_dim_lf,  # Dimension of the input data
    x_dim_hf,  # Dimension of the output data
    x_dim_out,
    theta_dim,
    early_stopping_patience=20, # Like in SBI
    nb_epochs=2**31 - 1, # Like in SBI
    print_every=10,
    clip_max_norm: Optional[float] = 5.0,
    plot_loss=False, 
    type_flow='HF',
    seed=False,
    device="cpu",  # Device to run the training on
):
    training_loss = []
    validation_loss = []
    best_validation_loss = float("inf")
    _converged = False
    
    epoch = 0
    epoch_validation_loss = float("inf")
    
    lambda_mmd = 10.00  # Based on paper's best result
    
    
    while epoch <= nb_epochs and not _converged: 
        network.train()
        train_loss_sum = 0
        for batch in training_dataset:
            optimizer.zero_grad()
            # domain_optimizer.zero_grad()
            
            # Get batches on current device.
            theta_batch, x_batch, masks_batch, domain_labels = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
                batch[3].to(device).long(), # domain label: 0=source, 1=target, should be long for cross entropy
            )

            # Separate source and target features
            src_mask = (domain_labels == 0)
            tgt_mask = (domain_labels == 1)
            
            lf_batch = x_batch[src_mask]
            hf_batch = x_batch[tgt_mask]
            
            # # Use the dimensions that were not expanded (we added 0's for the joint to concatenate for the train_ and val_loader, but they should be removed before the encoding)
            lf_batch = lf_batch[:, :x_dim_hf]
            hf_batch = hf_batch[:, :x_dim_hf]
            
            x_encoded_lf = lf_batch 
            x_encoded_hf = hf_batch
            
            # NLL Loss
            loss_lf = - network.log_prob(theta_batch[src_mask].unsqueeze(0), x_encoded_lf)[0]
            
            # In case HF samples are not present, because small batch size 
            if len(x_encoded_hf) == 0:
                loss_hf = torch.tensor(0.0, device=x_encoded_lf.device)
                losses = loss_lf
            else:
                loss_hf = - network.log_prob(theta_batch[tgt_mask].unsqueeze(0), x_encoded_hf)[0]
                losses = torch.cat([loss_lf, loss_hf], dim=0) 
                            
            lf_loss_mean = torch.mean(loss_lf)
            hf_loss_mean = torch.mean(loss_hf)

            train_loss_sum += losses.sum().item()
            
            # MMD loss: only compute if both source and target features are present
            # if len(x_encoded_hf) == 0 or len(x_encoded_lf) == 0:
            #     mmd_loss = None
            # else:
            #     mmd_loss = compute_mmd(x_encoded_lf, x_encoded_hf)
            
            
            loss = lf_loss_mean + hf_loss_mean
            # if mmd_loss is not None:
            #     loss = loss + lambda_mmd * mmd_loss

            loss.backward()
            
            
            # Clipping to avoid exploding gradients
            if clip_max_norm is not None:
                clip_grad_norm_(
                    network.parameters(), # this is the self._classifeir in the SBI package
                    max_norm=clip_max_norm,
                )
            optimizer.step()
        
        epoch += 1
        
        train_loss_average = train_loss_sum / (
            len(training_dataset) * training_dataset.batch_size  # type: ignore
        )

        epoch_training_loss = train_loss_average #np.mean(train_loss_average)
        training_loss.append(epoch_training_loss)
        
        # Calculate validation performance
        network.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch in validation_dataset:
                theta_batch, x_batch, masks_batch, domain_labels = (
                    batch[0].to(device),
                    batch[1].to(device),
                    batch[2].to(device),
                    batch[3].to(device).long(),  # domain labels
                )
                
                src_mask = (domain_labels == 0)
                tgt_mask = (domain_labels == 1)
                
                lf_batch = x_batch[src_mask]
                hf_batch = x_batch[tgt_mask]
                
                # Same as for training
                lf_batch = lf_batch[:, :x_dim_hf]
                hf_batch = hf_batch[:, :x_dim_hf]

                x_encoded_lf = lf_batch
                x_encoded_hf = hf_batch

                val_loss_lf = - network.log_prob(theta_batch[src_mask].unsqueeze(0), x_encoded_lf)[0]
                
                # In case HF samples are not present, because small batch size 
                if len(x_encoded_hf) == 0:
                    val_loss_hf = torch.tensor(0.0, device=x_encoded_lf.device)
                    val_losses = loss_lf
                else:
                    val_loss_hf = - network.log_prob(theta_batch[tgt_mask].unsqueeze(0), x_encoded_hf)[0]
                    val_losses = torch.cat([val_loss_lf, val_loss_hf], dim=0)

                val_loss_sum += val_losses.sum().item()

        # take the mean over all validation samples
        epoch_validation_loss = val_loss_sum / (
            len(validation_dataset) * validation_dataset.batch_size  # type: ignore
        )

        validation_loss.append(epoch_validation_loss)

        if epoch % print_every == 0:
            #epoch_domain_acc = domain_correct_sum / domain_total_sum
            # if mmd_loss is not None:
            #     print(
            #         f"Epoch: {epoch}, Train Loss: {epoch_training_loss}, Val Loss: {epoch_validation_loss}", end='\r' #  , MMD Loss: {mmd_loss.item():.4f}
            #     )
            # else:
            print(
                f"Epoch: {epoch}, Train Loss: {epoch_training_loss}, Val Loss: {epoch_validation_loss}", end='\r' #  
            )


        if epoch == 0 or epoch_validation_loss < best_validation_loss:
            best_validation_loss = epoch_validation_loss
            # Save the best model so far in a variable
            best_model = copy.deepcopy(network) # .state_dict()
            #best_model = torch.save(network.state_dict(), 'best_flow.pth') #save('best_autoencoder.pth')
            _epochs_since_last_improvement = 0
        else:
            _epochs_since_last_improvement += 1
        
        # If no validation improvement over many epochs, stop training.
        if _epochs_since_last_improvement > early_stopping_patience - 1:
            _converged = True
            print() # New line
            print(
                f"Early stopping after {epoch} epochs. "
                f"Best validation loss: {best_validation_loss:.4f})"
            )
            
    if plot_loss:
        plot_loss_func(training_loss, validation_loss, type_flow)
    
    # Avoid keeping the gradients in the resulting network, which can
    # cause memory leakage when benchmarking.
    network.zero_grad(set_to_none=True)
    
    # print("training loss", training_loss)
    # print("validation loss", validation_loss)

    return best_model
