#%%
import pickle
import time
import torch
import os
from torch.nn import functional as F
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from bayes_opt import BayesianOptimization
import numpy as np
from tqdm import tqdm
from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior
from mf_npe.flows.build_flows import build_zuko_flow
from bayes_opt import acquisition

# from mf_npe.flows.custom_ensemble_posterior import CustomEnsemblePosterior
from mf_npe.flows.train_flows import XEncoder, fit_conditional_normalizing_flow,create_train_val_dataloaders, fit_pretrained_conditional_normalizing_flow
# from mf_npe.plot.plot_gaussprocess import plot_bo_2d, plot_gp
from mf_npe.simulator.task2.simulation_func import Integrator
from sbi import utils as utils
import copy
from sbi.inference.posteriors.direct_posterior import DirectPosterior
import mf_npe.config.plot as plot_config
from torch.distributions import Categorical

# Flows
from sbi import analysis as analysis
import matplotlib
matplotlib.use("Agg")  # non-interactive; no windows, fewer semaphores
import matplotlib.pyplot as plt
from torch import nn
import zuko
from sbi.inference import NPE
from sbi.utils.user_input_checks import (
    process_prior,
)
from sbi.neural_nets import posterior_nn

from sbi.inference import NPE
from sbi.utils import RestrictedPrior, get_density_thresholder

import torch

from sbi.analysis import pairplot
from sbi.inference import NPE
from sbi.utils.user_input_checks import (
    process_prior,
)
import warnings

from sbi.inference import NLE


from mf_npe.utils.utils import dump_pickle

from mf_npe.utils.mf_abc import MFABC, make_mfabc_cloud

import math
import math
from typing import Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from botorch.optim import optimize_acqf


from sbi.inference import (
    NPE_C
)   


class Pipeline():
    def __init__(self, true_xen, task_setup): 
        
        self.obs = true_xen # only needed for mf-abc
        
        self.config_data = task_setup.config_data
        self.config_model = task_setup.config_model
        self.lf_simulator = task_setup.lf_simulator
        self.hf_simulator = task_setup.hf_simulator
        
        self.batch_lf_sims = task_setup.batch_lf_sims
        self.batch_hf_sims = task_setup.batch_hf_sims
        self.batch_mf_sims = task_setup.batch_mf_sims
        
        self.x_dim_lf = task_setup.config_data['x_dim_lf']
        self.x_dim_hf = task_setup.config_data['x_dim_hf']
        self.x_dim_out = task_setup.config_data['x_dim_out']
        self.theta_dim = task_setup.config_data['theta_dim']
        
        self.device = 'cpu' #task_setup.config_model['device'] (cpu is fastest)
       
        self.task = task_setup.config_data['task']
        self.sim_name = task_setup.config_data['sim_name']
        
        self.n_samples_to_generate = task_setup.config_data['n_samples_to_generate']
        
        self.hf_prior = self.hf_simulator.prior()
        
        # If lf_simulator is a dict, go over all simulators
        if isinstance(self.lf_simulator, dict):
            lf_priors = {}
            for fidelity, sim in self.lf_simulator.items():
                lf_priors[fidelity] = sim.prior()

            warnings.warn(
                "Using the prior of the lowest lf simulator.",
                category=UserWarning,   # or a custom subclass
                stacklevel=2            # points warning at the caller
            )
            
            preferred = ("lf", "lf0", "low")
            chosen_key = next((k for k in preferred if k in lf_priors), next(iter(lf_priors)))
            self.lf_prior = lf_priors[chosen_key] # Just use the first prior for now. This can be expanded further if needed.
        else:
            self.lf_prior = self.lf_simulator.prior()

        self.validation_fraction = task_setup.config_model['validation_fraction']
        
        self.use_wandb = False
        self.use_finetuned_model = False
        
        self.max_num_epochs = task_setup.config_model['max_num_epochs']
        self.batch_size = task_setup.config_model['batch_size']
        self.early_stopping = task_setup.config_model['patience']
        self.clip_max_norm= task_setup.config_model['clip_max_norm']
        
        self.z_score_theta: bool = task_setup.config_model['z_score_theta']
        self.z_score_x: bool = task_setup.config_model['z_score_x']
        self.logit_transform_theta_net = task_setup.config_model['logit_transform_theta_net']
        
        self.main_path = task_setup.main_path
        self.CURR_TIME = task_setup.CURR_TIME

        self.type_embedding_lf = task_setup.config_data['lf_embedding']# 'cnn' # xEncoder, identity, cnn
        self.type_embedding_hf = task_setup.config_data['hf_embedding'] # 'cnn' # xEncoder, identity, cnn

        self.task_setup = task_setup
        
    
    def _generate_embedding_networks(self, x_lf=None, x_hf=None):
        def build(kind: str, in_dim: int, which: str):
            if kind == "xEncoder":
                return XEncoder(
                    input_dim=in_dim,
                    hidden_dim=self.config_model["n_hidden_features"],
                    output_dim=self.x_dim_out,
                ).to(self.device)
            if kind == "identity":
                return nn.Identity()
            
            if kind == "cnn":
                from sbi.neural_nets.embedding_nets import CNNEmbedding
                import numpy as np
                
                x_dim_hf_unflattened = int(np.sqrt(self.x_dim_hf))
                
                # For gaussian blob
                embedding_net = CNNEmbedding(
                    input_shape=(x_dim_hf_unflattened, x_dim_hf_unflattened), # 1024, you can also put a tuple such as 32x32 for instance
                    in_channels=1,
                    out_channels_per_layer=[6],
                    num_conv_layers=1,
                    num_linear_layers=1,
                    output_dim=self.x_dim_out,
                    kernel_size=5,
                    pool_kernel_size=8
                )
                
                return embedding_net
            
            raise ValueError(
                f"Unknown type_embedding_{which}: {kind}. Choose 'xEncoder' or 'identity'."
            )
        
        # Check if data and dimensions are the same
        # If x_dim_lf is not none
        # if x_lf is not None:
        #     if self.x_dim_lf != x_lf.shape[1]:
        #         raise ValueError(f"LF data has {x_lf.shape[1]} features, but {self.x_dim_lf} were expected. Try to regenerate the data (by adding --generate_true_xen and --generate_train_data to the train.py command) or change the x_dim_lf parameter.")
        
        
        if x_hf is not None:
            if self.x_dim_hf != x_hf.shape[1]:
                raise ValueError(f"HF data has {x_hf.shape[1]} features, but {self.x_dim_hf} were expected. Try to regenerate the data (by adding --generate_true_xen and --generate_train_data to the train.py command) or change the x_dim_hf parameter.")

        embedding_lf = build(self.type_embedding_lf, self.x_dim_hf, "lf")
        embedding_hf = build(self.type_embedding_hf, self.x_dim_hf, "hf")
        
        return embedding_lf, embedding_hf


    def _write_time_file(self, type_algorithm, hf_start_time, hf_end_time, n_hf_samples, 
                         lf_start_time=None, lf_end_time=None,n_lf_samples=None, 
                         total_start_time=None, total_end_time=None, 
                         one_round_start_time=None, one_round_end_time=None, n_rounds=None):
        
        
        path = self.main_path + "/time"
        if not os.path.exists(path):
                os.makedirs(path)
        # Save the time in a txt file      
        with open(f"{path}/{type_algorithm}_time_LF{n_lf_samples}_HF{n_hf_samples}.txt", "a") as f:
            if lf_start_time is not None and lf_end_time is not None:
                f.write(f"Time taken for LF training: {lf_end_time - lf_start_time} seconds\n")
                
            if lf_start_time is not None and lf_end_time is not None:
                f.write(f"Time taken for MF training: {hf_end_time - lf_start_time} seconds\n")

            if total_start_time is not None and total_end_time is not None:
                f.write(f"Time taken for {n_rounds} rounds for 1 xo: {total_end_time - total_start_time} seconds\n")
                
            if one_round_start_time is not None and one_round_end_time is not None:
                f.write(f"Time taken for 1 round of HF training: {one_round_end_time - one_round_start_time} seconds\n")
                
            f.write(f"Time taken for HF training: {hf_end_time - hf_start_time} seconds\n")

            f.write("---------------------------------------\n")


        
    def run_npe(self, data):
        posteriors = []
        for i, n_simulations in enumerate(self.batch_hf_sims):  
            
            curr_dataset = data[i]
            x_t, theta_t = curr_dataset['x'], curr_dataset['theta']
            
            _, x_embedding_hf = self._generate_embedding_networks(x_hf=x_t)
            print("embedding hf:", x_embedding_hf)

            start_time = time.time()

            direct_flow = build_zuko_flow(theta_t, x_t, x_embedding_hf, 
                                        z_score_theta=self.z_score_theta, 
                                        z_score_x=self.z_score_x, 
                                        logit_transform_theta=self.logit_transform_theta_net,
                                        nf_type="NSF", 
                                        hidden_features=self.config_model['n_hidden_features'],
                                        num_transforms=self.config_model['n_transforms'],
                                        num_bins=self.config_model['n_bins'],
                                        prior=self.hf_prior)
            
            print("direct_flow:", direct_flow)
                        

            optimizer = torch.optim.Adam(direct_flow.parameters(), lr=self.config_model['learning_rate'])
        
            train_loader, val_loader = create_train_val_dataloaders(
                theta_t.to(self.device),
                x_t.to(self.device),
                validation_fraction = self.validation_fraction,
                batch_size=self.batch_size,
            )

            direct_flow = fit_conditional_normalizing_flow(
                direct_flow,
                optimizer,
                train_loader,
                val_loader,
                x_embedder=x_embedding_hf,
                early_stopping_patience=self.early_stopping,
                nb_epochs=self.max_num_epochs,
                print_every=1,
                clip_max_norm=self.clip_max_norm,
                plot_loss=False, 
                type_flow='NPE',
            )
            
            end_time = time.time()

            self._write_time_file("npe", start_time, end_time, n_simulations)
                        
            posterior = DirectPosterior(direct_flow, self.hf_prior) 
            posteriors.append(posterior)
            
            
        return posteriors
    
    
    # def _join_lf_hf(self, x_lf, x_hf, theta_lf, theta_hf):
    #     """
    #     Join low fidelity and high fidelity data.
    #     """
        # # Add zero's to the x-dimensions if they don't have the same shape
        # if x_lf.shape[1] < x_hf.shape[1]:
        #     x_lf = F.pad(x_lf, (0, x_hf.shape[1] - x_lf.shape[1]))

        # elif x_hf.shape[1] < x_lf.shape[1]:
        #     x_hf = F.pad(x_hf, (0, x_lf.shape[1] - x_hf.shape[1]))
            
        # print("x_lf padded shape:", x_lf.shape)
        # print("x_hf padded shape:", x_hf.shape)
        
        # Pretrain on joint LF and HF
        # Joint the lf and hf data for pretraining
        # theta_joint = torch.cat([theta_lf, theta_hf], dim=0) #theta_hf
        # x_joint     = torch.cat([x_lf, x_hf], dim=0) # x_hf # 
        # domain_labels = torch.cat([torch.zeros(len(theta_lf)),  
        #                             torch.ones(len(theta_hf))], dim=0).to(self.device) # torch.zeros(len(theta_hf)) 
        
        # Pretrain only on LF
        # theta_joint = theta_lf
        # x_joint     = x_lf
        # domain_labels = torch.zeros(len(theta_lf)).to(self.device)
        
        # return theta_joint, x_joint, domain_labels
    
    def _pretrain_model_on_lower_fidelities(self, lf_datasets, i):
        lf_start_time = time.time()
        
        pretrained_flow = []

        # Loop over all keys in lf_datasets, create each time a train dataloader and train the network, use this to retrain again.
        for fidelity, lf_data in lf_datasets.items():
            print(f"pretraining network with fidelity {fidelity}...")
                                         
            x_lf = lf_data[i]['x']
            theta_lf = lf_data[i]['theta']
            domain_labels = torch.zeros(len(theta_lf))
            
            n_lf_samples = lf_data[i]['n_samples']

            # Create train dataloader
            train_loader, val_loader = create_train_val_dataloaders(
                theta_lf.to(self.device),
                x_lf.to(self.device),
                validation_fraction=self.validation_fraction,
                domain_labels=domain_labels.to(self.device),
                batch_size=self.batch_size,
            )
            
            x_embedding_lf, _ = self._generate_embedding_networks(x_lf=x_lf)
            x_embedding_lf = x_embedding_lf.to(self.device)
            
            # If first time pretraining
            if pretrained_flow == []:
                pretrained_flow = build_zuko_flow(theta_lf, x_lf, x_embedding_lf,
                                        z_score_theta=self.z_score_theta, # Only z-score LF samples, not HF
                                        z_score_x=self.z_score_x,  
                                        logit_transform_theta=self.logit_transform_theta_net,
                                        nf_type="NSF_PRETRAIN", 
                                        hidden_features=self.config_model['n_hidden_features'],
                                        num_transforms=self.config_model['n_transforms'],
                                        num_bins=self.config_model['n_bins'],
                                        prior=self.lf_prior) #.to(self.device)
            else:
                pretrained_flow = build_zuko_flow(theta_lf, x_lf, x_embedding_lf,
                                        z_score_theta=self.z_score_theta, # Only z-score LF samples, not HF
                                        z_score_x=self.z_score_x,  
                                        logit_transform_theta=self.logit_transform_theta_net,
                                        nf_type="NSF_FINETUNE", 
                                        hidden_features=self.config_model['n_hidden_features'],
                                        num_transforms=self.config_model['n_transforms'],
                                        num_bins=self.config_model['n_bins'],
                                        base_model=pretrained_flow,
                                        prior=self.lf_prior) 
                       
            parameters = list(pretrained_flow.parameters())  
            pretrained_optimizer = torch.optim.Adam(parameters, lr=1e-4)  # Adjust learning rate
            
            pretrained_flow = fit_pretrained_conditional_normalizing_flow(
                pretrained_flow,
                pretrained_optimizer,
                train_loader,
                val_loader,
                x_dim_lf=self.x_dim_lf,  # Dimension of the low-fidelity data
                x_dim_hf=self.x_dim_hf,  # Dimension of the high-fidelity data
                x_dim_out=self.x_dim_out,  # Dimension of the input data
                theta_dim=self.theta_dim,
                nb_epochs=self.max_num_epochs,
                print_every=1,
                early_stopping_patience=self.early_stopping,
                clip_max_norm=self.clip_max_norm,
                plot_loss=False, 
                type_flow='MF-NPE (LF)',
                device=self.device,
            )
            lf_end_time = time.time()
        
        # Return the latest fidelity trained model
        base_model = copy.deepcopy(pretrained_flow)
            
        # return the n_lf_samples, which is the same for all lf_simulators!
        return base_model, lf_start_time, lf_end_time, x_embedding_lf, n_lf_samples
    
    
    def _convert_to_mf_data(self, data):
        
        # hf data: data where key == 'hf'
        hf_data = data['hf']  
        # lf data are all keys except data['hf']
        lf_data = {k: v for k, v in data.items() if k != 'hf'}

        # For all non-hf keys, make a lf_mf_dataset and hf_mf_dataset
        lf_mf_data = {}
        for fidelity, l_data in lf_data.items():
            lf_mf_data[fidelity] = [l_data[lf] for lf in l_data for _ in hf_data]
        
        hf_mf_data = [hf_data[hf] for _ in data['lf'] for hf in hf_data]
        
        return lf_mf_data, hf_mf_data


    def _train_mf_npe(self, x_hf, theta_hf, n_hf_samples, lf_mf_data, i):
        print("training low fidelity model...")

        # Pretrain on LF data. Give all data datasets, and loop over them inside this function.
        # Note, if multiple fidelities: it will return the pretrained flow of the latest model (e.g., mid-fidelity, if 3 fidelities used. otherwise, it's simply always the 'lf' model)
        pretrained_flow, lf_start_time, lf_end_time, x_embedding_lf, n_lf_samples = self._pretrain_model_on_lower_fidelities(lf_mf_data, i) # x_hf is passed for the embedding
        lf_pretrained_flow = copy.deepcopy(pretrained_flow) # Save the pretrained flow for returning later, to compare the distance between the LF and HF model.

        print("training high fidelity model...")
        hf_train_loader, hf_val_loader = create_train_val_dataloaders(
            theta_hf.to(self.device),
            x_hf.to(self.device),
            validation_fraction = self.validation_fraction,
            batch_size=self.batch_size,
        )
                
        hf_start_time = time.time()
        
        print("theta_hf shape:", theta_hf.shape, "x_hf shape:", x_hf.shape)
        print("x_embedding_hf shape:", x_embedding_lf(x_hf).shape)
                    
        hf_flow = build_zuko_flow(theta_hf, x_hf, x_embedding_lf, 
                                    z_score_theta=self.z_score_theta, 
                                    z_score_x=self.z_score_x,  
                                    logit_transform_theta=self.logit_transform_theta_net,
                                    nf_type="NSF_FINETUNE",
                                    hidden_features=self.config_model['n_hidden_features'],
                                    num_transforms=self.config_model['n_transforms'],
                                    num_bins=self.config_model['n_bins'],
                                    base_model=pretrained_flow, # Use the pretrained flow as a base model
                                    # additional_hf_transform=additional_hf_transform, 
                                    prior=self.hf_prior)


        hf_optimizer = torch.optim.Adam(list(hf_flow.parameters()), lr=self.config_model['learning_rate'])

        hf_flow = fit_conditional_normalizing_flow(
            hf_flow,
            hf_optimizer,
            hf_train_loader,
            hf_val_loader,
            x_embedder=x_embedding_lf,
            nb_epochs=self.max_num_epochs,
            print_every=1,
            early_stopping_patience=self.early_stopping,
            clip_max_norm=self.clip_max_norm,
            plot_loss=False, 
            type_flow='MF-NPE (HF)',
        )
        
        hf_end_time = time.time()
        self._write_time_file("mf_npe", hf_start_time, hf_end_time, n_hf_samples, lf_start_time, lf_end_time, n_lf_samples)
        
        mf_posterior = DirectPosterior(hf_flow, self.hf_simulator.prior())
        
        
        # lf_posterior are the pretrained models (or model, if only 1 lf simulator)
        if isinstance(self.lf_simulator, dict): # i.e., if there are multiple lf simulators
            pretrained_posterior = {}
            for fidelity, sim in self.lf_simulator.items():
                pretrained_posterior[fidelity] = DirectPosterior(lf_pretrained_flow, sim.prior())
        else:
            pretrained_posterior = DirectPosterior(lf_pretrained_flow, self.lf_simulator.prior())
            
        return pretrained_posterior, mf_posterior
        
    
    def run_mf_npe(self, data):
        '''
        # The finetuned model gave worse accuracy. So I'm keeping the code but it's not used.
        This method generates all possible combinations of the low-fidelity (LF) and high-fidelity (HF) datasets
        for the multi-fidelity model. It then trains a base model using the LF data and fine-tunes it using the HF data.

        Parameters:
        data (dict): Dictionary containing the low-fidelities data and high-fidelity data, with keys like {'lf':..., 'mid'..., 'hf':...}

        Returns:
        Tuple: A tuple containing the multi-fidelity posteriors and the multi-fidelity flows.
        
        '''
        mf_posteriors = []
        pretrained_posteriors = []
        
        # only for 1 (the lowest) lf dataset
        lf_mf_data, hf_mf_data = self._convert_to_mf_data(data)

        # The length of hf_mf_data is the same as lf_mf_data and is a permutation of both of the datasets.
        for i, _ in enumerate(hf_mf_data):    
            
            x_hf, theta_hf = hf_mf_data[i]['x'], hf_mf_data[i]['theta']
            n_hf_samples = hf_mf_data[i]['n_samples']

            pretrained_posterior, mf_posterior = self._train_mf_npe(x_hf, theta_hf, n_hf_samples, lf_mf_data, i)

            pretrained_posteriors.append(pretrained_posterior)
            mf_posteriors.append(mf_posterior)
                    
        return mf_posteriors, pretrained_posteriors
    
    
    

    def plot_sweeps_single_figure(self, results_summary, n_hf_sims, out_dir,
                                target_aspect=1.6,  # width/height of the whole figure
                                min_cols=2, max_cols=10,
                                panel_w=2.6, panel_h=2.2):
        """
        Make ONE figure with an auto-sized grid for any number of sweeps.
        - target_aspect ~ 1.6 gives a landscape figure; tweak for tall/wide.
        - min_cols/max_cols: clamp columns so panels don't get too tiny.
        """
        import math
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import os
    
        if not results_summary:
            return

        n = len(results_summary)
        # Choose columns near sqrt(n), then clamp
        n_cols = int(round(math.sqrt(n)))
        n_cols = max(min_cols, min(max_cols, n_cols))
        n_rows = math.ceil(n / n_cols)

        # Figure size from panel dims
        fig_w = max(6, n_cols * panel_w)
        fig_h = max(5, n_rows * panel_h)
        # Optionally nudge toward a target aspect by adding columns if very tall
        while (fig_w / fig_h) < target_aspect and n_cols < max_cols and (n_rows-1) * n_cols >= n:
            # Try to widen (only if it doesn’t force new empty row)
            n_cols += 1
            n_rows = math.ceil(n / n_cols)
            fig_w = max(6, n_cols * panel_w)
            fig_h = max(5, n_rows * panel_h)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                                sharex=True, sharey=True, constrained_layout=False)

        # normalize axes to 2D
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        # Shared color scale across all panels
        all_lp = np.concatenate([d["logprobs"].detach().cpu().numpy() for d in results_summary])
        vmin, vmax = float(all_lp.min()), float(all_lp.max())
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap()

        last_sc = None
        for i, d in enumerate(results_summary):
            r, c = divmod(i, n_cols)
            ax = axes[r, c]
            th = d["theta_hf"].detach().cpu().numpy()
            lp = d["logprobs"].detach().cpu().numpy()
            last_sc = ax.scatter(th[:, 0], th[:, 1], c=lp, s=12, alpha=0.7, cmap=cmap, norm=norm)
            ax.set_xlim(0, 3)
            ax.set_ylim(0.1, 0.6)
            ax.grid(True, linewidth=0.3, alpha=0.5)
            ax.set_title(fr"$\kappa$={d['kappa']}, $\alpha$={d['alpha']}, init={d['init']}", fontsize=8)

        # Hide any empty axes
        total_slots = n_rows * n_cols
        for j in range(n, total_slots):
            r, c = divmod(j, n_cols)
            axes[r, c].axis("off")

        # Label outer edges only
        for r in range(n_rows):
            axes[r, 0].set_ylabel(r"$\sigma$")
        for c in range(n_cols):
            axes[-1, c].set_xlabel(r"$\mu$")

        # Shared colorbar
        if last_sc is not None:
            cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
            cbar.set_label("Log Probability")

        fig.suptitle(f"BO Samples (all {n} sweeps)", y=0.995, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}bo_samples_logprob_GRID_{n_hf_sims}_ALL.png", dpi=200)
        plt.close(fig)


    

   

    def plot_sweeps_by_alpha(self, results_summary, kappa_list, alpha_list, init_points_list,
                                n_hf_sims, out_dir,
                                panel_w=2.6, panel_h=2.2):
        """
        Grid layout:
        - Columns = alpha (increasing left to right)
        - Rows    = (kappa, init) combinations stacked along y-axis
        Each subplot = one config (kappa, alpha, init)
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import numpy as np
        import os
        n_rows = len(kappa_list) * len(init_points_list)
        n_cols = len(alpha_list)

        fig_w = max(8, n_cols * panel_w)
        fig_h = max(6, n_rows * panel_h)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                                sharex=True, sharey=True)

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        # Shared color scale across all panels
        all_lp = np.concatenate([d["logprobs"].detach().cpu().numpy() for d in results_summary])
        vmin, vmax = float(all_lp.min()), float(all_lp.max())
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")

        last_sc = None
        for r_kappa, kappa_val in enumerate(kappa_list):
            for r_init, init_pts in enumerate(init_points_list):
                row = r_kappa * len(init_points_list) + r_init
                for c, alpha_val in enumerate(alpha_list):
                    ax = axes[row, c]

                    # find this exact config
                    conf = next((d for d in results_summary
                                if d["kappa"] == kappa_val
                                and d["alpha"] == alpha_val
                                and d["init"] == init_pts), None)
                    if conf is None:
                        ax.axis("off")
                        continue

                    th = conf["theta_hf"].detach().cpu().numpy()
                    lp = conf["logprobs"].detach().cpu().numpy()
                    last_sc = ax.scatter(th[:, 0], th[:, 1], c=lp, alpha=0.6, s=12,
                                        cmap=cmap, norm=norm)

                    ax.set_xlim(0, 3)
                    ax.set_ylim(0.1, 0.6)
                    ax.grid(True, linewidth=0.3, alpha=0.5)

                    # only label outer edges
                    if row == n_rows - 1:
                        ax.set_xlabel(r"$\mu$")
                    if c == 0:
                        ax.set_ylabel(r"$\sigma$")

                    # compact title
                    ax.set_title(fr"$\kappa$={kappa_val}, init={init_pts}, $\alpha$={alpha_val}", fontsize=7)

        # Shared colorbar
        if last_sc is not None:
            cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
            cbar.set_label("Log Probability")

        fig.suptitle(f"BO Samples: grid with α along x-axis, (κ, init) along y-axis "
                    f"(total {len(results_summary)} sweeps)", y=0.995, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}bo_samples_logprob_GRID_byAlphaKappaInit_{n_hf_sims}.png", dpi=200)
        plt.close(fig)

    
    def run_parameter_sweep(self, n_hf_sims, black_box_function, pbounds, x_hf_simulated):
        # Run parameter sweep and plot results
        kappa_list = [0.001, 0.1]
        alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9] # # alpha_list = [0.001, 0.01, 0.1, 1.0]
        init_points_list = [5, 10, 20]
        
        results_summary = []   # to keep track across configs
        
        for kappa_val in kappa_list:
            for alpha_val in alpha_list:
                for init_pts in init_points_list:
                    theta_hf, x_hf, logprobs = self._run_single_round_bo(kappa_val, alpha_val, n_hf_sims, init_pts, black_box_function, pbounds, x_hf_simulated)
                    
                    # ----------- collect for giant plot at the end ----------
                    results_summary.append({
                        "theta_hf": theta_hf,                 # (N_i, 2)
                        "logprobs": logprobs,                 # (N_i,)
                        "kappa": kappa_val,
                        "alpha": alpha_val,
                        "init": init_pts,
                        "i_kappa": kappa_list.index(kappa_val),
                        "i_alpha": alpha_list.index(alpha_val),
                        "i_init": init_points_list.index(init_pts),
                    })
        
        dir_plots = f"{self.main_path}/ob_sweep/"
        #self.plot_sweeps_single_figure(results_summary, n_hf_sims, dir_plots)
        self.plot_sweeps_by_alpha(results_summary, kappa_list, alpha_list, init_points_list, n_hf_sims, dir_plots)       
        
        return None # Do not continue to NPE after BO sweep, just for plotting and exploring space                
                    
    
    
    def _run_single_round_bo(self, kappa_val, alpha_val, n_hf_sims, init_pts, black_box_function, pbounds, x_hf_simulated):
        run_bo = True
        
        acquisition_function = acquisition.UpperConfidenceBound(kappa=kappa_val)
        # acquisition_function = acquisition.ExpectedImprovement(xi=0.0)

        optimizer = BayesianOptimization(
            f=black_box_function, # I removed the lambda here
            pbounds=pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
            acquisition_function=acquisition_function, # give your custom acquisition function here
        )
        
        # increase, because optimizer needs to be able to handle more noise
        optimizer.set_gp_params(alpha=alpha_val, n_restarts_optimizer=5)

        
        save_dir = f"{self.main_path}/train_data/"
        name = f"bo_samples_{n_hf_sims}_{kappa_val}_{alpha_val}_{init_pts}" # f"bo_simulations_{n_hf_sims}.p"

        if run_bo:
            #  maximize method is simply a wrapper around the methods suggest, probe, and register,
            # Where it suggests the next point to use
            optimizer.maximize(
                init_points=init_pts, # put back to 10 later
                n_iter=n_hf_sims,
            )
            
            
            y_mean, y_std = optimizer._gp.predict(np.linspace(-10, 10, 1000).reshape(-1, 1), return_std=True)

            # # Plot the GP posterior
            # plt.figure(figsize=(6, 4))
            # plt.plot(np.linspace(-10, 10, 1000), y_mean, label='Mean')
            # plt.fill_between(np.linspace(-10, 10, 1000), y_mean - 1.96 * y_std, y_mean + 1.96 * y_std, alpha=0.2, label='95% CI')
            # plt.title(f"GP Posterior (Kappa: {kappa_val}, Alpha: {alpha_val}, Init: {init_pts})")
            # plt.xlabel("x")
            # plt.ylabel("f(x)")
            # plt.legend()
            # plt.grid(True)
            # plt.savefig(f"{self.main_path}/gp_posteriors/gp_posterior_{kappa_val}_{alpha_val}_{init_pts}.png")
            # plt.close()
            

            # Save BO samples
            theta = []
            logprobs = []
            for _, res in enumerate(optimizer.res):
                # Convert the params to a nested torch array
                # torch_val = [res['params']['mu'], res['params']['sigma']]
                torch_val = [param for param in res['params'].values()]
                torch_target = res['target']
                theta.append(torch_val)
                logprobs.append(torch_target)

            # Torchify
            theta_hf = torch.tensor(theta, dtype=torch.float32)       
            x_hf = torch.stack(x_hf_simulated).squeeze()
            logprobs = torch.tensor(logprobs, dtype=torch.float32)

            # Save as dict pickle
            dump_pickle(save_dir, name, {
                'theta_hf': theta_hf,
                'x_hf': x_hf,
                'logprobs': logprobs
            })
        else:
            # load saved data: keeps everything fixed
            open_pickles_simulations = open(f"{save_dir}/{name}", "rb")
            bo_samples = pickle.load(open_pickles_simulations)
            theta_hf = bo_samples["theta_hf"]
            x_hf = bo_samples["x_hf"]
        

        # Plot the theta_hf samples on axis 1 and 2, and the color should be based on logprobs
        plt.figure(figsize=(6, 4))
        scatter = plt.scatter(theta_hf[:, 0].numpy(), theta_hf[:, 1].numpy(), c=logprobs.numpy(), alpha=0.6)
        plt.title(f"BO Samples (Total: {len(theta_hf)})")
        plt.xlabel(r"$\mu$")
        plt.ylabel(r"$\sigma$")
        # ranges fixed
        plt.xlim(0, 3)
        plt.ylim(0.1, 0.6)
        plt.grid(True)
        plt.colorbar(scatter, label="Log Probability")
        
        # save the plot
        dir_plots = f"{self.main_path}/ob_sweep/"
        if not os.path.exists(dir_plots):
            os.makedirs(dir_plots)
        plt.savefig(f"{dir_plots}bo_samples_logprob_{n_hf_sims}_{kappa_val}_{alpha_val}_{init_pts}.png")
        plt.close()

        # Pairplot of theta_hf samples
        _ = pairplot(
            theta_hf,
            limits=self.hf_simulator.parameter_ranges(self.theta_dim), #[[0.0, 3.3], [0.0, 0.7]],  # limits=[[0.1, 3.], [.1, .6]],
            figsize=(5, 5),
            labels=[r"$\theta_1$", r"$\theta_2$"]   
        )
        plt.savefig(f"pairplot_bo_samples_{n_hf_sims}.png")
        plt.close()
        
        return theta_hf, x_hf, logprobs

    

    def run_bo_npe(self, simulations, true_xen, true_thetas, use_mf_npe=False):        
        posteriors = []
        FLOOR  = -1e8           # fallback for non-finite vals
        lf_mf_data, hf_mf_data = self._convert_to_mf_data(simulations)
        
  
        # Loop over batch iterations
        for i, _ in enumerate(hf_mf_data):   
            # 1. Get all x-theta pairs
            lf_x, lf_theta = lf_mf_data['lf'][i]['x'], lf_mf_data['lf'][i]['theta']
            print("lf_x shape:", lf_x.shape, "lf_theta shape:", lf_theta.shape)

            # Get the number of HF simulations
            n_hf_sims = hf_mf_data[i]['n_samples']
            
            # 2. Run NLE on LF samples
            lf_inference = NLE(self.lf_prior, density_estimator='zuko_nsf')
            lf_posterior_net = lf_inference.append_simulations(lf_theta, lf_x).train()
            lf_likelihood = lf_inference.build_posterior() # .set_default_x(true_x) # loop over multiple true_xen
            x_hf_simulated = []
            
            # Blackbox function should be the LF likelihood
            def black_box_function(**params):
                theta = torch.tensor([[v for v in params.values()]], dtype=torch.float32)
                
                try:
                    # Generate an x with the mu and sigma ran over here
                    x_hf = self.hf_simulator.simulator(theta)
                    
                    # potential ≈ log p(θ|x) + const(x)
                    unnorm_logprob = lf_likelihood.potential(theta, x=x_hf)
                except Exception:
                    return FLOOR
                
                # Make sure it doesn't break in very low fidelity region
                if not np.isfinite(unnorm_logprob):
                    return FLOOR

                # Logprior
                logprior = self.lf_prior.log_prob(theta)     
                # log p(x|θ) = log p(θ|x) - log p(θ) + const(x). 
                logp = (unnorm_logprob - logprior)
                
                # Save x_hf. Theta's are known since it's the output of OB (checked)
                x_hf_simulated.append(x_hf)
                # Return negative log probability
                return float(-logp.item())

            
            # in case the prior bounds are None (like in SIR task)
            pbounds = {}
            for p, (param, ranges) in enumerate(self.config_data['all_prior_ranges'].items()):
                if p >= self.theta_dim:
                    break
                low, high = ranges
                if low is None or high is None:  # special case
                    pbounds[param] = (0.0, 1.0)
                    warnings.warn(
                        "Using default pbounds since prior bounds are None.",
                        category=UserWarning,   # or a custom subclass
                        stacklevel=2            # points warning at the caller
                    )
                else:  # shrink interval inward
                    pbounds[param] = (low + 0.01, high - 0.01)

            # Get prior bounds
            print("pbounds:", pbounds)
            
            RUN_SWEEP = False
            
            if not RUN_SWEEP:
                # Run single BO with fixed params
                kappa_val = 0.01
                alpha_val = 0.7
                init_pts = 10
                
                theta_hf, x_hf, logprobs = self._run_single_round_bo(kappa_val, alpha_val, n_hf_sims, init_pts, black_box_function, pbounds, x_hf_simulated)
                

                magnitude_uncertanty = logprobs.max() - logprobs.min()
                # Plot magnitude of the uncertainty
                plt.figure(figsize=(6, 4))
                scatter = plt.scatter(theta_hf[:, 0].numpy(), theta_hf[:, 1].numpy(), c=logprobs.numpy(), alpha=0.6)
                plt.title(f"BO Samples (Total: {len(theta_hf)})")
                plt.xlabel(r"$\mu$")
                plt.ylabel(r"$\sigma$")
                # ranges fixed
                plt.xlim(0, 3)
                plt.ylim(0.1, 0.6)
                plt.grid(True)
                plt.colorbar(scatter, label="Log Probability")
                plt.savefig(f"bo_samples_logprob_{n_hf_sims}_FINAL.png")

                # raise ValueError("Do not continue to NPE after BO, just for plotting and exploring space")
            else:
                # Run parameter sweep and plot results
                self.run_parameter_sweep(n_hf_sims, black_box_function, pbounds, x_hf_simulated)
                raise ValueError("Do not continue to NPE after BO sweep, just for plotting and exploring space")
            
            
            # # Plot x_hf samples over time
            # plt.figure(figsize=(6, 4))
            # for i in range(min(20, x_hf.shape[0])):  #
            #     plt.plot(x_hf[i].numpy(), alpha=0.6)
            # plt.title(f"BO HF Simulated Data (Total: {len(x_hf)})")
            # plt.xlabel("Time")
            # plt.ylabel("x_hf")
            # plt.grid(True)
            # plt.savefig(f"bo_x_hf_samples_{n_hf_sims}.png")
            # plt.close()
            
            # # Get samples from prior and simulate x_hf for comparison
            # n_prior_samples = 100
            # theta_prior = self.hf_prior.sample((n_prior_samples,))
            # x_hf_prior = self.hf_simulator.simulator(theta_prior)   
            # # Plot x_hf samples over time
            # plt.figure(figsize=(6, 4))
            # for i in range(min(20, x_hf_prior.shape[0])):  #
            #     plt.plot(x_hf_prior[i].numpy(), alpha=0.6)
            # plt.title(f"Prior HF Simulated Data (Total: {len(x_hf_prior)})")
            # plt.xlabel("Time")
            # plt.ylabel("x_hf")
            # plt.grid(True)
            # plt.savefig(f"prior_x_hf_samples_{n_hf_sims}.png")
            
            # Make the same parameters as mf_npe 
            # density_estimator_build_fun = posterior_nn(
            #             model="zuko_nsf",
            #             hidden_features=50,
            #             num_transforms=5,
            #             z_score_theta="transform_to_unconstrained",  # Transforms parameters to unconstrained space
            #             x_dist=self.lf_prior  # For NPE, this specifies bounds for parameters (internally called 'x')
            #             )
            
            # Make samples a categorical distribution for SBI
            weights = torch.ones(len(theta_hf)) / len(theta_hf)
            proposal_cat = Categorical(weights)
            
            # Use mf_NPE to train now
            if use_mf_npe:
                print("using MF NPE in combination with BO")
                                
                # Train a lf_flow
                lf_flow, lf_start_time, lf_end_time, x_embedding_lf, n_lf_samples = self._pretrain_model_on_lower_fidelities(lf_mf_data, i)                
                prior = self.hf_prior                
                prior, _, prior_returns_numpy = process_prior(prior)
                def lf_builder(batch_theta, batch_x):
                    return copy.deepcopy(lf_flow)
                
                pretrained_net = NPE_C(prior, show_progress_bars=False, density_estimator=lf_builder)
                
                # Train on hf data with proposal
                density_estimator = pretrained_net.append_simulations(theta_hf, x_hf, proposal=proposal_cat).train(force_first_round_loss=True) # , proposal=proposal_cat
                posterior = pretrained_net.build_posterior(density_estimator) 
            else:
                # Run NPE now for those samples.
                inference = NPE_C(self.hf_prior, show_progress_bars=False, density_estimator="zuko_nsf")     
                # Train on the hf data                
                density_estimator = inference.append_simulations(theta_hf, x_hf, proposal=proposal_cat).train() # Note, proposal are theta_hf, there will be a warning appearing because the common way are samples from prior.
                #posterior = inference.build_posterior() 
                inference.build_posterior(density_estimator)
                                                
                
            for x_i, true_x in enumerate(true_xen):
                # sample from posterior
                samples = posterior.sample((1000,), x=true_x)

                print("samples", samples)
                print("samples shape", samples.shape)
                
                # print max value from samples
                print("max sample", samples.max(dim=0).values)
                print("min sample", samples.min(dim=0).values)
                
                # If max_samples is bigger than prior, throw an error
                if (samples.max(dim=0).values > torch.tensor([3.0, 0.6])).any():
                    raise ValueError("Samples are outside prior range. Try to increase the bounds of the prior.")

                if (samples.min(dim=0).values < torch.tensor([0.1, 0.1])).any():
                    raise ValueError("Samples are outside prior range. Try to increase the bounds of the prior.")

                # Plot probability density with these samples
                _ = pairplot(
                    samples,
                    limits=[[0.0, 3.3], [0.0, 0.7]],  # limits=[[0.1, 3.], [.1, .6]],
                    points=true_thetas[x_i],
                    figsize=(5, 5),
                    labels=[r"$\theta_1$", r"$\theta_2$"]
                )
                
                # Save pairplot
                plt.savefig(f"pairplot_{x_i}.png")
                plt.close()

            posteriors.append(posterior)
        
        return posteriors

    
    def _run_active_mf_tsnpe_single_xo(self, lf_flow, lf_start_time, lf_end_time, n_lf_samples, hf_data, x_o, theta_o, xo_index,
                                      active_learning_pct=.8,
                                        n_rounds=5,
                                        n_hf_train_samples=1000,
                                        n_ensemble_members=5, 
                                        plot_thetas=False,
                                        save_posteriors=True,
                                        seed=None):
        
        if n_hf_train_samples > 10000:
            n_theta_samples = 25000 # Problems in accuracy. Try .9 and see how that works...
        elif n_hf_train_samples > 1000: 
            n_theta_samples = 2500
        elif n_hf_train_samples <= 1000:
            n_theta_samples = 250
        else:
            raise ValueError(f"n_samples: {n_hf_train_samples} is not in the right range")
        

        prior = self.hf_prior                
        prior, _, prior_returns_numpy = process_prior(prior)
        
        total_start_time = time.time()


        def lf_builder(batch_theta, batch_x):
            return copy.deepcopy(lf_flow)

        n_ensemble_members = n_ensemble_members
        ensemble = [NPE(prior, density_estimator=lf_builder) for _ in range(n_ensemble_members)]

        proposal = prior 
        
        active_posteriors = []
        n_rounds = n_rounds
        active_learning_pct = active_learning_pct
        n_theta_samples = n_theta_samples

        n_hf_samples = len(hf_data['x'])
        n_hf_samples_round = n_hf_samples // n_rounds
        n_hf_samples_prop = int(active_learning_pct * n_hf_samples_round)
        n_hf_samples_active = n_hf_samples_round - n_hf_samples_prop

        hf_start_time = time.time()
        
        for round in range(n_rounds):
            one_round_start_time = time.time()
            
            if round == 0:
                theta_hf = proposal.sample((n_hf_samples_round,))
            else:
                theta_hf = proposal.sample((n_hf_samples_prop,))
                theta_samples = self.hf_prior.sample((n_theta_samples,))
                post_samples = []
                for posterior in ensemble_posteriors:
                    with torch.no_grad():
                        post_samples.append(posterior.log_prob(theta_samples))
                post_samples = torch.stack(post_samples)
                ensemble_var = post_samples.var(dim=0)
                theta_ordered = theta_samples[torch.argsort(ensemble_var, descending=True)]
                theta_active = theta_ordered[:n_hf_samples_active]
                theta_hf = torch.cat([theta_hf, theta_active])

            if self.task == 'task1' or self.task == 'task4' or self.task == 'task5':
                x_hf = self.hf_simulator.simulator(theta_hf)
            elif self.task == 'task2':
                cell, _ = self.hf_simulator._jaxley_neuron()
                integrator_fn = Integrator(cell, self.config_data)
                if round == 0:
                    # Shape of theta_hf
                    # Approximatelly 0.05-0.07% has to be resampled. which is, for 2000 samples, +- 150 samples
                    x_hf, theta_hf, _ = self.hf_simulator.simulator(theta_hf, 
                                                                        integrator_fn, # param and noise param lambda inside
                                                                        allow_resampling_invalid_samples=True,
                                                                        proposal=proposal)
                else:
                    x_hf, theta_hf, _ = self.hf_simulator.simulator(
                        theta_hf,
                        integrator_fn,
                        allow_resampling_invalid_samples=True,
                        active_learning_list=theta_ordered
                    )
            elif self.task == 'task3':
                x_hf, theta_hf, _ = self.hf_simulator.summary_statistics(n_hf_samples_round, plot=True)
            elif self.task == 'task6' or self.task == 'task7':
                x_hf, theta_hf, _ = self.hf_simulator.summary_statistics(n_hf_samples_round, prior=proposal)
            else:
                raise ValueError("Task not implemented in active learning")

            ensemble_density_estimators,  ensemble_posteriors = [], []
            for inference in ensemble:
                density_estimator = inference.append_simulations(
                    theta_hf, x_hf, proposal
                ).train(force_first_round_loss=True)
                posterior = inference.build_posterior(density_estimator).set_default_x(x_o) 
                ensemble_density_estimators.append(density_estimator)
                ensemble_posteriors.append(posterior)

            ensemble_posterior = EnsemblePosterior(ensemble_posteriors)
            active_posteriors.append(ensemble_posterior)
            
            if self.task == 'task2':
                accept_reject_fn = get_density_thresholder(ensemble_posterior, quantile=1e-3) # was 1e-6, lower for test
            else:
                accept_reject_fn = get_density_thresholder(ensemble_posterior, quantile=1e-6)

            # Correct for the fact that we are not sampling from prior
            # weights = torch.ones(len(theta_hf)) / len(theta_hf)
            # proposal_cat = Categorical(weights)
            # proposal_cat = process_prior(proposal_cat)
            # print("weights", weights)
            # print("proposal" , proposal)
            
            proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
            

            
            one_round_end_time = time.time()

        hf_end_time = time.time()
        total_end_time = time.time()
        
        # Save the training duration in a txt file  
        self._write_time_file("active_mf_tsnpe", hf_start_time, hf_end_time, hf_data['n_samples'], lf_start_time, lf_end_time, n_lf_samples, total_start_time, total_end_time, one_round_start_time, one_round_end_time, n_rounds)

        if save_posteriors:
            save_dir = f"{self.main_path}/non_amortized_posteriors/"
            name = f"a_mf_tsnpe_LF{n_lf_samples}_HF{hf_data['n_samples']}_xo{xo_index}_seed{seed}_{self.CURR_TIME}.p" 
            dump_pickle(save_dir, name, {
                'posterior': ensemble_posterior,
                'true_x': x_o,
                'true_theta': theta_o,
                'type_estimator': 'a_mf_tsnpe',
                'n_rounds': n_rounds,
                'n_simulations': [n_lf_samples, hf_data['n_samples']],
            })
            
        return ensemble_posterior

    
    def run_active_mf_tsnpe(self, 
                        data,
                        true_xen,
                        true_thetas,
                        active_learning_pct=.8,
                        n_rounds=5,
                        n_theta_samples=250, # Not used: inferred dependent on n_hf_samples
                        n_ensemble_members=5, 
                        plot_thetas=False,
                        save_posteriors=True,
                        seed=None
                 ):
        '''
        This method is not amortized, which means that we have to train the model for each new x_o that we want to evaluate on.
        We return a nested array of posteriors, where the inner array corresponds to posteriors over the different xen,
        and the outer array corresponds to the different number of simulations used to train the model for model comparison.
        '''
        posteriors = []   # The posteriors across different batch sizes
        
        # only for 1 (the lowest) lf dataset
        lf_mf_data, hf_mf_data = self._convert_to_mf_data(data)
                 
        for hf_index, hf_samples in enumerate(hf_mf_data):
            non_amortized_posteriors = []
            
            curr_hf_data = hf_mf_data[hf_index]
            n_hf_samples = curr_hf_data['n_samples'] # Is the same as curr_hf_data ?

            # Note: We use the same n_lf_samples for all lower fidelity models
            lf_flow, lf_start_time, lf_end_time, x_embedding_lf, n_lf_samples = self._pretrain_model_on_lower_fidelities(lf_mf_data, hf_index)
                        
            # run over all xen that you want to evaluate on 
            for i_xo, _ in enumerate(tqdm(true_xen)):
                curr_xo = true_xen[i_xo]
                curr_theta = true_thetas[i_xo]
   
                ensemble_posterior =  self._run_active_mf_tsnpe_single_xo(lf_flow=lf_flow,
                                                                          lf_start_time=lf_start_time,
                                                                          lf_end_time=lf_end_time,
                                                                          n_lf_samples=n_lf_samples,
                                                                        hf_data=curr_hf_data, 
                                                                        x_o=curr_xo, 
                                                                        theta_o=curr_theta, 
                                                                        xo_index=i_xo,
                                                                        active_learning_pct=active_learning_pct,
                                                                        n_rounds=n_rounds,
                                                                        n_hf_train_samples=n_hf_samples,
                                                                        n_ensemble_members=n_ensemble_members, 
                                                                        plot_thetas=plot_thetas,
                                                                        save_posteriors=save_posteriors,
                                                                        seed=seed)

                non_amortized_posteriors.append(ensemble_posterior)
            posteriors.append(non_amortized_posteriors)

        return posteriors 
    
    
    def run_mf_tsnpe(self, 
                        data,
                        true_xen,
                        true_thetas,
                        n_rounds=5,
                        plot_thetas=False,
                        save_stuff=True,
                        save_posteriors=True,
                        seed=None
                 ):
        '''
         mf snpe
        '''
        posteriors = []   
        
        # only for 1 (the lowest) lf dataset
        lf_mf_data, hf_mf_data = self._convert_to_mf_data(data)
                 
        for i, data in enumerate(hf_mf_data):
            non_amortized_posteriors = []
            
            """ low-fidelity pre-training is amortized """
            theta_hf, x_hf = hf_mf_data[i]['theta'], hf_mf_data[i]['x']

            lf_flow, lf_start_time, lf_end_time, x_embedding_lf, n_lf_samples = self._pretrain_model_on_lower_fidelities(lf_mf_data, i)
            
            # run over all xen that you want to evaluate on 
            for i_xo, x_o in enumerate(tqdm(true_xen)):
                
                total_start_time = time.time()
                
                """ set up for sbi """
                prior = self.hf_prior


                """ high fidelity training loop """
                # NPE requires a function that return a density estimator
                def lf_builder(batch_theta, batch_x):
                    # just return a copy of the trained flow
                    return copy.deepcopy(lf_flow)

                # create ensemble of inference objects
                inference = NPE(prior, density_estimator=lf_builder) 

                proposal = prior 
                n_hf_samples = len(hf_mf_data[i]['x']) # total number of hf samples
                n_hf_samples_round = n_hf_samples // n_rounds # hf sample per round
                
                print('n_rounds: ', n_rounds)
                print('n_hf_samples: ', n_hf_samples)
                print('n_hf_samples_round: ', n_hf_samples_round)
                
                hf_start_time = time.time()
                
                for round in range(n_rounds):
                    print("round", round)
                    
                    one_round_start_time = time.time()
                    theta_hf = proposal.sample((n_hf_samples_round,))

                    if self.task == 'task1' or self.task == 'task4' or self.task == 'task5':
                        # generate new simulations
                        x_hf = self.hf_simulator.simulator(theta_hf)
                    elif self.task == 'task6' or self.task == 'task7':
                        x_hf, theta_hf, _ = self.hf_simulator.summary_statistics(n_hf_samples_round, prior=proposal)
                    elif self.task == 'task2':
                        cell, _ = self.hf_simulator._jaxley_neuron()
                        integrator_fn = Integrator(cell, self.config_data)
                        # Sample eacht ime from the proposal in the  method
                        # Method: Uses the samples from the proposal, if not enough samples, it samples new samples from the proposal
                        x_hf, theta_hf, _ = self.hf_simulator.simulator(theta_hf, 
                                                                        integrator_fn,#lambda params, noise_params: simulate_neuron(params, noise_params, cell), 
                                                                        allow_resampling_invalid_samples=True, # Resample from active_learning list if invalid number
                                                                        proposal=proposal, # Sample from the proposal for all rounds
                                                                        )

                        print(f"generated len(x_hf)={len(x_hf)} new high-fidelity samples")
                        # Check if x_active has the right size
                        assert (len(x_hf) == len(theta_hf)), f"len(x_hf)={len(x_hf)} != len(theta_hf)={len(theta_hf)}"
                    else:
                        raise ValueError("Task not implemented in active learning")
                    
                    
                    # retrain your density estimator 
                    density_estimator = inference.append_simulations(
                        theta_hf, x_hf, proposal
                    ).train(force_first_round_loss=True)

                    non_amortized_posterior = inference.build_posterior(density_estimator).set_default_x(x_o)  # 
                    
                    if self.task == 'task2':
                        accept_reject_fn = get_density_thresholder(non_amortized_posterior, quantile=1e-3) # was 1e-6, lower for test
                    else:
                        accept_reject_fn = get_density_thresholder(non_amortized_posterior, quantile=1e-6)

                    proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
                    
                    one_round_end_time = time.time()
                
                hf_end_time = time.time()

                total_end_time = time.time()
                print(f"Time taken for {n_rounds} rounds for 1 xo: {total_end_time - total_start_time} seconds")
                print(f"Time taken for LF training: {lf_end_time - lf_start_time} seconds")
                print(f"Time taken for HF training: {hf_end_time - hf_start_time} seconds")
                print(f"Time taken for 1 round: {one_round_end_time - one_round_start_time} seconds")
                print("Number of HF samples: ", hf_mf_data[i]['n_samples'])
                print("Number of LF samples: ", lf_mf_data['lf'][i]['n_samples'])
                
                
                self._write_time_file("mf_tsnpe", hf_start_time, hf_end_time, hf_mf_data[i]['n_samples'], lf_start_time, lf_end_time, lf_mf_data['lf'][i]['n_samples'], total_start_time, total_end_time, one_round_start_time, one_round_end_time, n_rounds)
                
                # Save the posterior in a pickle file! 
                if save_posteriors:
                    ensemble_post_save = {
                        'posterior': non_amortized_posterior, 
                        'true_x': x_o,
                        'true_theta': true_thetas[i_xo],
                        'type_estimator': 'mf_tsnpe',
                        'n_rounds': n_rounds,
                        'n_simulations': [lf_mf_data['lf'][i]['n_samples'], hf_mf_data[i]['n_samples']],
                    }
                    # Save a pickle with posterior samples (for 1 xo)
                    save_dir = f"{self.main_path}/non_amortized_posteriors/"
                    name = f"mf_tsnpe_LF{lf_mf_data['lf'][i]['n_samples']}_HF{hf_mf_data[i]['n_samples']}_xo{i_xo}_seed{seed}_{self.CURR_TIME}.p" 
                    dump_pickle(save_dir, name, ensemble_post_save)
                    
                non_amortized_posteriors.append(non_amortized_posterior)
            posteriors.append(non_amortized_posteriors)

        return posteriors 
    
    
    
    def run_tsnpe(self, hf_data, 
                        true_xen,
                        true_thetas,
                        n_rounds=5,
                        plot_thetas=False,
                        save_stuff=True,
                        save_posteriors=True,
                        seed=None
                 ):
        '''
         TSNPE SBI
        '''
        posteriors = []   
        
                 
        for i, data in enumerate(hf_data):
            non_amortized_posteriors = []
            
            # run over all xen that you want to evaluate on 
            for i_xo, x_o in enumerate(tqdm(true_xen)):
                
                total_start_time = time.time()
                
                """ set up for sbi """
                prior = self.hf_prior

                """ high fidelity training loop """

                # create ensemble of inference objects
                inference = NPE(prior, density_estimator='zuko_nsf') 

                proposal = prior 
                n_hf_samples = len(hf_data[i]['x']) # total number of hf samples
                n_hf_samples_round = n_hf_samples // n_rounds # hf sample per round
                
                print('n_rounds: ', n_rounds)
                print('n_hf_samples: ', n_hf_samples)
                print('n_hf_samples_round: ', n_hf_samples_round)
                
                hf_start_time = time.time()
                
                for round in range(n_rounds):
                    print("round", round)
                    
                    one_round_start_time = time.time()
                    theta_hf = proposal.sample((n_hf_samples_round,))
                    
                    # generate new simulations
                    if self.task == 'task1' or self.task == 'task4' or self.task == 'task5':
                        x_hf = self.hf_simulator.simulator(theta_hf)
                    elif self.task == 'task6' or self.task == 'task7':
                        x_hf, theta_hf, _ = self.hf_simulator.summary_statistics(n_hf_samples_round, prior=proposal)
                        
                    elif self.task == 'task2':
                        cell, _ = self.hf_simulator._jaxley_neuron()
                        integrator_fn = Integrator(cell, self.config_data)
                        # Sample eacht ime from the proposal in the  method
                        # Method: Uses the samples from the proposal, if not enough samples, it samples new samples from the proposal
                        x_hf, theta_hf, _ = self.hf_simulator.simulator(theta_hf, 
                                                                        integrator_fn,#lambda params, noise_params: simulate_neuron(params, noise_params, cell), 
                                                                        allow_resampling_invalid_samples=True, # Resample from active_learning list if invalid number
                                                                        proposal=proposal, # Sample from the proposal for all rounds
                                                                        )
                        
                        print("x_hf")

                        print(f"generated len(x_hf)={len(x_hf)} new high-fidelity samples")
                        # Check if x_active has the right size
                        assert (len(x_hf) == len(theta_hf)), f"len(x_hf)={len(x_hf)} != len(theta_hf)={len(theta_hf)}"
                    else:
                        raise ValueError("Task not implemented in active learning")
                    
                    # retrain your density estimator 
                    density_estimator = inference.append_simulations(
                        theta_hf, x_hf, proposal
                    ).train(force_first_round_loss=True)
                    
                    print("density_estimator computed")

                    non_amortized_posterior = inference.build_posterior(density_estimator).set_default_x(x_o)  # 
                    
                    print("posterior built")

                    if self.task == 'task2':
                        accept_reject_fn = get_density_thresholder(non_amortized_posterior, quantile=1e-3) # was 1e-6, lower for test
                    else:
                        accept_reject_fn = get_density_thresholder(non_amortized_posterior, quantile=1e-6)

                    print("accept_reject_fn computed")

                    proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
                    
                    print("proposal generated")
                    
                    one_round_end_time = time.time()
                
                hf_end_time = time.time()

                total_end_time = time.time()
                
                self._write_time_file("tsnpe", hf_start_time, hf_end_time, hf_data[i]['n_samples'], total_start_time, total_end_time, one_round_start_time, one_round_end_time, n_rounds)
                
                # Save the posterior in a pickle file! 
                if save_posteriors:
                    ensemble_post_save = {
                        'posterior': non_amortized_posterior, 
                        'true_x': x_o,
                        'true_theta': true_thetas[i_xo],
                        'type_estimator': 'tsnpe',
                        'n_rounds': n_rounds,
                        'n_simulations': hf_data[i]['n_samples'],
                    }
                    # Save a pickle with posterior samples (for 1 xo)
                    save_dir = f"{self.main_path}/non_amortized_posteriors/"
                    name = f"tsnpe_{hf_data[i]['n_samples']}_xo{i_xo}_seed{seed}_{self.CURR_TIME}.p" 
                    dump_pickle(save_dir, name, ensemble_post_save)
                    
                non_amortized_posteriors.append(non_amortized_posterior)
            posteriors.append(non_amortized_posteriors)

        return posteriors 


    # Distance to observed data
    def distance(self, sim, obs):
        # Root mean squared error
        rmse = torch.sqrt(torch.mean((sim - obs) ** 2))
        return rmse


    # def lofi(self, theta, xo, mean, std):
    #     # put theta in right shape
    #     theta = theta.unsqueeze(0)
    #     if self.task == 'task1':
    #         x_lf = self.lf_simulator.simulator(theta) 
    #         # z-score x_lf
    #         x_lf = self.z_score(x_lf, mean, std)
    #     else:
    #         raise ValueError("Task not implemented yet for mf-abc")

    #     return self.distance(x_lf, xo), x_lf
    
    
    # def hifi(self, theta, pass_lo, xo, mean, std): 
    #     theta = theta.unsqueeze(0)   
    #     if self.task == 'task1':
    #         x_hf = self.hf_simulator.simulator(theta) 
    #         # z-score x_hf
    #         x_hf = self.z_score(x_hf, mean, std)
    #     else:
    #         raise ValueError("Task not implemented yet for mf-abc")
        
    #     return self.distance(x_hf, xo) 
    
    def lofi(self, theta, xo, mean, std):
        # put theta in right shape
        theta = theta.unsqueeze(0)
        if self.task == 'task1' or self.task == 'task4' or self.task == 'task5':
            # Does not work currently if lf_simulator is a dict
            x_lf = self.lf_simulator.simulator(theta) 
        elif self.task == 'task7':
            x_lf = self.lf_simulator.simulator_wrapper(theta)
        else:
            ValueError("Task not implemented yet for mf-abc")
        # z-score x_lf     
        x_lf = self.z_score(x_lf, mean, std)

        return self.distance(x_lf, xo), x_lf
    
    
    def hifi(self, theta, pass_lo, xo, mean, std): 
        theta = theta.unsqueeze(0)   
        if self.task == 'task1'or self.task == 'task4' or self.task == 'task5' or self.task == 'task6':
            x_hf = self.hf_simulator.simulator(theta) 
        elif self.task == 'task7':
            x_hf = self.hf_simulator.simulator_wrapper(theta)
        else:
            ValueError("Task not implemented yet for mf-abc")
        # z-score x_hf    
        x_hf = self.z_score(x_hf, mean, std)
        
        return self.distance(x_hf, xo) 
    
    def parameter_sampler(self):
        dist = self.hf_prior
    
        return dist.sample()
    
    def get_stats(self, x):
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        return mean, std

    def z_score(self, x, mean, std):
        z =(x - mean) / std #(abs(x)-abs(mean))/abs(std)
        
        return z
    
    def z_inv(self, z, mean, std):
        x = z * std + mean
        return x
    
                    
    def run_mf_abc(self, lf_data, true_thetas):
        posterior_samples_all_batches = []
        posterior_weights_all_batches = []
        num_hifi_total = []
        
        # loop over n_lf simulations, hf simulations are actively sampled
        for k, n_simulations in enumerate(self.batch_lf_sims):  
            curr_dataset = lf_data[k]
            
            # Make sure to put a large enough HF dataset
            x_t, theta_t = curr_dataset['x'], curr_dataset['theta']
            
            # Z-score stats for the data: Make a data-dependent approximation (like in SBI)
            # To then dynamically transform the generated given this mean/std
            mean_x, std_x = self.get_stats(x_t)
            #mean_theta, std_theta = self.get_stats(theta_t) # transform true thetas with the same mean and std
            # x_z, theta_z = self.z_score(x_t, mean_x, std_x), self.z_score(theta_t, mean_theta, std_theta)
            

            posterior_samples_k = []
            weights_k = []
            
            for j, xo in enumerate(self.obs):     
                xo_z = self.z_score(xo, mean_x, std_x)

                mfabc = MFABC(
                    parameter_sampler=theta_t, #self.parameter_sampler, # plug in our data
                    lofi=lambda theta: self.lofi(theta, xo_z, mean_x, std_x), # pass in the observed data
                    hifi=lambda theta, pass_lo: self.hifi(theta, pass_lo, xo_z, mean_x, std_x) 
                )
                             
                # 30% HF
                # eps 1: looser, eps 2: accept simulations that are within ~1 SD
                epsilons = (1.0, 1.0) # (eps_lofi, eps_hifi)

                print(f"[batch {k}, obs {j}] Estimated epsilons: {epsilons}")
                etas = (0.9, 0.3)
                cloud = make_mfabc_cloud(mfabc, theta_t, epsilons, etas, N=n_simulations)
                
                for i, particle in enumerate(cloud):
                    print(f"Particle {i}: eta={particle.eta:.2f}, weight={particle.w:.2f}, distances={particle.p.dist}, cost={particle.p.cost}")

                # Print how many samples have weight > 0
                print(f"Number of particles with weight > 0: {sum(1 for p in cloud if p.w > 0)}")

                params = torch.stack([p.p.theta for p in cloud])  # (N, D)
                weights = torch.tensor([p.w for p in cloud], dtype=torch.float32)
                weights = torch.clamp(weights, min=0.0)
                weights = weights / torch.sum(weights)
                
                print("weights before importance sampling", weights.shape)
                
                # importance resampling 
                # use only a subset of reweighted samples (= how much we want to generate)
                # Try multinomial sampling, with fallback in case of error
                try:
                    idxs_post = torch.multinomial(weights, num_samples=self.n_samples_to_generate, replacement=True)
                except RuntimeError as e:
                    print(f"Multinomial sampling failed: {e}. Trying again.")
                    weights = torch.ones(len(cloud), dtype=torch.float32)
                    weights = weights / torch.sum(weights)
                    idxs_post = torch.multinomial(weights, num_samples=self.n_samples_to_generate, replacement=True)
                
                                
                post_samples = params[idxs_post]
                
                # Transform posterior samples back to original space
                print("post samples", post_samples)
                print("mean_x", mean_x)
                print("std_x", std_x)
                
                posterior_samples_k.append(post_samples)
                #weights_k.append(wx_counts)
                

                # Optional plot for first few obs in first batch
                if getattr(self.hf_simulator, "prior_ranges", None) is not None:
                    limits = self.hf_simulator.parameter_ranges(self.theta_dim)
                else:
                    limits = None

                # Optional plot for first few obs in first batch
                if k == 0 and j < 3:
                    _ = pairplot(
                        post_samples.numpy(),
                        **({"limits": limits} if limits is not None else {}),
                        figsize=(6, 6),
                        labels=[rf"$\theta_{i+1}$" for i in range(post_samples.shape[1])],
                        bins=30,
                        points=true_thetas[j].numpy(),
                        title=f"method: MF-abc (n_sims: {n_simulations})",
                    )
                # if k == 0 and j < 3:
                #     _ = pairplot(
                #         post_samples.numpy(),
                #         limits=self.hf_simulator.parameter_ranges(self.theta_dim),
                #         ticks=self.hf_simulator.parameter_ranges(self.theta_dim),
                #         figsize=(6, 6),
                #         labels=[rf"$\theta_{i+1}$" for i in range(post_samples.shape[1])],
                #         bins=30,
                #         points=true_thetas[j].numpy(),
                #         title=f"method: MF-abc (n_sims: {n_simulations})",
                #     )
                    
            # Stack obs-wise: (n_obs, n_samples, n_params)
            posterior_samples_k = torch.stack(posterior_samples_k)
            posterior_samples_all_batches.append(posterior_samples_k)
            
            # Count number of high-fidelity simulations used
            num_hifi = sum(1 for p in cloud if len(p.p.dist) == 2)
            print(f"Number of hf simulations: {num_hifi}")
            
            num_hifi_total.append(num_hifi)
            
        # print the size/shape of posterior_samples_all_batches
        print(f"Size of posterior_samples_all_batches: {posterior_samples_all_batches[0].shape}")
        
        return posterior_samples_all_batches, None, num_hifi_total

    

    

    # # Distance to observed data
    # def distance(self, sim, obs):
    #     # Root mean squared error
    #     rmse = torch.sqrt(torch.mean((sim - obs) ** 2))
    #     return rmse


    # def lofi(self, theta, xo, mean, std):
    #     # put theta in right shape
    #     theta = theta.unsqueeze(0)
    #     if self.task == 'task1' or self.task == 'task4' or self.task == 'task5':
    #         # Does not work currently if lf_simulator is a dict
    #         x_lf = self.lf_simulator.simulator(theta) 
    #     elif self.task == 'task7':
    #         x_lf = self.lf_simulator.simulator_wrapper(theta)
    #     else:
    #         ValueError("Task not implemented yet for mf-abc")
    #     # z-score x_lf     
    #     x_lf = self.z_score(x_lf, mean, std)

    #     return self.distance(x_lf, xo), x_lf
    
    
    # def hifi(self, theta, pass_lo, xo, mean, std): 
    #     theta = theta.unsqueeze(0)   
    #     if self.task == 'task1'or self.task == 'task4' or self.task == 'task5' or self.task == 'task6':
    #         x_hf = self.hf_simulator.simulator(theta) 
    #     elif self.task == 'task7':
    #         x_hf = self.hf_simulator.simulator_wrapper(theta)
    #     else:
    #         ValueError("Task not implemented yet for mf-abc")
    #     # z-score x_hf    
    #     x_hf = self.z_score(x_hf, mean, std)
        
    #     return self.distance(x_hf, xo) 
    
    # def parameter_sampler(self):
    #     dist = self.hf_prior
    
    #     return dist.sample()
    
    # def get_stats(self, x):
    #     mean = x.mean(dim=0)
    #     std = x.std(dim=0)
        
    #     # Add some noise to std if std is 0
    #     std[std == 0] = 1e-6
        
    #     return mean, std

    # def z_score(self, x, mean, std):
    #     z =(x - mean) / std #(abs(x)-abs(mean))/abs(std)
        
        
    #     return z
    
    # def z_inv(self, z, mean, std):
    #     x = z * std + mean
    #     return x
    
    
    # def run_mf_abc(self, lf_data, true_thetas):
    #     posterior_samples_all_batches = []
    #     num_hifi_total = []
        
    #     print(f"Warning: mf_ABC has only been implemented for the OUprocess task.")
        
    #     # loop over n_lf simulations, hf simulations are actively sampled
    #     for k, n_simulations in enumerate(self.batch_lf_sims):  
    #         curr_dataset = lf_data[k]
            
    #         # Make sure to put a large enough HF dataset (e.g. batch_hf_sims = 1000) in the task_setup file
    #         x_t, theta_t = curr_dataset['x'], curr_dataset['theta']
                        
    #         # Z-score stats for the data: Make a data-dependent approximation (like in the SBI package)
    #         mean_x, std_x = self.get_stats(x_t)
    #         posterior_samples_k = []
            
    #         for j, xo in enumerate(self.obs):     
                
    #             print("xo", xo.shape)
    #             print("self observation", self.obs.shape)
                
                
    #             xo_z = self.z_score(xo, mean_x, std_x)
                
    #             # If NaN in xo_z, make a 0
    #             if torch.isnan(xo_z).any():
    #                 xo_z = torch.zeros_like(xo_z)
    #                 print("Warning: xo_z contains NaN values, setting to zero vector.")
                    

    #             mfabc = MFABC(
    #                 parameter_sampler=theta_t, #self.parameter_sampler, # plug in our data
    #                 lofi=lambda theta: self.lofi(theta, xo_z, mean_x, std_x), # pass in the observed data
    #                 hifi=lambda theta, pass_lo: self.hifi(theta, pass_lo, xo_z, mean_x, std_x) 
    #             )
                             
    #             # 30% HF approximately
    #             # eps 1: looser, eps 2: accept simulations that are within ~1 SD
    #             if self.task == 'task1': 
    #                 epsilons = (1.0, 1.0) # OUprocess
    #             if self.task == 'task5':
    #                 epsilons = (4.0, 0.8) # SLCP # Epsilon not really tested indeptly
    #             if self.task == 'task7':
    #                 # ~61% are HF samples for SIR task
    #                 epsilons = (2.0, 0.5) # SIR: Epsilon not really tested indeptly

    #             print(f"[batch {k}, obs {j}] Estimated epsilons: {epsilons}")
                
    #             # etas = (0.9, 0.3)
    #             etas = (0.9, 0.3)
    #             cloud = make_mfabc_cloud(mfabc, theta_t, epsilons, etas, N=n_simulations)
                
    #             for i, particle in enumerate(cloud):
    #                 print(f"Particle {i}: eta={particle.eta:.2f}, weight={particle.w:.2f}, distances={particle.p.dist}, cost={particle.p.cost}")
                    
                    
    #             params = torch.stack([p.p.theta for p in cloud])  # (N, D)
    #             weights = torch.tensor([p.w for p in cloud], dtype=torch.float32)
                
    #             print("weights", weights)
                
    #             weights = torch.clamp(weights, min=0.0)
    #             weights = weights / torch.sum(weights)
                
    #             # importance resampling 
    #             idxs_post = torch.multinomial(weights, num_samples=n_simulations, replacement=True)
    #             post_samples = params[idxs_post]
    #             posterior_samples_k.append(post_samples)
                
                
    #             if getattr(self.hf_simulator, "prior_ranges", None) is not None:
    #                 limits = self.hf_simulator.parameter_ranges(self.theta_dim)
    #             else:
    #                 limits = None

    #             # Optional plot for first few obs in first batch
    #             if k == 0 and j < 3:
    #                 _ = pairplot(
    #                     post_samples.numpy(),
    #                     **({"limits": limits} if limits is not None else {}),
    #                     #limits=self.hf_simulator.parameter_ranges(self.theta_dim) if self.hf_simulator.parameter_ranges(self.theta_dim) is not None else {},
    #                     # ticks=self.hf_simulator.parameter_ranges(self.theta_dim) if self.hf_simulator.parameter_ranges(self.theta_dim) is not None else {},
    #                     figsize=(6, 6),
    #                     labels=[rf"$\theta_{i+1}$" for i in range(post_samples.shape[1])],
    #                     bins=30,
    #                     points=true_thetas[j].numpy(),
    #                     title=f"method: MF-abc (n_sims: {n_simulations})",
    #                 )
                    
    #                 path_pairplots = f"{self.main_path}/pairplots/"
    #                 if not os.path.exists(path_pairplots):
    #                     # Make directory if it doesn't exist
    #                     os.makedirs(path_pairplots)
                        
    #                 # Save the plot
    #                 plt.savefig(f"{path_pairplots}/mf_abc_{n_simulations}_{k}_{j}.svg")
    #                 plt.savefig(f"{path_pairplots}/mf_abc_{n_simulations}_{k}_{j}.pdf")
                    
                    
    #         # Stack obs-wise: (n_obs, n_samples, n_params)
    #         posterior_samples_k = torch.stack(posterior_samples_k)
    #         posterior_samples_all_batches.append(posterior_samples_k)
            
    #         # Count number of high-fidelity simulations used
    #         num_hifi = sum(1 for p in cloud if len(p.p.dist) == 2)
    #         print()
    #         print(f"Number of hf simulations: {num_hifi}")
            
    #         dic_save = {
    #             'posterior_samples': posterior_samples_k,
    #             'n_true_x': len(true_thetas),
    #             'true_theta': true_thetas,
    #             'type_estimator': 'mf_abc',
    #             'num_hifi_total': num_hifi
    #         }
    #         # Save a pickle with posterior samples (just one network init)
    #         save_dir = f"{self.main_path}/posterior_samples/"
    #         name = f"thetas_0_{len(true_thetas)}_mf_abc_{n_simulations}.p"
    #         dump_pickle(save_dir, name, dic_save)
            
    #         num_hifi_total.append(num_hifi)
            
    #     return posterior_samples_all_batches, num_hifi_total                    
            
                
    def run_sbi(self, hf_data):
        # Estimate the high fidelity posteriors with SBI
        posteriors = []
        
        z_score_theta = 'independent' if self.z_score_theta else None
        z_score_x = 'independent' if self.z_score_x else None
        
        for i in range(len(self.batch_hf_sims)):
            # Check prior, return PyTorch prior.
            prior, num_parameters, prior_returns_numpy = process_prior(self.hf_prior)
                        
            # Define the density estimator where thetas are not z-scored (since data is logit-transformed) and x is z-scored
            posterior_settings = posterior_nn(model="zuko_nsf", 
                                              z_score_theta=z_score_theta,
                                              z_score_x=z_score_x, #can also be structured, but independent corresponds to my mf-npe case.
                                              hidden_features=self.config_model['n_hidden_features'], 
                                              num_transforms=self.config_model['n_transforms'], 
                                              num_bins=self.config_model['n_bins'], 
                                              embedding_net=nn.Identity())       
            
            inference = NPE(prior=prior, density_estimator=posterior_settings) 
            
            theta, x = hf_data[i]['theta'], hf_data[i]['x']
            
            inference = inference.append_simulations(theta, x)
            density_estimator = inference.train()
            
            posterior = inference.build_posterior(density_estimator) #
            
            print("inference summary", inference.summary)
            # Plot the training and validation log probs
            plt.plot(inference.summary['training_loss'], label='training')
            plt.plot(inference.summary['validation_loss'], label='validation')
            
            # Make directory if does not exist yet
            path_plot = self.main_path + "/loss/"
            if not os.path.exists(path_plot):
                os.makedirs(path_plot)
            
            plt.savefig(f"{path_plot}/SBI loss_{self.config_data['sim_name']}_{self.batch_hf_sims[i]}.png")
            plt.legend()
            
            if plot_config.show_plots:
                plt.show()
            
            posteriors.append(posterior)
            
            # Save posterior as a pickle file
            path = f"{self.main_path}/sbi_posterior/"
            name = f"sbi_posterior_{self.batch_hf_sims[i]}.p"
            dump_pickle(path=path, name_pickle=name, variable=posteriors)
            
        return posteriors
    
    




    # def run_mf_npe(self, lf_data, hf_data):
    #     '''
    #     # The finetuned model gave worse accuracy. So I'm keeping the code but it's not used.
    #     This method generates all possible combinations of the low-fidelity (LF) and high-fidelity (HF) datasets
    #     for the multi-fidelity model. It then trains a base model using the LF data and fine-tunes it using the HF data.

    #     Parameters:
    #     lf_data (dict): Dictionary containing the low-fidelity data.
    #     hf_data (dict): Dictionary containing the high-fidelity data.

    #     Returns:
    #     Tuple: A tuple containing the multi-fidelity posteriors and the multi-fidelity flows.
        
    #     '''
    #     mf_posteriors = []
    #     lf_posteriors = []
        
    #     # Generate all possible combinations of the datasets for the multifidelity model
    #     lf_mf_data = [lf_data[lf] for lf in lf_data for _ in hf_data]
    #     hf_mf_data = [hf_data[hf] for _ in lf_data for hf in hf_data]

    #     # The length of hf_mf_data is the same as lf_mf_data and is a permutation of both of the datasets.
    #     for i, _ in enumerate(hf_mf_data):                   
    #         print("training low fidelity model...")            
    #         x_lf, theta_lf = lf_mf_data[i]['x'], lf_mf_data[i]['theta']
    #         x_hf, theta_hf = hf_mf_data[i]['x'], hf_mf_data[i]['theta']
            
    #         lf_start_time = time.time()
            
    #         # create dataloader
    #         lf_train_loader, lf_val_loader = create_train_val_dataloaders(
    #             theta_lf.to(self.device),
    #             x_lf.to(self.device),
    #             validation_fraction = self.validation_fraction,
    #             batch_size=self.batch_size,
    #         )
            
    #         hf_train_loader, hf_val_loader = create_train_val_dataloaders(
    #             theta_hf.to(self.device),
    #             x_hf.to(self.device),
    #             validation_fraction = self.validation_fraction,
    #             batch_size=self.batch_size,
    #         )
            
            

            
    #         # Embedding network to decrease the representational gap with the HF model
    #         # embedding_network = EmbeddingNetwork(input_dim=theta_lf.shape[1], output_dim=theta_hf.shape[1]).to(self.device)
    #         # embedding_network.train_embed(lf_train_loader, hf_train_loader, epochs=100, lr=0.001) 
    #         # print("Embedding network trained successfully.")
    #         # # Project the low-fidelity data into the high-fidelity space
    #         # lf_data_embedded = embedding_network.evaluate_embed(lf_val_loader)  
    #         # print("low fidelity embedded data shape:", lf_data_embedded.shape)
            

    #         lf_flow = build_zuko_flow(theta_lf, x_lf, nn.Identity(), 
    #                                     z_score_theta=self.z_score_theta, # Only z-score LF samples, not HF
    #                                     z_score_x=self.z_score_x,  
    #                                     logit_transform_theta=self.logit_transform_theta_net,
    #                                     nf_type="NSF_PRETRAIN", 
    #                                     hidden_features=self.config_model['n_hidden_features'],
    #                                     num_transforms=self.config_model['n_transforms'],
    #                                     num_bins=self.config_model['n_bins'],
    #                                     prior=self.lf_prior) #.to(self.device)

    #         lf_optimizer = torch.optim.Adam(lf_flow.parameters(), lr=self.config_model['learning_rate']) 

            

    #         lf_flow = fit_conditional_normalizing_flow(
    #             lf_flow,
    #             lf_optimizer,
    #             lf_train_loader,
    #             lf_val_loader,
    #             nb_epochs=self.max_num_epochs,
    #             print_every=1,
    #             early_stopping_patience=self.early_stopping,
    #             clip_max_norm=self.clip_max_norm,
    #             plot_loss=False, 
    #             type_flow='MF-NPE (LF)',
    #         )
            
    #         lf_end_time = time.time()

    #         base_model = copy.deepcopy(lf_flow)
    #         print("training high fidelity model...")            
    #         # If needed (we saw empirically that additional hf transform were not helping
    #         additional_hf_transform = zuko.flows.NSF(features=self.theta_dim, 
    #                                                  context=self.x_dim, 
    #                                                  bins=self.config_model['n_bins'],
    #                                        transforms=1, 
    #                                        hidden_features=(50, 50) # [50]*4 
    #                                        ).transform 
            
            
            
    #         hf_start_time = time.time()
                        
    #         hf_flow = build_zuko_flow(theta_hf, x_hf, nn.Identity(), 
    #                                     z_score_theta=self.z_score_theta, 
    #                                     z_score_x=self.z_score_x,  
    #                                     logit_transform_theta=self.logit_transform_theta_net,
    #                                     nf_type="NSF_FINETUNE",
    #                                     hidden_features=self.config_model['n_hidden_features'],
    #                                     num_transforms=self.config_model['n_transforms'],
    #                                     num_bins=self.config_model['n_bins'],
    #                                     base_model=base_model, 
    #                                     additional_hf_transform=additional_hf_transform, 
    #                                     prior=self.hf_prior)
            
    #         hf_optimizer = torch.optim.Adam(hf_flow.parameters(), lr=self.config_model['learning_rate']) 
            
            
            
    #         hf_flow = fit_conditional_normalizing_flow(
    #             hf_flow,
    #             hf_optimizer,
    #             hf_train_loader,
    #             hf_val_loader,
    #             nb_epochs=self.max_num_epochs,
    #             print_every=1,
    #             early_stopping_patience=self.early_stopping,
    #             clip_max_norm=self.clip_max_norm,
    #             plot_loss=False, 
    #             type_flow='MF-NPE (HF)',
    #         )
            
    #         hf_end_time = time.time()
            
    #         path = self.main_path + "/time"
    #         if not os.path.exists(path):
    #                 os.makedirs(path)
    #         # Save the time in a txt file      
    #         with open(f"{path}/mf_npe_time_LF{lf_mf_data[i]['n_samples']}_HF{hf_mf_data[i]['n_samples']}.txt", "a") as f:
    #             f.write(f"Time taken for LF training: {lf_end_time - lf_start_time} seconds\n")
    #             f.write(f"Time taken for HF training:: {hf_end_time - hf_start_time} seconds\n")
    #             f.write(f"Time taken for MF training: {hf_end_time - lf_start_time} seconds\n")
    #             f.write("---------------------------------------\n")

            
    #         mf_posterior = DirectPosterior(hf_flow, self.hf_simulator.prior())
    #         mf_posteriors.append(mf_posterior)
            
    #         lf_posterior = DirectPosterior(lf_flow, self.lf_simulator.prior())
    #         lf_posteriors.append(lf_posterior)

                    
    #     return mf_posteriors, lf_posteriors

    # def _train_ensemble(self, lf_flow, theta_hf, x_hf, n_ensemble_members):
    #     """
    #     train an ensemble of high-fidelity networks using the low-fidelity as the base model
    #     """
    #     ensemble = []
    #     hf_train_loader, hf_val_loader = create_train_val_dataloaders(
    #         theta_hf.to(self.device),
    #         x_hf.to(self.device),
    #         validation_fraction = self.validation_fraction,
    #         batch_size=self.batch_size,
    #     )
    #     # train n_ensemble_members networks on the same data, independently
    #     for _ in range(n_ensemble_members):
    #         base_model = copy.deepcopy(lf_flow)

    #         # If needed (we saw empirically that additional hf transform were not helping
    #         # additional_hf_transform = zuko.flows.NSF(features=self.theta_dim, 
    #         #                                     context=self.x_dim, 
    #         #                                     bins=8,
    #         #                                     transforms=1, 
    #         #                                     hidden_features=(50, 50) # [50]*4 
    #         #                                     ).transform 
                        
    #         # define the high hidelity model using the pre-trained base model
    #         hf_flow = build_zuko_flow(theta_hf, x_hf, nn.Identity(), 
    #                                 z_score_theta=self.z_score_theta, 
    #                                 z_score_x=self.z_score_x,  
    #                                 logit_transform_theta=self.logit_transform_theta_net,
    #                                 nf_type="NSF_FINETUNE",
    #                                 hidden_features=self.config_model['n_hidden_features'],
    #                                 num_transforms=self.config_model['n_transforms'],
    #                                 num_bins=self.config_model['n_bins'],
    #                                 base_model=base_model, 
    #                                 # additional_hf_transform=additional_hf_transform, 
    #                                 prior=self.hf_prior)
        
    #         hf_optimizer = torch.optim.Adam(hf_flow.parameters(), lr=self.config_model['learning_rate']) 

    #         hf_flow = fit_conditional_normalizing_flow(
    #             hf_flow,
    #             hf_optimizer,
    #             hf_train_loader,
    #             hf_val_loader,
    #             nb_epochs=self.max_num_epochs,
    #             print_every=1,
    #             early_stopping_patience=self.early_stopping,
    #             clip_max_norm=self.clip_max_norm,
    #             plot_loss=False, 
    #             type_flow='Active-NPE (HF)',
    #         )
    #         ensemble.append(hf_flow)
        
    #     # returns list of trained networks
    #     return ensemble
    
# %%
