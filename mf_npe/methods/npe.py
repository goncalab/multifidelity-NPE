
import time
from mf_npe.methods.embedding import _generate_embedding_networks
from mf_npe.methods.helpers import _write_time_file
import torch
from build.lib.mf_npe.flows.train_flows import fit_conditional_normalizing_flow
from mf_npe.flows.train_flows import create_train_val_dataloaders
from mf_npe.flows.build_flows import build_zuko_flow
from sbi.inference.posteriors.direct_posterior import DirectPosterior


def train_npe(self, curr_dataset, n_simulations):
    x_t, theta_t = curr_dataset['x'], curr_dataset['theta']
    _, x_embedding_hf = _generate_embedding_networks(self, x_hf=x_t)

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
    _write_time_file(self.time_path, "npe", start_time, end_time, n_simulations)
    posterior = DirectPosterior(direct_flow, self.hf_prior) # sbi wrapper
    
    return posterior