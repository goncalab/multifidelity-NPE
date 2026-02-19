import copy
import os
import time
import torch

from mf_npe.flows.build_flows import build_zuko_flow
from mf_npe.flows.train_flows import create_train_val_dataloaders, fit_pretrained_conditional_normalizing_flow
from mf_npe.methods.embedding import _generate_embedding_networks

def _convert_data_to_mf_format(data):
    hf_data = data['hf']  
    
    # lf data are all keys except data['hf'], in case there are multiple lf datasets (e.g., 'lf', 'mid1', 'mid2' etc.)
    lf_data = {k: v for k, v in data.items() if k != 'hf'}

    # For all non-hf keys, make a lf_mf_dataset and hf_mf_dataset
    lf_mf_data = {}
    for fidelity, l_data in lf_data.items():
        lf_mf_data[fidelity] = [l_data[lf] for lf in l_data for _ in hf_data]
    
    # currently only 1 low-fidelity dataset is supported in MF-NPE, so we take the first (and only) key in lf_mf_data
    hf_mf_data = [hf_data[hf] for _ in data['lf'] for hf in hf_data]
    
    return lf_mf_data, hf_mf_data



def _write_time_file(path, type_algorithm, hf_start_time, hf_end_time, n_hf_samples, 
                        lf_start_time=None, lf_end_time=None,n_lf_samples=None, 
                        total_start_time=None, total_end_time=None, 
                        one_round_start_time=None, one_round_end_time=None, n_rounds=None):
    
    
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




def _pretrain_model_on_lower_fidelities(pipeline, lf_datasets, i):
    """
        In case we have multiple fidelities, we loop over the LF datasets and keep pretraining the model on each of them sequentially.
        For example, if we have 2 LF datasets (e.g., 'lf' and 'mid'), we first train a model on the 'lf' data, 
        then we use that model as a base model to train on the 'mid' data, and return the pretrained model 
        after training on the 'mid' data. If we only have 1 LF dataset, 
        we simply train on that dataset and return the pretrained model.
    """
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
            theta_lf.to(pipeline.device),
            x_lf.to(pipeline.device),
            validation_fraction=pipeline.validation_fraction,
            domain_labels=domain_labels.to(pipeline.device),
            batch_size=pipeline.batch_size,
        )
        
        x_embedding_lf, _ = _generate_embedding_networks(pipeline, x_lf=x_lf)
        x_embedding_lf = x_embedding_lf.to(pipeline.device)
        
        # If first time pretraining
        if pretrained_flow == []:
            pretrained_flow = build_zuko_flow(theta_lf, x_lf, x_embedding_lf,
                                    z_score_theta=pipeline.z_score_theta, # Only z-score LF samples, not HF
                                    z_score_x=pipeline.z_score_x,  
                                    logit_transform_theta=pipeline.logit_transform_theta_net,
                                    nf_type="NSF_PRETRAIN", 
                                    hidden_features=pipeline.config_model['n_hidden_features'],
                                    num_transforms=pipeline.config_model['n_transforms'],
                                    num_bins=pipeline.config_model['n_bins'],
                                    prior=pipeline.lf_prior) 
        else:
            pretrained_flow = build_zuko_flow(theta_lf, x_lf, x_embedding_lf,
                                    z_score_theta=pipeline.z_score_theta, # Only z-score LF samples, not HF
                                    z_score_x=pipeline.z_score_x,  
                                    logit_transform_theta=pipeline.logit_transform_theta_net,
                                    nf_type="NSF_FINETUNE", 
                                    hidden_features=pipeline.config_model['n_hidden_features'],
                                    num_transforms=pipeline.config_model['n_transforms'],
                                    num_bins=pipeline.config_model['n_bins'],
                                    base_model=pretrained_flow,
                                    prior=pipeline.lf_prior) 
                    
        parameters = list(pretrained_flow.parameters())  
        pretrained_optimizer = torch.optim.Adam(parameters, lr=1e-4)  # Adjust learning rate
        
        pretrained_flow = fit_pretrained_conditional_normalizing_flow(
            pretrained_flow,
            pretrained_optimizer,
            train_loader,
            val_loader,
            x_dim_lf=pipeline.x_dim_lf,  # Dimension of the low-fidelity data
            x_dim_hf=pipeline.x_dim_hf,  # Dimension of the high-fidelity data
            x_dim_out=pipeline.x_dim_out,  # Dimension of the input data
            theta_dim=pipeline.theta_dim,
            nb_epochs=pipeline.max_num_epochs,
            print_every=1,
            early_stopping_patience=pipeline.early_stopping,
            clip_max_norm=pipeline.clip_max_norm,
            plot_loss=False, 
            type_flow='MF-NPE (LF)',
            device=pipeline.device,
        )
        lf_end_time = time.time()
    
    # Return the latest fidelity trained model
    base_model = copy.deepcopy(pretrained_flow)
        
    # return the n_lf_samples, which is the same for all lf_simulators!
    return base_model, lf_start_time, lf_end_time, x_embedding_lf, n_lf_samples