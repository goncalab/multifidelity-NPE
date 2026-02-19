import copy
import time
from mf_npe.methods.embedding import _generate_embedding_networks
from mf_npe.methods.helpers import _pretrain_model_on_lower_fidelities, _write_time_file
import torch
from mf_npe.flows.build_flows import build_zuko_flow
from mf_npe.flows.train_flows import create_train_val_dataloaders, fit_conditional_normalizing_flow, fit_pretrained_conditional_normalizing_flow
from sbi.inference.posteriors.direct_posterior import DirectPosterior


def train_mf_npe(pipeline, 
                 x_hf, 
                 theta_hf, 
                 n_hf_samples, 
                 lf_mf_data:dict, # Since it can contain data for multiple LF datasets
                 i:int):
    """MF-NPE trains a base model using the LF data and fine-tunes it using the HF data.

    Args:
        x_hf (_type_): high-fidelity observations
        theta_hf (_type_): high-fidelity parameters
        n_hf_samples (_type_): number of high-fidelity samples
        lf_mf_data (_type_): low-fidelity data, in the form of a dictionary with keys corresponding to the different fidelities (e.g., 'lf', 'mid', etc.) and values being the datasets for each fidelity.
        i (_type_): index of the current HF dataset (if multiple HF datasets are used)

    Returns:
        _type_: tuple of (pretrained_posterior, mf_posterior), where pretrained_posterior is the posterior obtained from the LF data (or the latest fidelity if multiple LF datasets are used) and mf_posterior is the posterior obtained from the HF data using MF-NPE.
    """
    
    # Pretrain on LF data. Give all data datasets, and loop over them inside this function.
    # Note, if multiple fidelities: it will return the pretrained flow of the latest model (e.g., mid-fidelity, if 3 fidelities used. otherwise, it's simply always the 'lf' model)
    pretrained_flow, lf_start_time, lf_end_time, x_embedding_lf, n_lf_samples = _pretrain_model_on_lower_fidelities(pipeline, lf_mf_data, i) # x_hf is passed for the embedding
    lf_pretrained_flow = copy.deepcopy(pretrained_flow) # Save the pretrained flow for returning later, to compare the distance between the LF and HF model.

    print("training high fidelity model...")
    hf_train_loader, hf_val_loader = create_train_val_dataloaders(
        theta_hf.to(pipeline.device),
        x_hf.to(pipeline.device),
        validation_fraction = pipeline.validation_fraction,
        batch_size=pipeline.batch_size,
    )
            
    hf_start_time = time.time()                
    hf_flow = build_zuko_flow(theta_hf, x_hf, x_embedding_lf, 
                                z_score_theta=pipeline.z_score_theta, 
                                z_score_x=pipeline.z_score_x,  
                                logit_transform_theta=pipeline.logit_transform_theta_net,
                                nf_type="NSF_FINETUNE",
                                hidden_features=pipeline.config_model['n_hidden_features'],
                                num_transforms=pipeline.config_model['n_transforms'],
                                num_bins=pipeline.config_model['n_bins'],
                                base_model=pretrained_flow, # Use the pretrained flow as a base model
                                prior=pipeline.hf_prior)


    hf_optimizer = torch.optim.Adam(list(hf_flow.parameters()), lr=pipeline.config_model['learning_rate'])
    hf_flow = fit_conditional_normalizing_flow(
        hf_flow,
        hf_optimizer,
        hf_train_loader,
        hf_val_loader,
        x_embedder=x_embedding_lf,
        nb_epochs=pipeline.max_num_epochs,
        print_every=1,
        early_stopping_patience=pipeline.early_stopping,
        clip_max_norm=pipeline.clip_max_norm,
        plot_loss=False, 
        type_flow='MF-NPE (HF)',
    )
    
    hf_end_time = time.time()
    _write_time_file(pipeline.time_path, "mf_npe", hf_start_time, hf_end_time, n_hf_samples, lf_start_time, lf_end_time, n_lf_samples)
    
    mf_posterior = DirectPosterior(hf_flow, pipeline.hf_simulator.prior())
    
    # lf_posterior are the pretrained models (or model, if only 1 lf simulator)
    if isinstance(pipeline.lf_simulator, dict): # i.e., if there are multiple lf simulators
        pretrained_posterior = {}
        for fidelity, sim in pipeline.lf_simulator.items():
            pretrained_posterior[fidelity] = DirectPosterior(lf_pretrained_flow, sim.prior())
    else:
        pretrained_posterior = DirectPosterior(lf_pretrained_flow, pipeline.lf_simulator.prior())
        
    return pretrained_posterior, mf_posterior


