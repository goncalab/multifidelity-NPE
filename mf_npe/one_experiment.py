
#%%
from sbi import utils as utils # pip3 lis to check what the local path is of sbi
from sbi import analysis as analysis
from mf_npe import evaluation
from mf_npe.config.TaskSetup import TaskSetup
from mf_npe.diagnostics.histogram import plot_xen_histogram
from mf_npe.evaluation import Evaluation
from mf_npe.pipeline import Pipeline
from mf_npe.utils.utils import get_n_samples, set_global_seed

def run_one_experiment(seed, models_to_run, 
                       lf_data, hf_data, 
                       true_xen, true_thetas, true_add_ons, 
                       batch_lf_sims, 
                       batch_hf_sims,
                       sim_name,
                       config_model,
                       main_path,
                       net_init, 
                       b_load_model=False):
    
    key = set_global_seed(seed)
    
    # Initialize the pipeline
    task_setup = TaskSetup(sim_name=sim_name, 
                 config_model=config_model, 
                 main_path=main_path, 
                 batch_lf_datasize=batch_lf_sims, 
                 batch_hf_datasize=batch_hf_sims, 
                 n_network_initializations=net_init,
                 theta_dim=true_thetas.shape[1],
                 n_true_xen=true_xen.shape[0],
                 seed=seed)
    
    
    print("task_setup:", task_setup.config_data)
    pipeline = Pipeline(true_xen, task_setup)
    
    
    if sim_name == 'MultiCompartmentalNeuron':
            for i, batch_size in enumerate(batch_lf_sims):
                plot_xen_histogram(lf_data[i]['x'], f'LF train data on {batch_size}', xo=None)

            for i, batch_size in enumerate(batch_hf_sims):
                plot_xen_histogram(hf_data[i]['x'], f'HF train data on {batch_size}', xo=None)

    
    # Create a dictionary of all data
    simulations = {}
    # make sure to have a 'lf' in the dictionary
    if isinstance(lf_data, dict) and "lf" in lf_data:
        for key in lf_data.keys():
            simulations[key] = lf_data[key]
        simulations['hf'] = hf_data
    else:
        # This is for the old simulations
        simulations['lf'] = lf_data
        simulations['hf'] = hf_data
        
    
    print("simulations", simulations)
    
    # get real n_sims of data trained on   
    n_lf_samples, n_hf_samples, n_mf_samples = get_n_samples(simulations)


    if b_load_model is False: 
        hf_posteriors = []
        mf_posteriors = []
        sbi_posteriors = []
        active_posteriors = []
        active_snpe_posteriors = []
        mf_snpe_posteriors = []
        mf_abc_posteriors = []
        lf_posteriors = [] # For plotting in OU process task
        hf_tsnpe_posteriors = []
        bo_posteriors = []
        num_hifi = 0
        
        # Multifidelity approaches
        if 'mf_npe' in models_to_run: 
            mf_posteriors, lf_posteriors = pipeline.run_mf_npe(simulations) 
            
        if 'mf_tsnpe' in models_to_run:
            mf_snpe_posteriors = pipeline.run_mf_tsnpe(simulations, true_xen, true_thetas, 
                                                      n_rounds=config_model['n_rounds_AL'],
                                                      seed=seed)
        if 'a_mf_tsnpe' in models_to_run:
            active_snpe_posteriors = pipeline.run_active_mf_tsnpe(simulations, true_xen, true_thetas,
                                                                  n_rounds=config_model['n_rounds_AL'],
                                                                  n_ensemble_members=config_model['n_ensemble_members'],
                                                                  seed=seed) 
            
        if 'bo_npe' in models_to_run:
            bo_posteriors = pipeline.run_bo_npe(simulations, true_xen, true_thetas, use_mf_npe=True) 
            
        # Non-multifidelity Approaches
        if 'npe' in models_to_run: # logit transformed
            hf_posteriors = pipeline.run_npe(simulations['hf'])
        if 'sbi_npe' in models_to_run:
            sbi_posteriors = pipeline.run_sbi(simulations['hf'])
        
        
        if 'tsnpe' in models_to_run:
            # This is the non-amortized version of TSNPE
            hf_tsnpe_posteriors = pipeline.run_tsnpe(simulations['hf'], true_xen, true_thetas, 
                                                         n_rounds=config_model['n_rounds_AL'],
                                                         seed=seed)
        # Non-neural Approaches 
        if 'mf_abc' in models_to_run:
            mf_abc_posteriors, mf_abc_weights, num_hifi = pipeline.run_mf_abc(simulations['lf'], true_thetas)

        
    all_methods = dict(
        hf_posteriors=hf_posteriors,
        mf_posteriors=mf_posteriors,
        sbi_posteriors=sbi_posteriors,
        active_posteriors=active_posteriors,
        active_snpe_posteriors=active_snpe_posteriors,
        mf_snpe_posteriors=mf_snpe_posteriors,
        mf_abc=mf_abc_posteriors,
        hf_tsnpe_posteriors=hf_tsnpe_posteriors, # This is the non-amortized version of TSNPE
        # To compare low and high fidelity with SBC
        lf_posteriors=lf_posteriors,  
        bo_posteriors=bo_posteriors
        )
        
    train_data = dict(
        task_setup=task_setup,
        true_xen=true_xen,
        true_thetas=true_thetas,
        true_add_ons=true_add_ons,
        n_lf_samples=n_lf_samples,
        n_hf_samples=n_hf_samples,
        n_mf_samples=n_mf_samples,
        all_methods=all_methods,
        net_init=net_init,
        num_hifi=num_hifi,
        mf_abc_weights=mf_abc_weights if 'mf_abc' in models_to_run else None,
        hf_data=hf_data,        
    )
    
    return train_data


def run_evaluation(task_setup, 
                   true_xen, true_thetas, true_add_ons,
                   n_lf_samples, n_hf_samples, n_mf_samples, 
                   all_methods, 
                   net_init, 
                   mf_abc_weights,
                   num_hifi, 
                   hf_data,
                   eval_metric):
    
    evaluation = Evaluation(true_xen, task_setup, eval_metric)
    
    df_one_seed = evaluation.evaluate_methods(true_xen, true_thetas,
                                            n_lf_samples, n_hf_samples, n_mf_samples,
                                            true_add_ons, all_methods, net_init, 
                                            mf_abc_weights,
                                            num_hifi, # num_hifi is used for mf-abc
                                            hf_data) # hf_data is used for sbc
    
    return df_one_seed
    
    

def run_comparison_lf_to_hf_posteriors(task_setup, 
                                true_xen, true_thetas,
                                n_lf_samples, n_hf_samples,
                                all_methods, 
                                net_init, 
                                eval_metrics,
                                simulator_name='simulator'):
    
    # Compare LF and HF posterior distance/difference
    for metric in eval_metrics:
        evaluation = Evaluation(true_xen, task_setup, eval_metric=metric) # eval_metric is not used here, since we give a list of metrics below
        df_one_seed = evaluation.compare_lf_hf_posteriors(metric, n_hf_sims=n_hf_samples, n_lf_sims=n_lf_samples, hf_posteriors=all_methods['mf_posteriors'], lf_posteriors=all_methods['lf_posteriors'], true_xen=true_xen, true_thetas=true_thetas, net_init=net_init, type_estimator='lf_and_hf_npe', simulator_name=simulator_name)

    return df_one_seed
