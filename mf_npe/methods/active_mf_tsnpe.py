import time
from tqdm import tqdm
from mf_npe.methods.helpers import _pretrain_model_on_lower_fidelities, _write_time_file
from mf_npe.simulator.task2.simulation_func import Integrator
from mf_npe.utils.utils import dump_pickle
from sbi.inference import NPE
from sbi.utils import RestrictedPrior, get_density_thresholder
from sbi.utils.user_input_checks import process_prior
import copy
import torch
from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior

def train_active_mf_tsnpe(self, 
                    curr_hf_dataset, 
                    curr_lf_datasets, 
                    i, 
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
                
    # for hf_index, hf_samples in enumerate(hf_mf_data):
    non_amortized_posteriors = []
    n_hf_samples = curr_hf_dataset['n_samples'] 

    # Note: We use the same n_lf_samples for all lower fidelity models
    lf_flow, lf_start_time, lf_end_time, x_embedding_lf, n_lf_samples = _pretrain_model_on_lower_fidelities(self, curr_lf_datasets, i)
                
    # run over all xen that you want to evaluate on 
    for i_xo, _ in enumerate(tqdm(true_xen)):
        curr_xo = true_xen[i_xo]
        curr_theta = true_thetas[i_xo]

        ensemble_posterior =  _run_active_mf_tsnpe_single_xo(self, lf_flow=lf_flow,
                                                                lf_start_time=lf_start_time,
                                                                lf_end_time=lf_end_time,
                                                                n_lf_samples=n_lf_samples,
                                                                hf_data=curr_hf_dataset, 
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
        
    return non_amortized_posteriors # List of posteriors for each xo, for the same number of simulations (n_hf_samples) used to train the model. We return a list of posteriors across xen, for the same number of simulations, to be able to compare the posteriors across xen for the same number of simulations. We can then compare the posteriors across different numbers of simulations by comparing the lists of posteriors returned for different n_hf_samples.



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

    # Wrapper function for sbi
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
    _write_time_file(self.time_path, "active_mf_tsnpe", hf_start_time, hf_end_time, hf_data['n_samples'], lf_start_time, lf_end_time, n_lf_samples, total_start_time, total_end_time, one_round_start_time, one_round_end_time, n_rounds)

    if save_posteriors:
        save_dir = f"{self.main_path}/non_amortized_posteriors/"
        name = f"a_mf_tsnpe_LF{n_lf_samples}_HF{hf_data['n_samples']}_xo{xo_index}_seed{seed}_{self.CURR_TIME}.p" 
        dump_pickle(save_dir, name, {
            'posterior': ensemble_posterior,
            'true_x': x_o,
            'true_theta': theta_o,
            'inference_method': 'a_mf_tsnpe',
            'n_rounds': n_rounds,
            'n_simulations': [n_lf_samples, hf_data['n_samples']],
        })
        
    return ensemble_posterior

