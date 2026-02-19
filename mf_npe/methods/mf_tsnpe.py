import time
import copy
from mf_npe.methods.helpers import _pretrain_model_on_lower_fidelities, _write_time_file
from tqdm import tqdm
from mf_npe.simulator.task2.simulation_func import Integrator
from mf_npe.utils.utils import dump_pickle
from sbi.inference import NPE
from sbi.utils import RestrictedPrior, get_density_thresholder


def train_mf_tsnpe(self, 
                    curr_hf_dataset,
                    curr_lf_datasets,
                    i,
                    true_xen,
                    true_thetas,
                    n_rounds=5,
                    save_posteriors=True,
                    seed=None
                ):
    '''
        MF-TSNPE
    '''
    non_amortized_posteriors = []
    
    """ low-fidelity pre-training is amortized """
    theta_hf, x_hf = curr_hf_dataset['theta'], curr_hf_dataset['x']
    lf_flow, lf_start_time, lf_end_time, x_embedding_lf, n_lf_samples = _pretrain_model_on_lower_fidelities(self, curr_lf_datasets, i)
    
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
        n_hf_samples = len(curr_hf_dataset['x']) # total number of hf samples
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
                # Sample each time from the proposal in the method
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
        print("Number of HF samples: ", curr_hf_dataset['n_samples'])
        print("Number of LF samples: ", curr_lf_datasets['lf'][i]['n_samples']) # only lowest fidelity for now
        
        _write_time_file(self.time_path, "mf_tsnpe", hf_start_time, hf_end_time, curr_hf_dataset['n_samples'], lf_start_time, lf_end_time, curr_lf_datasets['lf'][i]['n_samples'], total_start_time, total_end_time, one_round_start_time, one_round_end_time, n_rounds)
        
        # Save the posterior in a pickle file! 
        if save_posteriors:
            ensemble_post_save = {
                'posterior': non_amortized_posterior, 
                'true_x': x_o,
                'true_theta': true_thetas[i_xo],
                'inference_method': 'mf_tsnpe',
                'n_rounds': n_rounds,
                'n_simulations': [curr_lf_datasets['lf'][i]['n_samples'], curr_hf_dataset['n_samples']],
            }
            # Save a pickle with posterior samples (for 1 xo)
            save_dir = f"{self.main_path}/non_amortized_posteriors/"
            name = f"mf_tsnpe_LF{curr_lf_datasets['lf'][i]['n_samples']}_HF{curr_hf_dataset['n_samples']}_xo{i_xo}_seed{seed}_{self.CURR_TIME}.p" 
            dump_pickle(save_dir, name, ensemble_post_save)
            
        non_amortized_posteriors.append(non_amortized_posterior)
    return non_amortized_posteriors
