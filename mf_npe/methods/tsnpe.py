import time
from tqdm import tqdm
from mf_npe.methods.helpers import _write_time_file
from mf_npe.simulator.task2.simulation_func import Integrator
from mf_npe.utils.utils import dump_pickle
from sbi.inference import NPE
from sbi.utils import RestrictedPrior, get_density_thresholder


def train_tsnpe(self, curr_hf_data, 
                    true_xen,
                    true_thetas,
                    n_rounds=5,
                    save_posteriors=True,
                    seed=None
                ):
    '''
        TSNPE SBI
    '''
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
        n_hf_samples = len(curr_hf_data['x']) # total number of hf samples
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
            
            non_amortized_posterior = inference.build_posterior(density_estimator).set_default_x(x_o)  # 

            if self.task == 'task2':
                accept_reject_fn = get_density_thresholder(non_amortized_posterior, quantile=1e-3) # was 1e-6, lower for test
            else:
                accept_reject_fn = get_density_thresholder(non_amortized_posterior, quantile=1e-6)

            proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
            one_round_end_time = time.time()
        
        hf_end_time = time.time()
        total_end_time = time.time()
        
        _write_time_file(self.time_path, "tsnpe", hf_start_time, hf_end_time, curr_hf_data['n_samples'], total_start_time, total_end_time, one_round_start_time, one_round_end_time, n_rounds)
        
        # Save the posterior in a pickle file! 
        if save_posteriors:
            ensemble_post_save = {
                'posterior': non_amortized_posterior, 
                'true_x': x_o,
                'true_theta': true_thetas[i_xo],
                'inference_method': 'tsnpe',
                'n_rounds': n_rounds,
                'n_simulations': curr_hf_data['n_samples'],
            }
            # Save a pickle with posterior samples (for 1 xo)
            save_dir = f"{self.main_path}/non_amortized_posteriors/"
            name = f"tsnpe_{curr_hf_data['n_samples']}_xo{i_xo}_seed{seed}_{self.CURR_TIME}.p" 
            dump_pickle(save_dir, name, ensemble_post_save)
            
        non_amortized_posteriors.append(non_amortized_posterior)
    return non_amortized_posteriors
