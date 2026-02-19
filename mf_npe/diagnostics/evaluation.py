import pickle
from typing import Any, List, Optional, Tuple, Union
import pandas as pd
from seaborn import pairplot
import torch
from tqdm import tqdm 
from mf_npe.diagnostics.pairplots import PairPlot
from mf_npe.diagnostics.ppc import PPC
from mf_npe.diagnostics.sbc import SimulationBasedCalibration
from mf_npe.pipeline import Pipeline
from sbi.samplers.rejection.rejection import rejection_sample
from sbi import utils as utils
from typing import List, Union
from mf_npe.utils.mmd import batched_biased_mmd, batched_weighted_mmd, estimate_sigma_median_heuristic
from mf_npe.utils.utils import dump_pickle
from scipy.stats import wasserstein_distance_nd
from mf_npe.benchmarking.c2st_sbi import c2st
from sbi import analysis as analysis
from mf_npe.utils.calculate_error import get_mean_ci
import os
class Evaluation(Pipeline):
    
    def __init__(self, true_xen, task_setup, eval_metric='nltp'):
        super().__init__(true_xen, task_setup)
        
        self.true_xen = true_xen
        self.task_setup = task_setup
        self.evaluation_metric = eval_metric
        
        if self.evaluation_metric == 'c2st' or self.evaluation_metric == 'wasserstein' or self.evaluation_metric == 'mmd':
            self.ground_truth_available = True
        else:   
            self.ground_truth_available = False
    

    def compare_lf_hf_posteriors(self, metric, n_hf_sims, n_lf_sims, hf_posteriors, lf_posteriors, true_xen, true_thetas, net_init, inference_method='lf_and_hf_npe', simulator_name='simulator'):                
        if len(hf_posteriors) == 0:
            print("hf_posteriors", hf_posteriors)
            hf_posterior = hf_posteriors[0]
            n_hf_simulations = n_hf_sims[0]
        else:
            hf_posterior = hf_posteriors[-1]
            n_hf_simulations = n_hf_sims[-1]
            
        if len(lf_posteriors) == 0:
            lf_posterior = lf_posteriors[0]
            n_lf_simulations = n_lf_sims[0]
        else:
            lf_posterior = lf_posteriors[-1]
            n_lf_simulations = n_lf_sims[-1]

        # If lf_posteriors is a list, then posterior of first element (biggest difference with the high-fidelity model)
        if isinstance(lf_posterior, dict) and "lf" in lf_posterior:
            lf_posterior = lf_posterior["lf"]
        
        assert n_hf_simulations == 10**5, "n_hf_simulations is not 10**5"
        assert n_lf_simulations == 10**5, "n_lf_simulations is not 10**5"

        lf_posterior_samples = self.get_posterior_samples(true_xen, true_thetas, lf_posterior, inference_method, n_lf_simulations, net_init)
        hf_posterior_samples = self.get_posterior_samples(true_xen, true_thetas, hf_posterior, inference_method, n_hf_simulations, net_init)
        
        # Compute with function ground truth available, but use the lf and hf posterior samples to compute the metric
        n_samples = (n_lf_simulations, n_hf_simulations)
        
        # Plot posterior LF vs HF for the first true_xen
        for i in range(3):
            pairplot = PairPlot(self.true_xen, self.task_setup)
            pairplot._plot_pairplot_with_true_posterior(lf_posterior_samples[i], hf_posterior_samples[i], true_thetas[i], inference_method, n_train_sims=10**5, name=f"pairplot_lf_hf_comparison_{inference_method}_{n_lf_simulations}_{n_hf_simulations}_netinit{net_init}_truexen{i}.png", simulator_name=simulator_name)
        
        #self.check_posterior("sbc", mf_posterior, lf_posterior, inference_method, 10**5, true_xen, None, true_thetas, n_lf_samples=10**5, n_hifi_abc=0, net_init=net_init, true_posterior_samples=None)

        df_one_net_init = self.eval_ground_truth_available(true_xen, metric, lf_posterior_samples, hf_posterior_samples, n_samples, inference_method, net_init)

        return df_one_net_init
    
    
    
    def evaluate_methods(self, true_xen, true_thetas, n_lf_samples, n_hf_samples, n_mf_samples, true_add_ons, all_methods, net_init, mf_abc_weights=None, num_hifi_abc=0, hf_data=[]):  
        print("evaluating methods...")
        df_one_seed = pd.DataFrame()
        
        # Check if there are NaNs in the true data
        if torch.isnan(true_thetas).any() or torch.isnan(true_xen).any():
            raise ValueError("NaNs found in true_thetas or true_xen")

        # Calculate the true_posterior samples only once, and compare to the same samples of the other methods
        if self.ground_truth_available:
            true_posterior_samples = self.get_true_posterior_samples(true_xen, self.hf_prior, self.hf_simulator, self.n_samples_to_generate)
        else:
            true_posterior_samples = None
            
        # Compare HF and LF posterior distance/difference        
        posterior_configs = [
            (all_methods['hf_posteriors'], n_hf_samples, 'npe'),
            (all_methods['mf_posteriors'], n_mf_samples, 'mf_npe'),
            (all_methods['sbi_posteriors'], n_hf_samples, 'sbi_npe'),
            (all_methods['active_snpe_posteriors'], n_mf_samples, 'a_mf_tsnpe'),
            (all_methods['mf_snpe_posteriors'], n_mf_samples, 'mf_tsnpe'),
            (all_methods['hf_tsnpe_posteriors'], n_hf_samples, 'tsnpe'),
            (all_methods['mf_abc'], n_lf_samples, 'mf_abc'),
        ]

        # Only add if exists
        if 'bo_posteriors' in all_methods:
            posterior_configs.append((all_methods['bo_posteriors'], n_mf_samples, 'bo_npe'))

        # n_sims fixen: array van wat echte num sims is gebruikt
        for posteriors, n_sims, inference_method in posterior_configs:   
            if len(posteriors) != 0: # because posteriors are never empty atm
                for f, posterior in enumerate(posteriors): # because posteriors are never empty atm.                     
                    n_simulations = n_sims[f]
                    # mf_abc_weights_element = mf_abc_weights[f]
                    
                    print(f"Evaluating {inference_method} that has been trained on {n_simulations} samples...")
                    
                    # lf_posteriors are used to compare the lf and hf posteriors in the paper
                    lf_posteriors = all_methods['lf_posteriors']
                    if lf_posteriors is None or not lf_posteriors:
                        lf_posterior = None
                    else:
                        lf_posterior = lf_posteriors[f] # corresponds to n of hf_posteriors
                        
                    # n_hifi_abc is only used for the mf_abc method
                    if num_hifi_abc != 0:
                        n_hifi_abc = num_hifi_abc[f]
                    else:
                        n_hifi_abc = 0

                    if self.ground_truth_available:
                        self.check_posterior("pairplot", posterior, lf_posterior, inference_method, n_simulations, true_xen, true_add_ons, true_thetas, n_lf_samples, n_hifi_abc, net_init, true_posterior_samples)
                    else:
                        self.check_posterior("pairplot", posterior, lf_posterior, inference_method, n_simulations, true_xen, true_add_ons, true_thetas, n_lf_samples, n_hifi_abc, net_init)
                    
                    if self.ground_truth_available:
                        # Only estimate the posterior samples if we need it.
                        if inference_method == 'mf_abc':
                            # The posterior of mf-abc = actual samples from posterior: it's a sampling-based method
                            posterior_samples = posterior 
                        else: 
                            # All density-estimator methods
                            posterior_samples = self.get_posterior_samples(true_xen, true_thetas, posterior,
                                                                    inference_method, n_simulations, net_init)
                    
                        df = self.eval_ground_truth_available(true_xen, self.evaluation_metric, posterior_samples, true_posterior_samples, n_simulations, inference_method, mf_abc_weights=None, num_hifi_abc=n_hifi_abc, net_init=net_init)
                    else:
                        df = self.evaluate_no_ground_truth(true_xen, true_thetas, posterior, n_simulations, inference_method, net_init, num_hifi_abc=n_hifi_abc)
                    
                    # append the results with the current density_estimator (npe, mf_npe or sbi_npe)
                    df_one_seed = pd.concat([df_one_seed, df], ignore_index=True)
            
        return df_one_seed 


    def get_true_posterior_samples(self, true_xen, prior, simulator, n_samples):
        # Do not logit transform here
        true_posterior_samples = []
        
        # We can directly load the true posterior samples from the benchmarking paper of Lueckmann et al. (2019)
        if self.sim_name == 'SLCP' or self.sim_name == 'SIR' or self.sim_name == 'LotkaVolterra':
            true_posterior_samples = simulator.get_reference_posterior_samples(n_samples, true_xen, self.main_path)
        else: 
            save_dir = f"{self.main_path}/true_posterior_samples/"
            name = f"thetas_n_true_xen{len(true_xen)}.p"
            
            generate_new = True
            if generate_new:
                for i, x_o in tqdm(enumerate(true_xen), total=len(true_xen), desc="Posterior samples"):
                    # Generate samples from the true posterior
                    true_posterior_s, _ = rejection_sample(potential_fn=lambda theta: prior.log_prob(theta) + simulator.true_log_likelihood(theta, x_o),
                        proposal=prior, m=2.0, num_samples=n_samples)
                    true_posterior_samples.append(true_posterior_s)
                
                true_posterior_samples = torch.stack(true_posterior_samples)
                            
                # Save the true_posterior_samples as a file
                dump_pickle(save_dir, name, true_posterior_samples)
            else:
                file_path = os.path.join(save_dir, name)
                # Load the true posterior samples if they exist
                try:
                    with open(file_path, "rb") as f:
                        true_posterior_samples = pickle.load(f)
                except FileNotFoundError:
                    print("True posterior samples not found, please generate new true observations.")
        
        return true_posterior_samples


    def get_posterior_samples(self, true_xen, true_thetas, posterior, inference_method, n_train_sims, net_init) -> torch.Tensor:
        posterior_samples_over_x = []
        
        # The raw density estimator that does not have the reject-accept algorithm (needed when we have logit-transformed data)
        if isinstance(posterior, list) or inference_method in ['bo_npe'] or inference_method in ['mf_tsnpe'] or self.evaluation_metric == 'nrmse':
            density_estimator = posterior
        else:
            density_estimator = posterior.__dict__['posterior_estimator']
            
        for i, x in enumerate(tqdm(true_xen)):
            if isinstance(density_estimator, list) and len(density_estimator) != 0:
                # If density estimator is a list: For the sequential case: Then we have 
                # A density estimator for each x, so we iterate the estimators over the number of x'en.
                posterior_samples = density_estimator[i].sample((self.n_samples_to_generate,), x.unsqueeze(0))
            else:                    
                posterior_samples = density_estimator.sample((self.n_samples_to_generate,), x.unsqueeze(0)) # added unsqueeze after issues with CNN
            posterior_samples_over_x.append(posterior_samples)  
        
        posterior_samples = torch.stack(posterior_samples_over_x)        
        # Unsqueeze posterior samples
        posterior_samples = posterior_samples.squeeze(2)
                
        # Save the estimated posterior samples
        save_dir = f"{self.main_path}/posterior_samples/"
        name = f"thetas_{net_init}_{len(true_xen)}_{inference_method}_{n_train_sims}_{self.config_data['type_lf']}.p"
        post_samples = { 'posterior_samples': posterior_samples, 
                        'n_true_x': len(true_xen),
                        'true_theta': true_thetas,
                        'inference_method': inference_method,
                        'n_train_sims': n_train_sims}
        
        dump_pickle(save_dir, name, post_samples)
        
        return posterior_samples
    
    
        
    def eval_ground_truth_available(self, 
                      true_xen: List[Any], 
                      metric: str,
                      posterior_samples: List[Any], 
                      true_posterior_samples: List[Any], 
                      n_simulations: Union[int, Tuple[int, int]], 
                      inference_method: str,
                      gamma: Optional[float] = None, 
                      mu_offset: Optional[float] = None,
                      mf_abc_weights: Optional[List[Any]] = None,
                      num_hifi_abc: Optional[int] = 0,
                      net_init=12) -> pd.DataFrame:
        """
        Evaluate the distance between posterior samples and true posterior samples with available metrics (c2st, wasserstein, mmd).
        Gamma and mu_offset are only for the comparison between the analytical solutions of LF and HF of the OU process.

        Parameters:
        -----------
        true_xen : List[Any]
            List of true observation values.
        posterior_samples : List[Any]
            List of posterior samples generated by the model.
        true_posterior_samples : List[Any]
            List of true posterior samples.
        n_simulations : Union[int, Tuple[int, int]]
            Number of simulations. For 'mf_npe', it should be a tuple (n_lf_simulations, n_hf_simulations).
        inference_method : str
            The algorithm used for inference. Options are for instance 'npe', 'mf_npe', 'sbi_npe', 'a_mf_tsnpe'.
        gamma : Optional[float], default=None
            Gamma value for the comparison between the analytical solutions of LF and HF of the OU process.
        mu_offset : Optional[float], default=None
            Mu offset value for the comparison between the analytical solutions of LF and HF of the OU process.
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the evaluation results including mean, confidence intervals, and raw data over which mean and confidence intervals were calculated.
        """
                        
        metric_all_runs = []
        
        for i, x in enumerate(tqdm(true_xen)):
            posterior_s = posterior_samples[i]
            true_posterior_s = true_posterior_samples[i]
                        
            if metric == 'c2st':
                metric_vals = c2st(posterior_s, true_posterior_s)
            elif metric == 'wasserstein':
                metric_vals = wasserstein_distance_nd(posterior_s, true_posterior_s)
            elif metric == 'mmd':
                sigma = estimate_sigma_median_heuristic(posterior_s, true_posterior_s)
                metric_vals = batched_biased_mmd(posterior_s, true_posterior_s, sigma=sigma, batch_size=1000)
            else:
                raise ValueError(f"Evaluation metric {metric} not recognized.")
            
            metric_all_runs.append(metric_vals)
        
        m_metric = torch.tensor(metric_all_runs).float()
        mean, ci = get_mean_ci(m_metric)
                                    
        if inference_method == 'snpe' or inference_method == 'sbi_npe' or inference_method == 'npe' or inference_method == 'tsnpe':
            df = pd.DataFrame({
                'algorithm': inference_method,
                'evaluation_metric': self.evaluation_metric,
                'task': self.sim_name,
                'net_init': net_init,
                'n_lf_simulations': 0,
                'n_hf_simulations': n_simulations,
                'mean': mean,
                'ci_max': ci[0],
                'ci_min': ci[1],
                'raw_data': m_metric,
            })
            
        elif inference_method == 'a_mf_tsnpe' or inference_method == 'mf_tsnpe' or inference_method == 'bo_npe' or inference_method == 'active_snpe' or inference_method == 'mf_npe' or inference_method == 'lf_and_hf_npe':
            df = pd.DataFrame({
                'algorithm': inference_method,
                'evaluation_metric': self.evaluation_metric,
                'task': self.sim_name,
                'net_init': net_init,
                'n_lf_simulations': n_simulations[0],
                'n_hf_simulations': n_simulations[1],
                'mean': mean,
                'ci_max': ci[0],
                'ci_min': ci[1],
                'raw_data': m_metric,
            })

        # This dataframe is for evaluating the distance between the LF and HF models of the first task (OU-process)
        elif inference_method == 'true_comparison': 
            df = pd.DataFrame({
                'evaluation_metric': self.evaluation_metric,
                'task': self.sim_name,
                'net_init': net_init,
                'gamma': gamma,
                'mu_offset': mu_offset,
                'mean': mean,
                'ci_max': ci[0],
                'ci_min': ci[1],
                'raw_data': m_metric,
            })
            
        elif inference_method == 'mf_abc':
            df = pd.DataFrame({
                'algorithm': inference_method,
                'evaluation_metric': self.evaluation_metric,
                'task': self.sim_name,
                'net_init': net_init,
                'n_lf_simulations': n_simulations,
                'n_hf_simulations': num_hifi_abc, 
                'mean': mean,
                'ci_max': ci[0],
                'ci_min': ci[1],
                'raw_data': m_metric,
            })
        else:
            raise ValueError(f"Inference method not recognized. Please implement {inference_method} in eval_ground_truth_available.")
            
        print(f'evaluation for {inference_method} with {n_simulations} train simulations completed.')
                
        return df
        
        
    def evaluate_no_ground_truth(self, 
                                 true_xen: List[Any], 
                                 true_thetas: List[Any],
                                 posterior: Any, 
                                 n_simulations: Union[int, Tuple[int, int]],
                                 inference_method: str, 
                                 net_init: int, 
                                 num_hifi_abc: Optional[int] = 0) -> pd.DataFrame:     
        """Evaluate how good the performance of the density estimation is using available metrics (nltp, nrmse).

        Args:
            true_xen (List[Any]): List of true observation values.
            true_thetas (List[Any]): List of posterior samples generated by the model.
            posterior (Any): The posterior object (e.g., a trained neural network or an SBI posterior).
            n_simulations (Union[int, Tuple[int, int]]): Number of simulations used for training. For `mf_npe`, it should be a tuple (n_lf_simulations, n_hf_simulations).
            inference_method (str): The algorithm used for inference. Options are for instance `npe`, `mf_npe`, `sbi_npe`, `a_mf_tsnpe`.
            net_init (int): _description_
            num_hifi_abc (Optional[int], optional): _description_. Defaults to 0.

        Returns:
            pd.DataFrame: DataFrame containing the evaluation results including mean, confidence intervals, and raw data over which mean and confidence intervals were calculated.
        """
           
        # Give an error if true_x contains NaNs
        if torch.isnan(true_xen).any():
            raise ValueError("NaNs found in true_xen")
        if torch.isnan(true_thetas).any():
            raise ValueError("NaNs found in true_thetas")
        

        if self.evaluation_metric == 'nltp':
            if isinstance(posterior, list):
                # evaluate over single pair of true theta/true x for each posterior (log_prob takes care of the logprob for each ensemble).           
                log_probs = [ - posterior[i].log_prob(true_thetas[i], true_xen[i]) for i in range(len(posterior))]
                print("log_probs list", log_probs)
                # stack the list into a single tensor
                log_probs = torch.stack(log_probs)
                print("log_probs aaa", log_probs)
                log_probs = log_probs.clone().detach() 
            else:
                log_probs = - posterior.log_prob_batched(true_thetas.unsqueeze(0), true_xen, norm_posterior=False) # norm_posterior=False fails when a lot of leakage
                print("log_probs", log_probs)
                log_probs = log_probs[0].clone().detach() 
                
            mean, ci = get_mean_ci(log_probs)
            raw_data = log_probs
            
        elif self.evaluation_metric == 'nrmse':
            # Get the posterior samples
            posterior_samples = self.get_posterior_samples(true_xen, true_thetas, posterior,
                                                    inference_method, n_simulations, net_init)
            
            # (N, 1, D) - (N, S, D) → (N, S, D)
            diff = posterior_samples - true_thetas[:, None, :]

            # MSE over samples → (N, D)
            mse = diff.pow(2).mean(dim=1)

            # RMSE → (N, D)
            rmse = mse.sqrt()

            # Normalize by global range over all true_thetas per dim → (D,)
            rng = (true_thetas.max(dim=0).values - true_thetas.min(dim=0).values).clamp_min(1e-12)

            # Final (N, D)
            nrmse_over_thetas = rmse / rng
                    
            nrmse_over_true_thetas = torch.mean(nrmse_over_thetas)
            print("NRMSE over true thetas", nrmse_over_true_thetas)

            mean = [nrmse_over_true_thetas.item() if torch.is_tensor(nrmse_over_true_thetas) else nrmse_over_true_thetas]
            raw_data = mean
            ci = (0, 0) # no CI, since NRMSE is deterministic
            
        else:
            raise ValueError(f"Evaluation metric {self.evaluation_metric} not recognized for evaluate_no_ground_truth.")
        
        
        if inference_method == 'npe' or inference_method == 'sbi_npe' or inference_method == 'tsnpe': 
            df = pd.DataFrame({
                'algorithm': inference_method,
                'evaluation_metric': self.evaluation_metric,
                'task': self.sim_name,
                'net_init': net_init,
                'n_lf_simulations': 0,
                'n_hf_simulations': n_simulations, 
                'mean': mean,
                'ci_max': ci[0],
                'ci_min': ci[1],
                'raw_data': raw_data,
                })
            
        elif inference_method == 'mf_npe' or inference_method == 'bo_npe':       
            df = pd.DataFrame({
                'algorithm': inference_method,
                'evaluation_metric': self.evaluation_metric,
                'task': self.sim_name,
                'net_init': net_init,
                'n_lf_simulations': n_simulations[0],
                'n_hf_simulations': n_simulations[1],
                'mean': mean,
                'ci_max': ci[0],
                'ci_min': ci[1],
                'raw_data': raw_data,
            })
        elif inference_method == 'a_mf_tsnpe' or inference_method == 'mf_tsnpe':   
            print("MEAAAN", mean)
            print("CI", ci)
            print("n simulations", n_simulations)
            print("net init", net_init)
            df = pd.DataFrame({
                'algorithm': inference_method,
                'evaluation_metric': self.evaluation_metric,
                'task': self.sim_name,
                'net_init': net_init,
                'n_lf_simulations': n_simulations[0],
                'n_hf_simulations': n_simulations[1],
                'mean': mean,
                'ci_max': ci[0],
                'ci_min': ci[1],
                'raw_data': mean,
            })
            
        elif inference_method == 'mf_abc':
            df = pd.DataFrame({
                'algorithm': inference_method,
                'evaluation_metric': self.evaluation_metric,
                'task': self.sim_name,
                'net_init': net_init,
                'n_lf_simulations': n_simulations,
                'n_hf_simulations': num_hifi_abc, 
                'mean': mean,
                'ci_max': ci[0],
                'ci_min': ci[1],
                'raw_data': raw_data,
            })
            
        else:
            raise ValueError(f"Inference method not recognized. Please implement {inference_method} in the evaluate_no_ground_truth.")
        
        print(df)
        
        return df
             


    def check_posterior(self, test_method, posterior, lf_posterior, inference_method, n_simulations, true_xen, true_add_ons, true_thetas, n_lf_samples, n_hifi_abc, net_init, true_posterior_samples=None):
        """
        Test how the posterior estimation works with several tests.
            1. SBC
            2. PPC
            3. Pairplots
        """
        
        if test_method == 'sbc':
            # Should be tested
            sbc = SimulationBasedCalibration(self.true_xen, self.task_setup)
            sbc.run_sbc(posterior, inference_method, n_simulations, net_init)
            
        elif test_method == 'ppc':
            # Posterior predictive checks
            ppc = PPC(self.true_xen, self.task_setup)
            for i in range(1):
                full_trace_true = true_add_ons['full_trace'][i] if 'full_trace' in true_add_ons else None
                x_pp = ppc._posterior_predictive_check(posterior_samples[i], true_xen[i], full_trace_true, inference_method, n_simulations, i)
            
        elif test_method == 'pairplot':
            # Generate posterior samples for pairplots
            if inference_method == 'mf_abc':
                # The posterior of mf-abc = actual samples from posterior: it's a sampling-based method
                posterior_samples = posterior 
            else: 
                # All density-estimator methods (shape: (n_x, n_samples, n_theta_dim))
                posterior_samples = self.get_posterior_samples(true_xen, true_thetas, posterior,
                                                        inference_method, n_simulations, net_init)
            
            # Run n number of posterior pairplots
            for i in range(1): # zero based, so 2x
                pairplot = PairPlot(self.true_xen, self.task_setup)
                
                if true_posterior_samples is not None:                
                    if posterior_samples[i].shape[1] == 1:
                        posterior_samples_i = torch.squeeze(posterior_samples[i], dim=1)
                    else:
                        posterior_samples_i = posterior_samples[i]

                    # Check for NaNs
                    if torch.isnan(posterior_samples[i]).any():
                        print("NaNs found in posterior samples")
                        raise ValueError("NaNs found in posterior samples")
                    if torch.isnan(true_posterior_samples[i]).any():
                        print("NaNs found in true posterior samples")
                        raise ValueError("NaNs found in true posterior samples")
                    if torch.isnan(true_thetas[i]).any():
                        print("NaNs found in true thetas")
                        raise ValueError("NaNs found in true thetas")
                    
                    # if true posterior exists
                    pairplot._plot_pairplot_with_true_posterior(posterior_samples_i, true_posterior_samples[i], true_thetas[i], inference_method, n_simulations)
                    # if true posterior does not exist
                    # pairplot._plot_pairplot(posterior_samples[i], true_thetas[i], inference_method, n_simulations, i)
        else:
            raise ValueError(f"Test method {test_method} not recognized.")
        


