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
    

    def compare_lf_hf_posteriors(self, metric, n_hf_sims, n_lf_sims, hf_posteriors, lf_posteriors, true_xen, true_thetas, net_init, type_estimator='lf_and_hf_npe', simulator_name='simulator'):        
        print("LF posteriors", len(lf_posteriors)) 
        print("HF posteriors", len(hf_posteriors)) 
        
        print("n hf sims", n_hf_sims)
        print("n_lf_sims", n_lf_sims)
        
        
        print("len(hf_posteriors)", len(hf_posteriors))
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

        # Compute the metric between the LF and HF posterior 
        # 1. Get lf and hf posterior samples
        print("Comparing LF and HF posteriors with train sizes {} and {}...".format(n_lf_simulations, n_hf_simulations))

        # raise ValueError("Stop here")
        lf_posterior_samples = self.get_posterior_samples(true_xen, true_thetas, lf_posterior, type_estimator, n_lf_simulations, net_init)
        hf_posterior_samples = self.get_posterior_samples(true_xen, true_thetas, hf_posterior, type_estimator, n_hf_simulations, net_init)
        print("lf posterior samples shape", lf_posterior_samples.shape)
        print("hf posterior samples shape", hf_posterior_samples.shape)
        
        # Compute with function ground truth available, but use the lf and hf posterior samples to compute the metric
        n_samples = (n_lf_simulations, n_hf_simulations)
        
        # Plot posterior LF vs HF for the first true_xen
        for i in range(3):
            pairplot = PairPlot(self.true_xen, self.task_setup)
            pairplot._plot_pairplot_with_true_posterior(lf_posterior_samples[i], hf_posterior_samples[i], true_thetas[i], type_estimator, n_train_sims=10**5, name=f"pairplot_lf_hf_comparison_{type_estimator}_{n_lf_simulations}_{n_hf_simulations}_netinit{net_init}_truexen{i}.png", simulator_name=simulator_name)
        #self.check_posterior(mf_posterior, lf_posterior, type_estimator, 10**5, true_xen, None, true_thetas, n_lf_samples=10**5, n_hifi_abc=0, net_init=net_init, true_posterior_samples=None)

        df_one_net_init = self.eval_ground_truth_available(true_xen, metric, lf_posterior_samples, hf_posterior_samples, n_samples, type_estimator, net_init)
        print(df_one_net_init)

        return df_one_net_init
    
    def evaluate_methods(self, true_xen, true_thetas, n_lf_samples, n_hf_samples, n_mf_samples, true_add_ons, all_methods, net_init, mf_abc_weights=None, num_hifi_abc=0, hf_data=[]):  
        print("evaluating methods...")
        df_one_seed = pd.DataFrame()
        
        # Check if there are NaNs in the true data
        if torch.isnan(true_thetas).any() or torch.isnan(true_xen).any():
            raise ValueError("NaNs found in true_thetas or true_xen")

        # REMOVED TEMPORALY. IDK IF that's the issue: mf-abc should overwrite it anyway...
        # Calculate the true_posterior samples only once, and compare to the same samples of the other methods
        if self.evaluation_metric == 'c2st' or self.evaluation_metric == 'wasserstein' or self.evaluation_metric == 'mmd':
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
        for posteriors, n_sims, type_estimator in posterior_configs:   
            if len(posteriors) != 0: # because posteriors are never empty atm
                for f, posterior in enumerate(posteriors): # because posteriors are never empty atm.                     
                    n_simulations = n_sims[f]
                    # mf_abc_weights_element = mf_abc_weights[f]
                    
                    print(f"Evaluating {type_estimator} that has been trained on {n_simulations} samples...")
                    
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

                    if self.evaluation_metric == 'c2st' or self.evaluation_metric == 'wasserstein' or self.evaluation_metric == 'mmd':
                        self.check_posterior(posterior, lf_posterior, type_estimator, n_simulations, true_xen, true_add_ons, true_thetas, n_lf_samples, n_hifi_abc, net_init, true_posterior_samples)
                    else:
                        self.check_posterior(posterior, lf_posterior, type_estimator, n_simulations, true_xen, true_add_ons, true_thetas, n_lf_samples, n_hifi_abc, net_init)
                    
                    
                    if self.evaluation_metric == 'c2st' or self.evaluation_metric == 'wasserstein' or self.evaluation_metric == 'mmd':
                        # Only estimate the posterior samples if we need it.
                        if type_estimator == 'mf_abc':
                            # The posterior of mf-abc = actual samples from posterior: it's a sampling-based method
                            posterior_samples = posterior 
                        else: 
                            # All density-estimator methods
                            posterior_samples = self.get_posterior_samples(true_xen, true_thetas, posterior,
                                                                    type_estimator, n_simulations, net_init)


                        # print("n of posterior samples", posterior_samples.shape)
                        # print("n of true posterior samples", true_posterior_samples.shape)
                        # raise ValueError("Stop here")
                            
                        print("input ps,", posterior_samples.shape)
                        print("input tps,", true_posterior_samples.shape)
                        
                        # raise ValueError("Stop here")
                        df = self.eval_ground_truth_available(true_xen, self.evaluation_metric, posterior_samples, true_posterior_samples, n_simulations, type_estimator, mf_abc_weights=None, num_hifi_abc=n_hifi_abc, net_init=net_init)
                    elif self.evaluation_metric == 'nltp' or self.evaluation_metric == 'nrmse':
                        df = self.evaluate_no_ground_truth(true_xen, true_thetas, posterior, n_simulations, type_estimator, net_init, num_hifi_abc=n_hifi_abc)
                    
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
                # load the true posterior samples if they exist
                try:
                    with open(file_path, "rb") as f:
                        true_posterior_samples = pickle.load(f)
                except FileNotFoundError:
                    print("True posterior samples not found, please generate new true observations.")

            print("true posterior samples shape", true_posterior_samples.shape)
        
        return true_posterior_samples


    def get_posterior_samples(self, true_xen, true_thetas, posterior, type_estimator, n_train_sims, net_init) -> torch.Tensor:
        posterior_samples_over_x = []
        
        # The raw density estimator that does not have the reject-accept algorithm (needed when we have logit-transformed data)
        if isinstance(posterior, list) or type_estimator in ['bo_npe'] or type_estimator in ['mf_tsnpe'] or self.evaluation_metric == 'nrmse':
            density_estimator = posterior
        else:
            density_estimator = posterior.__dict__['posterior_estimator']
            
        for i, x in enumerate(tqdm(true_xen)):
            if isinstance(density_estimator, list) and len(density_estimator) != 0:
                # If density estimator is a list: For the sequential case: Then we have 
                # A density estimator for each x, so we iterate the estimators over the number of x'en.
                posterior_samples = density_estimator[i].sample((self.n_samples_to_generate,), x.unsqueeze(0))
            else:                    
                # Afh van embedding of geen embedding, gaat dit werken of niet
                posterior_samples = density_estimator.sample((self.n_samples_to_generate,), x.unsqueeze(0)) # added unsqueeze after issues with CNN
            posterior_samples_over_x.append(posterior_samples)  
        
        posterior_samples = torch.stack(posterior_samples_over_x)        
        # Unsqueeze posterior samples
        # If 4 dimensional posterior samples, unsqueeze last dim
        posterior_samples = posterior_samples.squeeze(2)
                
        # Save the estimated posterior samples
        save_dir = f"{self.main_path}/posterior_samples/"
        name = f"thetas_{net_init}_{len(true_xen)}_{type_estimator}_{n_train_sims}_{self.config_data['type_lf']}.p"
        post_samples = { 'posterior_samples': posterior_samples, 
                        'n_true_x': len(true_xen),
                        'true_theta': true_thetas,
                        'type_estimator': type_estimator,
                        'n_train_sims': n_train_sims}
        
        dump_pickle(save_dir, name, post_samples)
        
        return posterior_samples
    
    
        
    def eval_ground_truth_available(self, 
                      true_xen: List[Any], 
                      metric: str,
                      posterior_samples: List[Any], 
                      true_posterior_samples: List[Any], 
                      n_simulations: Union[int, Tuple[int, int]], 
                      type_estimator: str,
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
            List of true xen values.
        posterior_samples : List[Any]
            List of posterior samples generated by the model.
        true_posterior_samples : List[Any]
            List of true posterior samples.
        n_simulations : Union[int, Tuple[int, int]]
            Number of simulations. For 'mf_npe', it should be a tuple (n_lf_simulations, n_hf_simulations).
        type_estimator : str
            Type of estimator used. Options are 'npe', 'mf_npe', 'sbi_npe', 'true_comparison'.
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
                
                # print("posterior shape, truepost shape:", posterior_s.shape, true_posterior_s.shape)
                #
                
                # if type_estimator == 'mf_abc':
                #     # Use weighted MMD
                #     print("posterior_shape", posterior_s.shape)
                #     print("true posterior shape", true_posterior_s.shape)
                #     # print("mf_abc_weights[i] shape", mf_abc_weights[i].shape)

                #     med = estimate_sigma_median_heuristic(posterior_s, true_posterior_s)
                #     sigmas = [0.5*med, med, 2*med, 4*med]
                    
                #     metric_vals = batched_weighted_mmd(posterior_s, true_posterior_s, sigma=sigmas, batch_size=1000)
                # else:
                sigma = estimate_sigma_median_heuristic(posterior_s, true_posterior_s)
                metric_vals = batched_biased_mmd(posterior_s, true_posterior_s, sigma=sigma, batch_size=1000)
                
                #metric_vals = biased_mmd(posterior_s, true_posterior_s)
                #print("MMD value", metric_vals)
                #raise ValueError("Stop here")
            else:
                raise ValueError(f"Evaluation metric {metric} not recognized.")
            
            metric_all_runs.append(metric_vals)
        
        m_metric = torch.tensor(metric_all_runs).float()
        mean, ci = get_mean_ci(m_metric)
                                    
        if type_estimator == 'snpe' or 'type_estimator' == 'sbi_npe' or type_estimator == 'npe' or type_estimator == 'tsnpe':
            df = pd.DataFrame({
                'algorithm': type_estimator,
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
            
        elif type_estimator == 'a_mf_tsnpe' or type_estimator == 'mf_tsnpe' or type_estimator == 'bo_npe' or type_estimator == 'active_snpe' or type_estimator == 'mf_npe' or type_estimator == 'lf_and_hf_npe':
            df = pd.DataFrame({
                'algorithm': type_estimator,
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
        elif type_estimator == 'true_comparison': 
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
            
            
        elif type_estimator == 'mf_abc':
            df = pd.DataFrame({
                'algorithm': type_estimator,
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
            raise ValueError(f"Type estimator not recognized. Please implement {type_estimator} in eval_ground_truth_available.")
            
        print(f'evaluation for {type_estimator} with {n_simulations} train simulations completed.')
                
        return df
        
        
    def evaluate_no_ground_truth(self, true_xen, true_thetas, posterior, n_simulations, type_estimator, net_init, num_hifi_abc: Optional[int] = 0) -> pd.DataFrame:        
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
                # Return mean of logprobs
                # log_probs = log_probs.mean(dim=0)
                # print("log_probs bbb", log_probs)
            else:
                log_probs = - posterior.log_prob_batched(true_thetas.unsqueeze(0), true_xen, norm_posterior=False) # norm_posterior=False fails when a lot of leakage
                print("log_probs", log_probs)
                log_probs = log_probs[0].clone().detach() 
                
            mean, ci = get_mean_ci(log_probs)
            raw_data = log_probs
            
        elif self.evaluation_metric == 'nrmse':
            # Get the posterior samples
            posterior_samples = self.get_posterior_samples(true_xen, true_thetas, posterior,
                                                    type_estimator, n_simulations, net_init)
            
            
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
        
        
        if type_estimator == 'npe' or type_estimator == 'sbi_npe' or type_estimator == 'tsnpe': 
            df = pd.DataFrame({
                'algorithm': type_estimator,
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
            
        elif type_estimator == 'mf_npe' or type_estimator == 'bo_npe':       
            df = pd.DataFrame({
                'algorithm': type_estimator,
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
        elif type_estimator == 'a_mf_tsnpe' or type_estimator == 'mf_tsnpe':   
            print("MEAAAN", mean)
            print("CI", ci)
            print("n simulations", n_simulations)
            print("net init", net_init)
            df = pd.DataFrame({
                'algorithm': type_estimator,
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
            
        elif type_estimator == 'mf_abc':
            df = pd.DataFrame({
                'algorithm': type_estimator,
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
            raise ValueError(f"Type estimator not recognized. Please implement {type_estimator} in the evaluate_no_ground_truth.")
        
        print(df)
        
        return df
             


    def check_posterior(self, posterior, lf_posterior, type_estimator, n_simulations, true_xen, true_add_ons, true_thetas, n_lf_samples, n_hifi_abc, net_init, true_posterior_samples=None):
        """
        Test how the posterior estimation works with several tests. This function is not complete yet. 
            1. SBC
            2. PPC
            3. Pairplots
        """
        
        ### Pairplots (works, check other pairplot variations later)
        # Generate posterior samples for pairplots
        if type_estimator == 'mf_abc':
            # The posterior of mf-abc = actual samples from posterior: it's a sampling-based method
            posterior_samples = posterior 
        else: 
            # All density-estimator methods (shape: (n_x, n_samples, n_theta_dim))
            posterior_samples = self.get_posterior_samples(true_xen, true_thetas, posterior,
                                                    type_estimator, n_simulations, net_init)
        
        
        # Posterior predictive checks
        # ppc = PPC(self.true_xen, self.task_setup)
        # for i in range(1):
        #     full_trace_true = true_add_ons['full_trace'][i] if 'full_trace' in true_add_ons else None
        #     print("full_trace_true", full_trace_true)
        #     print("posterior samples", posterior_samples)
        #     print("true_xen", true_xen)
        #     print("fulltracetrue", full_trace_true)
        #     x_pp = ppc._posterior_predictive_check(posterior_samples[i], true_xen[i], full_trace_true, type_estimator, n_simulations, i)
        
        
        
        # PPC with subsampled traces
        # ppc = PPC(self.true_xen, self.task_setup)
        # for i in range(1):
        #     # full_trace_true = true_add_ons['full_trace'][i] if 'full_trace' in true_add_ons else None
            
            
        #     print("full_trace_true", full_trace_true)
        #     print("posterior samples", posterior_samples)
        #     print("true_xen", true_xen)
        #     print("fulltracetrue", full_trace_true)
        #     x_pp = ppc._posterior_predictive_check(posterior_samples[i], true_xen[i], full_trace_true, type_estimator, n_simulations, i)
        
        
        
        
        #print("posterior_samples", posterior_samples)
        
        # raise ValueError("Stop here")
 
        # # Run n number of posterior pairplots
        for i in range(1): # zero based, so 2x
            # Pairplots
            pairplot = PairPlot(self.true_xen, self.task_setup)
            # pairplot._plot_pairplot(posterior_samples[i], true_thetas[i], type_estimator, n_simulations, i)
    
            if true_posterior_samples is not None:                
                # Squeeze if needed
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

                
                # if prior ranges exist
                pairplot._plot_pairplot_with_true_posterior(posterior_samples_i, true_posterior_samples[i], true_thetas[i], type_estimator, n_simulations)
        





