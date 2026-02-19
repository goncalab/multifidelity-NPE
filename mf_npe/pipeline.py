import warnings
import matplotlib
matplotlib.use("Agg")  # non-interactive; no windows, fewer semaphores
from sbi import utils as utils

from mf_npe.methods import (
    embedding,
    helpers,
    npe,
    mf_npe,
    active_mf_tsnpe,
    mf_tsnpe,
    npe_from_sbi,
    tsnpe,
    mf_abc,
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
        self.time_path = task_setup.main_path + "/time"
        self.CURR_TIME = task_setup.CURR_TIME

        self.type_embedding_lf = task_setup.config_data['lf_embedding']# 'cnn' # xEncoder, identity, cnn
        self.type_embedding_hf = task_setup.config_data['hf_embedding'] # 'cnn' # xEncoder, identity, cnn

        self.task_setup = task_setup
    
    # ========================================================================
    # Main inference methods
    # ========================================================================
    
    def run_npe(self, data:dict) -> list:
        """
        Run Neural Posterior Estimation
        
        Args:
            data: high-fidelity data
            
        Returns:
            List of posteriors
        """    
        npe_posteriors = []
    
        for i, n_simulations in enumerate(self.batch_hf_sims):  
            posterior = npe.train_npe(self, data[i], n_simulations)
            npe_posteriors.append(posterior)
            
        return npe_posteriors
    
    
    def run_mf_npe(self, data: dict) -> tuple[list, list]:
        """
        Run Multi-Fidelity Neural Posterior Estimation (MF-NPE) on all combinations of LF and HF datasets 
        (e.g., trained on 50, 100 and 1000 samples).
        For each HF dataset, it trains a MF-NPE model using the LF data as a pretrained model.
        
        Args:
            data: Dictionary containing the low-fidelities data and high-fidelity data, with keys like {'lf':..., 'mid'..., 'hf':...}
            
        Returns:
            Tuple of (mf_posteriors, pretrained_posteriors)
        """
        
        mf_posteriors = []
        pretrained_posteriors = [] # keep pretrained model in case it is needed in further analysis
        
        # currently only 1 low-fidelity dataset is supported in MF-NPE (but it's easy to extend if needed, since the dictionary structure is already incorporated)
        lf_mf_data, hf_mf_data = helpers._convert_data_to_mf_format(data)

        # The length of hf_mf_data is the same as lf_mf_data and is a permutation of both of the datasets.
        for i, _ in enumerate(hf_mf_data):    
            x_hf, theta_hf = hf_mf_data[i]['x'], hf_mf_data[i]['theta']
            n_hf_samples = hf_mf_data[i]['n_samples']

            pretrained_posterior, mf_posterior = mf_npe.train_mf_npe(self, x_hf, theta_hf, n_hf_samples, lf_mf_data, i)

            pretrained_posteriors.append(pretrained_posterior)
            mf_posteriors.append(mf_posterior)
                    
        return mf_posteriors, pretrained_posteriors
            
    def run_tsnpe(self, hf_data, true_xen, true_thetas, n_rounds=5,
                  plot_thetas=False, save_stuff=True,
                  save_posteriors=True, seed=None):
        """
        Run TSNPE
        
        Args:
            hf_data: High-fidelity data
            true_xen: True observations
            true_thetas: True parameters
            n_rounds: Number of rounds
            save_posteriors: Whether to save posteriors
            seed: Random seed
            
        Returns:
            List of list of posteriors for each observation x_o
        """
        tsnpe_posteriors = []  
        
        for i, data in enumerate(hf_data):
            posteriors_per_xo = tsnpe.train_tsnpe(
                self, hf_data[i], true_xen, true_thetas, n_rounds,
                save_posteriors, seed
            )
            tsnpe_posteriors.append(posteriors_per_xo)
            
        return tsnpe_posteriors
    
    
    def run_mf_tsnpe(self, data, true_xen, true_thetas, n_rounds=5,
                     plot_thetas=False, save_stuff=True, 
                     save_posteriors=True, seed=None):
        """
        Run Multi-Fidelity TSNPE
        
        Args:
            data: Multi-fidelity data
            true_xen: True observations
            true_thetas: True parameters
            n_rounds: Number of rounds
            save_posteriors: Whether to save posteriors
            seed: Random seed
            
        Returns:
            List of list of posteriors
        """
        mf_tsnpe_posteriors = []   
    
        # only for 1 (the lowest) lf dataset
        lf_mf_data, hf_mf_data = helpers._convert_data_to_mf_format(data)
                    
        for i, _ in enumerate(hf_mf_data):
            posteriors_per_xo = mf_tsnpe.train_mf_tsnpe(
                self, hf_mf_data[i], lf_mf_data, i, 
                true_xen, true_thetas, n_rounds, save_posteriors, seed
            )
            mf_tsnpe_posteriors.append(posteriors_per_xo)
            
        return mf_tsnpe_posteriors
    
    
    def run_active_mf_tsnpe(self, data, true_xen, true_thetas, 
                           active_learning_pct=0.8, n_rounds=5,
                           n_theta_samples=250, n_ensemble_members=5,
                           plot_thetas=False, save_posteriors=True, seed=None):
        """
        Run Active Multi-Fidelity TSNPE
        
        Args:
            data: Multi-fidelity data
            true_xen: True observations
            true_thetas: True parameters
            active_learning_pct: Percentage for active learning
            n_rounds: Number of rounds
            n_theta_samples: Number of theta samples (inferred from n_hf_samples)
            n_ensemble_members: Number of ensemble members
            plot_thetas: Whether to plot thetas
            save_posteriors: Whether to save posteriors
            seed: Random seed
            
        Returns:
            List of list of posteriors
        """
        a_mf_tsnpe_posteriors = []   
    
        # only for 1 (the lowest) lf dataset
        lf_mf_data, hf_mf_data = helpers._convert_data_to_mf_format(data)
        
        for i, hf_samples in enumerate(hf_mf_data):
        
            posteriors_per_xo = active_mf_tsnpe.train_active_mf_tsnpe(
                self, hf_mf_data[i], lf_mf_data, i, 
                true_xen, true_thetas,
                active_learning_pct, n_rounds, n_theta_samples,
                n_ensemble_members, plot_thetas, save_posteriors, seed
            )
            a_mf_tsnpe_posteriors.append(posteriors_per_xo)
        
        return a_mf_tsnpe_posteriors
    
    
    def run_mf_abc(self, lf_data, true_thetas):
        """
        Run Multi-Fidelity ABC
        
        Args:
            lf_data: Low-fidelity data
            true_thetas: True parameters
            
        Returns:
            Tuple of (posterior_samples, posterior_weights, num_hifi_total)
        """
        return mf_abc.run_mf_abc(self, lf_data, true_thetas)
    
    
    def run_sbi(self, hf_data):
        """
        Run standard NPE with the SBI api, to compare with our implementation of NPE. 
        This is not used in the main paper, but can be used for sanity check and further analysis.
        
        Args:
            hf_data: High-fidelity data
            
        Returns:
            List of posteriors
        """
        sbi_posteriors = []

        for i in range(len(self.batch_hf_sims)):
            posterior = npe_from_sbi.train_npe_with_sbi(self, hf_data[i])
            sbi_posteriors.append(posterior)
        
        return sbi_posteriors   