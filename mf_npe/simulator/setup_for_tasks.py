import torch
from mf_npe.simulator.task1.GaussianSamples import GaussianSamples
from mf_npe.simulator.task1.OUprocess import OUprocess
from mf_npe.simulator.task1.OUprocessNoise import OUprocessNoise
from mf_npe.simulator.task1.OUprocessXinverse import OUprocessXinverse
from mf_npe.simulator.task1.OUprocessTshift import OUprocessTshift
from mf_npe.simulator.task1.OUprocessXinverseNoise import OUprocessXinverseNoise
from mf_npe.simulator.task2.L5PC import L5PC # second experiment. This is not a low/high fidelity model

from mf_npe.simulator.task3.MeanField import MeanField
from mf_npe.simulator.task3.PolynomialNetwork import PolynomialNetwork
from mf_npe.simulator.task4.GaussianBlobMidResolution import GaussianBlobMidResolution
from mf_npe.simulator.task4.GaussianBlobLowResolution import GaussianBlobLowResolution
from mf_npe.simulator.task4.GaussianBlob import GaussianBlob
from mf_npe.simulator.task4.GaussianBlobTshift import GaussianBlobTshift
from mf_npe.simulator.task4.GaussianBlobXinverse import GaussianBlobXinverse
from mf_npe.simulator.task5.SLCP import SLCP
from mf_npe.simulator.task5.SLCP_lf import SLCP_lf
from mf_npe.simulator.task6.LotkaVolterra import LotkaVolterra
from mf_npe.simulator.task7.SIR_lf import SIR_lf
from mf_npe.simulator.task7.SIR import SIR
from mf_npe.simulator.task7.SIR_lowres import SIR_lowres

def get_ouprocess_setup(theta_dim, n_true_xen, experiment_name):
    """
    Get the setup for the OU process simulation.
    """
    # Set the random seed for reproducibility
    # torch.manual_seed(seed)
    
    mu_offset = torch.tensor([3.0]) 
    gamma = torch.tensor([0.5])
    
    config_data = dict(
        sim_name=experiment_name,
        task = 'task1',
        x_dim_out= 10, 
        x_dim_lf = 10,
        x_dim_hf = 10,
        # Summary stats: Take only first 100 samples and take 10 subsamples
        length_total_trace = 101, # Must be 101 or 1001, 10001, 100001
        theta_dim= theta_dim, # 2-4 number of trainable parameters for HF model
        subsample_rate = 10,
        val_fraction = 0.1, # 0.1 is same as in sbi
        n_true_x = n_true_xen, # Number of true samples to generate
        logspace = False, # sample not linearly each 10 steps but logaritmicely
        first_n_samples = 100, # Number of samples to take from the full trace
        n_samples_to_generate = 1000, 
        mu_offset= mu_offset, # Offset for the mean of the Gaussian
        gamma= gamma, # Offset for the mean of the Gaussian
        # Prior ranges of HF model
        all_prior_ranges = {
            'mu': (0.1, 3.0),
            'sigma': (0.1, 0.6),
            'gamma': (0.1, 1.0),
            'mu_offset': (0.0, 4.0)
        },
        noise = torch.tensor([0.01]), # Noise level for the simulation for test why transfer learning works
        #evaluation_metric = 'mmd', # use `c2st`, `wasserstein` or `mmd`: since for this task we have a ground truth
        type_lf = '', # Just for tests of for how close the lf model is to the hf model, else 'gs', 'x_inv', 't_shift', 'hf'
        lf_embedding = 'identity',
        hf_embedding = 'identity',
        n_fidelities = 2, # Number of fidelities
    )
    
    if config_data['type_lf'] == 'gs':
        lf_simulator = GaussianSamples(config_data)
    elif config_data['type_lf'] == 'x_inv':
        lf_simulator = OUprocessXinverse(config_data, gamma, mu_offset)  
    elif config_data['type_lf'] == 't_shift':
        lf_simulator = OUprocessTshift(config_data, gamma, mu_offset)
    elif config_data['type_lf'] == 'hf':
        lf_simulator = OUprocess(config_data, gamma, mu_offset) 
    elif config_data['type_lf'] == 'noise':
        lf_simulator = OUprocessNoise(config_data, gamma, mu_offset, config_data['noise'])
    elif config_data['type_lf'] == 'noise_inv':
        lf_simulator = OUprocessXinverseNoise(config_data, gamma, mu_offset, config_data['noise'])
    else:
        lf_simulator = GaussianSamples(config_data)
        
    hf_simulator = OUprocess(config_data, gamma, mu_offset)  
    
    return config_data, lf_simulator, hf_simulator


def get_l5pc_setup(theta_dim, n_true_xen, experiment_name):
    config_data = dict(
        sim_name=experiment_name,
        task = 'task2',
        x_dim_out= 4, 
        x_dim_lf = 4,
        x_dim_hf = 4,
        theta_dim=theta_dim, 
        n_true_x = n_true_xen, # Number of true samples to generate
        val_fraction = 0.1, # same as in sbi
        dt = 0.025, # Is fixed, because errors when 0.01
        t_max = 120,
        n_samples_to_generate = 1000, # 10 000
        # Prior ranges of HF model
        all_prior_ranges = { # Because the conductances are in S/cm^2. And would be for instance 0.36 or 0.012 for a nicely spiking neuron
            'g_Na': (0.005, 0.8),
            'g_K': (1e-6, 0.15) 
        },
        multiprocessing = False,
        #evaluation_metric = 'nltp',
        type_lf = '',
        lf_embedding = 'identity',
        hf_embedding = 'identity',
        n_fidelities = 2, # Number of fidelities
    )
    lf_simulator = L5PC(config_data, n_compartments=1) # You can also try MultiCompartmentalNeuron(config_data, n_compartments=1)
    hf_simulator = L5PC(config_data, n_compartments=8) # You can also try MultiCompartmentalNeuron(config_data, n_compartments=100)
        
    return config_data, lf_simulator, hf_simulator
    

def get_synaptic_plasticity_setup(theta_dim, n_true_xen, experiment_name):
    config_data = dict(
        sim_name=experiment_name,
        task = 'task3',
        x_dim_out= 2, 
        x_dim_lf = 2,
        x_dim_hf = 2,
        theta_dim=theta_dim, 
        n_true_x = n_true_xen,
        val_fraction = 0.1, # same as in sbi
        n_samples_to_generate = 1000, # 10 000
        # Prior ranges of HF model: Is 24 dimensional....
        all_prior_ranges = { # Because the conductances are in S/cm^2. And would be for instance 0.36 or 0.012 for a nicely spiking neuron
            'tpre_EE': (0.01, 0.1),
            'tpost_EE': (0.01, 0.1),
            'alpha_EE': (-2., 2.),
            'beta_EE': (-2., 2.),
            'gamma_EE': (-2., 2.),
            'kappa_EE': (-2., 2.),
            
            'tpre_EI': (0.01, 0.1),
            'tpost_EI': (0.01, 0.1),
            'alpha_EI': (-2., 2.),
            'beta_EI': (-2., 2.),
            'gamma_EI': (-2., 2.),
            'kappa_EI': (-2., 2.),
            
            'tpre_IE': (0.01, 0.1),
            'tpost_IE': (0.01, 0.1),
            'alpha_IE': (-2., 2.),
            'beta_IE': (-2., 2.),
            'gamma_IE': (-2., 2.),
            'kappa_IE': (-2., 2.),
            'tpre_II': (0.01, 0.1),
            'tpost_II': (0.01, 0.1),
            'alpha_II': (-2., 2.),
            'beta_II': (-2., 2.),
            'gamma_II': (-2., 2.),
            'kappa_II': (-2., 2.),
        },
        #evaluation_metric = 'nltp',
        type_lf = '',
        lf_embedding = 'identity',
        hf_embedding = 'identity',
        n_fidelities = 2, # Number of fidelities
    )
    lf_simulator = MeanField(config_data) 
    hf_simulator = PolynomialNetwork(config_data) 
    
    return config_data, lf_simulator, hf_simulator



def get_gaussian_blob_setup(theta_dim, n_true_xen, experiment_name):
    x_shift = 2
    y_shift = 2
    gamma_shift = .3
    type_lf = 'low_res' # t_shift, x_inv, low_res, hf, ''

    # gamma_min = 0.5
    # gamma_max = 2
    gamma_min = 0.2
    gamma_max = 2
    lf_img_size = 32
    mid_img_size = 64
    img_size = 256

    prior_ranges = {}
    if type_lf == 't_shift':
        # expand prior for you are never shifted outside of the prior bounds
        all_prior_ranges = {
            'xoff': (0 - abs(x_shift), img_size + abs(x_shift)),
            'yoff': (0 - abs(y_shift), img_size + abs(y_shift)),
            'gamma': (gamma_min - abs(gamma_shift), gamma_max + abs(gamma_shift)),
        }
    else:
        all_prior_ranges = {
            'xoff': (0, img_size), 
            'yoff': (0, img_size),
            'gamma': (gamma_min, gamma_max),
        }
    config_data = dict(
        sim_name=experiment_name,
        task = 'task4',
        x_dim_out= 32, 
        x_dim_lf = lf_img_size*lf_img_size, # Will be upscaled, that's the init dim minimum resolution = 31: domension should be bigger than prior!
        x_dim_mid = mid_img_size*mid_img_size,
        x_dim_hf = img_size*img_size,
        # Summary stats: None
        theta_dim= theta_dim, # 2-4 number of trainable parameters for HF model
        val_fraction = 0.1, # 0.1 is same as in sbi
        n_true_x = n_true_xen, 
        logspace = False, # sample not linearly each 10 steps but logaritmicely
        n_samples_to_generate = 1000,  # TODO: Put to 10000
        # Prior ranges of HF model
        all_prior_ranges = all_prior_ranges,
        # evaluation_metric = 'nltp', # use `c2st`, `wasserstein` or `mmd`: since for this task we have a ground truth
        type_lf = type_lf, # Just for tests of for how close the lf model is to the hf model, else 'gs', 'x_inv', 't_shift', 'hf'
        decrease_resolution_option = 'blur', # Only for LowResolution LF model: 'blur' or 'avg_pool'. Avg_pool will decrease the size of x, and the width/height of the image should be half of the high-fidelity one
        lf_embedding = 'cnn', # identity or cnn
        hf_embedding = 'cnn', # identity or cnn
        n_fidelities = 3, # Number of fidelities
    )
    
    if config_data['type_lf'] == 'x_inv':
        lf_simulator = GaussianBlobXinverse(config_data)

    elif config_data['type_lf'] == 't_shift':
        lf_simulator = GaussianBlobTshift(config_data, x_shift, y_shift, gamma_shift)
        
    elif config_data['type_lf'] == 'low_res':
        lf_simulator = GaussianBlobLowResolution(config_data)

    else: # use standard simulator
        raise NotImplementedError("For the Gaussian blob task, please choose a low-fidelity model among 'x_inv', 't_shift', 'low_res'.")
        
    mid_simulator = GaussianBlobMidResolution(config_data)
    
    # If multiple fidelities, wrap lf_simulator and mid_simulator into a dictionary
    # lf_simulators = {'lf': lf_simulator,
    #                 'lf_2': mid_simulator
    #                } 
    lf_simulators = lf_simulator # mid_simulator
    hf_simulator = GaussianBlob(config_data)

    return config_data, lf_simulators, hf_simulator



def get_slcp_setup(theta_dim, n_true_xen, experiment_name):
    if n_true_xen > 10:
        raise ValueError("SLCP task is designed for 10 true samples, please set n_true_xen to 10.")
    
    # Benchmarking task for the SLCP model
    all_prior_ranges = {
            'theta1': (-3, 3),
            'theta2': (-3, 3),
            'theta3': (-3, 3),
            'theta4': (-3, 3),
            'theta5': (-3, 3),
    }
    config_data = dict(
        sim_name=experiment_name,
        task = 'task5',
        x_dim_out= 8, 
        x_dim_lf = 8,
        x_dim_hf = 8,
        theta_dim= theta_dim, # number of high-fidelity thetas
        val_fraction = 0.1, # 0.1 is same as in sbi
        n_true_x = n_true_xen, 
        n_samples_to_generate = 10000, # in sbiBM, we have 10 observations and 10_000 reference samples
        all_prior_ranges = all_prior_ranges,
        type_lf = '',
        lf_embedding = 'identity',
        hf_embedding = 'identity',
        n_fidelities = 2, # Number of fidelities
    )
    
    lf_simulator = SLCP_lf(config_data, distractors=False)  # Use distractors to indicate low fidelity variant
    hf_simulator = SLCP(config_data, distractors=False)  # High fidelity variant without distractors
    
    return config_data, lf_simulator, hf_simulator



def get_lotka_volterra_setup(theta_dim, n_true_xen, experiment_name):
    if n_true_xen > 10:
        raise ValueError("Lotka-Volterra task is designed for 10 true samples, please set n_true_xen to 10.")
    
    all_prior_ranges = {
            'alpha': (None, None),
            'beta': (None, None),
            'gamma': (None, None),
            'delta': (None, None),
    }
    config_data = dict(
        sim_name=experiment_name,
        task = 'task6',
        subsample_rate = None, 
        x_dim_out= 20, 
        x_dim_lf = 20,
        x_dim_hf = 20,
        theta_dim= theta_dim, # number of high-fidelity thetas
        val_fraction = 0.1, # 0.1 is same as in sbi
        n_true_x = n_true_xen, 
        n_samples_to_generate = 10000, # in sbiBM, we have 10 observations and 10_000 reference samples
        all_prior_ranges = all_prior_ranges,
        type_lf = '',
        lf_embedding = 'identity',
        hf_embedding = 'identity',
        n_fidelities = 2, # Number of fidelities
    )
    
    lf_simulator = LotkaVolterra(config_data)  # Use distractors to indicate low fidelity variant
    hf_simulator = LotkaVolterra(config_data)  # High fidelity variant without distractors
    
    return config_data, lf_simulator, hf_simulator


def get_sir_setup(theta_dim, n_true_xen, experiment_name):
    if n_true_xen > 10:
        raise ValueError("Lotka-Volterra task is designed for 10 true samples, please set n_true_xen to 10.")
    
    all_prior_ranges = {
            'b': (None, None),
            'g': (None, None),
    }
    config_data = dict(
        sim_name=experiment_name,
        task = 'task7',
        subsample_rate = None, 
        x_dim_out= 10, 
        x_dim_lf = 10,
        x_dim_hf = 10,
        theta_dim= theta_dim, # number of high-fidelity thetas
        val_fraction = 0.1, # 0.1 is same as in sbi
        n_true_x = n_true_xen, 
        n_samples_to_generate = 10000, # in sbiBM, we have 10 observations and 10_000 reference samples
        all_prior_ranges = all_prior_ranges,
        type_lf = '',
        lf_embedding = 'identity',
        hf_embedding = 'identity',
        n_fidelities = 2, # Number of fidelities
    )
    
    
    lowres_simulator = SIR_lowres(config_data)
    #midres_simulator = SIR_midres(config_data)  # Use distractors to indicate low fidelity variant
    
    # lf_simulator = {'lf': lowres_simulator,
    #                  #'lf_2': mid_simulator
    #                 } 
    
    lf_simulator = lowres_simulator # SIR_lf(config_data)  # Use distractors to indicate low fidelity variant
    hf_simulator = SIR(config_data)  # High fidelity variant without distractors
    
    return config_data, lf_simulator, hf_simulator