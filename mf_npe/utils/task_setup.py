import torch
from mf_npe.simulator.setup_for_tasks import get_gaussian_blob_setup, get_l5pc_setup, get_lotka_volterra_setup, get_ouprocess_setup, get_sir_setup, get_slcp_setup, get_synaptic_plasticity_setup


def load_task_setup(experiment_name: str, theta_dim, n_true_xen):
    if experiment_name == 'OUprocess':
        config_data, lf_simulator, hf_simulator = get_ouprocess_setup(theta_dim, n_true_xen, experiment_name)
    elif experiment_name == 'L5PC':
        config_data, lf_simulator, hf_simulator = get_l5pc_setup(theta_dim, n_true_xen, experiment_name)
    elif experiment_name == 'SynapticPlasticity':
        config_data, lf_simulator, hf_simulator = get_synaptic_plasticity_setup(theta_dim, n_true_xen, experiment_name)
    elif experiment_name == 'GaussianBlob':
        config_data, lf_simulator, hf_simulator = get_gaussian_blob_setup(theta_dim, n_true_xen, experiment_name)
    elif experiment_name == 'SLCP':
        config_data, lf_simulator, hf_simulator = get_slcp_setup(theta_dim, n_true_xen, experiment_name)
    elif experiment_name == 'LotkaVolterra':
        config_data, lf_simulator, hf_simulator = get_lotka_volterra_setup(theta_dim, n_true_xen, experiment_name)
    elif experiment_name == 'SIR':
        config_data, lf_simulator, hf_simulator = get_sir_setup(theta_dim, n_true_xen, experiment_name)
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    
    return config_data, lf_simulator, hf_simulator


def process_device():
    if torch.backends.mps.is_available(): # Device configuration for GPU acceleration
        device = torch.device("mps")  # Check if MPS is available, for Apple Silicon devices
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")  # Check if CUDA is available for Intel and AMD GPUs
    else:
        device = torch.device("cpu")
    # device = torch.device("gpu") # CPU is fastest atm. So set device to cpu by default
    print(f"Using device: {device}")
    return device