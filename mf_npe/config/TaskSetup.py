
from datetime import datetime
from mf_npe.config.plot import width_plots, height_plots, font_size, title_size, gridwidth, axis_color, show_plots
from mf_npe.utils.task_setup import load_task_setup
from mf_npe.utils.utils import set_global_seed

class TaskSetup:
    def __init__(self, 
                 sim_name: str, 
                 config_model, 
                 main_path, 
                 batch_lf_datasize: list = [10**3], 
                 batch_hf_datasize: list = [100], 
                 n_network_initializations: int = 1,
                 theta_dim: int = 0,
                 n_true_xen: int = 0,
                 seed: int = 42, 
                 ):
        self.sim_name = sim_name
        self.seed = seed
        self.key = set_global_seed(seed)
        self.CURR_TIME = datetime.now().strftime("%Y-%m-%d %Hh%M")

        # Plot settings
        self.width_plots = width_plots
        self.height_plots = height_plots
        self.font_size = font_size
        self.title_size = title_size
        self.gridwidth = gridwidth
        self.axis_color = axis_color
        self.show_plots = show_plots

        # Simulation config
        self.batch_lf_sims = batch_lf_datasize
        self.batch_hf_sims = batch_hf_datasize
        self.batch_mf_sims = [[lf, hf] for lf in self.batch_lf_sims for hf in self.batch_hf_sims]
        self.n_network_initializations = n_network_initializations

        # Model config
        self.config_model = config_model
        
        # theta dimension and true xen
        self.theta_dim = theta_dim
        self.n_true_xen = n_true_xen
        
        # Load task-specific configuration
        self.config_data, self.lf_simulator, self.hf_simulator = load_task_setup(self.sim_name, theta_dim=theta_dim, n_true_xen=n_true_xen)
        
        self.prior_ranges = {
            k: self.config_data['all_prior_ranges'][k]
            for k in list(self.config_data['all_prior_ranges'].keys())[:self.config_data['theta_dim']]
        }
        self.main_path = main_path
