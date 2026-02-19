import argparse
from mf_npe.one_experiment import run_one_experiment
from mf_npe.utils.utils import dump_pickle, generate_train_data, generate_true_data, set_global_seed
from mf_npe.utils.task_setup import load_task_setup, process_device


# --- macOS fork/thread safety prelude (must run before heavy imports) ---
import os, platform

if platform.system() == "Darwin":
    # Tame native thread pools to avoid condvar issues
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # Apple Accelerate

# ------------------------------------------------------------------------


######## USAGE ########
# python train.py --models_to_run npe mf_npe --simulator_task OUprocess --lf_datasize 1000 --hf_datasize 50 --n_true_xen 10 --seed 12 --n_net_inits 1
# if you want to generate new training data and/or true data, add these flags:
# --generate_true_data --generate_train_data


def parse_args():
    parser = argparse.ArgumentParser(description='Run one multifidelity simulator_task.')

    parser.add_argument('--seed', type=int, required=False, default=12) # 12 is the seed for reproducibility
    parser.add_argument('--models_to_run', type=str, nargs='+', required=True)
    parser.add_argument('--simulator_task', type=str, required=True) # e.g., OUprocess or L5PC or SyanpticPlasticity
    parser.add_argument('--theta_dim', type=int, default=None) # 2, 3,4 for OUprocess, 2 for L5PC, or 12 for SynapticPlasticity
    parser.add_argument('--lf_datasize', type=int, nargs='+', required=True, help="Provide integers, e.g. 100 1000.") # e.g., 10**3
    parser.add_argument('--hf_datasize', type=int, nargs='+', required=True, help="Provide integers, e.g. 100 1000") 
    parser.add_argument('--n_true_xen', type=int, default=None) 
    parser.add_argument('--n_net_inits', type=int, default=1) # in paper, typically 10
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--generate_true_data', action='store_true', default=False) # Load the true data
    parser.add_argument('--generate_train_data', action='store_true', default=False) # Load the training data
    
    args = parser.parse_args()
    
    # Set theta_dim defaults based on experiment name
    if args.theta_dim is None:
        if args.simulator_task in ['OUprocess', 'L5PC']:
            args.theta_dim = 2
        elif args.simulator_task == 'SynapticPlasticity':
            args.theta_dim = 24
        elif args.simulator_task == 'GaussianBlob':
            args.theta_dim = 3
        elif args.simulator_task == 'SLCP':
            args.theta_dim = 5
        elif args.simulator_task == 'LotkaVolterra':
            args.theta_dim = 4
        elif args.simulator_task == 'SIR':
            args.theta_dim = 2
        else:
            raise ValueError(f"No default theta_dim for simulator_task: {args.simulator_task}. Make sure the experiment exists.")


    if args.n_true_xen is None:
        print("args.simulator_task", args.simulator_task)
        if args.simulator_task == 'OUprocess':
            args.n_true_xen = 30
        elif args.simulator_task == 'L5PC':
            args.n_true_xen = 100
        elif args.simulator_task == 'SynapticPlasticity':
            args.n_true_xen = 300_000
        elif args.simulator_task == 'GaussianBlob':
            args.n_true_xen = 30
        elif args.simulator_task == 'SLCP':
            args.n_true_xen = 10
        elif args.simulator_task == 'LotkaVolterra':
            args.n_true_xen = 10
        elif args.simulator_task == 'SIR':
            args.theta_dim = 2
        else:
            raise ValueError(f"No default n_true_xen for simulator_task: {args.simulator_task}. Make sure the experiment exists.")

    return args


def load_task_setup_and_data(simulator_task, theta_dim, n_true_xen, lf_datasize, hf_datasize, gener_true_xen=False, gener_train_data=False):  
    """
    Load the task setup and data based on the simulator_task name and data sizes.
    """
    
    path_data = f'./data/{simulator_task}/{theta_dim}_dimensions'
            
    config_data, lf_simulator, hf_simulator = load_task_setup(simulator_task, theta_dim=theta_dim, n_true_xen=n_true_xen) # 42 is the seed for reproducibility
    
    # Load the true data
    true_xen, true_thetas, true_add_ons = generate_true_data(simulate_true_data=gener_true_xen, 
                                                             path_to_pickles=path_data,
                                                             n_true_xen=n_true_xen,
                                                             hf_simulator=hf_simulator,
                                                             config_data=config_data)
    
    # Load training data
    lf_data, hf_data = generate_train_data(simulate_train_data=gener_train_data, 
                                           path_to_pickles=path_data, 
                                           batch_lf_sims=lf_datasize,
                                            batch_hf_sims=hf_datasize,
                                            hf_simulator=hf_simulator,
                                            lf_simulator=lf_simulator,
                                            config_data=config_data,)
    
    return true_xen, true_thetas, true_add_ons, lf_data, hf_data, config_data, lf_simulator, hf_simulator


def main():
    args = parse_args()
    
    # Load experiment setup & data based on experiment name and data sizes
    true_xen, true_thetas, true_add_ons, lf_data, hf_data, config_data, lf_simulator, hf_simulator = load_task_setup_and_data(
        simulator_task=args.simulator_task,
        theta_dim=args.theta_dim,  # Assuming theta_dim is the same as lf_datasize
        n_true_xen=args.n_true_xen,
        lf_datasize=args.lf_datasize,
        hf_datasize=args.hf_datasize,
        gener_true_xen=args.generate_true_data,
        gener_train_data=args.generate_train_data
    )
    
    main_path = f"./data/{args.simulator_task}/{config_data['theta_dim']}_dimensions"
    
    config_model = dict(
        max_num_epochs=2**31 - 1, # high number since we have early stopping
        batch_size = 200, # increasing the batch size will speed up the training, but the model will be less accurate
        learning_rate= 5e-4, # Learning rate for Adam optimizer
        device = process_device(),
        validation_fraction = 0.1, # Fraction of the data to use for validation
        patience=20, # The number of epochs to wait for improvement on the validation set before terminating training.
        n_transforms = 5, 
        n_bins=8,
        n_hidden_features = 50,
        clip_max_norm = 5.0, # value to which to clip total gradient norm to prevent exploding gradients. Use None for no clipping
        
        # Choose between logit transforming or z_scoring thetas, not both
        logit_transform_theta_net = True, # for training in unbound space: Then we do not have that much leakage in posterior
        z_score_theta = False, 
        z_score_x = True,
        # For active learning
        active_learning_pct=0.8,
        n_rounds_AL = 5, # From 1 to 5 
        n_theta_samples = 1000, #250,
        n_ensemble_members = 5, # put to 2 for 10**5, but it back for all the others
        )

    train_data_over_n_inits = []
    seed_max = args.seed + args.n_net_inits - 1

    for net_init in range(args.n_net_inits):
        print(f"Running network initialization {net_init + 1} of {args.n_net_inits}")
        seed = args.seed + net_init
        set_global_seed(seed)
        
        # Run the experiment
        train_data = run_one_experiment(seed=seed,
                        models_to_run=args.models_to_run,
                        lf_data=lf_data,
                        hf_data=hf_data,
                        true_xen=true_xen,
                        true_thetas=true_thetas,
                        true_add_ons=true_add_ons,
                        batch_lf_sims=args.lf_datasize, 
                        batch_hf_sims=args.hf_datasize,
                        sim_name=args.simulator_task,
                        config_model=config_model,
                        main_path=main_path,
                        net_init=args.n_net_inits,
                        b_load_model=args.load_model)
        
        train_data_over_n_inits.append(train_data)

        # Save the trained models    
        models_str = '+'.join(sorted(args.models_to_run))
        lf_str = '+'.join(map(str, sorted(args.lf_datasize)))
        hf_str = '+'.join(map(str, sorted(args.hf_datasize)))
        
    # Adjusts the path to the model file you requested
    if seed_max == args.seed:
        name = f"train_{models_str}_LF{lf_str}_HF{hf_str}_Ninits{args.n_net_inits}_seed{args.seed}.pkl"
    else:
        name = f"train_{models_str}_LF{lf_str}_HF{hf_str}_Ninits{args.n_net_inits}_seed{args.seed}-{seed_max}.pkl"
    save_dir = f"{main_path}/models"
    dump_pickle(save_dir, name, train_data_over_n_inits)

    print("pickle of trained models saved at", f'{main_path}/{name}')


if __name__ == '__main__':
    main()