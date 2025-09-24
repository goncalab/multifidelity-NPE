import argparse
import pickle
import os

import pandas as pd
from mf_npe.one_experiment import run_comparison_lf_to_hf_posteriors, run_evaluation
from mf_npe.plot.method_performance import plot_methods_performance_paper
from mf_npe.utils.calculate_error import mean_confidence_interval
from mf_npe.utils.utils import dump_pickle


##### USAGE #####
# e.g., python eval_lf_hf_distance.py --models_to_run mf_npe --simulator_task SLCP --lf_datasize 100000 --hf_datasize 100000 --n_true_xen 10 --eval_metrics mmd c2st --seed 12 --n_net_inits 2
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate distance between low and high-fidelity model.')

    parser.add_argument('--models_to_run', type=str, default=None, nargs='+', required=True,)
    parser.add_argument('--simulator_task', type=str, required=True)
    parser.add_argument('--theta_dim', type=int, default=None)
    parser.add_argument('--lf_datasize', type=int, nargs='+', required=True)
    parser.add_argument('--hf_datasize', type=int, nargs='+', required=True)
    parser.add_argument('--n_net_inits', type=int, default=10) # in paper, typically 10
    parser.add_argument('--seed', type=int, default=12) # Default seed for reproducibility is 12
    parser.add_argument('--eval_metrics', type=str, nargs='+', required=True, help='evaluation metric. For toy tasks, use: c2st, wasserstein, mmd, nltp. For others: nltp')
    parser.add_argument('--n_true_xen', type=int, default=10, required=True) # number of true xen to evaluate on


    args = parser.parse_args()
    

    if args.theta_dim is None:
        if args.simulator_task == 'OUprocess' or args.simulator_task == 'L5PC':
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
            raise ValueError(f"No default theta_dim for simulator_task: {args.simulator_task}.")

    return args



def main():
    args = parse_args()

    main_path = f"./data/{args.simulator_task}/{args.theta_dim}_dimensions/models"
    
    # Sort the strings so that the order does not matter
    # when loading the model
    # e.g., 'npe+mf_npe' and 'mf_npe+npe' are the same
    models_str = '+'.join(sorted(args.models_to_run))
    lf_str = '+'.join(map(str, sorted(args.lf_datasize)))
    hf_str = '+'.join(map(str, sorted(args.hf_datasize)))
    
    # Adjusts the path to the model file you requested
    print("n of net inits", args.n_net_inits)
    

    seed_max = args.seed + args.n_net_inits - 1
    
    print("seed max", seed_max)
    
        
    # Adjusts the path to the model file you requested
    if seed_max == args.seed:
        name = f"train_{models_str}_LF{lf_str}_HF{hf_str}_Ninits{args.n_net_inits}_seed{args.seed}"
    else:
        name = f"train_{models_str}_LF{lf_str}_HF{hf_str}_Ninits{args.n_net_inits}_seed{args.seed}-{seed_max}"
    
    file_path = os.path.join(main_path, f"{name}.pkl")

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Train data pickle not found at {file_path}")

    # Load pickle
    with open(file_path, "rb") as f:
        batch_train_data = pickle.load(f)
    
    
    batch_df_results = pd.DataFrame()
        
    for net_i in range(args.n_net_inits): #range(2): # TODO: Put back to range(args.n_net_inits):
        # Extract the data for the current network initialization

        print("net initialization:", net_i + 1)
        train_data = batch_train_data[net_i] # because zero-based
            
        # Unpack everything
        task_setup     = train_data['task_setup']
        true_xen       = train_data['true_xen']
        true_thetas    = train_data['true_thetas']
        true_add_ons   = train_data['true_add_ons']
        n_lf_samples   = train_data['n_lf_samples']
        n_hf_samples   = train_data['n_hf_samples']
        n_mf_samples   = train_data['n_mf_samples']
        all_methods    = train_data['all_methods']
        net_init       = train_data['net_init']
        num_hifi       = train_data['num_hifi']
        hf_data        = train_data['hf_data']
        
        # If true_xen not the same as given in args, raise error
        if true_xen.shape[0] != args.n_true_xen:
            raise ValueError(f"Number of true_xen in train ({true_xen.shape[0]}) does not match the number of true_xen in eval ({args.n_true_xen}).")
        

        # if net inits of eval and train are different, raise error
        if net_init != args.n_net_inits:
            raise ValueError(f"Number of network initializations in train ({net_init}) does not match the number of network initializations in eval ({args.n_net_inits}).")
        
        
        # Raise error if n_lf_samples and n_hf_samples are not the same as the arguments passed
        if n_lf_samples != args.lf_datasize:
            raise ValueError(f"n_lf_samples {n_lf_samples} does not match the lf_datasize {args.lf_datasize} passed as argument.")
        if n_hf_samples != args.hf_datasize:
            raise ValueError(f"n_hf_samples {n_hf_samples} does not match the hf_datasize {args.hf_datasize} passed as argument.")


        # Run evaluation        
        df_results = run_comparison_lf_to_hf_posteriors(task_setup, 
                                true_xen, true_thetas,
                                n_lf_samples, n_hf_samples,
                                all_methods, 
                                net_i, 
                                args.eval_metrics,
                                simulator_name=args.simulator_task)

        # Save the trained models    
        models_str = '+'.join(args.models_to_run)
        lf_str = '+'.join(map(str, args.lf_datasize))
        hf_str = '+'.join(map(str, args.hf_datasize))
        
        # Append the df results to the batch DataFrame
        batch_df_results = pd.concat([batch_df_results, df_results], ignore_index=True)
        
        
    if seed_max == args.seed:
        name = f"eval_lf_hf_distance_{args.eval_metrics}_{models_str}_Ninits{net_init}_seed{args.seed}.pkl"
    else:
        name = f"eval_lf_hf_distance_{args.eval_metrics}_{models_str}_Ninits{net_init}_seed{args.seed}-{seed_max}.pkl"

    dump_pickle(main_path, name, df_results)
    print("Saved evaluation DataFrame to", f'{main_path}/{name}')

    # group pickle over the number of network initializations by calculating the mean and confidence interval
    grouped_df = batch_df_results.groupby(['evaluation_metric', 'task', 'n_lf_simulations', 'n_hf_simulations', 'algorithm'])['raw_data'].apply(mean_confidence_interval).reset_index()

    print("Grouped df", grouped_df)

    plot_methods_performance_paper(grouped_df, f"{task_setup.sim_name}_({task_setup.config_data['theta_dim']} dims, avg over {net_i + 1} networks)", task_setup.batch_lf_sims, df_results['evaluation_metric'][0], task_setup)


if __name__ == '__main__':
    main()
