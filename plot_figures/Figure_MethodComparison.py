#%%
# =============================================================================
#       SCRIPT TO REPRODUCE MAIN FIGURES PAPER
# =============================================================================

# DESCRIPTION: Evaluation of the performance of our methods and baseline methods, 
# including the performance of the amortized and non-amortized methods, described in the paper.
# > Tasks: OU process, L5PC, and SynapticPlasticity

# USAGE:
# >> python Figure_MethodComparison.py --tasks <task_name> --theta_dim <theta_dim> --plot_amortized_and_non_amortized_seperately --eval_metric <eval_metric>
# Example (as in paper): python Figure_MethodComparison.py --tasks OUprocess SIR SLCP --theta_dim 2 2 5 --plot_amortized_and_non_amortized_seperately --eval_metric c2st mmd


import os
import re
import pickle
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from types import SimpleNamespace
from sbi import analysis
# from mf_npe import task_setup
from mf_npe.config.TaskSetup import TaskSetup
from mf_npe.utils.calculate_error import mean_confidence_interval
from mf_npe.plot.method_performance import plot_methods_performance_paper
from mf_npe.evaluation import Evaluation
import argparse
import numpy as np

from mf_npe.utils.load_from_eval import group_df, load_from_eval_file
# from mf_npe.utils.task_setup import process_device

batch_lf_sims = [10**3, 10**4, 10**5]
batch_hf_sims = [50, 100, 10**3, 10**4, 10**5]
net_init = 9
config_model = dict(
    max_num_epochs=2**31 - 1, # high number since we have early stopping
    batch_size = 200, # increasing the batch size will speed up the training, but the model will be less accurate
    learning_rate= 5e-4, # Learning rate for Adam optimizer
    type_estimator='npe', # we always compute the posterior (npe), and do not evaluated likelihood or ratio methods (e.g., NLE, NRE)
    device = 'cpu', #process_device(),
    validation_fraction = 0.1, # Fraction of the data to use for validation
    patience=20, # The number of epochs to wait for improvement on the validation set before terminating training.
    n_transforms = 5, 
    n_bins=8,
    n_hidden_features = 50,
    clip_max_norm = 5.0, # value to which to clip total gradient norm to prevent exploding gradients. Use None for no clipping
    
    # Choose between logit transforming or z_scoring thetas, not both
    # logit_transform_theta_net = True, # for training in unbound space: Then we do not have that much leakage in posterior
    # z_score_theta = False, 
    # z_score_x = True,
    logit_transform_theta_net = True, # for training in unbound space: Then we do not have that much leakage in posterior
    z_score_theta = False, 
    z_score_x = True,
    # For active learning
    active_learning_pct=0.8,
    n_rounds_AL = 5, # From 1 to 5 
    n_theta_samples = 1000, #250,
    n_ensemble_members = 5, 
    )


#############################################
# Parse Arguments
#############################################

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and plot method performance for simulation-based inference tasks.")
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        nargs='+',
        choices=["OUprocess", "L5PC", "SynapticPlasticity", "SLCP", "LotkaVoterra", "GaussianBlob", "SIR"],
        help="Name of the simulator task to run."
    )
    parser.add_argument(
        "--theta_dim",
        type=int,
        default=None,
        nargs='+',
        help="Dimensionality of the parameter space theta. If not set, defaults will be chosen based on the task."
    )
    parser.add_argument(
        "--plot_amortized_and_non_amortized_seperately",
        action="store_true",
        help="If set, plots amortized and non-amortized results in separate subplots."
    )
    parser.add_argument(
        "--eval_metrics",
        type=str,
        default="nltp",
        nargs='+',
        choices=["c2st", "mmd", "nltp", "nrmse"],
        help="Evaluation metric to use. Options are 'c2st', 'mmd', 'nltp', or 'nrmse'. Default is 'nltp'."
    )
    return parser.parse_args()

#############################################

def configure_simulation(sim_name, custom_theta_dim):

    if sim_name == 'OUprocess':
        theta_dim = custom_theta_dim if custom_theta_dim is not None else 2
    elif sim_name == 'L5PC':
        theta_dim = custom_theta_dim if custom_theta_dim is not None else 2
    elif sim_name == 'SynapticPlasticity':
        theta_dim = custom_theta_dim if custom_theta_dim is not None else 24
    elif sim_name == 'SLCP':
        theta_dim = custom_theta_dim if custom_theta_dim is not None else 5
    elif sim_name == 'SIR':
        theta_dim = custom_theta_dim if custom_theta_dim is not None else 2
    elif sim_name == 'GaussianBlob':
        theta_dim = custom_theta_dim if custom_theta_dim is not None else 3
    else:
        raise ValueError(f"Unknown simulator task: {sim_name}. Please choose from 'OUprocess', 'L5PC', or 'SynapticPlasticity', 'GaussianBlob.")

    
    main_path = f"./../data/{sim_name}/{theta_dim}_dimensions"

    plot_setup = SimpleNamespace(
        main_path=main_path,
        show_plots=True,
        CURR_TIME=datetime.now().strftime("%Y-%m-%d %Hh%M"),
        width_plots=400,
        height_plots=200,
        axis_color='#6A798F',
        font_size=20,
        title_size=20,
        gridwidth=2,
        show_legend=True,
    )
    
    return plot_setup, theta_dim, main_path





#############################################
# Function 1: Load and Evaluate Non-Amortized Posteriors
#############################################

def load_and_evaluate_non_amortized_posteriors(post_path, eval_metric, sim_name):
    # 1. Load non-amortized posterior samples if they exist
    print(f"Loading non-amortized posterior samples from {post_path}... this may take ~1-2min.")
    records = []
    file_names = [f for f in os.listdir(post_path) if f.startswith(('a_mf_tsnpe', 'mf_tsnpe', 'tsnpe'))]

    pattern = re.compile(
            r"(?P<method>a_mf_tsnpe|mf_tsnpe|(?:a_)?tsnpe)"
            r"(?:_LF(?P<lf>\d+))?"
            r"(?:_HF(?P<hf>\d+)|_(?P<hf_alt>\d+))?"
            r"_xo(?P<xo>\d+)"
            r"(?:_seed(?P<seed>\d+))?" # Optional seed for reproducibility. L5PC and SynapticPlasticity did not use seeds.
            r"_(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}h\d{2})\.p$"
        )

    parsed_data = []        
    for name in file_names:
        match = pattern.fullmatch(name)
        if match:
            parsed_data.append({
                "file": name,
                "method": match.group("method"),
                "timestamp": match.group("timestamp"),
            })
        
        print("match", match)
            
    print("file names", file_names)

    for item in parsed_data:
        data = pd.read_pickle(f'{post_path}/{item["file"]}')
        
        print(f"Processing file: {data}")
        
        # if tsnpe, load differently
        if item["method"] == 'tsnpe':
            print("tsnpe data")
            records.append({
                        "true_x": data['true_x'],
                        "true_theta": data['true_theta'],
                        "posterior": data['posterior'],
                        "n_simulations": data['n_simulations'],
                        "n_lf": 0,
                        "n_hf": data['n_simulations'],
                        "type_estimator": data['type_estimator'],
                        "file": item["file"],
                        "timestamp": item["timestamp"],
                    })
        else:
            records.append({
            "true_x": data['true_x'],
            "true_theta": data['true_theta'],
            "posterior": data['posterior'],
            "n_simulations": data['n_simulations'],
            "n_lf": data['n_simulations'][0],
            "n_hf": data['n_simulations'][1],
            "type_estimator": data['type_estimator'],
            "file": item["file"],
            "timestamp": item["timestamp"],
        })
        
        true_xen = data['true_x']
        
        # Will be reloaded for each non-amortized posterior
        task_setup = TaskSetup(sim_name=sim_name, 
                config_model=config_model, 
                main_path=main_path, 
                batch_lf_datasize=batch_lf_sims, 
                batch_hf_datasize=batch_hf_sims, 
                n_network_initializations=net_init,
                theta_dim=theta_dim,
                n_true_xen=true_xen.shape[0],
                seed=None)
    
        evaluation = Evaluation(true_xen, task_setup, eval_metric=eval_metric)
    
    
    # Print the records
    print("Records loaded:", len(records))
    for record in records:
        print(f"File: {record['file']}, Method: {record['type_estimator']}, LF: {record['n_lf']}, HF: {record['n_hf']}, Timestamp: {record['timestamp']}")

    df_trials = pd.DataFrame(records)
    print("df_trials", df_trials)
    unique_combinations = df_trials[["type_estimator", "n_lf", "n_hf"]].drop_duplicates()
    df_all_evaluated = pd.DataFrame()

    for _, row in unique_combinations.iterrows():
        df_filtered = df_trials[
            (df_trials["type_estimator"] == row["type_estimator"]) &
            (df_trials["n_lf"] == row["n_lf"]) &
            (df_trials["n_hf"] == row["n_hf"])
        ]
            
        if sim_name == 'OUprocess':
            n_samples = 1000 # Default
            print("df_filtered['true_x']", df_filtered["true_x"].tolist())
            true_posterior_samples = evaluation.get_true_posterior_samples(df_filtered["true_x"].tolist(), 
                                                                        task_setup.hf_simulator.prior(), 
                                                                        task_setup.hf_simulator,
                                                                        n_samples)
            

            posterior_samples = evaluation.get_posterior_samples(df_filtered["true_x"].tolist(), 
                                                                df_filtered["true_theta"].tolist(), 
                                                                df_filtered["posterior"].tolist(), 
                                                                row["type_estimator"], 
                                                                [row["n_lf"], row["n_hf"]],
                                                                net_init)
            
            
            if row["type_estimator"] == 'tsnpe':
                df_evaluated = evaluation.eval_ground_truth_available( 
                        true_xen=df_filtered["true_x"].tolist(),
                        metric=eval_metric,
                        posterior_samples=posterior_samples,
                        true_posterior_samples=true_posterior_samples,
                        n_simulations=row["n_hf"],
                        type_estimator=row["type_estimator"]) 
            else:
                df_evaluated = evaluation.eval_ground_truth_available( 
                        true_xen=df_filtered["true_x"].tolist(),
                        metric=eval_metric,
                        posterior_samples=posterior_samples,
                        true_posterior_samples=true_posterior_samples,
                        n_simulations=[row["n_lf"], row["n_hf"]],
                        type_estimator=row["type_estimator"]) 
        else:
            if row["type_estimator"] == 'tsnpe':
                df_evaluated = evaluation.evaluate_no_ground_truth(
                    torch.stack(df_filtered["true_x"].tolist()),
                    torch.stack(df_filtered["true_theta"].tolist()),
                    df_filtered["posterior"].tolist(),
                    row["n_hf"],
                    type_estimator=row["type_estimator"],
                    net_init=net_init
                )
            else:
                df_evaluated = evaluation.evaluate_no_ground_truth(
                    torch.stack(df_filtered["true_x"].tolist()),
                    torch.stack(df_filtered["true_theta"].tolist()),
                    df_filtered["posterior"].tolist(),
                    [row["n_lf"], row["n_hf"]],
                    type_estimator=row["type_estimator"],
                    net_init=net_init
                )
                
            # raise ValueError("check")

        df_evaluated["n_lf_simulations"] = row["n_lf"]
        df_evaluated["n_hf_simulations"] = row["n_hf"]
        df_evaluated["algorithm"] = row["type_estimator"]

        df_all_evaluated = pd.concat([df_all_evaluated, df_evaluated], ignore_index=True)
        
    print("df_all_evaluated", df_all_evaluated)
    
    
    # Problem lays here!!
    # Remove raw_data column if it exists
    if 'task' not in df_all_evaluated.columns:
        df_all_evaluated['task'] = sim_name
    
    # drop raw data
    if 'raw_data' in df_all_evaluated.columns:
        df_all_evaluated = df_all_evaluated.drop(columns=['raw_data'])
        
    print("df_all_evaluated", df_all_evaluated)

    df_non_amortized = group_df(df_all_evaluated, eval_metric)
    
    # Put it table with ascending order based on 1. n_hf_sims and 2. n_lf_sims
    df_non_amortized = df_non_amortized.sort_values(by=['n_hf_simulations', 'n_lf_simulations'])

    return df_non_amortized


#############################################
# Function 2: Plot amortized posteriors
#############################################

def _as_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def _sanitize_algo(a: str) -> str:
    # keep it filename-friendly
    return str(a).strip().replace(" ", "_").replace("/", "_")

def _join_sims(vals, skip_zeros=True):
    vals = [int(v) for v in _as_list(vals)]
    if skip_zeros:
        vals = [v for v in vals if v != 0]
    # keep order as given; if you prefer sorted unique: vals = sorted(set(vals))
    return "+".join(str(v) for v in vals) if vals else "0"

def make_eval_filename(eval_metric, algorithms, lf_sims, hf_sims, prefix="evaluate", ext=".pkl"):
    algos_part = "+".join(_sanitize_algo(a) for a in _as_list(algorithms))
    lf_part = _join_sims(lf_sims, skip_zeros=True)   # drop 0 to match your example
    hf_part = _join_sims(hf_sims, skip_zeros=False)  # keep all HF values
    return f"{prefix}_{eval_metric}_{algos_part}_LF{lf_part}_HF{hf_part}{ext}"


def load_amortized_posteriors(amortized_path, eval_metric):
    all_trials = pd.DataFrame()
    for file in os.listdir(f'{amortized_path}'):
        if file.endswith(".pkl"):
            try:
                df = pd.read_pickle(f'{amortized_path}/{file}')
                df["evaluation_metric"] = eval_metric
                # print(f"Loaded {file} with shape {df.shape}")
                all_trials = pd.concat([all_trials, df], ignore_index=True)
            except Exception as e:
                print(f"Failed to load {file}: {e}")

    # replace fidelity by algorithm in dataframe
    all_trials = all_trials.rename(columns={'fidelity': 'algorithm'})
    print("all_trials", all_trials)
    amortized_df = group_df(all_trials, eval_metric)
    
    print("amortized_df", amortized_df)
    
    return amortized_df





#############################################
# Function 3: Plot Posterior Distributions
#############################################

def plot_posteriors(n_true_x=3, net_init=9, method='a_mf_npe', n_samples=[10000, 1000], theta_dim=2):
    with open(f'{posterior_path}/thetas_{net_init}_{n_true_x}_{method}_{n_samples}.p', 'rb') as f:
        data = pickle.load(f)

    posterior_samples = data['posterior_samples']
    true_theta = data['true_theta']
    type_estimator = data['type_estimator']
    n_train_sims = data['n_train_sims']

    with open(f'{posterior_path}/true_thetas_{n_true_x}.p', 'rb') as f:
        data = pickle.load(f)

    true_posterior_samples = data['true_posterior_samples']

    all_param_ranges = {'mu': (0.1, 3.0), 'sigma': (0.1, 0.6), 'gamma': (0.1, 1.0), 'mu_offset': (0.0, 4.0)}
    prior_ranges = {k: all_param_ranges[k] for k in list(all_param_ranges.keys())[:theta_dim]}
    parameter_ranges = [list(prior_ranges[key]) for key in prior_ranges]

    index = 2

    analysis.pairplot(
        [true_posterior_samples[index], posterior_samples[index]],
        upper=["contour", "hist"],
        offdiag=["contour", "hist"],
        contour_offdiag={"levels": [0.68], "percentile": True},
        limits=parameter_ranges,
        ticks=parameter_ranges,
        figsize=(5, 5),
        points=true_theta[index],
        points_offdiag={"markersize": 6},
        points_colors="r",
        samples_colors=["#FFA15A", "b"],
        title=f"method: {type_estimator} (n_sims: {n_train_sims})",
        labels=[rf"$\\theta_{{{d}}}$" for d in range(theta_dim)],
    )
    





#############################################
# Main Execution
#############################################

if __name__ == "__main__":
    args = parse_args()
    all_tasks_df = pd.DataFrame()
    
    for i, task in enumerate(args.tasks):
        for eval_metric in args.eval_metrics:
            
            # Define path
            plot_setup, theta_dim, main_path = configure_simulation(task, args.theta_dim[i])
            amortized_path = f"{main_path}/{eval_metric}"
            non_amortized_posterior_samples_path = f"{main_path}/non_amortized_posteriors"
            posterior_path = f"{main_path}/posterior_samples"
            eval_models_path = f"{main_path}/models"

            # Non-amortized posteriors
            if task in ['L5PC', 'SynapticPlasticity']:
                # TODO: Check the df_amortized: there is something odd
                
                ######### AMORTIZED #########
                # Put to load_from_eval when evaluating for the first time
                load_from_eval = True # Keep on false, otherwise the true stuff disapplears for L5PC, idk. why.
                if load_from_eval:
                    df_amortized = load_from_eval_file(eval_models_path, eval_metric)
                else:    
                    df_amortized = load_amortized_posteriors(amortized_path, eval_metric)

                # Add to df_amortized a column 'task' = task
                df_amortized['task'] = task
                # add column net init
                df_amortized['net_init'] = net_init
                # Remove level_4 column if it exists
                if 'level_4' in df_amortized.columns:
                    df_amortized = df_amortized.drop(columns=['level_4'])

                ##### NON-AMORTIZED #########
                df_non_amortized = load_and_evaluate_non_amortized_posteriors(non_amortized_posterior_samples_path, eval_metric, task)
                # TODO: Save them as an eval file!
                print("df_non_amortized", df_non_amortized)
                
                # if raw_data column exists, remove it
                if 'raw_data' in df_non_amortized.columns:
                    df_non_amortized = df_non_amortized.drop(columns=['raw_data'])
                
                print("df_amortized", df_amortized)
                
                df_non_amortized['task'] = task
                # add column net init
                df_non_amortized['net_init'] = net_init
                # Remove level_4 column if it exists
                if 'level_4' in df_non_amortized.columns:
                    df_non_amortized = df_non_amortized.drop(columns=['level_4'])

                diff1 = set(df_amortized.columns) - set(df_non_amortized.columns)
                diff2 = set(df_non_amortized.columns) - set(df_amortized.columns)
                if diff1 or diff2:
                    print("Columns only in df_amortized:", diff1)
                    print("Columns only in df_non_amortized:", diff2)

                # raise ValueError("check")
                if set(df_amortized.columns) == set(df_non_amortized.columns):
                    df_from_eval = pd.concat([df_amortized, df_non_amortized], ignore_index=True)
                else:
                    raise ValueError("Columns do not match between amortized and non-amortized dataframes.")

                print("df from_eval", df_from_eval)
                # algorithm: all unique algorithms in df_from_eval
                algorithms = df_from_eval['algorithm'].unique()
                print("algorithm", algorithms)
                #  LF simulations: all unique LF simulations in df_from_eval
                lf_sims = df_from_eval['n_lf_simulations'].unique()
                print("lf_sims", lf_sims)
                #  HF simulations: all unique HF simulations in df_from_eval
                hf_sims = df_from_eval['n_hf_simulations'].unique()
                print("hf_sims", hf_sims)
                
                fname = make_eval_filename(eval_metric, algorithms, lf_sims, hf_sims)
                print("fname", fname)   

                # Save this file as eval file
                with open(f"{eval_models_path}/{fname}", "wb") as f:
                    pickle.dump(df_from_eval, f)
            else:
                df_from_eval = load_from_eval_file(eval_models_path, eval_metric)
                
                
            # Add row wich task it comes from (will be used for plotting)
            # if task column does not exist
            if 'task' not in df_from_eval.columns:
                df_from_eval['task'] = task

            all_tasks_df = pd.concat([all_tasks_df, df_from_eval], ignore_index=True)
            
            
    print("all_tasks_df", all_tasks_df)

    
    # If multiple tasks or multiple metrics:
    if len(args.tasks) > 1 or len(args.eval_metrics) > 1:
        fig = plot_methods_performance_paper(all_tasks_df, sim_name=None, lf_simulations=batch_lf_sims, evaluation_metric=None, task_setup=plot_setup, plot_amortized_and_non_amortized_seperately=args.plot_amortized_and_non_amortized_seperately)
    else:
        fig = plot_methods_performance_paper(all_tasks_df, sim_name=task, lf_simulations=batch_lf_sims, evaluation_metric=eval_metric, task_setup=plot_setup, plot_amortized_and_non_amortized_seperately=args.plot_amortized_and_non_amortized_seperately)

