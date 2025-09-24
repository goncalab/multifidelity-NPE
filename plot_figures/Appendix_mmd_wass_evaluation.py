#%%
# =============================================================================
#       SCRIPT TO REPRODUCE FIGURES PAPER
# =============================================================================

# DESCRIPTION: Evaluation of the MMD test statistics.
# > Tasks: OU process
# > Experiments:
#   - exp 1: OU process with 3 or 4 parameter dimensions

# USAGE:
# >> python Figure_mmd_wass_evaluation.py

# LOAD PICKLES
from datetime import datetime
import os
import pandas as pd
import torch
from mf_npe.config.TaskSetup import TaskSetup
from mf_npe.utils.calculate_error import mean_confidence_interval
from mf_npe.plot.method_performance import plot_methods_performance_paper
import matplotlib.pyplot as plt
from types import SimpleNamespace
from mf_npe.evaluation import Evaluation
import re
import ast

# ------------------------------
# USER CONFIGURATION
# ------------------------------
load_from_pickles = True  # Set this to False to recompute everything
theta_dim = 3
batch_lf_sims = [10**3, 10**4, 10**5]
metric = 'wasserstein'
sim_name = 'OUprocess'

# ------------------------------
# PATH SETUP
# ------------------------------
main_path = f"./../data/{sim_name}/{theta_dim}_dimensions"
path_posterior_samples = f"{main_path}/pickles/posterior_samples"
path_save_plot = f"{main_path}/{metric}"
path_metrics = f"{path_save_plot}/data"

df_all_trials = pd.DataFrame()

# ------------------------------
# LOAD FROM PICKLE OPTION
# ------------------------------
if load_from_pickles:
    print("Loading all metrics from saved pickles...")
    for file in os.listdir(path_metrics):
        if file.endswith('.p'):
            df = pd.read_pickle(os.path.join(path_metrics, file))
            df_all_trials = pd.concat([df_all_trials, df], ignore_index=True)
else:
    print("Computing distances from scratch...")
    file_names = [f for f in os.listdir(path_posterior_samples) if f.startswith("thetas")]

    parsed_data = []
    for name in file_names:
        match = re.match(r"thetas_(\d+)_(\d+)_((?:[a-z]+_)*[a-z]+)_(\[[\d,\s]+\]|\d+)\.p", name)
        if match:
            net_init = int(match.group(1))
            xos = int(match.group(2))
            method = match.group(3)
            sims_str = match.group(4)
            n_sims = ast.literal_eval(sims_str) if "[" in sims_str else int(sims_str)

            parsed_data.append({
                "file": name,
                "net_init": net_init,
                "xos": xos,
                "method": method,
                "n_sims": n_sims,
            })

    for item in parsed_data:
        data = pd.read_pickle(f'{path_posterior_samples}/{item["file"]}')
        posterior_s = data['posterior_samples']
        n_hf_simulations = data.get('num_hifi_total', 0) if item["method"] == 'mf_abc' else 0

        true_xen = pd.read_pickle(f"./../data/OUprocess/{theta_dim}_dimensions/pickles/true_data/true_xen_{item['xos']}.p")['true_xen']
        true_posterior_s = pd.read_pickle(f'{path_posterior_samples}/true_thetas_{item["xos"]}.p')['true_posterior_samples']

        def has_nan_or_inf(tensor):
            return torch.isnan(tensor).any() or torch.isinf(tensor).any()

        if has_nan_or_inf(posterior_s) or has_nan_or_inf(true_posterior_s) or has_nan_or_inf(true_xen):
            print(f"Invalid values found in one of the tensors for file: {item['file']}")
            continue
        
        task_setup = []
        
        evaluation = Evaluation(true_xen, task_setup)

        dataframe = evaluation.eval_ground_truth_available(
            true_xen=true_xen,
            metric=metric,
            posterior_samples=posterior_s,
            true_posterior_samples=true_posterior_s,
            n_simulations=item["n_sims"],
            type_estimator=item["method"],
            num_hifi_abc=n_hf_simulations,
        )

        # Save for reuse
        if not os.path.exists(path_metrics):
            os.makedirs(path_metrics)
        dataframe.to_pickle(f"{path_metrics}/{item['method']}_{item['xos']}_{item['n_sims']}.p")

        df_all_trials = pd.concat([df_all_trials, dataframe], ignore_index=True)

# ------------------------------
# PLOTTING
# ------------------------------
grouped_df = df_all_trials.groupby(
    ['n_lf_simulations', 'n_hf_simulations', 'fidelity']
)['raw_data'].apply(mean_confidence_interval).reset_index()

grouped_df = grouped_df[grouped_df['fidelity'] != 'sbi_npe']
grouped_df = grouped_df[~((grouped_df['fidelity'] == 'active_npe') &
                          (grouped_df['n_lf_simulations'].isin([1000, 10000, 100000])))]

plot_setup = SimpleNamespace(
    main_path=main_path,
    axis_color='#6A798F',
    CURR_TIME=datetime.now().strftime("%Y-%m-%d %Hh%M"),
    width_plots=600,
    height_plots=400,
    font_size=20,
    title_size=20,
    gridwidth=2,
    show_legend=True,
    show_plots=True,
)

fig1 = plot_methods_performance_paper(
    grouped_df, f"OU{theta_dim}", batch_lf_sims, df_all_trials['evaluation_metric'][0], plot_setup
)

#%%
