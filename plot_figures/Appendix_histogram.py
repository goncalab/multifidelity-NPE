#%%
# =============================================================================

#       SCRIPT TO REPRODUCE HISTOGRAMS FOR THE APPENDIX OF THE PAPER

# =============================================================================

# DESCRIPTION: Plotting histograms for the Appendix.
# > Tasks: Ornstein-Uhlenbeck process with 2, 3, and 4 dimensions

# USAGE:
# >> python Appendix_histogram.py 



# LOAD PICKLES
from datetime import datetime
import os
import numpy as np
import pandas as pd
import mf_npe.task_setup as task_setup
from mf_npe.utils.calculate_error import mean_confidence_interval
from mf_npe.plot.method_performance import plot_methods_performance_paper

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


sim_name = 'OUprocess'
theta_dim = 2 # Can be 2, 3 or 4
main_path = f"./../data/{sim_name}/{theta_dim}_dimensions/c2st"
batch_lf_sims = [10**3, 10**4, 10**5] # The ranges on which the lf-model was trained on.
from types import SimpleNamespace


plot_setup = SimpleNamespace(main_path=f"{main_path}",
                        CURR_TIME=datetime.now().strftime("%Y-%m-%d %Hh%M"),
                        width_plots=1600,
                        height_plots=300,
                        )

# Merge all the pkl file results from the different test trials in one dataframe
all_trials = []
for file in os.listdir(f'{main_path}'):
    if file.endswith(".pkl"):
        df = pd.read_pickle(f'{main_path}/{file}')
        print(f"Loaded {file}")
        all_trials.append(df)
        
df_all_trials = pd.concat(all_trials, ignore_index=True)

# Group the dataframe by the number of simulations and calculate the mean and CI-95
grouped_df = df_all_trials.groupby(['n_lf_simulations', 'n_hf_simulations', 'fidelity'])['raw_data'].apply(mean_confidence_interval).reset_index()

# Remove results from fidelity = active_npe with lf_simulations = 1000 and 100000
grouped_df = grouped_df[~((grouped_df['fidelity'] == 'active_npe') & (grouped_df['n_lf_simulations'] == 1000))]
grouped_df = grouped_df[~((grouped_df['fidelity'] == 'active_npe') & (grouped_df['n_lf_simulations'] == 100000))]


def plot_histograms():
    fig = make_subplots(rows=1, cols=5, shared_xaxes=True, subplot_titles=("50 HF simulations", "100 HF simulations", "1000 HF simulations", "10000 HF simulations", "100000 HF simulations"))

    n_hf_sims_list = [50, 100, 1000, 10000, 100000]
    colors = {'npe': '#0000A6', 'mf_npe': '#B6E880'}
    show_legend = {'npe': True, 'mf_npe': True}

    for i, n_hf_sims in enumerate(n_hf_sims_list, start=1):
        npe = df_all_trials[(df_all_trials['fidelity'] == 'npe') & (df_all_trials['n_hf_simulations'] == n_hf_sims)]
        mf_npe5 = df_all_trials[(df_all_trials['fidelity'] == 'mf_npe') & (df_all_trials['n_lf_simulations'] == 100000) & (df_all_trials['n_hf_simulations'] == n_hf_sims)]

        x0 = npe['raw_data']
        x1 = mf_npe5['raw_data']

        fig.add_trace(go.Histogram(x=x0, name='NPE', opacity=0.75, marker_color=colors['npe'], showlegend=show_legend['npe']), row=1, col=i)
        fig.add_trace(go.Histogram(x=x1, name='MF-NPE5', opacity=0.75, marker_color=colors['mf_npe'], showlegend=show_legend['mf_npe']), row=1, col=i)

        show_legend['npe'] = False
        show_legend['mf_npe'] = False

    fig.update_traces(nbinsx=30)
    fig.update_xaxes(title_text="C2ST", range=[0.49, 1.01], showline=True, linewidth=1, linecolor='#E5ECF6')
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(barmode='overlay', plot_bgcolor='#fff', 
                      legend=dict(x=0, y=2, orientation='h'),
                      legend_font=dict(size=16),
                      width=plot_setup.width_plots, height=plot_setup.height_plots)

    path = f"{main_path}/results"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.write_image(f"{path}/histograms.svg")
    fig.write_html(f"{path}/histograms.html")

    fig.show()

plot_histograms()

#%%
