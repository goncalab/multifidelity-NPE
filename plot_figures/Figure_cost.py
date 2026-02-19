
#%%
import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mf_npe.config.TaskSetup import TaskSetup


sim_name = "GaussianBlob"
theta_dim = 3
net_init = 1
batch_lf_sims = [5000] # dummy, does not matter
batch_hf_sims = [50] # dummy, does not matter
true_xen = np.array([[0.0, 0.0, 0.0]])  # dummy, does not matter
main_path = f"./../data/{sim_name}/{theta_dim}_dimensions"

config_model = dict(
    max_num_epochs=2**31 - 1, # high number since we have early stopping
    batch_size = 200, # increasing the batch size will speed up the training, but the model will be less accurate
    learning_rate= 5e-4, # Learning rate for Adam optimizer
    device = 'cpu', #process_device(),
    validation_fraction = 0.1, # Fraction of the data to use for validation
    patience=20, # The number of epochs to wait for improvement on the validation set before terminating training.
    n_transforms = 5, 
    n_bins=8,
    n_hidden_features = 50,
    clip_max_norm = 5.0, # value to which to clip total gradient norm to prevent exploding gradients. Use None for no clipping
    logit_transform_theta_net = True, # for training in unbound space: Then we do not have that much leakage in posterior
    z_score_theta = False, 
    z_score_x = True,
    # For active learning
    active_learning_pct=0.8,
    n_rounds_AL = 5, # From 1 to 5 
    n_theta_samples = 1000, #250,
    n_ensemble_members = 5, 
    )

task_setup = TaskSetup(sim_name=sim_name, 
                config_model=config_model, 
                main_path=main_path, 
                batch_lf_datasize=batch_lf_sims, 
                batch_hf_datasize=batch_hf_sims, 
                n_network_initializations=net_init,
                theta_dim=theta_dim,
                n_true_xen=true_xen.shape[0],
                seed=None)

def plot_methods_performance_paper_dual_axis(
    df, sim_name, task_setup,
    cost_col="total_sim_cost"  # expects values in {1,2,10}
):
    """
    Single-panel figure for the 'GaussianBlob' task with dual y-axes:
      - Left y-axis: NLTP
      - Right y-axis: NRMSE
    X-axis is 'total simulation cost (a.u.)' with discrete values {1,2,10}.

    df must contain columns:
      ['task','evaluation_metric','algorithm','mean','ci_min','ci_max', cost_col]

    Notes:
    - Multifidelity approaches are assumed to have a single evaluation per cost value.
    - If your dataframe uses a different name for total cost, pass it via `cost_col`.
    """

    # -------------------- 0) Guardrails & filtering --------------------
    required_cols = {'task','evaluation_metric','algorithm','mean','ci_min','ci_max', cost_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    # Only 1 task: GaussianBlob
    df = df[df['task'] == 'GaussianBlob'].copy()
    if df.empty:
        raise ValueError("No rows for task 'GaussianBlob' found in df.")

    # Keep only nltp & nrmse
    df = df[df['evaluation_metric'].isin(['nltp','nrmse'])].copy()
    if df.empty:
        raise ValueError("No 'nltp' or 'nrmse' rows found in df.")

    # Ensure cost in {1,2,10} and with deterministic order on x
    allowed_costs = [1, 2, 10]
    df = df[df[cost_col].isin(allowed_costs)].copy()
    if df.empty:
        raise ValueError(f"No rows with {cost_col} in {allowed_costs}.")

    # Sort for consistent plotting
    df[cost_col] = df[cost_col].astype(float)
    df = df.sort_values(by=[cost_col, 'algorithm', 'evaluation_metric'])

    # -------------------- 1) Figure --------------------
    width  = task_setup.width_plots 
    height = task_setup.height_plots * 1.2

    fig = make_subplots(
        rows=1, cols=1, specs=[[{"secondary_y": True}]],
        subplot_titles=(["GaussianBlob"])
    )

    # -------------------- 2) Colors & legend names --------------------
    color_map = {
        'sbi_npe': '#5D5D5D',   # NPE (SBI) oracle-ish
        'npe':     '#000000',   # NPE
        'mf_npe':  '#198AF3',   # MF-NPE (single eval)
        'bo_npe':  '#FFA15A',   # BO-MF-NPE (single eval)
        'tsnpe':   '#0F9E9A',   # TSNPE
        'mf_tsnpe':'#AB63FA',   # MF-SNPE (single eval)
        'a_mf_tsnpe':'#F65009', # ACTIVE-MF-SNPE (single eval)
        'mf_abc':  '#7A7B7E',   # MF-ABC
    }

    legend_name = {
        'sbi_npe': 'NPE (SBI)',
        'npe': 'NPE',
        'mf_npe': 'MF-NPE',
        'bo_npe': 'BO-MF-NPE',
        'tsnpe': 'TSNPE',
        'mf_tsnpe': 'MF-SNPE',
        'a_mf_tsnpe': 'ACTIVE-MF-SNPE',
        'mf_abc': 'MF-ABC',
    }

    # -------------------- 3) Helper to add traces for a metric --------------------
    def add_metric_traces(metric, secondary_y):
        sub = df[df['evaluation_metric'] == metric]
        if sub.empty:
            return

        for algo, g in sub.groupby('algorithm'):
            name = legend_name.get(algo, algo)
            color = color_map.get(algo, '#333333')

            # Aggregate by cost value (in case multiple runs/replicates are pre-aggregated already)
            g = g.sort_values(cost_col)

            fig.add_trace(
                go.Scatter(
                    x=g[cost_col],
                    y=g['mean'],
                    name=name if metric == 'nltp' else f"{name} (NRMSE)",  # keep legend disambiguation optional
                    legendgroup=algo,
                    mode='lines+markers',
                    marker=dict(color=color),
                    line=dict(color=color, dash='dot' if algo in {'sbi_npe','npe','tsnpe'} else 'solid'),
                    error_y=dict(
                        type='data', symmetric=False,
                        array=(g['ci_max']).to_numpy(),
                        arrayminus=np.abs(g['ci_min']).to_numpy(),
                        width=0, thickness=2
                    ),
                    hovertemplate=f"Cost={{x}}<br>{metric.upper()}={{y:.3g}}<extra>{name}</extra>"
                ),
                row=1, col=1, secondary_y=secondary_y
            )

    # NLTP on left, NRMSE on right
    add_metric_traces('nltp',   secondary_y=False)
    add_metric_traces('nrmse',  secondary_y=True)

    # -------------------- 4) Axes & layout --------------------
    fig.update_xaxes(
        title_text="Total simulation cost (a.u.)",
        tickmode='array', tickvals=allowed_costs, ticktext=[str(v) for v in allowed_costs],
        showgrid=False, zeroline=False
    )

    # Left y-axis (NLTP)
    fig.update_yaxes(
        title_text="NLTP",
        showgrid=True, gridcolor='#DEE3EA',
        zeroline=True, zerolinewidth=1, zerolinecolor="#DEE3EA",
        tickfont=dict(size=17, family="Arial"),
        row=1, col=1, secondary_y=False
    )
    # Right y-axis (NRMSE)
    fig.update_yaxes(
        title_text="NRMSE",
        showgrid=False,
        tickfont=dict(size=17, family="Arial"),
        row=1, col=1, secondary_y=True
    )

    title_text = "" if sim_name is None else f"{sim_name}"
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        plot_bgcolor='#fff',
        paper_bgcolor='#fff',
        margin=dict(l=10, r=10, b=10, t=60),
        font_color=task_setup.axis_color,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='left', x=0.0),
        title_text=title_text, title_x=0.5, title_y=0.98, title_font_size=task_setup.font_size
    )

    # -------------------- 5) Save --------------------
    out_dir = f"{task_setup.main_path}/nltp_nrmse/plots"
    os.makedirs(out_dir, exist_ok=True)
    out_base = f"{out_dir}/GaussianBlob_dualaxis_nltp+nrmse_cost"
    fig.write_image(f"{out_base}.svg")
    fig.write_html (f"{out_base}.html")

    if getattr(task_setup, "show_plots", False):
        fig.show()

    print(f"Saved to {out_base}.svg and .html")
    return fig


# Load dataframe from the provided path
sim_name = "GaussianBlob"
task_setup = type('TaskSetup', (object,), {
    'width_plots': 800,
    'height_plots': 600,
    'axis_color': '#333333',
    'font_size': 17,
    'main_path': './plot_figures',          
})()    


import pickle
import pandas as pd

# List all files you want to load
paths = [
        "./../data/GaussianBlob/3_dimensions/models/evaluate_nrmse_npe_LF5000_HF100_Ninits1_seed12.pkl",
        "./../data/GaussianBlob/3_dimensions/models/evaluate_nrmse_mf_npe_LF5000_HF50_Ninits1_seed12.pkl",
        "./../data/GaussianBlob/3_dimensions/models/evaluate_nrmse_npe_LF10_HF200_Ninits1_seed12.pkl",
        "./../data/GaussianBlob/3_dimensions/models/evaluate_nrmse_mf_npe_LF10000_HF100_Ninits1_seed12.pkl",
        "./../data/GaussianBlob/3_dimensions/models/evaluate_nrmse_npe_LF10_HF1000_Ninits1_seed12.pkl",
        "./../data/GaussianBlob/3_dimensions/models/evaluate_nrmse_mf_npe_LF50000_HF500_Ninits1_seed12.pkl"
]

# manual mapping: which path gets which total_sim_cost
path_to_cost = {
    paths[0]: 1,
    paths[1]: 1,
    paths[2]: 2,
    paths[3]: 2,
    paths[4]: 10,
    paths[5]: 10
}

dfs = []
for p in paths:
    with open(p, "rb") as f:
        loaded = pickle.load(f)
        # If loaded is already a DataFrame, just append
        if isinstance(loaded, pd.DataFrame):
            df_tmp = loaded.copy()
        else:
            # If it’s dict-like, turn it into a DataFrame
            df_tmp = pd.DataFrame(loaded)
        # Keep track of the file it came from
        df_tmp["source_file"] = p
        dfs.append(df_tmp)

# Concatenate all dataframes
df = pd.concat(dfs, ignore_index=True)

# add column 'total_sim_cost'
# if the number of hf simulations < 101, then cost = 1, if between 101 - 999, then cost = 2, if > 999, then cost = 10
# df['total_sim_cost'] = pd.cut(df['n_hf_simulations'], bins=[0, 100, 1000, float('inf')], labels=[1, 2, 10])
df['total_sim_cost'] = df['source_file'].map(path_to_cost)

print(df)

plot_methods_performance_paper_dual_axis(
    df, sim_name, task_setup,
    cost_col="total_sim_cost"  # expects values in {1,2,10}
)
# %%
