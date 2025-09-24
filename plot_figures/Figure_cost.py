
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
    type_estimator='npe', # we always compute the posterior (npe), and do not evaluated likelihood or ratio methods (e.g., NLE, NRE)
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
# import os
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd

# def plot_methods_performance_paper_dual_axis(
#     df, sim_name, task_setup, cost_col="total_sim_cost",
#     pdf=True, svg=True, html=False, export_scale=0
# ):
#     """
#     Single-panel (GaussianBlob) with dual y-axes for paper:
#       - X: total simulation cost (a.u.) in {1,2,10} (discrete, ordered)
#       - Left Y: NLTP
#       - Right Y: NRMSE

#     df must include:
#       ['task','evaluation_metric','algorithm','mean','ci_min','ci_max', cost_col]
#     """

#     # -------- 0) Guardrails & filtering --------
#     req = {'task','evaluation_metric','algorithm','mean','ci_min','ci_max', cost_col}
#     miss = req - set(df.columns)
#     if miss:
#         raise ValueError(f"df is missing required columns: {miss}")

#     df = df[df['task'] == 'GaussianBlob'].copy()
#     if df.empty: raise ValueError("No rows for task 'GaussianBlob'.")

#     df = df[df['evaluation_metric'].isin(['nltp','nrmse'])].copy()
#     if df.empty: raise ValueError("No 'nltp' or 'nrmse' rows found.")

#     allowed_costs = [1, 2, 10]
#     df = df[df[cost_col].isin(allowed_costs)].copy()
#     if df.empty: raise ValueError(f"No rows with {cost_col} in {allowed_costs}.")

#     # Ensure deterministic x ordering (and prettier axis ticks)
#     df[cost_col] = pd.Categorical(df[cost_col], categories=allowed_costs, ordered=True)
#     df = df.sort_values(by=[cost_col, 'algorithm', 'evaluation_metric'])

#     # -------- 1) Figure --------
#     width  = min(520, int(task_setup.width_plots))     # sane minimums for journals
#     height = min(380, int(task_setup.height_plots*1.05))

#     fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

#     # -------- 2) Aesthetics (paper-friendly) --------
#     # Okabe–Ito palette (colorblind-safe)
#     palette = {
#         'black': "#000000", 'orange': "#E69F00", 'sky': "#198AF3",
#         'green': "#009E73", 'yellow': "#F0E442", 'vermilion': "#D55E00", 'purple': "#CC79A7"
#     }

#     # Map algorithms → colors & marker symbols (distinct in grayscale)
#     color_map = {
#         'sbi_npe':  palette['black'],
#         'npe':      palette['black'],
#         'mf_npe':   palette['sky'],
#         'bo_npe':   palette['orange'],
#         'tsnpe':    palette['green'],
#         'mf_tsnpe': palette['purple'],
#         'a_mf_tsnpe': palette['vermilion'],
#         'mf_abc':   "#7A7B7E",
#     }
#     legend_name = {
#         'sbi_npe': 'NPE (SBI)',
#         'npe': 'NPE',
#         'mf_npe': 'MF-NPE',
#         'bo_npe': 'BO-MF-NPE',
#         'tsnpe': 'TSNPE',
#         'mf_tsnpe': 'MF-SNPE',
#         'a_mf_tsnpe': 'ACTIVE-MF-SNPE',
#         'mf_abc': 'MF-ABC',
#     }
#     marker_map = {
#         'sbi_npe': "circle",
#         'npe': "square",
#         'mf_npe': "diamond",
#         'bo_npe': "triangle-up",
#         'tsnpe': "cross",
#         'mf_tsnpe': "x",
#         'a_mf_tsnpe': "star",
#         'mf_abc': "triangle-down",
#     }
#     dash_map = {
#         # dotted for single-fidelity baselines; solid for MF variants
#         'sbi_npe': "dot", 'npe': "dot", 'tsnpe': "dot",
#         'mf_npe': "solid", 'bo_npe': "solid",
#         'mf_tsnpe': "solid", 'a_mf_tsnpe': "solid", 'mf_abc': "solid",
#     }

#     line_width = 3
#     marker_size = 8
#     error_thickness = 3
#     error_cap = 0

#     # Order legend (consistent across metrics)
#     legend_order = [k for k in
#         ['sbi_npe','npe','tsnpe','mf_npe','bo_npe','mf_tsnpe','a_mf_tsnpe','mf_abc']
#         if k in df['algorithm'].unique()
#     ]

#     # -------- 3) Helper to add traces --------
#     def add_metric_traces(metric, secondary_y):
#         sub = df[df['evaluation_metric'] == metric]
#         for algo in legend_order:
#             g = sub[sub['algorithm'] == algo]
#             if g.empty: continue
#             g = g.sort_values(cost_col)
#             name = legend_name.get(algo, algo)
#             color = color_map.get(algo, "#333333")
#             mkr   = marker_map.get(algo, "circle")
#             dash  = dash_map.get(algo, "solid")

#             fig.add_trace(
#                 go.Scatter(
#                     x=g[cost_col].astype(float),
#                     y=g['mean'],
#                     name=name if metric == 'nltp' else f"{name} (NRMSE)",
#                     legendgroup=algo, mode='lines+markers',
#                     line=dict(color=color, width=line_width, dash=dash),
#                     marker=dict(symbol="diamond" if metric == "nrmse" else "circle",
#                                 size=marker_size, line=dict(width=1, color=color), color=color),
#                     error_y=dict(type='data', symmetric=False,
#                                  array=g['ci_max'].to_numpy(),
#                                  arrayminus=np.abs(g['ci_min']).to_numpy(),
#                                  thickness=error_thickness, width=error_cap),
#                     hoverinfo="skip"  # disable hover for clean exports
#                 ),
#                 row=1, col=1, secondary_y=secondary_y
#             )

#     # NLTP left, NRMSE right
#     add_metric_traces('nltp',  secondary_y=False)
#     add_metric_traces('nrmse', secondary_y=True)

#     # -------- 4) Axes & layout --------
#     fig.update_xaxes(
#         title_text="Total simulation cost (a.u.)",
#         tickmode='array', tickvals=allowed_costs, ticktext=[str(v) for v in allowed_costs],
#         showgrid=False, zeroline=False,
#         tickfont=dict(family="Arial", size=16), title_font=dict(family="Arial", size=18)
#     )
#     fig.update_yaxes(
#         title_text="NLTP", secondary_y=False,
#         showgrid=True, gridcolor="#E6E6E6", gridwidth=1,
#         zeroline=True, zerolinecolor="#CCCCCC", zerolinewidth=1,
#         tickfont=dict(family="Arial", size=16), title_font=dict(family="Arial", size=18)
#     )
#     fig.update_yaxes(
#         title_text="NRMSE", secondary_y=True,
#         showgrid=False,
#         tickfont=dict(family="Arial", size=16), title_font=dict(family="Arial", size=18)
#     )

#     title_text = "" if sim_name is None else f"{sim_name}"
#     fig.update_layout(
#         width=width, height=height,
#         margin=dict(l=10, r=10, b=10, t=10),
#         plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
#         font=dict(family="Arial", color="#000000"),
#         legend=dict(
#             orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0,
#             font=dict(size=14), itemwidth=50
#         ),
#         title=dict(text=title_text, x=0.5, y=0.98, font=dict(size=18)),
#         modebar_remove=[
#             "zoom","pan","select","lasso2d","zoomin","zoomout","autoscale","resetScale2d",
#             "hoverClosestCartesian","hoverCompareCartesian","toggleSpikelines"
#         ]
#     )

#     # -------- 5) Save --------
#     out_dir = f"{task_setup.main_path}/nltp_nrmse/plots"
#     os.makedirs(out_dir, exist_ok=True)
#     base = f"{out_dir}/GaussianBlob_dualaxis_nltp+nrmse_cost"
#     if svg:  fig.write_image(f"{base}.svg")
#     if pdf:  fig.write_image(f"{base}.pdf")
#     if html: fig.write_html (f"{base}.html", include_mathjax='cdn')

#     if getattr(task_setup, "show_plots", False):
#         fig.show()

#     print(f"Saved to: {base}" + (".svg " if svg else " ") + (".pdf " if pdf else " ") + (".html" if html else ""))
#     return fig



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
