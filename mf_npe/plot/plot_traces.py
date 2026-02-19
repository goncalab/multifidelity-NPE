import numpy as np
import plotly.express as px
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mf_npe.config.plot as plot_config
    
def plot_OU_xen(true_xen, lf_xen, full_trace, config_data, path_to_save):  
    max_n_traces = 3 # max n of traces to plot
    type_x = 'xen'
    n_traces = len(true_xen)
    
    logscale = config_data['logspace']
    subsample_rate = config_data['subsample_rate']
    first_n_samples = config_data['first_n_samples']
    
    l_trace = config_data['length_total_trace']
    xdim = config_data['x_dim_hf']
    
    df_true_xen = pd.DataFrame(true_xen[:max_n_traces]).T if n_traces > max_n_traces else pd.DataFrame(true_xen).T
    df_full_trace = pd.DataFrame(full_trace[:max_n_traces]).T if n_traces > max_n_traces else pd.DataFrame(full_trace).T
    df_lf_xen = pd.DataFrame(lf_xen[:max_n_traces]).T if n_traces > max_n_traces else pd.DataFrame(lf_xen).T
    
    # Initialize figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None],
           [ {},{}],
            ],
        subplot_titles=("Full trace","HF summary features", "LF summary features"),
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,   
    )
    
    max_logspace = np.log10(l_trace-1)

    for col in df_full_trace.columns:
        fig.add_trace(
            go.Scatter(x=np.arange(0, l_trace), y=df_full_trace[col]),
            row=1, col=[1,2],
        )
        
        if logscale:
            fig.add_trace(
                go.Scatter(x=np.logspace(0, max_logspace, num=xdim), y=df_true_xen[col]),
                row=2, col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(x=np.arange(0, first_n_samples, subsample_rate), y=df_true_xen[col]),
                row=2, col=1,
            )
            
        fig.add_trace(
            go.Scatter(x=np.arange(0, l_trace), y=df_lf_xen[col]),
            row=2, col=2
        )
        
    fig.update_layout(
        yaxis=dict(
            domain=[0.4, 1],
            range=[0, 6]
        ),
        yaxis2=dict(
            domain=[0, 0.30],
            range=[0, 6],
            anchor="x1"
        ),
        yaxis3=dict(
            domain=[0, 0.30],
            range=[0, 6]
        ),
        font_color=plot_config.axis_color,
        xaxis2_title="$t$",
        yaxis_title="$X(t)$",
        # No legend
        showlegend=False,
        # legend_title=f"$x_0$",
        # legend=dict(
        #     yanchor="top",
        #     y=0.99,
        #     xanchor="left",
        #     x=0.01
        # ),
        
        width=plot_config.width_plots, height=plot_config.height_plots,
    )
    
    if logscale:
        fig.update_layout(
            xaxis2 = dict(
                type="log"
                # title = 't'
                #anchor="x1"
            )
        )
    else:
        fig.update_layout(
            xaxis2 = dict(
                title = 't'
                #anchor="x1"
            )
        )

    path_plot = f"{path_to_save}/plot_traces"
    if not os.path.exists(path_plot):
        os.makedirs(path_plot)   
    fig.write_image(f"{path_plot}/{type_x}.svg")
    fig.write_html(f"{path_plot}/{type_x}.html")
        
    if plot_config.show_plots:
        fig.show()
    

def plot_CompNeuron_xen(true_xen, I_curr, full_trace, dt, title, inference_method, n_train_sims, i, path_to_save, true_x_trace=None):  
    max_n_traces = 10 # max n of traces to plot
    type_x = 'xen'
    n_traces = len(true_xen)
    
    if n_traces == 1: 
        df_full_trace = pd.DataFrame([full_trace]).T
    else:
        df_full_trace = pd.DataFrame(full_trace[:max_n_traces]).T if n_traces > max_n_traces else pd.DataFrame(full_trace).T
    
    df_current = pd.DataFrame(I_curr)
    
    bg_color = '#fff'

    fig = go.Figure()
    colors = px.colors.sequential.Blues_r

    for idx, col in enumerate(df_full_trace.columns):    
        fig.add_trace(
            go.Scatter(x=np.arange(0, len(df_full_trace[0]), dt), y=df_full_trace[col],
                       marker=dict(
                            color=colors[idx % len(colors)],
                            # colorscale="Viridis"
                        ),
                        mode="lines"
                        )
        )
        
        
    if true_x_trace is not None:
        fig.add_trace(
                    go.Scatter(x=np.arange(0, len(true_x_trace), dt), y=true_x_trace,
                            mode="lines", line=go.scatter.Line(color="black", dash='dash')),
        )
        
    fig.update_layout(
        title=title,
        xaxis_title="$t [ms]$ ",
        yaxis_title="$V(t)$",
        plot_bgcolor=bg_color,
        font_color=plot_config.axis_color,
        # No legend
        showlegend=True,
        # legend_title=f"$x_0$",
        # legend=dict(
        #     yanchor="top",
        #     y=0.99,
        #     xanchor="left",
        #     x=0.01
        # ),
        width=plot_config.width_plots, height=plot_config.height_plots // 1.5,
    )

    path_plot = f"{path_to_save}/plot_traces"
    if not os.path.exists(path_plot):
        os.makedirs(path_plot)   
    fig.write_image(f"{path_plot}/{type_x}{inference_method}{n_train_sims}_{i}.svg")
    fig.write_html(f"{path_plot}/{type_x}{inference_method}{n_train_sims}_{i}.html")
        
    if plot_config.show_plots:
        fig.show()

    
def plot_true_OU_traces(hf_full_trace, true_xen, lf_xen, gamma, mu_offset, path_to_save):  
    # Transpose to get the correct format for plotting.
    hf_trace_df = pd.DataFrame(hf_full_trace).T  
    true_xen_df = pd.DataFrame(true_xen).T
    lf_xen_df = pd.DataFrame(lf_xen).T
    
    dt = 1
    
    length_trace = len(hf_full_trace[0])    
    fig = go.Figure()
    
    for col in hf_trace_df.columns:
        fig.add_trace(
            go.Scatter(x=np.arange(0, length_trace, dt), y=hf_trace_df[col], name=f"HF trace {col+1}")
        )
        
    for col in true_xen_df.columns:
        fig.add_trace(
            go.Scatter(x=np.arange(0, length_trace, dt+9), y=true_xen_df[col], name=f"HF summary stats {col+1}")
        )
        
    for col in lf_xen_df.columns:
        fig.add_trace(
            go.Scatter(x=np.arange(0, length_trace, dt+9), y=lf_xen_df[col], name=f"LF summary stats {col+1}")
        )    
    
    fig.update_layout(
        title=f"gamma: {gamma}, mu_off: {mu_offset}",
        width=plot_config.width_plots, height=plot_config.height_plots,
        xaxis_title="t",
        yaxis_title="X(t)",
        font_color=plot_config.axis_color,
    )
    
    if plot_config.show_plots:
        fig.show()

    plot_traces = f"{path_to_save.main_path}/plot_traces"
    if not os.path.exists(plot_traces):
        os.makedirs(plot_traces)   
    fig.write_image(f"{plot_traces}/{gamma}_{mu_offset}.svg")
    fig.write_html(f"{plot_traces}/{gamma}_{mu_offset}.html")

