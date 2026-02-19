#%%
from datetime import datetime
import os
import pickle
from types import SimpleNamespace
import plotly.graph_objects as go

# ---- Parameters ----
simulator = 'OUprocess'
theta_dim = 3
metric = 'c2st'

path = f"./../data/{simulator}/{theta_dim}_dimensions/mf-abc/"


pickle_files = [
    f"{path}/eps_0.5,0.5.pkl",
    f"{path}/eps_0.5,1.pkl",
    f"{path}/eps_1,1.pkl",
    f"{path}/eps_2,1.pkl",
    f"{path}/eps_2,2.pkl"
]

fig_log = go.Figure()

for file_path in pickle_files:
    with open(file_path, "rb") as f:
        df = pickle.load(f)

    # Extract epsilon values from filename for legend label
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    eps_label = base_name.replace("eps_", "").replace(",", ", ")
    label = f"ε = ({eps_label})"

    # Group and extract representative values
    grouped = df.groupby("n_hf_simulations").agg({
        "mean": "mean",
        "ci_max": "first",
        "ci_min": "first"
    }).reset_index()

    fig_log.add_trace(go.Scatter(
        x=grouped["n_hf_simulations"],
        y=grouped["mean"],
        error_y=dict(
            type="data",
            symmetric=False,
            array=grouped["ci_max"],
            arrayminus=-grouped["ci_min"]
        ),
        mode="lines+markers",
        name=label
    ))

# Define your desired tick values (powers of 10)
log_ticks = [10**i for i in range(1, 6)]  # Adjust range as needed
log_tick_text = [str(tick) for tick in log_ticks]

# Update layout with logarithmic x-axis and grouped legend title
fig_log.update_layout(
    title="Varying epsilon in OU3 task",
    xaxis_title="Number of High-Fidelity Simulations (log scale)",
    yaxis_title="C2ST (mean ± CI)",
    template="plotly_white",
    xaxis=dict(
        type="log",
        tickvals=log_ticks,
        ticktext=log_tick_text
    ),
    legend_title_text="ε"
)

task_setup = SimpleNamespace(main_path=path,
                             show_plots=True,
                            CURR_TIME=datetime.now().strftime("%Y-%m-%d %Hh%M"),
                            width_plots=800,
                            height_plots=400,
                            font_size=14,
                            title_size=14,
                            gridwidth=2,
                            show_legend=True,
                            )

fig_log.update_layout(autosize=False,
                        width=task_setup.width_plots,
                        height=task_setup.height_plots,
                        plot_bgcolor='#ffffff',
                        margin=dict(
                            l=0,
                            r=10,
                            b=0,
                            t=50,
                            pad=4
                        ),
                        title_x=0.5, 
                        title_y=0.97, 
                        title_font_size=task_setup.font_size)


    
fig_log.update_xaxes(zeroline=True, 
                #zerolinewidth=6, 
                title_font_size=task_setup.title_size,
                tickfont=dict(
                    color='#2A3F5F',  # Set the color of numbers on the x-axis
                    size=task_setup.font_size,       # Optional: Set font size
                    family="Arial" # Optional: Set font family
                ),
                )
fig_log.update_yaxes(zeroline=True, 
                # title_font_color="red",
                linewidth=1,
                title_font_size=task_setup.title_size,
                tickfont=dict(
                    color='#2A3F5F',  # Set the color of numbers on the x-axis
                    size=task_setup.font_size,       # Optional: Set font size
                    family="Arial" # Optional: Set font family
                ),
                #zerolinewidth=6, 
                )

path_plots = f'{path}/mf-abc-plot' # evaluate with c2st here
evaluation_metric = "c2st"

if not os.path.exists(path_plots):
    os.makedirs(path_plots)
fig_log.write_image(f"{path_plots}/{evaluation_metric}.svg")
fig_log.write_html(f"{path_plots}/{evaluation_metric}.html")


fig_log.show()
#%%