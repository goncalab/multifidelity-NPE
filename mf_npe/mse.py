import os
from mf_npe.utils.calculate_error import ci95_t
import plotly.graph_objects as go

def plot_mse_barplot(df_mse, task_setup, net_init, true_xen, inference_method, n_train_sims, ppc_path):
    # Compute means and 95% CI using t-distribution
    mse_mean, mse_ci = ci95_t(df_mse["mse"])
    mse_lf_mean, mse_lf_ci = ci95_t(df_mse["mse_lf"])
    mse_prior_mean, mse_prior_ci = ci95_t(df_mse["mse_prior"])

    # Save means to text file
    mean_file_name = f"mse_mean_{net_init}_{len(true_xen)}_{inference_method}_{n_train_sims}_{task_setup.config_data['type_lf']}.txt"
    with open(os.path.join(ppc_path, mean_file_name), "w") as f:
        f.write(f"MSE: {mse_mean}\n")
        f.write(f"MSE LF: {mse_lf_mean}\n")
        f.write(f"MSE PRIOR: {mse_prior_mean}\n")

    # Plot setup
    bar_labels = ["MF-NPE", "NPE (LF)", "Prior"]
    bar_values = [mse_mean, mse_lf_mean, mse_prior_mean]
    error_bars = [mse_ci, mse_lf_ci, mse_prior_ci]
    bar_colors = ['#EA6340', '#234070', '#0F9E9A']
    bar_positions = [0, 1, 2]

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bar_positions,
        y=bar_values,
        error_y=dict(
            type='data',
            array=error_bars,
            visible=True,
            color='black',
            thickness=1.5,
            width=5,
        ),
        marker_color=bar_colors,
        textposition='auto'
    ))

    fig.update_layout(
        title=dict(
            text=f"Mean squared error over 1000 ppc, {len(true_xen)} x, LF: {task_setup.config_data['type_lf']}",
            font=dict(size=task_setup.title_size)
        ),
        font=dict(size=32),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            title="",
            tickvals=bar_positions,
            ticktext=bar_labels,
            showticklabels=True
        ),
        yaxis=dict(
            title="MSE",
            range=[0, 5],
            gridcolor='lightgray',
            zeroline=True,
        )
    )

    fig.show()

    # Save plot
    fig.write_html(os.path.join(ppc_path, f"mse_plot_{task_setup.config_data['type_lf']}.html"))
    fig.write_image(os.path.join(ppc_path, f"mse_plot_{task_setup.config_data['type_lf']}.svg"))