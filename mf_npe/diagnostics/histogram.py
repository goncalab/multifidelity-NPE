from turtle import pd
from plotly.subplots import make_subplots
import mf_npe.config.plot as plot_config
import plotly.graph_objects as go

def plot_xen_histogram(xen, title_plot, xo):
    df = pd.DataFrame(xen)
    
    n_plots = xen.size(dim=1)    

    # Define histogram titles
    titles = ['n spikes', 'mu RP', 'std RP', 'mean V',  'skewness V', 'kurtosis V'] #'std V',

    # Create subplots
    if n_plots > 6:
        fig = make_subplots(rows=3, cols=3, subplot_titles=titles)
    elif n_plots > 5:
        fig = make_subplots(rows=2, cols=3, subplot_titles=titles)
    elif n_plots > 3:
        fig = make_subplots(rows=2, cols=2, subplot_titles=titles)

    # Add histograms to subplots
    for i in range(n_plots):
        # row = i // 3 + 1
        # col = i % 3 + 1
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(go.Histogram(x=df[i], name=titles[i]), row=row, col=col)
        

    # TODO: For loop around does not work
    if xo is not None:
        # for i in range(n_plots):
            fig.update_layout(
                shapes=[
                    dict(type="line", xref=f"x1", yref=f"y1",
                        x0=xo[0], y0=0, x1=xo[0], y1=60, line_width=2),
                    dict(type="line", xref="x2", yref="y2",
                        x0=xo[1], y0=0, x1=xo[1], y1=60, line_width=2),
                    dict(type="line", xref="x3", yref="y3",
                        x0=xo[2], y0=0, x1=xo[2], y1=60, line_width=2),
                    dict(type="line", xref="x4", yref="y4",
                        x0=xo[3], y0=0, x1=xo[3], y1=60, line_width=2),
                    # dict(type="line", xref="x5", yref="y5",
                    #     x0=xo[4], y0=0, x1=xo[4], y1=60, line_width=2),
                    # dict(type="line", xref="x6", yref="y6",
                    #     x0=xo[5], y0=0, x1=xo[5], y1=60, line_width=2),
                    # dict(type="line", xref="x7", yref="y7",
                    #     x0=xo[6], y0=0, x1=xo[6], y1=60, line_width=2),
                    ]
                )

    # Update layout
    fig.update_layout(title_text=title_plot, width=plot_config.width_plots, height=plot_config.height_plots)

    # Show figure
    if plot_config.show_plots:
        fig.show()
    
    
    
   