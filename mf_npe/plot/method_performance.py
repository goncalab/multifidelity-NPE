# For paper
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_methods_performance_paper(
    df, sim_name, lf_simulations, evaluation_metric, task_setup,
    plot_amortized_and_non_amortized_seperately=False  # kept for API compat; ignored
):
    """
    Layout: rows = metrics; columns = 2 * tasks (per task: [Amortized | Non-amortized]).
    df must contain:
      ['task','evaluation_metric','algorithm','n_hf_simulations','n_lf_simulations',
       'mean','ci_min','ci_max']
    """
    
    show_legend = True

    metric_order = ['c2st', 'mmd', 'wasserstein', 'nltp', 'nrmse']
    
    print("df in plotmethod", df)
    
    present_metrics = df['evaluation_metric'].dropna().unique().tolist()

    # allow evaluation_metric to be None / str / list
    if evaluation_metric is None:
        metrics = [m for m in metric_order if m in present_metrics]
    elif isinstance(evaluation_metric, str):
        metrics = [m for m in metric_order if m == evaluation_metric and m in present_metrics]
    else:  # list/tuple
        wanted = list(evaluation_metric)
        metrics = [m for m in metric_order if m in wanted and m in present_metrics]

    if not metrics:
        raise ValueError("No supported evaluation_metric found in df.")

    task_order = ['OUprocess', 'L5PC', 'SynapticPlasticity', 'SLCP', 'LotkaVoterra', 'GaussianBlob', 'SIR']
    present_tasks = df['task'].dropna().unique().tolist()
    tasks = [t for t in task_order if t in present_tasks] or present_tasks
    if not tasks:
        raise ValueError("No tasks found in df.")

    n_rows = len(metrics)
    n_cols = 2 * len(tasks)  # per task: [Amortized | Non-amortized]

    # ---- 2) Figure & titles ----
    width  = task_setup.width_plots  * len(tasks) * 2.2  / 2  # mild scale
    height = task_setup.height_plots * n_rows

    # Only title the *left* col of each task pair;
    titles = []
    for r in range(n_rows):
        for t_idx, tname in enumerate(tasks):
            titles.append(tname)  # amortized column
            titles.append("")     # non-amortized column
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.00, vertical_spacing=0.10,
    )
    
    

    # Colors
    mf_colors      = ['#B6E880', '#19D3F3', '#198AF3', '#FF6692',  '#FFA15A']
    a_tsnpe_colors = ['#890909', '#F65109', ]
    tsnpe_colors   = [ '#FFA15A'] # '#FFA15A',

    # Traces
    def add_amortized_traces(cell, row, col, showlegend=False):
        oracle_df = cell[cell['algorithm'] == 'sbi_npe']
        hf_df     = cell[cell['algorithm'] == 'npe']
        mf_df     = cell[cell['algorithm'] == 'mf_npe']
        active_df = cell[cell['algorithm'] == 'bo_npe']

        if len(oracle_df):
            fig.add_trace(go.Scatter(
                x=oracle_df['n_hf_simulations'], y=oracle_df['mean'],
                name='NPE (SBI)', legendgroup='oracle', showlegend=showlegend,
                marker=dict(color='#5D5D5D'), line=dict(color='#5D5D5D', dash='dot'),
                error_y=dict(type='data', symmetric=False,
                             array=oracle_df['ci_max'], arrayminus=np.abs(oracle_df['ci_min']), width=0, thickness=2)),
                row=row, col=col
            )
        if len(hf_df):
            fig.add_trace(go.Scatter(
                x=hf_df['n_hf_simulations'], y=hf_df['mean'],
                name='NPE', legendgroup='npe', showlegend=showlegend,
                marker=dict(color='#000000'), line=dict(dash='dot'),
                error_y=dict(type='data', symmetric=False,
                             array=hf_df['ci_max'], arrayminus=np.abs(hf_df['ci_min']), width=0, thickness=2)),
                row=row, col=col
            )
        for i, n_lf in enumerate(lf_simulations):
            c = mf_df[mf_df['n_lf_simulations'] == n_lf]
            if len(c):
                fig.add_trace(go.Scatter(
                    x=c['n_hf_simulations'], y=c['mean'],
                    name=f"MF-NPE{int(np.log10(n_lf))}",
                    legendgroup=f"mf-npe-{n_lf}", showlegend=showlegend,
                    marker=dict(color=mf_colors[i % len(mf_colors)]),
                    error_y=dict(type='data', symmetric=False,
                                 array=c['ci_max'], arrayminus=np.abs(c['ci_min']), width=0, thickness=2)),
                    row=row, col=col
                )
        for i, n_lf in enumerate(lf_simulations):
            c = active_df[active_df['n_lf_simulations'] == n_lf]
            if len(c):
                fig.add_trace(go.Scatter(
                    x=c['n_hf_simulations'], y=c['mean'],
                    name=f"BO-MF-NPE{int(np.log10(n_lf))}",
                    legendgroup=f"bo-npe-{n_lf}", showlegend=showlegend,
                    marker=dict(color='#FFA15A'),
                    error_y=dict(type='data', symmetric=False,
                                 array=c['ci_max'], arrayminus=np.abs(c['ci_min']), width=0, thickness=2)),
                    row=row, col=col
                )

    def add_non_amortized_traces(cell, row, col, showlegend=False):
        tsnpe_df       = cell[cell['algorithm'] == 'tsnpe']
        mf_tsnpe_df     = cell[cell['algorithm'] == 'mf_tsnpe']
        active_snpe_df = cell[cell['algorithm'] == 'a_mf_tsnpe']
        mf_abc_df      = cell[cell['algorithm'] == 'mf_abc']

        if len(tsnpe_df):
            fig.add_trace(go.Scatter(
                x=tsnpe_df['n_hf_simulations'], y=tsnpe_df['mean'],
                name='TSNPE', legendgroup='tsnpe', showlegend=showlegend,
                marker=dict(color='#0F9E9A'), line=dict(dash='dot'),
                error_y=dict(type='data', symmetric=False,
                             array=tsnpe_df['ci_max'], arrayminus=np.abs(tsnpe_df['ci_min']), width=0, thickness=2)),
                row=row, col=col
            )
        for i, n_lf in enumerate(lf_simulations):
            c = mf_tsnpe_df[mf_tsnpe_df['n_lf_simulations'] == n_lf]
            if len(c):
                fig.add_trace(go.Scatter(
                    x=c['n_hf_simulations'], y=c['mean'],
                    name=f"MF-TSNPE{int(np.log10(n_lf))}",
                    legendgroup=f"mf-tsnpe-{n_lf}", showlegend=showlegend,
                    marker=dict(color=tsnpe_colors[i % len(tsnpe_colors)]),
                    error_y=dict(type='data', symmetric=False,
                                 array=c['ci_max'], arrayminus=np.abs(c['ci_min']), width=0, thickness=2)),
                    row=row, col=col
                )
        for i, n_lf in enumerate(lf_simulations):
            c = active_snpe_df[active_snpe_df['n_lf_simulations'] == n_lf]
            if len(c):
                fig.add_trace(go.Scatter(
                    x=c['n_hf_simulations'], y=c['mean'],
                    name=f"ACTIVE-MF-SNPE{int(np.log10(n_lf))}",
                    legendgroup=f"a-mf-tsnpe-{n_lf}", showlegend=showlegend,
                    marker=dict(color=a_tsnpe_colors[i % len(a_tsnpe_colors)]),
                    error_y=dict(type='data', symmetric=False,
                                 array=c['ci_max'], arrayminus=np.abs(c['ci_min']), width=0, thickness=2)),
                    row=row, col=col
                )
        if len(mf_abc_df):
            name = "MF-ABC"
            if 'n_lf_simulations' in mf_abc_df.columns and len(mf_abc_df):
                try:
                    name = f"MF-ABC{int(np.log10(int(mf_abc_df['n_lf_simulations'].iloc[0])))}"
                except Exception:
                    pass
            fig.add_trace(go.Scatter(
                x=mf_abc_df['n_hf_simulations'], y=mf_abc_df['mean'],
                name=name, legendgroup='mf-abc', showlegend=showlegend,
                mode='lines+markers', marker=dict(color='#7A7B7E'),
                line=dict(dash='dot'),
                error_y=dict(type='data', symmetric=False,
                                array=mf_abc_df['ci_max'], arrayminus=np.abs(mf_abc_df['ci_min']), width=0, thickness=2)),
                row=row, col=col
            )

    split = bool(plot_amortized_and_non_amortized_seperately)
    
    # For each metric row, every task contributes two side-by-side panels
    for r, metric in enumerate(metrics, start=1):
        for tj, task in enumerate(tasks):
            cell = df[(df['task'] == task) & (df['evaluation_metric'] == metric)]
            if not len(cell):
                continue
            



            if split:
                c_amort = 2*tj + 1
                c_non   = 2*tj + 2
                showlegend_here = (r == 1 and tj == 0)
                add_amortized_traces(cell, row=r, col=c_amort, showlegend=showlegend_here)
                add_non_amortized_traces(cell, row=r, col=c_non,   showlegend=showlegend_here)
            else:
                c = tj + 1
                showlegend_here = (r == 1 and tj == 0)
                # overlay both families into the same subplot
                add_amortized_traces(cell, row=r, col=c, showlegend=showlegend_here)
                add_non_amortized_traces(cell, row=r, col=c, showlegend=False)

    
    metric_titles = {'c2st': 'C2ST', 'nltp': 'NLTP', 'wasserstein': 'Wasserstein',
                     'mmd': 'MMD', 'nrmse': 'NRMSE'}
    
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=90))

    y_top = 1.06  # how high above the top; tweak with margin.t if needed
    if split:
        for tj, tname in enumerate(tasks):
            c1, c2 = 2*tj + 1, 2*tj + 2
            xax1, _ = fig.get_subplot(1, c1)
            xax2, _ = fig.get_subplot(1, c2)
            d1, d2 = xax1.domain, xax2.domain
            x_mid = (d1[0] + d2[1]) / 2.0
            fig.add_annotation(
                xref="paper", yref="paper", x=x_mid, y=y_top,
                text=f"<b>{tname}</b>", showarrow=False,
                xanchor="center", yanchor="bottom",
                font=dict(size=task_setup.title_size, color='#000000', family='Arial')
            )
    else:
        for tj, tname in enumerate(tasks):
            c = tj + 1
            xax, _ = fig.get_subplot(1, c)
            d = xax.domain
            x_mid = (d[0] + d[1]) / 2.0
            fig.add_annotation(
                xref="paper", yref="paper", x=x_mid, y=y_top,
                text=f"<b>{tname}</b>", showarrow=False,
                xanchor="center", yanchor="bottom",
                font=dict(size=task_setup.title_size, color='#000000', family='Arial')
            )
    
        
    # Custom spacing: small gap within a pair, big gap between pairs
    K = len(tasks)           # number of task pairs
    n_cols = 2 * K

    intra_gap = 0.005         # inside a pair [Amortized | Non-amortized]
    inter_gap = 0.03         # between pairs
    # width of each subplot domain so everything fits in [0, 1]
    w = (1.0 - K*intra_gap - (K-1)*inter_gap) / (2*K)
    if w <= 0:
        raise ValueError("Gaps too large for the number of columns; decrease inter_gap/intra_gap.")

    # build domains for each column
    col_domains = []
    x = 0.0
    for tj in range(K):
        # left col of pair
        col_domains.append([x, x + w]); x += w
        x += intra_gap
        # right col of pair
        col_domains.append([x, x + w]); x += w
        if tj < K - 1:
            x += inter_gap

    # apply same domains to every row
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_xaxes(domain=col_domains[c - 1], row=r, col=c, tickangle=-90)

    # x-axes: log scale everywhere; ticks only on bottom row    
    for r in range(1, n_rows+1):
        show_xticks = (r == n_rows)
        for c in range(1, n_cols+1):
            fig.update_xaxes(
                type="log",
                tickvals=[50, 1e2, 1e3, 1e4, 1e5],
                ticktext=["50", "10²", "10³", "10⁴", "10⁵"],
                showgrid=False, dtick=1,
                showticklabels=show_xticks,
                title_text=None,  # single shared label added via annotation below
                title_standoff=10,
                tickfont=dict(family="Arial", size=17, color='#2A3F5F'),
                row=r, col=c
            )

    # y-axes: per metric row; title on the very left column only
    for r, metric in enumerate(metrics, start=1):
        ya = dict(
            title_text=metric_titles.get(metric, metric.upper()),
            showgrid=True, gridwidth=task_setup.gridwidth, gridcolor='#DEE3EA',
            zeroline=True, linewidth=1,
            tickfont=dict(color='#2A3F5F', size=17, family="Arial"),
            title_font_size=task_setup.title_size
        )
        if metric == 'c2st':
            ya.update(dict(range=[0.5, 1.0],
                           tickmode='array',
                           tickvals=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           ticktext=['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                           showgrid=False))
            # light reference lines in every column of this row
            for c in range(1, n_cols+1):
                for yv in [0.51, 0.6, 0.7, 0.8, 0.9, 0.99]:
                    fig.add_hline(y=yv, line_color='#DEE3EA',
                                  line_width=task_setup.gridwidth,
                                  line_dash='solid', layer='below',
                                  row=r, col=c)
        elif metric == 'mmd' or metric == 'nrmse':
            def _nice_upper_bound(x):
                if not np.isfinite(x) or x <= 0:
                    return 1.0
                exp = np.floor(np.log10(x))
                m = x / (10 ** exp)
                for t in [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]:
                    if m <= t:
                        return float(t * (10 ** exp))
                return float(10 ** (exp + 1))

            def _nice_step(ymax, target_nticks=6):
                """Return a 'nice' step so ticks are readable and regular."""
                if ymax <= 0 or not np.isfinite(ymax):
                    return 0.2
                raw = ymax / max(target_nticks - 1, 2)
                exp = np.floor(np.log10(raw))
                m = raw / (10 ** exp)
                for base in [1, 2, 2.5, 5, 10]:
                    if m <= base:
                        return float(base * (10 ** exp))
                return float(10 ** (exp + 1))

            for tj, task in enumerate(tasks):
                c1, c2 = 2*tj + 1, 2*tj + 2

                cell = df[(df['task'] == task) & (df['evaluation_metric'] == 'mmd')]
                if len(cell):
                    mean = np.asarray(cell['mean'], dtype=float)
                    ci_plus = np.asarray(cell.get('ci_max', 0.0), dtype=float)
                    ci_plus = np.where(np.isfinite(ci_plus), ci_plus, 0.0)

                    y_top = np.nanmax(mean + ci_plus)
                    if not np.isfinite(y_top) or y_top <= 0:
                        y_top = float(np.nanmax(mean))
                    y_top *= 1.05  # small headroom
                    y_max = _nice_upper_bound(y_top)

                    # build nice ticks 0..y_max
                    step = _nice_step(y_max, target_nticks=6)  # ~5–6 ticks
                    n = int(np.floor(y_max / step))
                    tickvals = [i * step for i in range(n + 1)]
                    if tickvals[-1] < y_max:
                        tickvals.append(y_max)
                    ticktext = [f"{v:g}" for v in tickvals]

                    # apply same font as elsewhere via your global tickfont
                    for c in (c1, c2):
                        fig.update_yaxes(
                            row=r, col=c,
                            range=[0, y_max],
                            zeroline=True, zerolinewidth=2, zerolinecolor="#DEE3EA",
                            tickfont=dict(color='#2A3F5F', size=17, family="Arial"),
                            showgrid=True, gridwidth=task_setup.gridwidth, gridcolor="#DEE3EA",
                            tickmode='array',
                            tickvals=tickvals,
                            ticktext=ticktext,
                        )
                        # light reference lines at each tick (like C2ST)
                        for yv in tickvals:
                            if yv == 0:
                                continue  # zeroline already drawn
                            fig.add_hline(
                                y=float(yv),
                                line_color="#DEE3EA",
                                line_width=task_setup.gridwidth,
                                line_dash="solid",
                                layer="below",
                                row=r, col=c
                            )
                else:
                    # sane default if no data
                    tickvals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    for c in (c1, c2):
                        fig.update_yaxes(
                            row=r, col=c,
                            range=[0, 0.4],
                            zeroline=True, zerolinewidth=2, zerolinecolor="#DEE3EA",
                            showgrid=True, gridwidth=task_setup.gridwidth, gridcolor="#DEE3EA",
                            tickmode='array',
                            tickvals=tickvals,
                            ticktext=[f"{v:g}" for v in tickvals],
                        )
                        for yv in tickvals[1:]:
                            fig.add_hline(y=yv, line_color="#DEE3EA",
                                        line_width=task_setup.gridwidth,
                                        line_dash="solid", layer="below",
                                        row=r, col=c)

            # important: skip the generic y-axis application for MMD below
            # continue   
        elif metric == 'nrmse':
            ya['autorangeoptions_clipmin'] = 0.0

        # apply: left-most column gets title; others same without
        fig.update_yaxes(row=r, col=1, **ya)
        ya_no_title = {k: v for k, v in ya.items() if k != 'title_text'}
        for c in range(2, n_cols+1):
            fig.update_yaxes(row=r, col=c, **ya_no_title, showticklabels=False)

    # Shared bottom x-label (centered)
    fig.add_annotation(
        text="Number of high-fidelity simulations",
        xref="paper", yref="paper", x=0.5, y=-0.30, showarrow=False,
        font=dict(size=task_setup.title_size)
    )
    

    # Layout
    title_text = "" if sim_name is None else f"{sim_name}"
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        plot_bgcolor='#fff',
        paper_bgcolor='#fff',
        margin=dict(l=10, r=10, b=80, t=70),
        font_color=task_setup.axis_color,
        showlegend=show_legend,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        title_text=title_text, title_x=0.5, title_y=0.97, title_font_size=task_setup.font_size
    )
    
    
    title_anns = [a for a in fig.layout.annotations if getattr(a, "xref", None) == "x domain"]
    for i, a in enumerate(title_anns):
        if i >= n_cols:
            a.update(text="")

    # save
    path_plots = f"./../data/combined_plots" if evaluation_metric is None else f"{task_setup.main_path}/{evaluation_metric}/plots"
    os.makedirs(path_plots, exist_ok=True)
    safe_metrics = "_".join(metrics)
    fig.write_image(f"{path_plots}/{safe_metrics}_{task_setup.CURR_TIME}.svg")
    fig.write_html (f"{path_plots}/{safe_metrics}_{task_setup.CURR_TIME}.html")

    if getattr(task_setup, "show_plots", False):
        fig.show()

    print(f"image saved at {path_plots}")
    return fig


    
