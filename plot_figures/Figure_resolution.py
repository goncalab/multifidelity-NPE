#%% Paper-ready resolution comparison figure

from datetime import datetime
from types import SimpleNamespace
import os
import pickle
import string
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: your util
try:
    from mf_npe.utils.load_from_eval import load_from_eval_file  # optional fallback
except Exception:
    load_from_eval_file = None

# ============== CONFIG (edit here) ==============
path_to_data = "./resolution_files"
save_path = "./resolution_files"

FILES_AND_LABELS = [
    ("evaluate_2_fidelities_lf1_nltp_npe+mf_npe_LF1000+10000_HF100_Ninits10_seed12-21.pkl", "2 fidelities, LF1"),
    ("evaluate_2_fidelities_lf2_nltp_npe+mf_npe_LF1000+10000_HF100_Ninits10_seed12-21.pkl", "2 fidelities, LF2"),
    ("evaluate_3_fidelities_nltp_npe+mf_npe_LF1000+10000_HF100_Ninits10_seed12-21.pkl",      "3 fidelities"),
]

PLOT_MODE = "faceted"    # "faceted" (side-by-side columns) or "grouped" (single axis with hue)
ADD_JITTER = False        # jitter off for a cleaner paper figure
SHOW_MEAN = False          # overlay mean marker (diamond) on each box
X_LABEL = "Method"
Y_LABEL = "NLTP (↓)"      # set your metric name; example
TITLE = None              # Most journals discourage figure titles; use panel caption instead
OUT_BASENAME = "Figure_Resolution_Study_Boxplot"

# Figure size: pick one
SINGLE_COL_MM = 85        # ~single-column width (e.g., ~ 85–90 mm)
DOUBLE_COL_MM = 178       # ~double-column width
FIG_WIDTH_MM = SINGLE_COL_MM * 3   # 3 columns side-by-side; adjust as you like
FIG_HEIGHT_MM = 55                  # keep it compact; increase if labels collide

# Typography & style
USE_LATEX = False         # set True if your LaTeX build is available; otherwise keep False
BASE_FONTSIZE_PT = 8.5    # 8–9 pt works for most venues
AX_LINEWIDTH = 0.8
SPINE_LINEWIDTH = 0.8
GRID_LINEWIDTH = 0.6
BOX_LINEWIDTH = 0.8
Fliersize = 2
Whiskerwidth = 1.2
Whiskers = (5, 95)        # use percentiles for robustness; change to 1.5*IQR if you prefer

# Plot-only relabel for the x ticks:
tick_map = {0: "NPE", 1000: "MF-NPE3", 10000: "MF-NPE4"}

plot_setup = SimpleNamespace(
    main_path=path_to_data,
    show_plots=True,
    CURR_TIME=datetime.now().strftime("%Y-%m-%d_%Hh%M"),
)

# ============== UTILS ==============
NEEDED_COLS = {"n_lf_simulations", "mean"}

def mm_to_inches(mm: float) -> float:
    return mm / 25.4

def set_mpl_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 600,                    # high-res PNG if needed
        "font.size": BASE_FONTSIZE_PT,
        "axes.labelsize": BASE_FONTSIZE_PT,
        "axes.titlesize": BASE_FONTSIZE_PT,
        "xtick.labelsize": BASE_FONTSIZE_PT,
        "ytick.labelsize": BASE_FONTSIZE_PT,
        "legend.fontsize": BASE_FONTSIZE_PT,
        "axes.linewidth": AX_LINEWIDTH,
        "grid.linewidth": GRID_LINEWIDTH,
        "lines.linewidth": 1.0,
    })
    if USE_LATEX:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
        })
    else:
        # Use a serif that prints nicely even without LaTeX
        mpl.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        })

def _extract_df(obj) -> pd.DataFrame | None:
    if isinstance(obj, pd.DataFrame) and NEEDED_COLS.issubset(obj.columns):
        return obj.copy()
    if isinstance(obj, dict):
        for v in obj.values():
            df = _extract_df(v)
            if df is not None: return df
    for attr in ("data", "results", "df", "grouped_df"):
        if hasattr(obj, attr):
            df = _extract_df(getattr(obj, attr))
            if df is not None: return df
    if isinstance(obj, (list, tuple)):
        for v in obj:
            df = _extract_df(v)
            if df is not None: return df
    return None

def load_eval_df(pkl_path: str) -> pd.DataFrame:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    df = _extract_df(data)

    if df is None and load_from_eval_file is not None:
        try:
            folder = os.path.dirname(pkl_path)
            df = load_from_eval_file(folder, "nltp")
            if not isinstance(df, pd.DataFrame) or not NEEDED_COLS.issubset(df.columns):
                df = None
        except Exception:
            df = None

    if df is None:
        raise ValueError(
            f"Could not locate a DataFrame with {NEEDED_COLS} in {os.path.basename(pkl_path)}."
        )

    df["n_lf_simulations"] = pd.to_numeric(df["n_lf_simulations"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df = df.dropna(subset=["n_lf_simulations", "mean"])
    return df

def gather_all_data():
    dfs = []
    for filename, label in FILES_AND_LABELS:
        pth = os.path.join(path_to_data, filename)
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"Missing file: {pth}")
        df = load_eval_df(pth)
        df = df.copy()
        df["Condition"] = label
        dfs.append(df[["n_lf_simulations", "mean", "Condition"]])
    all_df = pd.concat(dfs, ignore_index=True)
    x_order = sorted(all_df["n_lf_simulations"].unique())
    all_df["n_lf_simulations"] = pd.Categorical(all_df["n_lf_simulations"], categories=x_order, ordered=True)
    return all_df, x_order

def add_panel_labels(axes, xpos=0.02, ypos=0.98):
    # Label subplots A, B, C...
    for ax, letter in zip(axes, string.ascii_uppercase):
        ax.text(xpos, ypos, letter, transform=ax.transAxes,
                va="top", ha="left", fontweight="bold")

# ============== PLOTTING ==============
def plot_faceted_box(all_df: pd.DataFrame, x_order):
    # figure size in inches
    fig_w = mm_to_inches(FIG_WIDTH_MM)
    fig_h = mm_to_inches(FIG_HEIGHT_MM)
    
    palette = {
        "0": "tab:red",      # NPE
        "1000": "tab:blue",  # MF-NPE3
        "10000": "tab:blue"  # MF-NPE4
    }

    # Create the plot
    g = sns.catplot(
        data=all_df, kind="box",
        x="n_lf_simulations", y="mean",
        col="Condition", col_order=[lab for _, lab in FILES_AND_LABELS],
        order=x_order, sharey=True, sharex=True,
        height=fig_h, aspect=fig_w / (fig_h * len(FILES_AND_LABELS)),
        fliersize=Fliersize, linewidth=BOX_LINEWIDTH,
        whis=Whiskers, palette=palette
    )

    # Ticks relabel for plot only
    for ax in g.axes.flat:
        ax.set_xticklabels([tick_map.get(v, str(v)) for v in x_order])

    # Optional jitter (light)
    if ADD_JITTER:
        for ax, label in zip(g.axes.flat, [lab for _, lab in FILES_AND_LABELS]):
            sub = all_df[all_df["Condition"] == label]
            sns.stripplot(
                data=sub, x="n_lf_simulations", y="mean",
                order=x_order, dodge=False, alpha=0.35, size=2, ax=ax
            )

    # Optional mean overlay
    if SHOW_MEAN:
        for ax, label in zip(g.axes.flat, [lab for _, lab in FILES_AND_LABELS]):
            sub = all_df[all_df["Condition"] == label]
            means = sub.groupby("n_lf_simulations")["mean"].mean().reindex(x_order)
            ax.plot(range(len(x_order)), means.values, marker="D", linestyle="None", markersize=3)

    # Labels & grid
    g.set_axis_labels(X_LABEL, Y_LABEL)
    g.set_titles("{col_name}" if TITLE is None else TITLE)
    for ax in g.axes.flat:
        ax.grid(True, axis="y", linewidth=GRID_LINEWIDTH, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LINEWIDTH)

    # Panel letters
    add_panel_labels(g.axes.flat)

    # Tight spacing
    g.fig.subplots_adjust(top=0.98, left=0.08, right=0.995, bottom=0.18, wspace=0.25)
    return g.fig

def plot_grouped_box(all_df: pd.DataFrame, x_order):
    fig_w = mm_to_inches(FIG_WIDTH_MM)
    fig_h = mm_to_inches(FIG_HEIGHT_MM)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    palette = {
        "0": "tab:red",      # NPE
        "1000": "tab:blue",  # MF-NPE3
        "10000": "tab:blue"  # MF-NPE4
    }

    sns.boxplot(
        data=all_df, x="n_lf_simulations", y="mean",
        order=x_order, hue="Condition",
        fliersize=Fliersize, linewidth=BOX_LINEWIDTH, whis=Whiskers, ax=ax,
        palette=palette
    )
    ax.set_xticklabels([tick_map.get(v, str(v)) for v in x_order])

    if ADD_JITTER:
        sns.stripplot(
            data=all_df, x="n_lf_simulations", y="mean",
            order=x_order, hue="Condition", dodge=True, alpha=0.35, size=2, ax=ax
        )
        # remove duplicate legends from strip overlay
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(FILES_AND_LABELS)], labels[:len(FILES_AND_LABELS)], frameon=False)

    if SHOW_MEAN:
        # mean per (x,hue)
        means = all_df.groupby(["n_lf_simulations", "Condition"])["mean"].mean().unstack("Condition").reindex(index=x_order)
        # plot as diamond markers at x positions with small offsets per hue
        conds = [lab for _, lab in FILES_AND_LABELS]
        offsets = np.linspace(-0.2, 0.2, len(conds))
        for off, cond in zip(offsets, conds):
            ax.plot(np.arange(len(x_order)) + off, means[cond].values, marker="D", linestyle="None", markersize=3)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    if TITLE:
        ax.set_title(TITLE)
    ax.grid(True, axis="y", linewidth=GRID_LINEWIDTH, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)
    fig.tight_layout()
    return fig

def main():
    os.makedirs(save_path, exist_ok=True)
    set_mpl_paper_style()
    all_df, x_order = gather_all_data()

    # lock y-range with a little padding for consistency across exports
    y_min, y_max = all_df["mean"].min(), all_df["mean"].max()
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_lims = (y_min - y_pad, y_max + y_pad)

    if PLOT_MODE == "grouped":
        fig = plot_grouped_box(all_df, x_order)
        suffix = "grouped"
        # Apply y limits after fig creation
        ax = fig.axes[0]
        ax.set_ylim(*y_lims)
    else:
        fig = plot_faceted_box(all_df, x_order)
        suffix = "faceted"
        for ax in fig.axes:
            ax.set_ylim(*y_lims)

    pdf_path = os.path.join(save_path, f"{OUT_BASENAME}_{suffix}.pdf")
    png_path = os.path.join(save_path, f"{OUT_BASENAME}_{suffix}.png")
    fig.savefig(pdf_path, bbox_inches="tight")      # vector for your paper
    fig.savefig(png_path, bbox_inches="tight")      # high-res raster (600 dpi from rcParams)

    if plot_setup.show_plots:
        plt.show()
    plt.close(fig)
    print(f"Saved:\n- {pdf_path}\n- {png_path}")

if __name__ == "__main__":
    main()


