
#%%import os
import os
import pandas as pd
import re
import torch
from types import SimpleNamespace
from datetime import datetime

from mf_npe.utils.calculate_error import mean_confidence_interval
from mf_npe.plot.method_performance import plot_MI_performance_paper

# Paths
path_c2st = "./../data/OUprocess/noise_tl/c2st_table"
path_mi = "./../data/OUprocess/noise_tl/mi_table"

# --- Helper Functions ---

def extract_noise(filename):
    match = re.search(r"noise=([0-9]+(?:\.[0-9]+)?)", filename)
    return float(match.group(1)) if match else None

def clean_noise(val):
    if isinstance(val, list) and isinstance(val[0], torch.Tensor):
        return round(val[0].item(), 2)
    elif isinstance(val, torch.Tensor):
        return round(val.item(), 2)
    else:
        return round(float(val), 2)

def load_pickled_data(folder, filename_filter, add_noise=False):
    """Load and optionally process noise value from pickled files in a folder."""
    dataframes = []
    for file in os.listdir(folder):
        if filename_filter(file):
            file_path = os.path.join(folder, file)
            data = pd.read_pickle(file_path)
            if add_noise:
                noise_value = extract_noise(file)
                data['noise'] = noise_value
            dataframes.append(data)
            
            print("dataframe", data)
    return dataframes

# --- Load and Process Data ---

# Load C2ST data
all_data = load_pickled_data(path_c2st, lambda f: f.startswith("c2st") and f.endswith("x.pkl"), add_noise=True)

all_data_inv = load_pickled_data(path_c2st, lambda f: f.startswith("c2st") and f.endswith("x_inv.pkl"), add_noise=True)


# Load MI data
all_MI = load_pickled_data(path_mi, lambda f: f.startswith("MI_dataframe_noise=") and f.endswith("_OUprocess.pkl"), add_noise=False)

# Load MI_inv data
# (You can easily add df_MI_inv now with the same function)
all_MI_inv = load_pickled_data(path_mi, lambda f: f.startswith("MI_dataframe_noise=") and f.endswith("_OUprocess_x_inv.pkl"), add_noise=False)

# --- Clean and Prepare ---

# Combine and process
df_data = pd.concat(all_data, ignore_index=True)
df_data = df_data.groupby(['noise'])['raw_data'].apply(mean_confidence_interval).reset_index()

df_data_inv = pd.concat(all_data_inv, ignore_index=True)
df_data_inv = df_data_inv.groupby(['noise'])['raw_data'].apply(mean_confidence_interval).reset_index()


df_MI = pd.concat(all_MI, ignore_index=True)
df_MI['type_lf'] = 'OUprocess'

df_MI_inv = pd.concat(all_MI, ignore_index=True)
df_MI_inv['type_lf'] = 'OUprocess_x_inv'

# print("df_data", df_MI.head())

print("df_data", df_data)
print("df_data_inv", df_data_inv)
print("df_data inv", df_MI_inv)




# Clean 'noise' columns
for df in [df_data, df_data_inv, df_MI, df_MI_inv]: # 
    df['noise'] = df['noise'].apply(clean_noise)

print("df_MI", df_MI)

# --- Merge DataFrames ---

merged_df_MI = df_data.merge(df_MI, on='noise', suffixes=('_data', '_MI'))

# merged_df_MI_inv
merged_df_MI_inv = df_data_inv.merge(df_MI_inv, on='noise', suffixes=('_data', '_MIinv'))

# concatenate the two dataframes
merged_df = pd.concat([merged_df_MI, merged_df_MI_inv], ignore_index=True)


# --- Plot Setup ---

plot_setup = SimpleNamespace(
    main_path=f"{path_c2st}",
    show_plots=True,
    CURR_TIME=datetime.now().strftime("%Y-%m-%d %Hh%M"),
    width_plots=400,
    height_plots=400,
    axis_color='#6A798F',
    font_size=20,
    title_size=20,
    gridwidth=2,
    show_legend=True,
)

# --- Plot ---
type_lf = ['OUprocess', 'OUprocess_x_inv']

axes = ['mi', 'c2st'] # 'c2st', 'mi', 'noise'
#axes = ['noise', 'mi']
# axes = ['noise', 'c2st']

if axes == ['mi', 'c2st']:
    npe_val = 0.785 # see main figure paper
elif axes == ['noise', 'mi']:
    npe_val = 0
elif axes == ['noise', 'c2st']:
     npe_val = 0.785 
else:
    npe_val = 0


plot_MI_performance_paper(merged_df, "OU2", type_lf, 'c2st', plot_setup, axes, npe_val)


# %%
