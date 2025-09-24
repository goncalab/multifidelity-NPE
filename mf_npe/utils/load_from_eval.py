import os
import pandas as pd
from mf_npe.utils.calculate_error import mean_confidence_interval

def load_from_eval_file(eval_path, eval_metric):
    # Load all the files that start with evaluate_ (all methods that have been saved for that experiment!)
    file_names = [f for f in os.listdir(eval_path) if f.startswith('evaluate')]
    
    all_dfs = pd.DataFrame()
    for file in file_names:
        try:
            df = pd.read_pickle(f'{eval_path}/{file}')
            all_dfs = pd.concat([all_dfs, df], ignore_index=True)
            print(f"Loaded {file} with shape {df.shape}")
        except Exception as e:
            print(f"Failed to load {file}: {e}")


    # only show element with correct metric
    print("all_dfs", all_dfs.head(100))
    
    
    grouped_df = group_df(all_dfs, eval_metric)
    print("grouped_df", grouped_df)
    
    return grouped_df


def group_df(df_all_evaluated, eval_metric):
    
    print("df_all_evaluated", df_all_evaluated)
    
    df_all_evaluated = df_all_evaluated[df_all_evaluated['evaluation_metric'] == eval_metric]
    
    print("df_all_evaluated", df_all_evaluated)
    
    if "raw_data" in df_all_evaluated.columns:
        # Needs aggregation
        grouped_df = df_all_evaluated.groupby(
            ['evaluation_metric', 'n_lf_simulations', 'n_hf_simulations', 'algorithm']
        )['raw_data'].apply(mean_confidence_interval).reset_index()
    else:
        # Already aggregated, just return
        grouped_df = df_all_evaluated.copy()

    grouped_df = grouped_df[grouped_df['algorithm'] != 'sbi_npe']
    
    return grouped_df