

#%%
import pickle

with open("./data/OUprocess/2_dimensions/models/10**1-3_A-MF-TSNPE.pkl", "rb") as f:
    df = pickle.load(f)

# rename fidelity to algorithm
df = df.rename(columns={'fidelity': 'algorithm'})

# Add column 'task' = 'OUprocess'
df['task'] = 'OUprocess'

# rename 'round' by 'net_init'
df = df.rename(columns={'round': 'net_init'})

print("df shape:", df)

# Rename algorithms: 'active_snpe' becomes 'a_mf_tsnpe', 'mf_snpe' becomes 'mf_tsnpe'
df['algorithm'] = df['algorithm'].replace({'active_snpe': 'a_mf_tsnpe', 'mf_snpe': 'mf_tsnpe'})

# Save file
with open("./data/OUprocess/2_dimensions/models/evaluate_c2st_mf_tsnpe+a_mf_tsnpe_LF10000+100000_HF50+100+1000+10000+100000_Ninits6_seed12-17.pkl", "wb") as f:
    pickle.dump(df, f)


#%%

with open("./data/OUprocess/2_dimensions/models/evaluate_c2st_tsnpe+mf_tsnpe+a_mf_tsnpe_LF10000_HF50+100+1000+10000+100000_Ninits5_seed12-16.pkl", "rb") as f:
    df_or = pickle.load(f)

# Remove all entries with evaluation_metric = 'mmd'
df_or = df_or[df_or['evaluation_metric'] != 'mmd']

# Save pickle as new file
# with open("./data/OUprocess/2_dimensions/models/evaluate_c2st_npe+mf_npe_LF1000+10000_HF50+100+1000+10000+100000_Ninits10_seed12-21_no_mmd.pkl", "wb") as f:
#     pickle.dump(df_or, f)   

# with open("./data/OUprocess/2_dimensions/models/evaluate_mmd_npe+mf_npe_LF1000+10000_HF50+100+1000+10000+100000_Ninits10_seed12-21.pkl", "rb") as f:
#     df_or = pickle.load(f)
    
    
# print("Original df shape:", df_or)


####################################################################################
#%%

# Load pickle train_a_mf_tsnpe+mf_tsnpe+tsnpe_LF10000_HF50+100+1000+10000+100000_Ninits1_seed12_LF+HF.pkl from SLCP
import pickle


with open("./data/SLCP/5_dimensions/models/LF+HF/evaluate_mmd_tsnpe+a_mf_tsnpe+mf_tsnpe_LF10000_HF50+100+1000+10000+100000_Ninits1_seed12_LF+HF.pkl", "rb") as f:
    df = pickle.load(f)

# print dataframe
# print(df)


# remove all columns except where tsnpe = method
df = df[df['fidelity'] == 'tsnpe']

# replace 'fidelity' column by 'algorithm' column with value 'tsnpe'
df = df.rename(columns={'fidelity': 'algorithm'})

# # Print new df
print(df)

# Save pickle as new file
with open("./data/SLCP/5_dimensions/models/evaluate_c2st_tsnpe+mf_tsnpe+a_mf_tsnpe_LF10000_HF50+100+1000+10000+100000_Ninits5_seed12-16.pkl", "wb") as f:
    pickle.dump(df, f)

# %%
