#%%
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

main_path = '../../data/OUprocess/2_dimensions'

path_lf_x = f'{main_path}/train_data/lf_simulations_10000.p'
path_hf_x = f'{main_path}/train_data/hf_simulations_10000.p'

# unpickle
import pickle
with open(path_lf_x, 'rb') as f:
    lf_data = pickle.load(f)
    
with open(path_hf_x, 'rb') as f:
    hf_data = pickle.load(f)
    

lf_X = lf_data['x']
hf_X = hf_data['x']

lf_theta = lf_data['theta']
hf_theta = hf_data['theta']

###### THETA AND X #######
# Concatenate x and theta along the feature axis for both fidelities
lf_features = np.concatenate([lf_X, lf_theta], axis=1)  # shape: [N_lf, D_x + D_theta]
hf_features = np.concatenate([hf_X, hf_theta], axis=1)  # shape: [N_hf, D_x + D_theta]
# Stack both fidelities along the sample axis
X = np.concatenate([lf_features, hf_features], axis=0)  # shape: [N_lf + N_hf, D_x + D_theta]

######  X #######
X = np.concatenate([lf_X, hf_X], axis=0)

tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)

n_lf = lf_X.shape[0]
X_embedded_lf = X_embedded[:n_lf]
X_embedded_hf = X_embedded[n_lf:]


plt.figure(figsize=(8, 6))
plt.scatter(X_embedded_lf[:, 0], X_embedded_lf[:, 1], alpha=0.5, label='lf data', s=10, color='#3333B8')
plt.scatter(X_embedded_hf[:, 0], X_embedded_hf[:, 1], alpha=0.5, label='hf data', s=10, color='#FF6D00')
plt.title('t-SNE visualization of lf and hf data')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Remove the frame
# Remove the frame
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Remove axis numbers and ticks
ax.set_xticks([])
ax.set_yticks([])

plt.legend()
plt.grid(False)

# if {main_path}/tsne/ does not exist, create

if not os.path.exists(f'{main_path}/tsne/'):
    os.makedirs(f'{main_path}/tsne/')
# Save the figure
plt.savefig(f'{main_path}/tsne/tsne_visualization.svg', format='svg', bbox_inches='tight')

plt.show()


# %%