import numpy as np
acts = np.load('goal_analysis_plots/env_net_activations.npy')
labels = np.load('goal_analysis_plots/env_net_labels.npy')
print('Activations shape:', acts.shape)
print('Labels shape:', labels.shape)
print('Unique labels and counts:', np.unique(labels, return_counts=True))
print('Activation min/max:', acts.min(), acts.max())
print('Number of unique activations:', np.unique(acts, axis=0).shape[0]) 