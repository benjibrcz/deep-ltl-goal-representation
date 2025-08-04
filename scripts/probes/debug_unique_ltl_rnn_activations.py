import numpy as np

acts = np.load('goal_analysis_plots/ltl_rnn_activations.npy')
labels = np.load('goal_analysis_plots/ltl_rnn_labels.npy')
unique_labels = np.unique(labels)

for color in unique_labels:
    color_acts = acts[labels == color]
    n_unique = np.unique(color_acts, axis=0).shape[0]
    print(f'{color}: {n_unique} unique activations out of {len(color_acts)}') 