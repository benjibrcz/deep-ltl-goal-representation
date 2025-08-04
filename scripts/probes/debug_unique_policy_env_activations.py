import numpy as np

for layer in ['policy_encoder', 'env_net']:
    print(f'\nLayer: {layer}')
    acts = np.load(f'goal_analysis_plots/{layer}_activations.npy')
    labels = np.load(f'goal_analysis_plots/{layer}_labels.npy')
    unique_labels = np.unique(labels)
    for color in unique_labels:
        color_acts = acts[labels == color]
        n_unique = np.unique(color_acts, axis=0).shape[0]
        print(f'{color}: {n_unique} unique activations out of {len(color_acts)}') 