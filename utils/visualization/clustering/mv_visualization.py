# Plot the clustering results
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def remove_outliers(data):
    # Calculate z-scores for each feature
    z_scores = np.abs((data - data.mean(axis=0)) / data.std(axis=0))
    # Define threshold for outlier detection (e.g., z-score > 3)
    threshold = 5
    # Detect outliers
    outliers = np.any(z_scores > threshold, axis=1)
    # Remove outliers
    cleaned_data = data[~outliers]
    return cleaned_data, outliers


def display_mv_plots(pre_title, data, labels):
    fig, ax = plt.subplots(1, 2, figsize=(22, 6))
    dot_size = 50

    # Assign colors based on cluster labels
    colors = ['skyblue','salmon']

    # Plot View 1 without outlier removal
    sns.scatterplot(x=data[0][:, 0], y=data[0][:, 1], hue=labels, palette=colors, ax=ax[0], s=dot_size)
    ax[0].set_title(pre_title + ' View 1')

    # Remove outliers only for View 2
    cleaned_data_1, outliers_1 = remove_outliers(data[1])

    # Plot View 2 with outlier removal
    sns.scatterplot(x=cleaned_data_1[:, 0], y=cleaned_data_1[:, 1], hue=labels[~outliers_1], palette=colors, ax=ax[1], s=dot_size)
    ax[1].set_title(pre_title + ' View 2')

    # Add legend
    ax[0].legend(title='Cluster')

    plt.show()