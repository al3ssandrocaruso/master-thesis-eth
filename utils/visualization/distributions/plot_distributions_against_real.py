import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from config.config import hrv_feat_dict, selected_features_pearson

save_dest = "/Users/alessandrocaruso/Desktop/RESULTS_BINARY_KMEANS"
n_bins = 100

results_depressed = {}
results_healthy = {}
results_c0 = {}
results_c1 = {}

def plot_distribution(df, feature_name,display, display_mean_std = True):
    # Filter data
    depressed_users = df[df['type'] == 'p'][feature_name]
    healthy_users = df[df['type'] == 'h'][feature_name]

    # Calculate statistics
    mean_depressed = depressed_users.mean()
    mean_healthy = healthy_users.mean()
    std_depressed = depressed_users.std()
    std_healthy = healthy_users.std()

    results_depressed[feature_name] = (mean_depressed,std_depressed)
    results_healthy[feature_name] = (mean_healthy,std_healthy)

    # Calculate 98th percentile
    percentile_98 = np.nanpercentile(df[feature_name], 95)
    max_val = percentile_98 if percentile_98 > 0 else 1

    # Plot histogram using Matplotlib
    plt.figure(figsize=(20, 10))
    if display_mean_std:
        plt.hist(depressed_users, bins=n_bins, color='salmon', alpha=0.7, label=f'Depressed (Mean: {mean_depressed:.2f}, Std: {std_depressed:.2f})', density=True, range=(0, max_val))
        plt.hist(healthy_users, bins=n_bins, color='skyblue', alpha=0.7, label=f'Healthy (Mean: {mean_healthy:.2f}, Std: {std_healthy:.2f})', density=True, range=(0, max_val))
    else:
        plt.hist(depressed_users, bins=n_bins, color='skyblue', alpha=0.7,
                 label=f'Depressed ', density=True,
                 range=(0, max_val))
        plt.hist(healthy_users, bins=n_bins, color='salmon', alpha=0.7,
                 label=f'Healthy ', density=True, range=(0, max_val))

    # Set x-axis limits

    plt.xlabel(feature_name, fontsize=30)
    plt.ylabel('Density', fontsize=30)
    feature_display_name = hrv_feat_dict.get(feature_name, feature_name) if feature_name.startswith('hrv_') else feature_name
    plt.title(f'Distribution of {feature_display_name} Among Depressed and Healthy Users', fontsize=35, pad=20)
    plt.legend(fontsize=30)
    plt.grid(True)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.tight_layout()


    if display:
        plt.show()
    else:
        os.makedirs(save_dest, exist_ok=True)
        save_name = os.path.join(save_dest, f"{feature_display_name}_true_distribution.png")
        plt.savefig(save_name)
        plt.close()

def plot_distribution_cluster(df, feature_name, display,display_mean_std = True):
    # Filter data
    cluster_0 = df[df['cluster_label'] == 0][feature_name]
    cluster_1 = df[df['cluster_label'] == 1][feature_name]



    # Calculate statistics
    mean_0 = cluster_0.mean()
    mean_1 = cluster_1.mean()
    std_0 = cluster_0.std()
    std_1 = cluster_1.std()

    results_c0[feature_name] = (mean_0,std_0)
    results_c1[feature_name] = (mean_1,std_1)

    # Calculate 98th percentile
    percentile_98 = np.nanpercentile(df[feature_name], 95)
    max_val = percentile_98 if percentile_98 > 0 else 1

    color_map = {0: 'salmon', 1: 'skyblue'}

    # Plot histogram using Matplotlib
    plt.figure(figsize=(20, 10))
    if display_mean_std:
        plt.hist(cluster_0, bins=n_bins, color='salmon', alpha=0.7, label=f'Cluster 0 (Mean: {mean_0:.2f}, Std: {std_0:.2f})', density=True,range=(0, max_val))
        plt.hist(cluster_1, bins=n_bins, color='skyblue', alpha=0.7, label=f'Cluster 1 (Mean: {mean_1:.2f}, Std: {std_1:.2f})', density=True, range=(0, max_val))

    else:
        plt.hist(cluster_0, bins=n_bins, color='salmon', alpha=0.7,
                 label=f'Cluster 0', density=True, range=(0, max_val))
        plt.hist(cluster_1, bins=n_bins, color='skyblue', alpha=0.7,
                 label=f'Cluster 1', density=True, range=(0, max_val))

    plt.xlabel(feature_name, fontsize=30)
    plt.ylabel('Density', fontsize=30)
    feature_display_name = hrv_feat_dict.get(feature_name, feature_name) if feature_name.startswith('hrv_') else feature_name
    plt.title(f'Distribution of {feature_display_name} Among Different Clusters', fontsize=35, pad=20)
    plt.legend(fontsize=30)
    plt.grid(True)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.tight_layout()
    if display:
        plt.show()
    else:
        os.makedirs(save_dest, exist_ok=True)
        save_name = os.path.join(save_dest, f"{feature_display_name}_predicted_distribution.png")
        plt.savefig(save_name)
        plt.close()

df_participants = pd.read_csv("/Users/alessandrocaruso/Desktop/Master Thesis Files/participants.csv")
df_participants.rename(columns={'key_smart_id': 'user'}, inplace=True)

df_results = pd.read_csv("/Users/alessandrocaruso/Documents/RESULTS/Binary Clustering/result_physical_binary_kmeans.csv")
df = pd.merge(df_participants, df_results, on='user')

feature_names = df_results.columns.tolist()[4:-1]
display = False

# Example usage:
for feature_name in selected_features_pearson:
    plot_distribution(df, feature_name, display)
    plot_distribution_cluster(df_results, feature_name, display)

# Initialize lists to store the normalized errors
mean_errors = []
std_errors = []

# Calculate normalized errors
for key in results_c1:
    pred_mean, pred_std = results_c1[key]
    true_mean, true_std = results_healthy[key]

    mean_error = abs(pred_mean - true_mean) / true_mean
    std_error = abs(pred_std - true_std) / true_std

    mean_errors.append((key, mean_error))
    std_errors.append((key, std_error))

# Print the results
print("Normalized Mean Errors:")
for key, error in mean_errors:
    print(f"{key}: {error}")

print("\nNormalized Standard Deviation Errors:")
for key, error in std_errors:
    print(f"{key}: {error}")