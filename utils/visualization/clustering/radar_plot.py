import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from math import pi

from utils.questionnaires.bdi import prepare_bdi

warnings.filterwarnings("ignore")

# Define file paths
# path_to_csv = "/Users/alessandrocaruso/Desktop/FINAL_RESULTS/config_1_fusion/csv/all_users_combined_clusters_lstm_early.csv"
# path_to_csv = "/Users/alessandrocaruso/PycharmProjects/master-thesis/experiments/results/csv/result_physiological_12.csv"
path_to_csv = "/Users/alessandrocaruso/PycharmProjects/master-thesis/experiments/embeddings_clustering/results_clustering/out.csv"
n = 12

def analyze_clusters(df, expanded_bdi):
    n = df['cluster_label'].nunique()  # Ensure 'cluster_label' is defined in your DataFrame
    # n = df['gmm_cluster'].nunique()  # Ensure 'cluster_label' is defined in your DataFrame
    # n = df['kmeans_cluster'].nunique()  # Ensure 'cluster_label' is defined in your DataFrame
    df_list = []
    bdi_scores = {}
    user_percentages = {}
    type_percentages = {}

    for i in range(n):
        df_cluster = df[df['cluster_label'] == i]
        # df_cluster = df[df['gmm_cluster'] == i]
        # df_cluster = df[df['kmeans_cluster'] == i]

        # Merge with BDI scores
        print(len(df_cluster))
        df_cluster = pd.merge(df_cluster, expanded_bdi, on=['user', 'day'], how='inner')
        print(len(df_cluster))

        total_samples_in_cluster = len(df_cluster)
        print(f"Number of elements in cluster {i} = {total_samples_in_cluster}")

        df_list.append(df_cluster)

        # Calculate median and mean BDI scores
        median_bdi = df_cluster['bdi_score'].median()
        mean_bdi = df_cluster['bdi_score'].mean()
        bdi_scores[f'df_{i}'] = {'median': median_bdi, 'mean': mean_bdi}

        # Count samples per user and calculate percentages
        user_counts = df_cluster['user'].value_counts()
        user_percentage = (user_counts / total_samples_in_cluster) * 100
        user_percentages[f'Cluster {i}'] = user_percentage.to_dict()

        # Calculate percentage of 'p' and 'h' types
        type_counts = df_cluster['type'].value_counts()
        type_percentage = (type_counts / total_samples_in_cluster) * 100
        type_percentages[f'Cluster {i}'] = type_percentage.to_dict()

    # Print BDI and user distribution information
    for df_name, scores in bdi_scores.items():
        print(f"Median BDI score for {df_name}: {scores['median']}")
        print(f"Mean BDI score for {df_name}: {scores['mean']}\n")

    for cluster, percentages in user_percentages.items():
        print(f"\nUser percentages in {cluster}:")
        for user, percentage in percentages.items():
            print(f"User {user}: {percentage:.2f}%")

    # Print type distribution information
    for cluster, percentages in type_percentages.items():
        print(f"\nType percentages in {cluster}:")
        for type_, percentage in percentages.items():
            print(f"Type {type_}: {percentage:.2f}%")

    return df_list, bdi_scores, type_percentages


def plot_radar_chart_multiple(df_list, bdi_scores, type_percentages, ge_2=True):
    mean_bdi_columns = [df[['bdi_' + str(i) for i in range(1, 22)]].mean() for df in df_list]

    # Convert list of Series to DataFrame
    mean_bdi_df = pd.DataFrame(mean_bdi_columns)

    # Step 3: Compute the intra-cluster variance
    intra_cluster_variance = [df[['bdi_' + str(i) for i in range(1, 22)]].var().mean() for df in df_list]
    mean_intra_cluster_variance = sum(intra_cluster_variance) / len(intra_cluster_variance)

    # Step 4: Compute the inter-cluster variance
    inter_cluster_variance = mean_bdi_df.var().mean()

    combined_metric = inter_cluster_variance / mean_intra_cluster_variance

    # Output results
    print(f"Mean Intra-Cluster Variance: {mean_intra_cluster_variance}")
    print(f"Inter-Cluster Variance: {inter_cluster_variance}")
    print(f"Combined Metric: {combined_metric}")

    labels = [f'BDI {i}' for i in range(1, 22)]
    num_vars = len(labels)
    angles = [i / float(num_vars) * 2 * pi for i in range(num_vars)]
    angles += angles[:1]

    scores = [list(mean_bdi.values) for mean_bdi in mean_bdi_columns]

    # Initialize a list to store the final counts (ratios or percentages)
    final_counts = []

    # Iterate over each DataFrame in df_list
    for df in df_list:
        # Calculate total number of elements in the current DataFrame
        total_elements = df.shape[0]

        # Calculate number of elements >= 2 in each 'bdi_' column
        count_bdi_ge_2 = df[['bdi_' + str(i) for i in range(1, 22)]].ge(2).sum()

        # Calculate ratio (or percentage) for each 'bdi_' column
        ratios = count_bdi_ge_2 / total_elements

        # Append ratios to the final_counts list
        final_counts.append(ratios)

    # display percentage of scores greater or equal two if set to true
    if ge_2:
        scores = [list(count.values) for count in final_counts]

    n = len(df_list)
    name_dict = {i: f"Cluster {i}" for i in range(n)}

    sorted_indices = sorted(range(n), key=lambda idx: bdi_scores[f'df_{idx}']['median'])
    max_score = max(max(scores[i]) for i in range(n))

    fig, axs = plt.subplots(1, n, figsize=(8*n, 10), subplot_kw=dict(polar=True))
    if n == 1:
        axs = [axs]

    # Define colors for clusters
    cmap = plt.get_cmap('tab20')
    color_map = {i: cmap(i % cmap.N) for i in range(n)}

    for plot_idx, idx in enumerate(sorted_indices):
        ax = axs[plot_idx]
        score = scores[idx] + scores[idx][:1]
        median_bdi = bdi_scores[f'df_{idx}']['median']
        mean_bdi = bdi_scores[f'df_{idx}']['mean']
        # type_percentage = type_percentages[f'Cluster {idx}']
        # type_percent_str = ', '.join([f'{type_}: {percentage:.2f}%' for type_, percentage in type_percentage.items()])

        color = color_map[idx]  # Use idx instead of plot_idx to get the correct color

        ax.plot(angles, score, linewidth=1, linestyle='solid', color=color)
        ax.fill(angles, score, alpha=0.4, color=color)

        ax.set_ylim(0, max_score)

        ax.set_title(f"{name_dict[idx]} (Median BDI: {median_bdi:.2f})", fontsize=30, y=1.1)
        # ax.set_title(f"{name_dict[idx]}", fontsize=30, y=1.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=14)

        # Set y-ticks to add concentric circles
        y_ticks = [i * (max_score / 5) for i in range(5)]  # Creates 5 equal intervals up to max_score
        ax.set_yticks(y_ticks)  # Set the concentric circles
        ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks], fontsize=14)  # Optional: add labels to the y-ticks

        # Display type percentage on the plot
        # ax.text(pi, max_score * 0.95, type_percent_str, horizontalalignment='center', size=24, color=color, weight='semibold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_sub, expanded_bdi = prepare_bdi(path_to_csv)

    type_distribution = df_sub['type'].value_counts()
    print(type_distribution)
    df_list, bdi_scores, type_percentages = analyze_clusters(df_sub, expanded_bdi)
    plot_radar_chart_multiple(df_list, bdi_scores, type_percentages)