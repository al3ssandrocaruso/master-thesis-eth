from datetime import timedelta
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from config.config import excluded_user_ids

bdi = pd.read_csv("/Users/alessandrocaruso/Desktop/Master Thesis Files/SMART_bdi_allparticipants.csv")
bdi = bdi[~bdi['study_id'].isin(excluded_user_ids)]

# Prepare BDI data
bdi['bdi_time'] = pd.to_datetime(bdi['bdi_time'])
bdi['day'] = bdi['bdi_time'].dt.date
bdi = bdi.drop(columns=['bdi_time'])
bdi.rename(columns={'study_id': 'user'}, inplace=True)

expanded_bdi = pd.DataFrame()
for _, row in bdi.iterrows():
    for i in range(7):
        expanded_row = {
            'day': row['day'] - timedelta(days=i),
            'bdi_score': row['bdi_score'],
            'user': row['user']
        }
        for j in range(1, 22):
            expanded_row[f'bdi_{j}'] = row[f'bdi_{j}']
        expanded_bdi = expanded_bdi.append(expanded_row, ignore_index=True)

expanded_bdi['day'] = expanded_bdi['day'].astype(str)
bdi['day'] = bdi['day'].astype(str)

import pandas as pd
import numpy as np

# Let's simulate random cluster assignments and compute the metric
def compute_metric(expanded_bdi, n_clusters, iterations=100):
    bdi_columns = ['bdi_' + str(i) for i in range(1, 22)]
    results = []

    for _ in range(iterations):
        # Step 1: Randomly assign clusters
        expanded_bdi['random_cluster'] = np.random.randint(0, n_clusters, expanded_bdi.shape[0])

        # Step 2: Split the dataframe based on random clusters
        df_list = [expanded_bdi[expanded_bdi['random_cluster'] == i] for i in range(n_clusters)]

        # Step 3: Compute the intra-cluster variance
        intra_cluster_variance = [df[bdi_columns].var().mean() for df in df_list if not df.empty]
        mean_intra_cluster_variance = np.mean(intra_cluster_variance)

        # Step 4: Compute the inter-cluster variance
        mean_bdi_df = expanded_bdi.groupby('random_cluster')[bdi_columns].mean()
        inter_cluster_variance = mean_bdi_df.var().mean()

        # Step 5: Compute combined metric
        combined_metric = inter_cluster_variance / mean_intra_cluster_variance
        results.append(combined_metric)

    # Return the average score over all iterations
    return np.mean(results), results


# Example of running the simulation for 5 random clusters with 100 iterations
random_baseline_mean, all_iterations = compute_metric(expanded_bdi, n_clusters=12, iterations=100)

print(f"Average Combined Metric (Random Baseline): {random_baseline_mean}")
