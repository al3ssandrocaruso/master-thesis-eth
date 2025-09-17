import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
from models.semi_supervised.constrained_kmeans import create_constraints, select_informative_constraints, constrained_kmeans
from utils.clustering.clustering_utils import get_embeddings, save_results
from utils.questionnaires.hamilton import get_hdrs_labels

def parse_args():
    parser = argparse.ArgumentParser(description='Constrained KMeans clustering with informative constraints')
    parser.add_argument('--max_constraints', type=int, default=100, help='Maximum number of constraints to use')
    return parser.parse_args()

def main():

    args = parse_args()

    # Load Hamilton labels and embeddings
    hamilton_labels = get_hdrs_labels()
    print(hamilton_labels['hamd_label'].value_counts())

    trained_embeddings_path = "/Users/alessandrocaruso/Desktop/BEST_33.pkl"

    embeddings, metadata = get_embeddings(trained_embeddings_path)
    df_metadata = pd.DataFrame(metadata, columns=['user', 'day', 'window'])
    df_metadata.drop(columns=['window'], inplace=True)
    merged_df = pd.merge(df_metadata, hamilton_labels, on=['user', 'day'], how='left')
    bdi_scores = merged_df['hamd_label'].to_numpy()

    labeled_indices = [i for i, score in enumerate(bdi_scores) if not np.isnan(score)]
    labels = [bdi_scores[i] for i in labeled_indices]
    must_link, cannot_link = create_constraints(labels)

    # Adjust indices in constraints to match the original embeddings
    must_link = [(labeled_indices[a], labeled_indices[b]) for a, b in must_link]
    cannot_link = [(labeled_indices[a], labeled_indices[b]) for a, b in cannot_link]

    # Select most informative constraints
    must_link, cannot_link = select_informative_constraints(embeddings, must_link, cannot_link, args.max_constraints)

    # Apply constrained KMeans
    n_clusters_main = len(set(labels))
    main_cluster_labels, centroids = constrained_kmeans(embeddings, n_clusters=n_clusters_main, must_link=must_link, cannot_link=cannot_link)

    # Further divide each main cluster into subclusters
    final_labels = np.zeros_like(main_cluster_labels)
    for cluster in range(n_clusters_main):
        cluster_indices = np.where(main_cluster_labels == cluster)[0]
        subcluster_labels = KMeans(n_clusters=3).fit_predict(embeddings[cluster_indices])

        final_labels[cluster_indices[subcluster_labels == 0]] = cluster * 3
        final_labels[cluster_indices[subcluster_labels == 1]] = cluster * 3 + 1
        final_labels[cluster_indices[subcluster_labels == 2]] = cluster * 3 + 2

    # Save results
    save_results(metadata, final_labels, "results_clustering/out.csv")


if __name__ == "__main__":
    main()
