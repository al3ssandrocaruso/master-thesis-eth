import argparse
import numpy as np
import random
import pandas as pd

from utils.clustering.clustering_utils import select_clustering_algorithm, get_embeddings, save_results
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

def get_args():
    parser = argparse.ArgumentParser(description='Clustering parameters')
    parser.add_argument('--algorithm_choice', type=int, default=4, choices=[1, 2, 3], help='Choice of clustering algorithm: 1 for KMeans, 2 for Birch, 3 for GaussianMixture')
    parser.add_argument('--n_clusters', type=int, default=12, help='Number of clusters')
    parser.add_argument('--tsne_viz', type=bool, default=False, help='Set to True to Display 2D TSNE visualization')
    return parser.parse_args()

def plot_tsne(embeddings, labels, n_clusters, output_path):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    cmap = plt.get_cmap('tab20')
    color_map = {i: cmap(i % cmap.N) for i in range(n_clusters)}

    plt.figure(figsize=(10, 6))
    for cluster in range(n_clusters):
        indices = [i for i, label in enumerate(labels) if label == cluster]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=[color_map[cluster]], label=f'Cluster {cluster}', alpha=0.7)

    plt.title('t-SNE visualization of clustered embeddings')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.legend()
    plt.savefig(output_path)
    plt.show()


def save_centroids_and_labels(model, embeddings, labels, file_path):
    if hasattr(model, 'cluster_centers_'):
        centroids = model.cluster_centers_
        unique_labels = np.unique(labels)

        # Create a dictionary to map each label to its corresponding centroid
        label_to_centroid = {}
        for label in unique_labels:
            # Find all points belonging to the current label
            label_indices = np.where(labels == label)
            # Compute the centroid for this label (to ensure alignment)
            centroid = np.mean(embeddings[label_indices], axis=0)
            label_to_centroid[label] = centroid

        # Create a DataFrame with two columns: 'centroid' and 'label'
        df = pd.DataFrame({
            'centroid': [list(label_to_centroid[label]) for label in unique_labels],
            'label': unique_labels
        })

        df.to_csv(file_path, index=False)


# create clusters based on trained embeddings
def main():
    args = get_args()
    trained_embeddings_path = "/Users/alessandrocaruso/Desktop/BEST_33.pkl"
    embeddings, metadata = get_embeddings(trained_embeddings_path)
    model = select_clustering_algorithm(args.algorithm_choice, args.n_clusters)
    labels = model.fit_predict(embeddings)
    save_results(metadata, labels, f"results_clustering/out.csv")

    if args.algorithm_choice == 1:
        save_centroids_and_labels(model, embeddings, labels, "results_clustering/centroids_and_labels.csv")

    if args.tsne_viz:
        tsne_output_path = "results_clustering/tsne_visualization.png"
        plot_tsne(embeddings, labels, args.n_clusters, tsne_output_path)

if __name__ == "__main__":
    main()
