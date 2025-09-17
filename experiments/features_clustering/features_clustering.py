from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import argparse
import os
from utils.clustering.clustering_utils import select_clustering_algorithm
from dataset.load_data_utils import load_data
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='Clustering parameters')
    parser.add_argument('--n_clusters', type=int, default=12, help='Number of clusters')
    parser.add_argument('--scaler', type=str, default='Standard', choices=['MinMax', 'Standard'],
                        help='Choice of scaler: MinMax for MinMaxScaler, Standard for StandardScaler')
    parser.add_argument('--pca_components', type=int, default=11, help='Number of PCA components to keep')
    parser.add_argument('--modality', type=str, default='physiological', choices=['physiological', 'social'],
                        help='Clustering modality')
    return parser.parse_args()


def run_clustering(X_pca, data, modality, n_clusters, algorithm_choice, algorithm_name, result_dir):
    # Select clustering algorithm
    model = select_clustering_algorithm(algorithm_choice, n_clusters)

    # Fit the model and predict clusters
    clusters = model.fit_predict(X_pca)
    data['cluster_label'] = clusters  # Add cluster labels to the dataset

    # Calculate and print the silhouette score
    silhouette_avg = silhouette_score(X_pca, clusters)
    print(f"The average silhouette score for {algorithm_name} with {n_clusters} clusters is:", silhouette_avg)

    # Save the results to a CSV file
    data.to_csv(os.path.join(result_dir, f"result_{modality}_{n_clusters}_{algorithm_name}.csv"))


def main():
    args = get_args()

    X, data = load_data(args.modality)  # Loading data

    # Selecting and applying the scaler based on arguments
    scaler = MinMaxScaler() if args.scaler == 'MinMax' else StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Applying PCA for dimensionality reduction
    pca = PCA(n_components=args.pca_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Define the clustering algorithms to test
    algorithms = {
        1: 'KMeans',
        2: 'Birch',
        3: 'GaussianMixture'
    }

    # Create the results directory if it doesn't exist
    result_dir = "../results/csv/"
    os.makedirs(result_dir, exist_ok=True)

    # Run clustering for both the specified number of clusters and for binary clustering
    for algorithm_choice, algorithm_name in algorithms.items():
        # Run for specified n_clusters
        run_clustering(X_pca, data, args.modality, args.n_clusters, algorithm_choice, algorithm_name, result_dir)

        # Run for binary clustering (n_clusters = 2)
        run_clustering(X_pca, data, args.modality, 2, algorithm_choice, algorithm_name, result_dir)


if __name__ == "__main__":
    main()
