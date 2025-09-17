import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
from dataset.load_data_utils import load_prepare_bdi
from utils.classification.classification_utils import prepare_data
from config.config import PATH_DEMOGRAPHIC
from kneed import KneeLocator

global_random_state = 42

# Dictionary to map algorithm names to their corresponding classes
clustering_algorithms = {
    'KMeans': KMeans,
    'Birch': Birch,
    'GaussianMixture': GaussianMixture,
    'AgglomerativeClustering': AgglomerativeClustering
}

# Add random_state if the algorithm supports it
def initialize_algorithm(algorithm_name, params):
    if 'random_state' in clustering_algorithms[algorithm_name]().get_params():
        params['random_state'] = global_random_state
    return clustering_algorithms[algorithm_name](**params)

expanded_bdi = load_prepare_bdi()
df_participants = pd.read_csv(PATH_DEMOGRAPHIC)

def get_embeddings(file_path):
    pickle = pd.read_pickle(file_path)
    metadata = pickle["metadata"]
    embeddings = pickle["embeddings"]
    return embeddings, metadata

def find_optimal_clusters(X, algorithm, model):
    """Determine the optimal number of clusters using elbow-like methods."""
    metric_values = []
    cluster_range = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    for n_clusters in cluster_range:
        
        if algorithm == "KMeans":
            model.n_clusters = n_clusters
            model.fit(X)
            metric_values.append(model.inertia_)  # Use inertia for KMeans
            
        elif algorithm == "GaussianMixture":
            model.n_components = n_clusters
            model.fit(X)
            metric_values.append(model.bic(X))  # Use BIC for Gaussian Mixture Models
            
        elif algorithm == "AgglomerativeClustering":
            model.n_clusters = n_clusters
            labels = model.fit_predict(X)
            silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
            metric_values.append(-silhouette)  # Use negative silhouette to find the "elbow"
    
    # Use KneeLocator or similar to find the "elbow"
    curve_direction = "decreasing" if algorithm in ["KMeans", "GaussianMixture"] else "increasing"
    knee_locator = KneeLocator(cluster_range, metric_values, curve="convex", direction=curve_direction)
    optimal_clusters = knee_locator.knee

    return optimal_clusters, model


# Function to perform grid search over multiple clustering algorithms
def grid_search_clustering(fold_data, param_grid):

    embeddings_train = fold_data["train_embeddings"]
    metadata_train = fold_data['train_metadata']
    embeddings_val =  fold_data["val_embeddings"]
    metadata_val = fold_data['val_metadata']

    # Prepare training data (train split)
    X_train, y_train, _ = prepare_data(
        embeddings_train, metadata_train, expanded_bdi, df_participants, demo=True, mood=False
    )
    
    # Prepare validation data (val split)
    X_val, y_val, _ = prepare_data(
        embeddings_val, metadata_val, expanded_bdi, df_participants, demo=True, mood=False
    )

    scaler = StandardScaler()

    # Standardize the embeddings
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    best_score = -np.inf

    # Iterate over clustering algorithms
    for algorithm_name, param_dict in param_grid.items():

        # Generate parameter combinations using ParameterGrid
        param_combinations = ParameterGrid(param_dict)
        
        # Iterate over each parameter combination
        for params in param_combinations:
            print(f"Testing {params}...")
            
            # Initialize the clustering algorithm with the current parameters
            model = initialize_algorithm(algorithm_name, params)
            #optimal_clusters, model = find_optimal_clusters(X_train, algorithm_name, model)
            model.fit(X_train)
            #params['n_clusters'] = optimal_clusters
            #print(f"Optimal clusters for {algorithm_name}: {optimal_clusters} based on elbow-like method.")        

            # Evaluate the model on validation data
            if algorithm_name == 'AgglomerativeClustering':
                # Use NearestNeighbors to find the closest training point for each validation sample
                nbrs = NearestNeighbors(n_neighbors=5).fit(X_train)
                distances, indices = nbrs.kneighbors(X_val)
                neighbor_labels = model.labels_[indices]  # Get labels of the 5 nearest neighbors
                val_labels = mode(neighbor_labels, axis=1).mode.flatten()  # Majority vote
            else:
                val_labels = model.predict(X_val)

            # Evaluate clustering performance on validation data
            try:
                dunn = dunn_index(X_val, val_labels)  # Negative to maximize
                custom = calculate_custom_metric(val_labels,y_val)
            except:
                break
            # Combine scores (you can add your custom metric if needed)
            #score = compute_combined_metric(dunn,custom)  # Replace with a combination if needed
            score = custom
            if score > best_score:
                best_score = score
                best_scores = {"dunn": dunn, "custom":custom}
                best_params = {"algorithm_choice": algorithm_name, "params": params}
    
    return best_params, best_scores


def dunn_index(data, labels):
    """
    Calculate the Dunn Index for a given clustering.
    
    Parameters:
    - data: numpy array of shape (n_samples, n_features) representing the dataset.
    - labels: numpy array of shape (n_samples,) representing cluster labels for each data point.
    
    Returns:
    - dunn: float, the Dunn Index value.
    """
    # Get unique clusters
    unique_labels = np.unique(labels)
    
    # Initialize intra-cluster distances (compactness) and inter-cluster distances (separation)
    intra_cluster_distances = []
    inter_cluster_distances = []

    # Compute intra-cluster distances
    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:  # Avoid zero-distance for single-point clusters
            intra_cluster_distances.append(np.max(cdist(cluster_points, cluster_points)))
        else:
            intra_cluster_distances.append(0)  # Single point cluster has no spread
    
    # Compute inter-cluster distances
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:  # Avoid redundant calculations
                cluster_i_points = data[labels == label_i]
                cluster_j_points = data[labels == label_j]
                inter_cluster_distances.append(np.min(cdist(cluster_i_points, cluster_j_points)))

    # Dunn Index = Min(inter-cluster distance) / Max(intra-cluster distance)
    min_inter_cluster_distance = np.min(inter_cluster_distances)
    max_intra_cluster_distance = np.max(intra_cluster_distances)

    dunn = min_inter_cluster_distance / max_intra_cluster_distance if max_intra_cluster_distance > 0 else 0
    return dunn

def calculate_bdi_cluster_list(labels, metadata):

    metadata['cluster_label'] = pd.DataFrame(labels)
    df = metadata.copy()
    n = df['cluster_label'].nunique()
    df_list = []

    for i in range(n):
        df_cluster = df[df['cluster_label'] == i]
        df_list.append(df_cluster)
    
    return df_list


def calculate_custom_metric(labels, metadata, num_bdi_columns=21):
    """
    Calculate a custom clustering metric based on BDI responses.

    Parameters:
    - df_list: list of pandas DataFrames, where each DataFrame corresponds to a cluster.
    - num_bdi_columns: int, number of BDI columns (default is 21 for BDI-21).

    Returns:
    - combined_metric: float, the ratio of inter-cluster variance to mean intra-cluster variance.
    """
    df_list = calculate_bdi_cluster_list(labels, metadata)

    # Step 1: Compute the mean of BDI columns for each cluster
    mean_bdi_columns = [df[['bdi_' + str(i) for i in range(1, num_bdi_columns + 1)]].mean() for df in df_list]
    
    # Convert list of Series to DataFrame
    mean_bdi_df = pd.DataFrame(mean_bdi_columns)
    
    # Step 2: Compute the intra-cluster variance for each cluster
    intra_cluster_variance = [
        df[['bdi_' + str(i) for i in range(1, num_bdi_columns + 1)]].var().mean() 
        for df in df_list
    ]
    mean_intra_cluster_variance = sum(intra_cluster_variance) / len(intra_cluster_variance)
    
    # Step 3: Compute the inter-cluster variance
    inter_cluster_variance = mean_bdi_df.var().mean()
    
    # Step 4: Calculate the custom metric
    combined_metric = inter_cluster_variance / mean_intra_cluster_variance if mean_intra_cluster_variance > 0 else 0
    
    return combined_metric

def compute_combined_metric(dunn, custom_metric, 
                            weights=[1,1]):
    """
    Compute a combined metric by summing individual scores.
    
    Parameters:
    - silhouette: float, Silhouette score for clustering.
    - davies_bouldin: float, Davies-Bouldin score for clustering.
    - custom_metric: float, A custom metric (e.g., Dunn index or another measure).
    - weights: tuple (optional), weights for the individual metrics.
    
    Returns:
    - combined_metric: float, the combined score.
    """
    
    # If weights are provided, use them
    if weights:
        alpha, beta = weights
    else:
        # Default equal weights if none are specified
        alpha, beta = 1, 1, 1
    
    # Compute combined metric by summing the weighted scores
    combined_metric = (alpha * dunn + beta * custom_metric)
    
    return combined_metric

def save_results(metadata, labels, output_path):
    out = pd.DataFrame(metadata, columns=['user', 'day', 'window'])
    out['cluster_label'] = labels
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)