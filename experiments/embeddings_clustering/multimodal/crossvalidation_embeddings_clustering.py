import numpy as np
import os
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pickle
from utils.clustering.clustering_utils import grid_search_clustering, dunn_index, calculate_custom_metric, initialize_algorithm
from collections import Counter
from utils.classification.classification_utils import prepare_data
from dataset.load_data_utils import load_prepare_bdi
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from config.config import PATH_DEMOGRAPHIC

param_grid = {
    'KMeans': {
        'n_clusters': [3,4,5,6,7,8,9,10,11,12],
        'init': ['k-means++'],
        'n_init': [10, 20, 30],
        'max_iter': [300, 500],
        'algorithm': ['lloyd', 'elkan']
    },
    'GaussianMixture': {
        'n_components': [3,4,5,6,7,8,9,10,11,12],
        'covariance_type': ['full'],
        'max_iter': [100, 200],
        'init_params': ['kmeans'],
        'tol': [1e-3, 1e-4, 1e-5]
    },
    'AgglomerativeClustering': {
        'n_clusters': [3,4,5,6,7,8,9,10,11,12],
        'linkage': ['ward', 'complete', 'average', 'single'],
        'metric': ['euclidean']
    }
}
folds_folder = '/Users/crisgallego/Desktop/SMART_deepRLearning/results_all_experiments/embeddings/multimodal/embeddings_128' #TODO: add folder where fold embeddings restuls from autoencoder are stored (10 pkl files)

def save_clustering(results,model_name,fusion,folder):
    path = '/Users/crisgallego/Desktop/SMART_deepRLearning/results_all_experiments/clustering/multimodal/' + folder
    with open(f"{path}/clusters_{model_name}_{fusion}.pkl", "wb") as f:
        pickle.dump(results, f)


expanded_bdi = load_prepare_bdi()
df_participants = pd.read_csv(PATH_DEMOGRAPHIC)

best_model_folds = []
best_params_folds = []
best_scores_folds = []

for root, dirs, files in os.walk(folds_folder):
    for dir_name in dirs:
        model_name = dir_name
        dir_path = os.path.join(root, dir_name)
        fold_idx = 0
        # List files inside this specific directory
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            try:
                with open(file_path, "rb") as f:
                    fold_data = pickle.load(f)
            except:
                break

            best_params, best_scores = grid_search_clustering(fold_data, param_grid)

            best_params_folds.append(best_params)
            best_scores_folds.append(best_scores)

        # Step 1: Count the most frequently chosen algorithm
        algorithm_counts = Counter([params['algorithm_choice'] for params in best_params_folds])
        most_common_algorithm = algorithm_counts.most_common(1)[0][0]

        print(f"Most frequently chosen algorithm: {most_common_algorithm}")
        print(f"Algorithm counts: {algorithm_counts}")

        # Step 2: Filter parameters for the most common algorithm
        most_common_algorithm_params = [
            params['params'] for params in best_params_folds if params['algorithm_choice'] == most_common_algorithm
        ]

        # Step 3: Count the most frequently chosen parameters for the most common algorithm
        parameter_counts = Counter(tuple(sorted(param.items())) for param in most_common_algorithm_params)
        most_common_params = parameter_counts.most_common(1)[0][0]

        print(f"Most frequently chosen parameters for algorithm {most_common_algorithm}: {dict(most_common_params)}")
        print(f"Parameter counts: {parameter_counts}")

        # Initialize to store test results
        results = []

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            try:
                with open(file_path, "rb") as f:
                    fold_data = pickle.load(f)
            except:
                break

            # Load data
            embeddings_train = fold_data["train_embeddings"]
            metadata_train = fold_data['train_metadata']
            embeddings_val =  fold_data["val_embeddings"]
            metadata_val = fold_data['val_metadata']
            embeddings_test = fold_data['test_embeddings']
            metadata_test = fold_data['test_metadata']

            # Combine training and validation embeddings
            combined_embeddings = np.vstack((embeddings_train, embeddings_val))
            combined_metadata = np.vstack((metadata_train,metadata_val))
            
            # Prepare training data (train split)
            X_train, y_train, _ = prepare_data(
                combined_embeddings, combined_metadata, expanded_bdi, df_participants, demo=True, mood=False
            )

            X_test, y_test, _ = prepare_data(
                embeddings_test, metadata_test, expanded_bdi, df_participants, demo=True, mood=False
            )

            # Standardize embeddings
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Refit the model with the most common algorithm and parameters
            best_algorithm = most_common_algorithm
            best_params = dict(most_common_params)  # Convert tuple back to dictionary

            model = initialize_algorithm(
                best_algorithm,
                best_params
            )
            
            # Fit the model
            model.fit(X_train)
            
            # Evaluate the model on validation data
            if best_algorithm == 'AgglomerativeClustering':
                # Use NearestNeighbors to find the closest training point for each validation sample
                nbrs = NearestNeighbors(n_neighbors=5).fit(X_train)
                distances, indices = nbrs.kneighbors(X_test)
                neighbor_labels = model.labels_[indices]  # Get labels of the 5 nearest neighbors
                test_labels = mode(neighbor_labels, axis=1).mode.flatten()  # Majority vote
            else:
                test_labels = model.predict(X_test)
        
            silhouette = silhouette_score(X_test, test_labels)
            dunn = dunn_index(X_test, test_labels)  # Negative to maximize
            custom = calculate_custom_metric(test_labels,y_test)

            # Store test scores
            results.append({
                "fold_idx": fold_idx,
                "silhouette": silhouette,
                "dunn": dunn,
                "custom": custom,
                "algorithm": best_algorithm,
                "params": dict(most_common_params)
            })

    
            print(f"Fold {fold_idx}: Silhouette = {silhouette:.2f}, Dunn = {dunn:.2f}, Custom = {custom:.2f}")
            fold_idx += 1

        save_clustering(results,model_name,'early','embeddings_128')



