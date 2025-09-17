import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

from config.config import selected_features_pearson

warnings.filterwarnings("ignore")

# Define file paths
data_path = "/Users/alessandrocaruso/PycharmProjects/master-thesis/experiments/results/csv/result_physiological_12.csv"


def load_and_prepare_data(data_path, selected_features):
    # Load the data
    data = pd.read_csv(data_path)
    data = data.dropna()

    # Select features and labels
    X = data.iloc[:, 4:-1]
    X = X[selected_features]
    clusters = data['cluster_label'].values

    # Determine the number of clusters
    n_clusters = data['cluster_label'].nunique()

    return X, clusters, n_clusters

def standardize_features(X):
    # Standardize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def apply_pca(X, n_components=3):
    # Apply PCA on filtered data for 3 components
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca

def compute_centroids(X_pca, clusters):
    # Calculate centroids
    centroids = pd.DataFrame(X_pca).groupby(clusters).mean().values
    return centroids

def plot_pca_results(X_pca, clusters, centroids, n_clusters):
    n_clusters += 1
    if n_clusters == 2:
        color_map = {0: 'salmon', 1: 'skyblue'}
    else:
        # Define colors for clusters
        cmap = plt.get_cmap('tab20')
        color_map = {i: cmap(i % cmap.N) for i in range(n_clusters)}

    # Create a DataFrame for Seaborn
    df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])
    df_pca['cluster'] = clusters

    # Create a 2D scatter plot using Seaborn
    plt.figure(figsize=(20, 12))
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='cluster', palette=color_map, s=120)

    # Add centroids to the plot
    for idx, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], color=color_map[idx], edgecolor='black', s=300)

    plt.title('2D Scatter Plot of PCA Components', fontsize=25, pad=10)
    plt.xlabel('PCA Component 1', fontsize=20)
    plt.ylabel('PCA Component 2', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Customize legend
    legend = plt.gca().legend()
    legend.set_title('Cluster', prop={'size': 15})
    plt.setp(legend.get_texts(), fontsize=15)

    plt.show()

if __name__ == "__main__":
    X, clusters, n_clusters = load_and_prepare_data(data_path, selected_features_pearson)
    X_scaled = standardize_features(X)
    X_pca = apply_pca(X_scaled)
    centroids = compute_centroids(X_pca, clusters)
    plot_pca_results(X_pca, clusters, centroids, n_clusters)
