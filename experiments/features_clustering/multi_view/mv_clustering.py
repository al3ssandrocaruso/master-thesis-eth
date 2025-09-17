import argparse
from mvlearn.cluster import MultiviewKMeans
from torchgen.gen_lazy_tensor import default_args
from dataset.load_data_utils import load_data
from utils.visualization.clustering.mv_visualization import display_mv_plots
from config.config import selected_features_pearson
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def get_args():

    parser = argparse.ArgumentParser(description='Clustering parameters')

    parser.add_argument('--n_clusters', type=int, default=2, help='Number of clusters')
    parser.add_argument('--scaler', type=str, default='MinMax', choices=['MinMax', 'Standard'], help='Choice of scaler: MinMax for MinMaxScaler, Standard for StandardScaler')
    parser.add_argument('--pca_components_view_1', type=int, default=11, help='Number of PCA components to keep for view 1')
    parser.add_argument('--pca_components_view_2', type=int, default=5, help='Number of PCA components to keep for view 2')

    return parser.parse_args()

def main():
    args = get_args()  # Getting parsed arguments
    X, data = load_data('multi-view')  # Loading data

    X_1 = X.iloc[:,:-9] # View 1
    X_1 = X_1[selected_features_pearson]
    scaler_1 = MinMaxScaler() if args.scaler == 'MinMax' else StandardScaler()
    X_1_scaled = scaler_1.fit_transform(X_1)
    pca_1 = PCA(n_components=args.pca_components_view_1, random_state=42)
    X_1_pca = pca_1.fit_transform(X_1_scaled)

    X_2 = X.iloc[:,-10:] # View 2
    scaler_2 = MinMaxScaler() if args.scaler == 'MinMax' else StandardScaler()
    X_2_scaled = scaler_2.fit_transform(X_2)
    pca_2 = PCA(n_components=args.pca_components_view_2, random_state=42)
    X_2_pca = pca_2.fit_transform(X_2_scaled)

    Xs = [X_1_pca, X_2_pca]

    model = MultiviewKMeans(n_clusters=args.n_clusters, random_state=42)

    clusters = model.fit_predict(Xs)

    display_mv_plots('Multiview KMeans Clusters (PCA for visualization)', [X_1_pca[:, :2], X_2_pca[:, :2]], clusters)


if __name__ == "__main__":
    main()