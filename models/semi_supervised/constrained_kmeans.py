import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm

def select_informative_constraints(X, must_link, cannot_link, max_constraints):
    def constraint_score(pair):
        i, j = pair
        return np.linalg.norm(X[i] - X[j])

    must_link_scores = [(pair, constraint_score(pair)) for pair in must_link]
    cannot_link_scores = [(pair, constraint_score(pair)) for pair in cannot_link]

    must_link_scores.sort(key=lambda x: x[1])
    cannot_link_scores.sort(key=lambda x: x[1], reverse=True)

    selected_must_link = [pair for pair, score in must_link_scores[:max_constraints]]
    selected_cannot_link = [pair for pair, score in cannot_link_scores[:max_constraints]]

    return selected_must_link, selected_cannot_link


def constrained_kmeans(X, n_clusters, must_link, cannot_link, max_iter=300):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
    kmeans.fit(X)

    for _ in tqdm(range(max_iter)):
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        for a, b in must_link:
            labels[b] = labels[a]

        for a, b in cannot_link:
            if labels[a] == labels[b]:
                _, distances = pairwise_distances_argmin_min(X[b].reshape(1, -1), centroids)
                alt_cluster = np.argmin(distances + np.inf * (np.arange(n_clusters) == labels[a]))
                labels[b] = alt_cluster

        for j in range(n_clusters):
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) > 0:
                centroids[j] = np.mean(points_in_cluster, axis=0)

        kmeans.cluster_centers_ = centroids
        kmeans.labels_ = labels
        kmeans.fit(X)

        if np.all(labels == kmeans.labels_):
            break

    return kmeans.labels_, kmeans.cluster_centers_


def create_constraints(labels):
    must_link = []
    cannot_link = []
    n = len(labels)

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                must_link.append((i, j))
            else:
                cannot_link.append((i, j))

    return must_link, cannot_link
