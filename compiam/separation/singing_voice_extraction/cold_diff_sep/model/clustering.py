import numpy as np
from sklearn.cluster import KMeans


def get_mask(normalized_feat, clusters, scheduler):
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(normalized_feat)
    centers = kmeans.cluster_centers_
    original_means = np.mean(centers, axis=1)
    ordered_means = np.sort(np.mean(centers, axis=1))
    means_and_pos = {}
    manual_weights = np.linspace(0, 1, clusters) ** scheduler
    for idx, j in zip(manual_weights, ordered_means):
        means_and_pos[j] = idx
    label_and_dist = []
    for j in original_means:
        label_and_dist.append(means_and_pos[j])
    weights = []
    for j in kmeans.labels_:
        weights.append(label_and_dist[j])
    return np.array(weights, dtype=np.float32) / float(clusters - 1)
