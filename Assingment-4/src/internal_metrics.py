import numpy as np

def silhouette_score(X, labels):
    """
    Silhouette Score (-1 to +1): Measures how similar points are to their own cluster 
    (cohesion) vs other clusters (separation). Higher is better.
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    sil_scores = np.zeros(n_samples)
    
    for i in range(n_samples): # for each point
        # Distance to all other points in same cluster (a_i: cohesion)
        same_cluster_mask = (labels == labels[i]) & (np.arange(n_samples) != i) # point is in same cluster and not itself
        if np.sum(same_cluster_mask) > 0:
            a_i = np.mean(np.linalg.norm(X[i] - X[same_cluster_mask], axis=1)) # vectorized
        else:
            a_i = 0
        
        # Distance to nearest neighboring cluster (b_i: separation)
        b_i_values = []
        for label in unique_labels:
            if label != labels[i]: # not same cluster
                cluster_mask = (labels == label)
                if np.sum(cluster_mask) > 0:
                    cluster_distances = np.linalg.norm(X[i] - X[cluster_mask], axis=1)
                    b_i_values.append(np.mean(cluster_distances)) # avg distance to other clusters
        
        if len(b_i_values) > 0:
            b_i = np.min(b_i_values) # b_i = distance to nearest neighboring cluster
            sil_scores[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            sil_scores[i] = 0 # only triggers if no other clusters exist
    
    return np.mean(sil_scores) # mean across all points -> +1=perfect, 0=boundary, -1=wrong cluster

def davies_bouldin_index(X, labels):
    """
    Davies-Bouldin Index: Lower values indicate better clustering.
    Ratio of within-cluster scatter to between-cluster separation.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Compute centroids and within-cluster scatter for each cluster
    centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
    scatter_r = np.zeros(n_clusters)
    
    for i, k in enumerate(unique_labels):
        cluster_points = X[labels == k]
        if len(cluster_points) > 1:
            # Within-cluster scatter (average distance to centroid)
            distances_to_centroid = np.linalg.norm(cluster_points - centroids[i], axis=1)
            scatter_r[i] = np.mean(distances_to_centroid)
        else:
            scatter_r[i] = 0
    
    # Compute pairwise similarity ratios
    db_sum = 0
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            # Between-cluster separation
            separation = np.linalg.norm(centroids[i] - centroids[j])
            ratio = (scatter_r[i] + scatter_r[j]) / separation
            db_sum += ratio
    
    return db_sum / n_clusters

def within_cluster_sum_squares(X, labels):
    """
    Lower is better.
    Total sum of squared distances from points to their cluster centroids.
    """
    unique_labels = np.unique(labels)
    centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels]) # mean of each cluster -> centroid
    wcss = 0
    # For each cluster, sum squared distances to its centroid
    for i, k in enumerate(unique_labels):
        cluster_mask = (labels == k)
        cluster_points = X[cluster_mask]  # all points in this cluster
        if len(cluster_points) > 0:
            distances_squared = np.sum((cluster_points - centroids[i]) ** 2, axis=1)
            wcss += np.sum(distances_squared)
    
    return wcss

def calinski_harabasz_index(X, labels):
    """
    Calinski–Harabasz Index (Variance Ratio Criterion).
    Higher is better: ratio of between-cluster dispersion to within-cluster dispersion.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples, n_features = X.shape

    if n_clusters < 2:
        return 0.0

    overall_mean = np.mean(X, axis=0)

    # Between-cluster dispersion (SSB)
    ss_between = 0.0
    ss_within = 0.0

    for k in unique_labels:
        cluster_points = X[labels == k]
        if cluster_points.shape[0] == 0:
            continue
        cluster_mean = np.mean(cluster_points, axis=0)
        n_k = cluster_points.shape[0]

        # Between-cluster contribution
        diff_mean = cluster_mean - overall_mean
        ss_between += n_k * np.sum(diff_mean ** 2)

        # Within-cluster contribution
        diff_points = cluster_points - cluster_mean
        ss_within += np.sum(diff_points ** 2)

    # Avoid division by zero
    if ss_within == 0 or n_clusters == 1 or n_samples == n_clusters:
        return 0.0

    # CH formula: (SSB / (k - 1)) / (SSW / (n - k))
    ch = (ss_between / (n_clusters - 1)) / (ss_within / (n_samples - n_clusters))
    return ch
