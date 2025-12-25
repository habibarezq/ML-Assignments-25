import numpy as np


def _comb2(n):
    """n choose 2."""
    return n * (n - 1) // 2


def _contingency_matrix(labels_true, labels_pred):
    """
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
    
    Returns:
        contingency: 2D array of shape (n_classes, n_clusters)
        classes: Unique values in labels_true
        clusters: Unique values in labels_pred
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    n_classes, n_clusters = len(classes), len(clusters)
    
    # Create mapping from label values to indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}
    
    # Build contingency matrix
    contingency = np.zeros((n_classes, n_clusters), dtype=int)
    
    for idx in range(len(labels_true)):
        i = class_to_idx[labels_true[idx]]
        j = cluster_to_idx[labels_pred[idx]]
        contingency[i, j] += 1
    
    return contingency, classes, clusters


def adjusted_rand_index(labels_true, labels_pred):
    """
    Adjusted Rand Index (ARI). Range [-1,1], 1 = perfect, 0 ≈ random.
    """
    contingency, _, _ = _contingency_matrix(labels_true, labels_pred)
    n = len(labels_true)
    n_classes, n_clusters = contingency.shape

    # Count same-cluster pairs in the intersection
    sum_comb = 0
    for i in range(n_classes):
        for j in range(n_clusters):
            sum_comb += _comb2(contingency[i, j])

    # Count same-cluster pairs in true labels
    sum_comb_c = 0
    for i in range(n_classes):
        row_sum = np.sum(contingency[i, :])
        sum_comb_c += _comb2(row_sum)

    # Count same-cluster pairs in predicted labels
    sum_comb_k = 0
    for j in range(n_clusters):
        col_sum = np.sum(contingency[:, j])
        sum_comb_k += _comb2(col_sum)

    total_pairs = _comb2(n)
    if total_pairs == 0:
        return 0.0

    prod_comb = (sum_comb_c * sum_comb_k) / total_pairs
    mean_comb = (sum_comb_c + sum_comb_k) / 2.0

    if mean_comb == prod_comb:
        return 1.0

    ari = (sum_comb - prod_comb) / (mean_comb - prod_comb)
    return float(ari)


def purity(labels_true, labels_pred):
    """
    Purity: fraction of samples correctly assigned if each cluster is labeled
    with the majority true class. Range [0,1], higher is better.
    """
    contingency, _, _ = _contingency_matrix(labels_true, labels_pred)
    
    # For each cluster, find the max count (majority class)
    max_per_cluster = np.max(contingency, axis=0)
    
    # Purity is the sum of majority counts divided by total samples
    total_samples = np.sum(contingency)
    return float(np.sum(max_per_cluster) / total_samples)


def _entropy_from_counts(counts):
    """
    Calculate entropy from an array of counts.
    H(X) = -Σ p(x) log(p(x))
    """
    counts = np.asarray(counts, dtype=float)
    total = np.sum(counts)
    if total == 0:
        return 0.0
    
    p = counts / total
    p = p[p > 0]  # Remove zeros to avoid log(0)
    return float(-np.sum(p * np.log(p)))


def normalized_mutual_information(labels_true, labels_pred):
    """
    Normalized Mutual Information (symmetric). Range [0,1], higher is better.
    NMI = MI(X,Y) / sqrt(H(X) * H(Y))
    """
    contingency, _, _ = _contingency_matrix(labels_true, labels_pred)
    n = len(labels_true)
    n_classes, n_clusters = contingency.shape

    # Marginal distributions
    class_counts = np.sum(contingency, axis=1)
    cluster_counts = np.sum(contingency, axis=0)

    # Calculate mutual information
    mi = 0.0
    for i in range(n_classes):
        for j in range(n_clusters):
            n_ij = contingency[i, j]
            if n_ij == 0:
                continue
            
            pij = n_ij / n
            pi = class_counts[i] / n
            pj = cluster_counts[j] / n
            mi += pij * np.log(pij / (pi * pj))

    # Calculate entropies
    h_true = _entropy_from_counts(class_counts)
    h_pred = _entropy_from_counts(cluster_counts)
    
    if h_true == 0 or h_pred == 0:
        return 0.0

    # Normalize by geometric mean of entropies
    return float(mi / np.sqrt(h_true * h_pred))