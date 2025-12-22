import numpy as np
from itertools import combinations
from collections import Counter

class ExternalMetrics:
    @staticmethod
    def contingency_matrix(labels_true,labels_pred):
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        
        n_classes=len(classes)
        n_clusters=len(clusters)
        
        # mapping to indices
        class_to_idx={c:i for i,c in enumerate(classes)}
        cluster_to_idx={c:i for i,c in enumerate(clusters)}
        
        matrix=np.zeros((n_classes,n_clusters),dtype=int)
        
        for idx in range(len(labels_true)):
            i = class_to_idx[labels_true[idx]]
            j = cluster_to_idx[labels_pred[idx]]
            matrix[i, j] += 1
        
        return matrix, classes, clusters
    
    @staticmethod
    def _comb2(n): # computes nC2
        return n * (n-1)//2
    
    @staticmethod
    def adjusted_rand_index(labels_true, labels_pred):
        n = len(labels_true)
        contingency, _, _ = ExternalMetrics.contingency_matrix(labels_true, labels_pred)

        n_classes, n_clusters = contingency.shape

        # Same–same pairs
        sum_comb = 0
        for i in range(n_classes):
            for j in range(n_clusters):
                sum_comb += ExternalMetrics._comb2(contingency[i][j])

        # True cluster pairs
        sum_comb_c = 0
        for i in range(n_classes):
            row_sum = 0
            for j in range(n_clusters):
                row_sum += contingency[i][j]
            sum_comb_c += ExternalMetrics._comb2(row_sum)

        # Predicted cluster pairs
        sum_comb_k = 0
        for j in range(n_clusters):
            col_sum = 0
            for i in range(n_classes):
                col_sum += contingency[i][j]
            sum_comb_k += ExternalMetrics._comb2(col_sum)

        total_pairs = ExternalMetrics._comb2(n)
        prod_comb = (sum_comb_c * sum_comb_k) / total_pairs
        mean_comb = (sum_comb_c + sum_comb_k) / 2

        if mean_comb == prod_comb:
            return 1.0

        ari = (sum_comb - prod_comb) / (mean_comb - prod_comb)
        return ari