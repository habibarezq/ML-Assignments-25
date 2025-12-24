import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
import time
def create_comparison_table(pca_res, ae_res, kmeans_res, gmm_res):
    table = []
    
    for b in pca_res.keys():
        table.append([
            b,
            pca_res[b]["mse"],
            pca_res[b]["explained_variance"],
            ae_res[b]["mse"],
            kmeans_res["PCA"][b],
            kmeans_res["AE"][b],
            gmm_res["PCA"][b],
            gmm_res["AE"][b]
        ])
    
    return np.array(table)
def paired_tests(metric_pca, metric_ae):
    t_stat, p_val = ttest_rel(metric_pca, metric_ae)
    return t_stat, p_val
t, p = paired_tests(
    [kmeans_res["PCA"][b] for b in bottleneck_sizes],
    [kmeans_res["AE"][b] for b in bottleneck_sizes]
)
print("Paired t-test (KMeans): p-value =", p)
def complexity_summary():
    print("""
    PCA:
      Time: O(n · d²) (covariance + eigen decomposition)
      Space: O(d²)
    
    Autoencoder:
      Time: O(n · epochs · parameters)
      Space: O(parameters)
    
    K-Means:
      Time: O(n · k · d · i)
      Space: O(n · d)
    
    GMM:
      Time: O(n · k · d² · i)
      Space: O(k · d²)
    """)
