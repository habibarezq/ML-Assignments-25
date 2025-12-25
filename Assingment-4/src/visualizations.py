import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
import time
from kmeans import NumpyKMeans
from gmm import GMM, CovarianceType


def plot_2d_clusters(Z, labels, title):
    plt.figure(figsize=(5,4))
    plt.scatter(Z[:,0], Z[:,1], c=labels, cmap='tab10', s=10)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()
def plot_elbow(Z):
    inertias = []
    ks = range(2, 15)
    
    for k in ks:
        km = NumpyKMeans(n_clusters=k)
        km.fit(Z)
        inertias.append(km.inertia_)
    
    plt.plot(ks, inertias, marker='o')
    plt.axvline(x=10, linestyle='--')  # expected optimal
    plt.title("Elbow Curve")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.show()
def plot_bic_aic(Z):
    ks = range(2, 15)
    bic, aic = [], []
    
    for k in ks:
        gmm = GMM(k, CovarianceType.FULL)
        gmm.fit(Z)
        bic.append(gmm.bic(Z))
        aic.append(gmm.aic(Z))
    
    plt.plot(ks, bic, label="BIC")
    plt.plot(ks, aic, label="AIC")
    plt.legend()
    plt.xlabel("Components")
    plt.ylabel("Score")
    plt.title("GMM Model Selection")
    plt.show()
def plot_ae_losses(ae_results):
    for b, res in ae_results.items():
        plt.plot(res["loss_curve"], label=f"B={b}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Autoencoder Training Curves")
    plt.legend()
    plt.show()
def plot_heatmap(pca_res, ae_res, kmeans_res, gmm_res):
    data = []
    labels = []
    
    for b in pca_res:
        data.append([
            pca_res[b]["mse"],
            ae_res[b]["mse"],
            kmeans_res["PCA"][b],
            kmeans_res["AE"][b],
            gmm_res["PCA"][b],
            gmm_res["AE"][b]
        ])
        labels.append(f"B={b}")
    
    sns.heatmap(data, annot=True, xticklabels=[
        "PCA MSE", "AE MSE",
        "KMeans PCA", "KMeans AE",
        "GMM PCA", "GMM AE"
    ], yticklabels=labels)
    
    plt.title("Method Comparison Heatmap")
    plt.show()
def plot_confusion(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
