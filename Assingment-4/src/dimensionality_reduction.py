import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
import time
def reconstruction_mse(X, X_hat):
    return np.mean((X - X_hat) ** 2)
def evaluate_pca(X, bottleneck_sizes):
    results = {}
    
    for b in bottleneck_sizes:
        pca = NumpyPCA(n_components=b)
        pca.fit(X)
        
        Z = pca.transform(X)
        X_rec = pca.inverse_transform(Z)
        
        results[b] = {
            "mse": reconstruction_mse(X, X_rec),
            "explained_variance": np.sum(pca.explained_variance_ratio_)
        }
    return results
def evaluate_autoencoder(X, bottleneck_sizes, epochs=50):
    results = {}
    
    for b in bottleneck_sizes:
        ae = Autoencoder(
            input_dim=X.shape[1],
            hidden_dims=[128, 64],
            bottleneck_dim=b,
            activation='relu',
            learning_rate=0.01
        )
        
        ae.train(X, epochs=epochs, batch_size=64)
        
        Z = ae.encode(X)
        X_rec = ae.decode(Z)
        
        results[b] = {
            "mse": reconstruction_mse(X, X_rec),
            "loss_curve": ae.loss_history
        }
    return results
