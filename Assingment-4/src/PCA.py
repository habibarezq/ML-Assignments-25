import numpy as np

class NumpyPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
    
    def fit(self, X):
        # Center the data (subtract mean)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        if self.n_components is not None:
            self.components_ = eigenvectors[:, :self.n_components]
            self.explained_variance_ = eigenvalues[:self.n_components]
        else:
            self.components_ = eigenvectors
            self.explained_variance_ = eigenvalues
        
        # Explained variance ratio
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        # Singular values (sqrt of eigenvalues)
        self.singular_values_ = np.sqrt(self.explained_variance_)
        
        return self
    
    def transform(self, X):
        # Transform data to principal components 
        if self.components_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def inverse_transform(self, X_transformed):
        # Transform data back to original feature space (reconstruction)
        if self.components_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return np.dot(X_transformed, self.components_.T) + self.mean_
    
    def reconstruction_error(self, X):
        # Compute reconstruction error (MSE) for given data
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        mse = np.mean((X - X_reconstructed) ** 2)
        return mse
