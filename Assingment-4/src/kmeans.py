import numpy as np

class NumpyKMeans:
    def __init__(self, n_clusters=8, init='k-means++', max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters # k
        self.init = init # k-means++ / random
        self.max_iter = max_iter
        self.tol = tol # tolerance on inertia change
        self.random_state = random_state
        self.cluster_centers_ = None # final centroid coordinates after fit
        self.labels_ = None 
        self.inertia_ = None
        self.n_iter_ = None
        self.inertia_history_ = None
    
    def _initialize_centers(self, X):
        n_samples, n_features = X.shape 
        rng = np.random.RandomState(self.random_state)

        if self.init == 'k-means++':
            centers = X[rng.choice(n_samples, 1, replace=False)]
            
            for _ in range(1, self.n_clusters):
                dists_to_centers = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2)
                min_dists = np.min(dists_to_centers, axis=1)
                
                if np.sum(min_dists) == 0:
                    next_center_idx = rng.choice(n_samples)
                else:
                    probs = min_dists / np.sum(min_dists)
                    next_center_idx = rng.choice(n_samples, p=probs)
                
                centers = np.vstack([centers, X[next_center_idx]])
                
        elif self.init == 'random':
            idx = rng.permutation(n_samples)[:self.n_clusters]
            centers = X[idx]
        
        return centers

    def fit(self, X):
        X = np.array(X, dtype=float)
        rng = np.random.RandomState(self.random_state)
        
        # Initialize centers
        self.cluster_centers_ = self._initialize_centers(X)
        inertia_history = []
        labels_old = None
        
        for i in range(self.max_iter):
            dists_to_centers = np.sum((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :])**2, axis=2)
            
            self.labels_ = np.argmin(dists_to_centers, axis=1)
            
            new_centers = np.zeros((self.n_clusters, X.shape[1]))
            for k in range(self.n_clusters):
                mask = (self.labels_ == k)
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    new_centers[k] = self.cluster_centers_[k]  # Keep old if empty
            
            self.cluster_centers_ = new_centers
            
            dists_to_new_centers = np.sum(
                (X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :])**2,
                axis=2)
            inertia = np.sum(dists_to_new_centers[np.arange(len(self.labels_)), self.labels_])
            inertia_history.append(inertia)

            #  Convergence checks
            if labels_old is not None and np.all(labels_old == self.labels_):
                break
            if i > 0 and np.abs(inertia_history[-2] - inertia) < self.tol:
                break
                
            labels_old = self.labels_.copy()
        
        self.inertia_ = inertia
        self.n_iter_ = i + 1
        self.inertia_history_ = np.array(inertia_history)
        return self
    
    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X = np.array(X, dtype=float)
        dists_to_centers = np.sum((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :])**2, axis=2)
        return np.argmin(dists_to_centers, axis=1)
    
    def transform(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X = np.array(X, dtype=float)
        return np.sum((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :])**2, axis=2)
