import numpy as np

class NumpyKMeans:

    def __init__(self, n_clusters=8, init='k-means++', max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.inertia_history_ = None
    
    def _initialize_centers(self, X):
        # Initialize cluster centers using K-Means++ or random

        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        
        if self.init == 'k-means++':
         centers = X[rng.choice(n_samples, 1, replace=False)]
    
         for _ in range(1, self.n_clusters):
        # Compute distance from each point to nearest existing center
           dist_sq = np.min(np.array([np.sum((X - c)**2, axis=1) for c in centers]), axis=0)
           probs = dist_sq / np.sum(dist_sq)
           next_center_idx = rng.choice(n_samples, p=probs)
           centers = np.vstack([centers, X[next_center_idx]])

                
        elif self.init == 'random':
            # Random initialization
            idx = rng.permutation(n_samples)[:self.n_clusters]
            centers = X[idx]
        
        return centers
    
    def fit(self, X):
        # Fit K-Means clustering to data

        X = np.array(X, dtype=float)
        rng = np.random.RandomState(self.random_state)
        
        # Initialize centers
        self.cluster_centers_ = self._initialize_centers(X)
        inertia_history = []
        
        for i in range(self.max_iter):
            # Assign labels
            distances = np.array([np.sum((X - center)**2, axis=1) 
                                for center in self.cluster_centers_])
            labels_old = self.labels_
            self.labels_ = np.argmin(distances, axis=0)
            
            # Update centers
            new_centers = np.array([X[self.labels_ == k].mean(axis=0) 
                                  if np.any(self.labels_ == k) else self.cluster_centers_[k]
                                  for k in range(self.n_clusters)])
            self.cluster_centers_ = new_centers
            
            # Compute inertia
            inertia = np.sum(distances[self.labels_, np.arange(len(self.labels_))])
            inertia_history.append(inertia)
            
            # Check convergence
            if labels_old is not None and np.all(labels_old == self.labels_):
                break
            if i > 0 and np.abs(inertia_history[-2] - inertia) < self.tol:
                break
        
        self.inertia_ = inertia
        self.n_iter_ = i + 1
        self.inertia_history_ = np.array(inertia_history)
        return self
    
    def predict(self, X):
        # Predict cluster labels for new data

        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = np.array(X, dtype=float)
        distances = np.array([np.sum((X - center)**2, axis=1) 
                            for center in self.cluster_centers_])
        return np.argmin(distances, axis=0)
    
    def transform(self, X):
        # Transform X to cluster-distance space

        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = np.array(X, dtype=float)
        return np.array([np.sum((X - center)**2, axis=1) 
                        for center in self.cluster_centers_]).T
