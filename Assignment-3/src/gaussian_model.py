

import numpy as np
from numpy.linalg import slogdet, inv

class GaussianGenerativeModel:
    def __init__(self, lambda_reg=1e-3):
        self.lambda_reg = lambda_reg
    #μ_k is the mean vector for class k (a 64-dimensional vector) 
    def fit(self, X, y):
        #hena bn define each class k in {0, 1, …, 9}
        self.classes = np.unique(y)
        self.K_Mean = len(self.classes)
        self.dim = X.shape[1]
        #  N(x ; μ_k, Σ)
        N = X.shape[0]

        # -> PRIORS : BNGEEB probability kol digit ll training set
        self.pi = np.zeros(self.K_Mean)
        for k in self.classes:
            self.pi[k] = np.sum(y == k) / N

        # -> MEANS : BNGEEB ll 64_ D el mean 
        self.mu = np.zeros((self.K_Mean, self.dim))
        for k in self.classes:
            self.mu[k] = np.mean(X[y == k], axis=0)

        # -> SHARED COVARIANCE : bngeeb el average ll points 
        #Accumulate (x_i – μ_{y_i}) (x_i – μ_{y_i})^T
        S = np.zeros((self.dim, self.dim))
        for i in range(N):
            k = y[i]
            diff = (X[i] - self.mu[k]).reshape(-1, 1)
            S += diff @ diff.T
        S /= N

        # -> REGULARIZATION : bn classify new points by inverse Σ−1 w log determination log∣Σ∣
        # Σ_λ = Σ + λ I
        self.Sigma = S + self.lambda_reg * np.eye(self.dim)
        self.Sigma_inv = inv(self.Sigma)
        sign, logdet = slogdet(self.Sigma)
        self.logdetSigma = logdet

    def _log_gaussian(self, x, k):
        # log p(y = k | x) ∝ log π_k + log N(x ; μ_k, Σ_λ) for prediction
        diff = (x - self.mu[k])
        t1 = -0.5 * (diff.T @ self.Sigma_inv @ diff)
        t2 = -0.5 * (self.dim * np.log(2 * np.pi) + self.logdetSigma)
        return t1 + t2

    def predict(self, X):
        preds = []
        for x in X:
            scores = []
            for k in self.classes:
                score_in_K = np.log(self.pi[k]) + self._log_gaussian(x, k)
                scores.append(score_in_K)
            preds.append(np.argmax(scores))
        return np.array(preds)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
