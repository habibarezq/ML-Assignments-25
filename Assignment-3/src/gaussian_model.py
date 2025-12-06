

import numpy as np
from numpy.linalg import slogdet, inv

class GaussianGenerativeModel:
    def __init__(self, lambda_reg=1e-3):
        self.lambda_reg = lambda_reg

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.K = len(self.classes)
        self.d = X.shape[1]

        N = X.shape[0]

        # PRIORS
        self.pi = np.zeros(self.K)
        for k in self.classes:
            self.pi[k] = np.sum(y == k) / N

        # MEANS
        self.mu = np.zeros((self.K, self.d))
        for k in self.classes:
            self.mu[k] = np.mean(X[y == k], axis=0)

        # SHARED COVARIANCE
        S = np.zeros((self.d, self.d))
        for i in range(N):
            k = y[i]
            diff = (X[i] - self.mu[k]).reshape(-1, 1)
            S += diff @ diff.T
        S /= N

        # REGULARIZATION
        self.Sigma = S + self.lambda_reg * np.eye(self.d)
        self.Sigma_inv = inv(self.Sigma)
        sign, logdet = slogdet(self.Sigma)
        self.logdetSigma = logdet

    def _log_gaussian(self, x, k):
        diff = (x - self.mu[k])
        t1 = -0.5 * (diff.T @ self.Sigma_inv @ diff)
        t2 = -0.5 * (self.d * np.log(2 * np.pi) + self.logdetSigma)
        return t1 + t2

    def predict(self, X):
        preds = []
        for x in X:
            scores = []
            for k in self.classes:
                score = np.log(self.pi[k]) + self._log_gaussian(x, k)
                scores.append(score)
            preds.append(np.argmax(scores))
        return np.array(preds)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
