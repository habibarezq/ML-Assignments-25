import numpy as np
from enum import Enum
import random

class CovarianceType(Enum):
    FULL="full"
    TIED="tied"
    DIAGONAL="diag"
    SPHERICAL="spherical"

class GMM:
    def __init__(self,n_components,cov_type:CovarianceType,tol,max_iter=50,reg_covar=1e-6):
        self.n_components=n_components
        self.cov_type=cov_type # FULL, TIED, DIAGONAL, SPHERICAL
        self.tol=tol
        self.max_iter=max_iter
        self.reg_covar=reg_covar # covariance regularization term added to the 
        #  diagonal of each covariance matrix to prevent singular matrices
        
        # EM tracking
        self.log_likelihoods=[]
        
        # Parameters ( initialized in fit by calling init_params)
        self.weights: np.ndarray | None=None # weights π_k
        self.means: np.ndarray | None=None # μ_k
        self.covariances: np.ndarray | None=None # Σ_k
        
    def _init_params(self,X):
        """ Initialize means , Covariances , weights"""
        n_samples,n_features=X.shape
        K=self.n_components # number of gaussian components
        
        # randomly pick K samples from X
        indices=np.random.choice(n_samples,K,replace=False)
        self.means=X[indices] # means is a matrix K x d
        
        self.weights = np.full(K,1.0/K) # prior probability assume all gaussian components are equally likely
        
        if self.cov_type== CovarianceType.FULL: # each component has full cov matrix
            self.covariances=np.array([np.cov(X,rowvar=False)+self.reg_covar * np.eye(n_features) for _ in range(K)]) # np.eye --> identity matrix
            # shape: (K, D, D)
        elif self.cov_type ==CovarianceType.TIED: # all components share one cov matrix
            self.covariances=np.cov(X,rowvar=False)+self.reg_covar* np.eye(n_features)
            # shape: (D, D)
        elif self.cov_type ==CovarianceType.DIAGONAL: # diagonal covariance per component
            variance=np.var(X,axis=0) +self.reg_covar
            self.covariances=np.tile(variance,(K,1)) # shape K x D x D but using the tile K x D
            
        elif self.cov_type==CovarianceType.SPHERICAL: # single variance per component
            variance=np.mean(np.var(X,axis=0))+self.reg_covar
            self.covariances=np.full(K,variance) # shape: (K,)
        
    def _multivariate_gaussian(self,x,mean,cov):
        D=x.shape[0]
        
        if self.cov_type == CovarianceType.TIED or self.cov_type == CovarianceType.FULL:
            determinant_cov=np.linalg.det(cov)
            cov_inverse=np.linalg.inv(cov)
            constant=np.pow(2*np.pi,D/2) * np.pow(determinant_cov,0.5)
            diff=x-mean
            exponent=-0.5 * diff.T @ cov_inverse @ diff
            prob_x=1/constant * np.exp(exponent)
            return prob_x
        
        elif self.cov_type ==CovarianceType.DIAGONAL:
            prod = np.prod(cov)  # product of variances
            diff = x - mean
            exponent = np.sum(diff**2 / cov) # vectorized form
            constant = np.sqrt((2 * np.pi)**D * prod)
            prob_x = np.exp(-0.5 * exponent) / constant
            return prob_x
        
        elif self.cov_type==CovarianceType.SPHERICAL:
            diff = x - mean
            exponent = np.sum(diff**2 / cov) # vectorized form
            constant = np.sqrt((2 * np.pi * cov)**D)
            prob_x = np.exp(-0.5 * exponent) / constant
            return prob_x

    def _log_multivariate_gaussian(self, x, mean, cov):
        D = x.shape[0]
        if self.cov_type == CovarianceType.TIED or self.cov_type == CovarianceType.FULL:
            diff = x - mean
            cov_inverse = np.linalg.inv(cov)
            determinant_cov = np.linalg.det(cov)
            
            log_prob = -0.5 * (diff.T @ cov_inverse @ diff) - 0.5 * (D * np.log(2*np.pi) + np.log(determinant_cov))
            return log_prob
        
        elif self.cov_type ==CovarianceType.DIAGONAL:
            diff = x - mean
            log_prob=-0.5*np.sum(diff**2/cov) -0.5 * (D*np.log(2*np.pi)+np.sum(np.log(cov)))
            return log_prob
        
        elif self.cov_type==CovarianceType.SPHERICAL:
            diff = x - mean
            log_prob=-0.5*np.sum(diff**2)/cov -0.5 * D *np.log(2*np.pi*cov)
            return log_prob

    def _e_step(self,X):
        n_samples,n_features=X.shape
        responsibilities=np.zeros((n_samples,self.n_components))
        
        
        for i in range(n_samples):
            log_probs=np.zeros(self.n_components) # log probabilites per sample i
            for k in range(self.n_components):
                if self.cov_type == CovarianceType.TIED:
                    cov = self.covariances  # Use the shared covariance matrix
                else:
                    cov = self.covariances[k]  # type: ignore # Use component-specific covariance
                
                log_probs[k]=(np.log(self.weights[k]) + self._log_multivariate_gaussian(X[i],self.means[k],cov)) # type: ignore
            
            # numerical stability 
            max_log=np.max(log_probs)
            probs=np.exp(log_probs-max_log)
            
            #normalize
            responsibilities[i,:]=probs/probs.sum()
        return responsibilities
    
    def _m_step(self,X,responsibilities):
        # responsiiblities shape(N,K)
        n_samples=X.shape[0]
        N_k=responsibilities.sum(axis=0) #shape(K,)
        self.weights=N_k/n_samples
        self.means=(responsibilities.T @ X)/N_k[:,np.newaxis] #shape should be (K,D)
        # [:, np.newaxis] converts (K,) → (K, 1)
        self.covariances=self._compute_covariance(X,responsibilities,N_k)
    def _compute_covariance(self,X,responsibilities,N_k):
        n_samples, n_features = X.shape
        K = self.n_components
    
        covariances = np.zeros((K, n_features, n_features))  # store all covs

        
        # Vectorized Form
        diffs = X[np.newaxis, :, :] - self.means[:, np.newaxis, :]  # type: ignore # shape (K, N, D)
        resp_reshaped = responsibilities.T[:, :, np.newaxis]  # shape (K, N, 1)
        weighted_diffs = diffs * resp_reshaped                 # shape (K, N, D)
        for k in range(K):
            covariances[k] = weighted_diffs[k].T @ diffs[k] / N_k[k] + self.reg_covar * np.eye(n_features)
        
        if self.cov_type == CovarianceType.TIED:
            return covariances.mean(axis=0)  # average across components
        
        elif self.cov_type == CovarianceType.DIAGONAL:
            diag_covs = np.array([np.diag(cov) for cov in covariances])
            diag_covs = np.maximum(diag_covs, self.reg_covar)
            return diag_covs  # Shape: (K, D)
        
        elif self.cov_type == CovarianceType.SPHERICAL:
            spherical_vars= np.array([np.mean(np.diag(cov)) for cov in covariances])
            spherical_vars = np.maximum(spherical_vars, self.reg_covar)
            return spherical_vars
        else:  # FULL
            return covariances
    
    def _compute_log_likelihood(self, X):
        n_samples = X.shape[0]
        log_likelihood = 0
        for i in range(n_samples):
            log_probs = np.zeros(self.n_components)
            for k in range(self.n_components):
                if self.cov_type == CovarianceType.TIED:
                    cov = self.covariances
                else:
                    cov = self.covariances[k] # type: ignore
                # log P(x_i | z_i=k, θ) + log P(z_i=k)
                log_probs[k] = (np.log(self.weights[k]) +  # type: ignore
                           self._log_multivariate_gaussian(X[i], self.means[k], cov))  # type: ignore
        
            max_log_prob = np.max(log_probs)
            log_likelihood += max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))
        return log_likelihood

    def fit(self,X):
        self._init_params(X)
        for iter in range(self.max_iter):
            responsibilities=self._e_step(X)
            self._m_step(X,responsibilities)
            log_likelihood=self._compute_log_likelihood(X)
            self.log_likelihoods.append(log_likelihood)
            
            if iter>0 and abs(self.log_likelihoods[-1]-self.log_likelihoods[-2] )< self.tol:
                print(f"Converged at iteration {iter}")
                break
            
    def predict(self,X):
        responsibilities=self._e_step(X)
        return np.argmax(responsibilities,axis=1)
    
    def predict_proba(self, X):
        return self._e_step(X)
    
    def bic(self,X): # Bayesian Information Criterion
        N,D=X.shape
        K=self.n_components
        # means: K * D
        # weights: K-1
        n_params=K * D + (K-1)
        
        # covariance count depends on the type
        if self.cov_type==CovarianceType.FULL:
            n_params+=K*D*(D+1)//2
        elif self.cov_type==CovarianceType.TIED:
            n_params+=D*(D+1)//2
        elif self.cov_type==CovarianceType.DIAGONAL:
            n_params+=K*D # (diagonal elements only)
        elif self.cov_type==CovarianceType.SPHERICAL:
            n_params+=K
            
        log_likelihood=self.log_likelihoods[-1]
        return -2*log_likelihood+np.log(N)*n_params
    
    def aic(self,X): # kaike Information Criterion (AIC)
        N,D=X.shape
        K=self.n_components
        # means: K * D
        # weights: K-1
        n_params=K * D + (K-1)
        
        # covariance count depends on the type
        if self.cov_type==CovarianceType.FULL:
            n_params+=K*D*(D+1)//2
        elif self.cov_type==CovarianceType.TIED:
            n_params+=D*(D+1)//2
        elif self.cov_type==CovarianceType.DIAGONAL:
            n_params+=K*D # (diagonal elements only)
        elif self.cov_type==CovarianceType.SPHERICAL:
            n_params+=K
            
        log_likelihood=self.log_likelihoods[-1]
        return -2*log_likelihood+2*n_params