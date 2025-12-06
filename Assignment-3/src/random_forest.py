import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))
from src.decision_tree import *

class RandomForestClassifier:
    def __init__(self,max_depth,min_samples_split,n_estimators,max_features,bootstrap=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators=n_estimators
        self.max_features=int(max_features)
        self.bootstrap=bootstrap
        
        self.trees=[]
        self.feature_indices=[] #  list of arrays each to store the feature indices used by each tree
        self.n_features=None
        self.n_classes=None
        
    def _bootstrap_sample(self,X,y): # create a bootstrap sample
        n_samples=X.shape[0]
        indices=np.random.choice(n_samples,size=n_samples,replace=True)
        return X[indices],y[indices] # returns the random selected rows
    
    def _get_feature_subset(self,n_features):
        n_features_to_sample=self.max_features
        feature_indices=np.random.choice(n_features,size=n_features_to_sample,replace=False) # cant use a single feature more than once
        return sorted(feature_indices)
    
    def fit(self,X,y):
        self.n_features=X.shape[1]
        self.n_classes=len(np.unique(y))
        
        # build each tree in the forest
        for i in range(self.n_estimators):
            # bootstrap sampling
            if self.bootstrap:
                X_sample,y_sample=self._bootstrap_sample(X,y)
            else:
                X_sample,y_sample=X,y
            
            # Random feature subset
            feature_indices=self._get_feature_subset(self.n_features)
            self.feature_indices.append(feature_indices)
            
            # select only the chosen features
            X_subset=X_sample[:,feature_indices]
            
            # train a decision tree
            tree=DecisionTreeClassifier(max_depth=self.max_depth,
                min_samples_split=self.min_samples_split)
            
            tree.fit(X_subset,y_sample)
            self.trees.append(tree)
            
            
    def predict(self,X):
        tree_predictions=np.zeros((X.shape[0],self.n_estimators))
        
        for i,tree in enumerate(self.trees):
            # use only the features this tree was trained on
            X_subset=X[:,self.feature_indices[i]]
            tree_predictions[:,i]=tree.predict(X_subset)
            
        # Majority Voting
        predictions=[]
        for i in range(X.shape[0]):
            #count votes for each class
            votes=tree_predictions[i,:]
            unique,counts=np.unique(votes,return_counts=True)
            index_max_count=np.argmax(counts)
            predictions.append(unique[index_max_count])    
        
        return np.array(predictions)
    
    def get_feature_importance(self):
        if self.n_features is not None:
            importance = np.zeros(self.n_features)      
        
            for i, tree in enumerate(self.trees):
                # Get feature importance from this tree
                tree_importance = tree.feature_importance
                
                # Map back to original feature indices
                for j, feature_idx in enumerate(self.feature_indices[i]):
                    if tree_importance is not None and j < len(tree_importance):
                        importance[feature_idx] += tree_importance[j]
            
            # # Normalize
            # if importance.sum() > 0:
            #     importance = importance / importance.sum()
            
            return importance
