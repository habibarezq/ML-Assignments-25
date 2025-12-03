import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, label=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left: Node | None = None
        self.right: Node | None = None
        self.label = label              # used only for leaf nodes
        self.samples_count = 0
        self.class_counts: np.ndarray | None = None       #ndarray is the class(data type) we use np.array to create an instance of that class

class DecisionTreeClassifier:
    def __init__(self,max_depth,min_samples_split):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        # max_depth , min_samples_split are the model parameters
        self.feature_importance: np.ndarray | None = None
        self.root:Node | None= None
        
    def _majority_class(self,y):
        arr=self._class_counts(y)
        if arr[0]>arr[1]:
            return 0
        else:
            return 1
        
    
    def _class_counts(self,y):
        count_of_0=0
        count_of_1=0
        for label in y:
            if label == 0:
                count_of_0 +=1
            elif label ==1:
                count_of_1 +=1
        return np.array([count_of_0, count_of_1])

    def _best_split(self,X,y):  # finds the feature and threshold thay gives best info gain
        n_features=X.shape[1]
        best_ig=float('-inf')
        best_feature_index = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        for i in range(n_features):
            values=np.unique(X[:,i]) # will be automatically sorted
            if len(values) < 2:  # Skip if only one unique value
                continue
            for j in range(len(values)-1):
                candidate_threshold= (values[j]+values[j+1]) /2
                left_idx=np.where(X[:,i] <=  candidate_threshold)[0]
                right_idx = np.where(X[:, i] > candidate_threshold)[0]
                ig=self.information_gain(y,y[left_idx],y[right_idx])
                if ig > best_ig:
                    best_ig=ig
                    best_feature_index=i
                    best_threshold=candidate_threshold
                    best_left_idx=left_idx
                    best_right_idx=right_idx
                    
        # left_idx array of indices of samples going left
        # right_idx array of indices of samples going right
        return best_feature_index, best_threshold, best_left_idx, best_right_idx, best_ig
    def entropy(self,y):
        # get the probailities
        total_samples=len(y)
        counts = self._class_counts(y)          # [count_0, count_1]
        probs = counts / total_samples          # probabilities
        eps=1e-10
        probs=np.maximum(probs,eps)
        return -np.sum(probs*np.log2(probs))
    
    def information_gain(self,parent_y,left_y,right_y):
        y_entropy=self.entropy(parent_y)
        y_left_entropy=self.entropy(left_y)
        y_right_entropy=self.entropy(right_y)
        
        p_left=len(left_y)/len(parent_y)
        p_right=len(right_y)/len(parent_y)
        
        return y_entropy-(p_left*y_left_entropy + p_right*y_right_entropy)
        
    def _build_tree(self,X,y,depth): # private function only called by the fit
        # check stopping critieria
        n_samples=len(X)
        class_counts=self._class_counts(y)
        if(depth>=self.max_depth or n_samples < self.min_samples_split or len(np.unique(y))==1 ):
            leaf=Node()
            leaf.label=self._majority_class(y)
            leaf.samples_count=n_samples
            leaf.class_counts=class_counts
            return leaf
        
        # find best feature to split upon amd threshold
        feature_index,threshold,left_idx,right_idx,best_info_gain=self._best_split(X,y)
        
        # if no valid split, make leaf node
        if feature_index is None:
            leaf=Node()
            leaf.label=self._majority_class(y)
            leaf.samples_count=n_samples
            leaf.class_counts=class_counts
            return leaf
        
        # upate feature importance
        if self.feature_importance is not None:
            self.feature_importance[feature_index] += best_info_gain
        
        # create internal node 
        node=Node(feature_index=feature_index,threshold=threshold)
        node.samples_count=n_samples
        node.class_counts=class_counts
        
        # recursively build the children
        # X[left_idx] selects the rows of X at positions in the left_idx array
        # y[left_idx] selects the corresponding labels
        
        node.left=self._build_tree(X[left_idx],y[left_idx],depth+1)
        node.right=self._build_tree(X[right_idx],y[right_idx],depth+1)
        
        return node
    def fit(self,X,y): #builds the tree
        num_features=X.shape[1]
        self.feature_importance=np.zeros(num_features)
        self.root=self._build_tree(X,y,depth=0)
        
    def _predict_single(self,x): #private function only called by the predict
        node=self.root
        if node is not None:
            while node.left is not None and node.right is not None:
                f=node.feature_index
                if x[f] <= node.threshold:
                    node=node.left
                else:
                    node=node.right
            return node.label
    def predict(self,X): # uses the built tree
        predictions=[]
        for i in range(len(X)):
            predictions.append(self._predict_single(X[i]))
        return np.array(predictions)