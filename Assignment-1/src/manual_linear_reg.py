import numpy as np

def predict(X, w,b=None):
    # Compute predictions
    if b is None:
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(X_b, w) # working with y=X.w where X is design matrix
    else:
        return np.dot(X, w) + b
    
def compute_mae_cost(X,y,w,b):
    m = X.shape[0]
    y_pred=predict(X,w,b)
    
    # Calculate MAE (Mean Absolute Error)
    mae_cost = (1 / m) * np.sum(np.abs(y_pred - y))
    return mae_cost

def compute_mse_cost(X,y,w,b):
    m = X.shape[0]
    y_pred=predict(X,w,b)
    
    # Calculate MSE (Mean Square Error)
    mse_cost=(1 / (2*m) * np.sum((y_pred - y)** 2))
    return mse_cost

def normal_equation(X,y):
    X_b = np.hstack((np.ones((X.shape[0], 1)), X))  # (m, n+1)
    # Compute weights (w = (XᵀX)⁻¹Xᵀy)
    w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return w

def compute_ridge_cost(X,y,w,b,lambda_ridge):
    ## should return total cost (MSE + L2 Penalty)
    m=X.shape[0]
    
    mse_cost=compute_mse_cost(X,y,w,b)
    l2_penalty=(lambda_ridge / (2*m)) * np.sum(w **2)
    total_cost = mse_cost + l2_penalty
    
    return total_cost

def compute_ridge_gradient(X,y,w,b,lambda_ridge):
    ## should return dw , db
    m=X.shape[0]
    
    y_pred=predict(X,w,b)
    dw=(1/m) * np.dot(X.T,(y_pred-y)) + (lambda_ridge/m) * w
    db=(1/m) * np.sum(y_pred-y)
    return dw,db

def compute_gradient(X,y,w,b):
    m=X.shape[0]
    y_pred=predict(X,w,b)
    dw = (1 / m) * np.dot(X.T, (y_pred - y))   # (n, 1)
    db = (1 / m) * np.sum(y_pred - y)         
    return dw, db
    

def gradient_descent(X,y,w,b,alpha,num_iterations,ridge=False):
    #update weights to minimize the cost
    mse_cost_history=[]
    
    for i in range(num_iterations):
        if(ridge):
            dw,db=compute_ridge_gradient(X,y,w,b,alpha)
        else:
            dw,db=compute_gradient(X,y,w,b)
        w = w - alpha * dw
        b = b - alpha * db
        mae_cost =compute_mae_cost(X,y,w,b)
        mse_cost =compute_mse_cost(X,y,w,b)
        mse_cost_history.append(mse_cost)
    
        # if i % 100 == 0:
        #     print(f"Iteration {i}, MSE Cost: {mse_cost:.4f},MAE Cost: {mae_cost:.4f}")
    return w,b,mse_cost_history

def train_model(X,y,alpha,num_iterations):
    m,n=X.shape
    w=np.zeros((n,1))
    b=0
    w,b,cost_history=gradient_descent(X,y,w,b,alpha,num_iterations,ridge=False)
    return w,b,cost_history

def evaluate_model(X_test,y_test,w,b=None):
    #to compute MSE, MAE ON Test set after training
    y_pred=predict(X_test,w,b)
    mse=np.mean((y_test-y_pred)**2)
    mae=np.mean(np.abs(y_test-y_pred))
    return mse,mae

def train_ridge_model(X,y,alpha,lambda_ridge,num_iterations):
    m,n=X.shape
    w=np.zeros((n,1))
    b=0
    
    w,b,cost_history=gradient_descent(X,y,w,b,alpha,num_iterations,ridge=True)
    return w,b,cost_history
