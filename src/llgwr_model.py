import numpy as np
from scipy.linalg import inv
from sklearn.metrics import pairwise_distances

class LLGWRModel:
    """Implementation of the Local Linear Geographically Weighted Regression model"""
    
    def __init__(self, coords, X, y):
        self.coords = coords
        self.X = np.column_stack([np.ones(len(X)), X])  # Add intercept term
        self.y = y.reshape(-1, 1)
        self.u = coords[:, 0]
        self.v = coords[:, 1]
        self.n, self.p = self.X.shape
        self.results = None
        
    def gaussian_kernel(self, distance, bandwidth):
        return np.exp(-distance**2 / (2 * bandwidth**2))
    
    def fit(self, bandwidth=0.5):
        """Fit the LL-GWR model"""
        y_hat = np.zeros(self.n)
        beta_hat_local = np.zeros((self.n, self.p))
        
        for i in range(self.n):
            # Calculate distances
            distances = np.sqrt((self.u - self.u[i])**2 + (self.v - self.v[i])**2)
            
            # Calculate weights
            weights = self.gaussian_kernel(distances, bandwidth)
            W = np.diag(weights)
            
            # Construct local design matrix
            delta_u = self.u - self.u[i]
            delta_v = self.v - self.v[i]
            
            # Extended design matrix [X, X*Δu, X*Δv]
            X_local = np.zeros((self.n, 3*self.p))
            for j in range(self.p):
                X_local[:, j] = self.X[:, j]  # X
                X_local[:, self.p + j] = self.X[:, j] * delta_u  # X * Δu
                X_local[:, 2*self.p + j] = self.X[:, j] * delta_v  # X * Δv
            
            # Weighted least squares estimation
            try:
                XWX = X_local.T @ W @ X_local
                XWy = X_local.T @ W @ self.y
                beta_local = inv(XWX + 1e-8*np.eye(3*self.p)) @ XWy
                
                # Extract coefficient estimates for current point
                beta_hat_local[i, :] = beta_local[:self.p].flatten()
                
                # Predict response value for current point
                x0 = np.zeros(3*self.p)
                x0[:self.p] = self.X[i, :]
                y_hat[i] = x0 @ beta_local
            except:
                # If matrix is singular, use simple average
                beta_hat_local[i, :] = np.mean(self.X, axis=0)
                y_hat[i] = np.mean(self.y)
        
        self.results = {
            'y_hat': y_hat.reshape(-1, 1),
            'beta_hat_local': beta_hat_local
        }
        
        return self.results
    
    def get_local_coefs(self):
        if self.results is None:
            raise ValueError("Model is not fitted yet")
        return self.results['beta_hat_local']