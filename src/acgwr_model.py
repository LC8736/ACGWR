### acgwr_model.py
```python
import numpy as np
from scipy.linalg import inv
from sklearn.metrics import pairwise_distances

class ACGWRModel:
    """Implementation of the Adaptive Coefficient Geographically Weighted Regression model"""
    
    def __init__(self, coords, X, y):
        self.coords = coords
        self.X = np.column_stack([np.ones(len(X)), X])  # Add intercept term
        self.y = y.reshape(-1, 1)
        self.n, self.p = self.X.shape
        self.results = None
        
    def gaussian_kernel(self, distance, bandwidth):
        return np.exp(-distance**2 / (2 * bandwidth**2))
    
    def construct_D(self):
        """Construct the D matrix"""
        dim = self.p * self.n
        D = np.eye(dim)
        for j in range(self.p):
            idx = np.arange(j, dim, self.p)
            for i in idx:
                for k in idx:
                    D[i, k] -= 1 / self.n
        return D
    
    def construct_Z_matrix(self):
        """Construct the Z matrix"""
        Z = np.zeros((self.n, self.n * self.p))
        for i in range(self.n):
            for j in range(self.p):
                Z[i, i*self.p + j] = self.X[i, j]
        return Z
    
    def compute_S_block(self, W_matrix):
        """Compute the S block matrix"""
        S_block = np.zeros((self.p * self.n, self.n))
        for i in range(self.n):
            W = np.diag(W_matrix[i])
            XtWX = self.X.T @ W @ self.X
            try:
                XtWX_inv = np.linalg.inv(XtWX + 1e-8*np.eye(self.p))
            except:
                XtWX_inv = np.linalg.pinv(XtWX)
            S_block[i*self.p:(i+1)*self.p, :] = XtWX_inv @ self.X.T @ W
        return S_block
    
    def estimate_L1_L2(self, bandwidth_u, bandwidth_v):
        """Estimate L1 and L2 matrices"""
        distances_u = pairwise_distances(self.coords[:, 0].reshape(-1, 1))
        distances_v = pairwise_distances(self.coords[:, 1].reshape(-1, 1))
        
        W_u = self.gaussian_kernel(distances_u, bandwidth_u)
        W_v = self.gaussian_kernel(distances_v, bandwidth_v)
        
        D = self.construct_D()
        Z = self.construct_Z_matrix()
        S_u = self.compute_S_block(W_u)
        S_v = self.compute_S_block(W_v)
        
        A = Z @ D @ S_u
        B = Z @ D @ S_v
        I = np.eye(self.n)
        
        try:
            L1 = np.linalg.inv(I - A @ B + 1e-8*I) @ A @ (I - B)
            L2 = np.linalg.inv(I - B @ A + 1e-8*I) @ B @ (I - A)
        except:
            L1 = np.linalg.pinv(I - A @ B + 1e-8*I) @ A @ (I - B)
            L2 = np.linalg.pinv(I - B @ A + 1e-8*I) @ B @ (I - A)
        
        return L1, L2, Z, D, S_u, S_v
    
    def fit(self, bandwidth_u=0.5, bandwidth_v=0.5):
        """Fit the ACGWR model"""
        L1, L2, Z, D, S_u, S_v = self.estimate_L1_L2(bandwidth_u, bandwidth_v)
        H = np.eye(self.n) - L1 - L2
        
        # Estimate global parameters
        alpha_hat = np.linalg.lstsq(H @ self.X, H @ self.y, rcond=None)[0]
        
        # Estimate local components
        r = self.y - self.X @ alpha_hat
        M1_hat = L1 @ r
        M2_hat = L2 @ r
        
        # Predicted values
        y_hat = self.X @ alpha_hat + M1_hat + M2_hat
        
        # Estimate local coefficients
        f_flat = D @ S_u @ (r - M2_hat)
        g_flat = D @ S_v @ (r - M1_hat)
        f_est = f_flat.reshape(self.n, self.p)
        g_est = g_flat.reshape(self.n, self.p)
        beta_hat_local = alpha_hat.T + f_est + g_est
        
        self.results = {
            'alpha_hat': alpha_hat, 
            'y_hat': y_hat, 
            'beta_hat_local': beta_hat_local,
            'M1_hat': M1_hat, 
            'M2_hat': M2_hat, 
            'f_est': f_est, 
            'g_est': g_est
        }
        
        return self.results
    
    def get_local_coefs(self):
        if self.results is None:
            raise ValueError("Model is not fitted yet")
        return self.results['beta_hat_local']
data_generation.py