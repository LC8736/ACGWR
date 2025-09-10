import numpy as np

def generate_grid_data(grid_size=24):
    """Generate grid coordinates"""
    grid_1d = np.linspace(0, 1, grid_size + 1)
    u_grid, v_grid = np.meshgrid(grid_1d, grid_1d)
    u = u_grid.flatten()
    v = v_grid.flatten()
    return u, v, np.column_stack((u, v))

def generate_covariates(n):
    """Generate covariates x1, x2"""
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    return x1, x2

def generate_separable_beta(u, v, x1, x2, alpha=[1.0, 0.5, -0.7]):
    """Generate true beta values and response y for separable model"""
    alpha0, alpha1, alpha2 = alpha
    
    f0 = np.sin(np.pi * u) - np.mean(np.sin(np.pi * u))
    g0 = np.cos(np.pi * v) - np.mean(np.cos(np.pi * v))
    f1 = (u - 0.5)**2 - np.mean((u - 0.5)**2)
    g1 = (v - 0.5)**2 - np.mean((v - 0.5)**2)
    f2 = np.exp(-2 * u) - np.mean(np.exp(-2 * u))
    g2 = np.log(v + 2) - np.mean(np.log(v + 2))
    
    beta0 = alpha0 + f0 + g0
    beta1 = alpha1 + f1 + g1
    beta2 = alpha2 + f2 + g2
    
    y = beta0 + beta1 * x1 + beta2 * x2 + np.random.normal(0, 0.5, len(u))
    
    # Return f/g for visualization
    return y, beta0, beta1, beta2, f0, f1, f2, g0, g1, g2

def generate_nonseparable_beta(u, v, x1, x2, alpha=[1.2, -0.8, 0.7]):
    """Generate true beta values and response y for non-separable model"""
    alpha0, alpha1, alpha2 = alpha
    beta0 = alpha0 + np.sin(np.pi * (u + v) / 3)
    beta1 = alpha1 + np.log((u * v) + 0.5)
    beta2 = alpha2 + np.exp(-((u - 0.5)**2 + (v - 0.5)**2))
    
    y = alpha0 + beta0 + (alpha1 + beta1) * x1 + (alpha2 + beta2) * x2 + np.random.normal(0, 0.5, len(u))
    
    return y, beta0, beta1, beta2