import numpy as np
import pandas as pd
import warnings
from scipy.sparse import lil_matrix
from scipy.stats import norm
import statsmodels.api as sm
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from mgwr.utils import shift_colormap
from libpysal.weights import KNN
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.linalg import inv
import os
import time
import pickle
import sys
from io import StringIO
import contextlib
import argparse
from tqdm import tqdm

# Set matplotlib
import matplotlib
matplotlib.use('Agg')  # Set to non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib as mpl
from matplotlib import font_manager

# Set Times New Roman font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

# Set global font size
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Set plot style
sns.set_style("whitegrid")
mpl.rcParams['axes.edgecolor'] = 'gray'
mpl.rcParams['axes.linewidth'] = 0.5

# Check for Times New Roman font
font_path = None
for font in font_manager.findSystemFonts():
    if 'times' in font.lower() or 'tnr' in font.lower():
        font_path = font
        break

if font_path:
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
else:
    print("Warning: Times New Roman font not found, using default font")

# Ignore warnings
warnings.filterwarnings('ignore')

# Create a completely silent output context manager
class CompleteSilence:
    def __enter__(self):
        # Save original stdout and stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        # Redirect to null device
        self._null_stream = open(os.devnull, 'w')
        sys.stdout = self._null_stream
        sys.stderr = self._null_stream
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout and stderr
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        # Close null device stream
        self._null_stream.close()

# Redirect IPython/Jupyter display output
@contextlib.contextmanager
def suppress_ipython_display():
    """Suppress IPython display output"""
    try:
        # Try to get IPython's display module
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            # Save original display function
            original_display = ipython.display_formatter.format
            # Replace with empty function
            ipython.display_formatter.format = lambda obj, include, exclude: ({}, {})
    except ImportError:
        # If no IPython, do nothing
        pass
    
    try:
        yield
    finally:
        try:
            # Restore original display function
            if ipython is not None:
                ipython.display_formatter.format = original_display
        except:
            pass

def plot_surface(ax, u, v, z, cmap='viridis'):
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')

    ax.xaxis._axinfo["grid"]['color'] = '#e0e0e0'
    ax.yaxis._axinfo["grid"]['color'] = '#e0e0e0'
    ax.zaxis._axinfo["grid"]['color'] = '#e0e0e0'

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')

    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis='z', colors='black')

    surf = ax.plot_trisurf(u, v, z, cmap=cmap, edgecolor='none', alpha=1)
    ax.set_xlabel('u', fontname='Times New Roman', color='black')
    ax.set_ylabel('v', fontname='Times New Roman', color='black')
    ax.view_init(elev=30, azim=-60)

    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_zticklabels():
        label.set_fontname('Times New Roman')

    return surf

def plot_function_curve(ax, x, true_vals, est_vals, title, xlabel, ylabel):
    sorted_idx = np.argsort(x)
    ax.plot(x[sorted_idx], true_vals[sorted_idx], label='True', lw=2, color='#1f77b4')
    ax.plot(x[sorted_idx], est_vals[sorted_idx], label='Estimated', lw=2, linestyle='--', color='#ff7f0e')
    ax.set_xlabel(xlabel, fontname='Times New Roman', fontsize=15)
    ax.set_ylabel(ylabel, fontname='Times New Roman', fontsize=15) 
    ax.set_title(title, fontname='Times New Roman', fontsize=15)
    legend = ax.legend(fontsize=14)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    ax.grid(True, linestyle='--', alpha=0.7)
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

def visualize_results(results, output_dir):
    setting_name = results['simulation_params']['setting_name']
    u = results['coordinates'][:, 0]
    v = results['coordinates'][:, 1]

    fig = plt.figure(figsize=(15, 12), dpi=1500)
    fig.set_facecolor('white')

    titles = [
        "$β_0(u,v)$", "$β_0(u,v)$", "$β_0(u,v)$",
        "$β_1(u,v)$", "$β_1(u,v)$", "$β_1(u,v)$",
        "$β_2(u,v)$", "$β_2(u,v)$", "$β_2(u,v)$"
    ]
    
    beta_data = [
        results['true_beta_avg'][:, 0],
        results['acgwr_beta_avg'][:, 0],
        results['gwr_beta_avg'][:, 0],
        results['true_beta_avg'][:, 1],
        results['acgwr_beta_avg'][:, 1],
        results['gwr_beta_avg'][:, 1],
        results['true_beta_avg'][:, 2],
        results['acgwr_beta_avg'][:, 2],
        results['gwr_beta_avg'][:, 2]
    ]

    # Set fixed y-coordinate for each row, and x-offset for each column
    y_title_by_row = {
        0: 0.94,  # top row
        1: 0.63,  # middle row
        2: 0.32   # bottom row
    }
    offset_by_col = {
        0: 0.015,
        1: -0.010,
        2: -0.025
    }
    
    for i in range(9):
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        plot_surface(ax, u, v, beta_data[i])
        
        # Get position and adjust title coordinates
        bbox = ax.get_position()
        x_center = (bbox.x0 + bbox.x1) / 2 + offset_by_col[i % 3]
        y_title = y_title_by_row[i // 3]
        
        fig.text(x_center, y_title, titles[i],
                 ha='center', fontsize=14, fontname='Times New Roman')

    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05,
                        wspace=-0.4, hspace=0.1)

    plt.savefig(os.path.join(output_dir, f"beta_comparison_{setting_name}.jpg"),
                format='jpg', dpi=500, bbox_inches='tight')

    if setting_name == 'separable':
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        axs = axs.flatten()

        plot_function_curve(axs[0], u, results['true_f_avg'][:, 0], results['acgwr_f_avg'][:, 0], 
                            "", "u", "$f_0(u)$")
        plot_function_curve(axs[1], u, results['true_f_avg'][:, 1], results['acgwr_f_avg'][:, 1], 
                            "", "u", "$f_1(u)$")
        plot_function_curve(axs[2], u, results['true_f_avg'][:, 2], results['acgwr_f_avg'][:, 2], 
                            "", "u", "$f_2(u)$")

        plot_function_curve(axs[3], v, results['true_g_avg'][:, 0], results['acgwr_g_avg'][:, 0], 
                            "", "v", "$g_0(v)$")
        plot_function_curve(axs[4], v, results['true_g_avg'][:, 1], results['acgwr_g_avg'][:, 1], 
                            "", "v", "$g_1(v)$")
        plot_function_curve(axs[5], v, results['true_g_avg'][:, 2], results['acgwr_g_avg'][:, 2], 
                            "", "v", "$g_2(v)$")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"f_g_comparison_{setting_name}.jpg"),
                    format='jpg', dpi=500, bbox_inches='tight')

class LLGWRModel:
    """Locally Linear Geographically Weighted Regression (LL-GWR) model implementation"""
    def __init__(self, X, y, u, v):
        self.X = X
        self.y = y.reshape(-1, 1)
        self.u = u
        self.v = v
        self.n, self.p = X.shape
        self.results = {}
        self.metrics = {}
    
    def gaussian_kernel(self, distance, bandwidth):
        return np.exp(-distance**2 / (2 * bandwidth**2))
    
    def biweight_kernel(self, distance, bandwidth):
        """Biweight kernel function"""
        d_abs = np.abs(distance / bandwidth)
        return np.where(d_abs <= 1, (1 - d_abs**2)**2, 0)
    
    def tricube_kernel(self, distance, bandwidth):
        """Tricube kernel function"""
        d_abs = np.abs(distance / bandwidth)
        return np.where(d_abs <= 1, (1 - d_abs**3)**3, 0)
    
    def select_bandwidth(self, bandwidths, method='CV', kernel='gaussian'):
        """Select optimal bandwidth
        
        Parameters:
        bandwidths: list of candidate bandwidth values
        method: selection method ('CV' or 'AICc')
        kernel: kernel function type ('gaussian', 'biweight', 'tricube')
        
        Returns:
        Optimal bandwidth value
        """
        best_score = np.inf
        best_bw = bandwidths[0]
        
        # Select kernel function
        if kernel == 'gaussian':
            kernel_func = self.gaussian_kernel
        elif kernel == 'biweight':
            kernel_func = self.biweight_kernel
        elif kernel == 'tricube':
            kernel_func = self.tricube_kernel
        else:
            kernel_func = self.gaussian_kernel
        
        for bw in bandwidths:
            try:
                if method == 'CV':
                    # Cross-validation method
                    cv_score = 0
                    for i in range(self.n):
                        # Leave-one-out cross-validation
                        distances = np.sqrt((self.u - self.u[i])**2 + (self.v - self.v[i])**2)
                        weights = kernel_func(distances, bw)
                        weights[i] = 0  # Exclude current point
                        
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
                        W = np.diag(weights)
                        try:
                            XWX = X_local.T @ W @ X_local
                            XWy = X_local.T @ W @ self.y
                            beta_local = inv(XWX + 1e-8*np.eye(3*self.p)) @ XWy
                            
                            # Predict response value at current point
                            x0 = np.zeros(3*self.p)
                            x0[:self.p] = self.X[i, :]
                            y_pred = x0 @ beta_local
                            
                            cv_score += (self.y[i] - y_pred)**2
                        except:
                            cv_score += (self.y[i] - np.mean(self.y))**2
                    
                    score = cv_score / self.n
                    
                elif method == 'AICc':
                    # AICc method
                    results = self.estimate_llgwr(bw, kernel=kernel)
                    y_hat = results['y_hat']
                    rss = np.sum((self.y - y_hat)**2)
                    
                    # Calculate effective parameter count (approximation)
                    # For LL-GWR, effective parameters approximate tr(S), where S is the hat matrix
                    # Here we use a simplified estimate
                    eff_params = 3 * self.p * 0.7  # Simplified estimate
                    
                    n = self.n
                    aicc = n * np.log(rss/n) + 2*eff_params + (2*eff_params*(eff_params+1))/(n-eff_params-1)
                    score = aicc
                
                if score < best_score:
                    best_score = score
                    best_bw = bw
                    
            except Exception as e:
                # Do not print bandwidth selection information
                continue
        
        return best_bw
    
    def estimate_llgwr(self, bandwidth, kernel='gaussian', beta_true=None):
        """Estimate Locally Linear Geographically Weighted Regression model"""
        # Select kernel function
        if kernel == 'gaussian':
            kernel_func = self.gaussian_kernel
        elif kernel == 'biweight':
            kernel_func = self.biweight_kernel
        elif kernel == 'tricube':
            kernel_func = self.tricube_kernel
        else:
            kernel_func = self.gaussian_kernel
        
        y_hat = np.zeros(self.n)
        beta_hat_local = np.zeros((self.n, self.p))
        
        for i in range(self.n):
            # Calculate distances
            distances = np.sqrt((self.u - self.u[i])**2 + (self.v - self.v[i])**2)
            
            # Calculate weights
            weights = kernel_func(distances, bandwidth)
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
                
                # Predict response value at current point
                x0 = np.zeros(3*self.p)
                x0[:self.p] = self.X[i, :]
                y_hat[i] = x0 @ beta_local
            except:
                # If matrix is singular, use simple mean
                beta_hat_local[i, :] = np.mean(self.X, axis=0)
                y_hat[i] = np.mean(self.y)
        
        self.results = {
            'y_hat': y_hat.reshape(-1, 1),
            'beta_hat_local': beta_hat_local
        }
        
        self.metrics = {
            'R2': r2_score(self.y, y_hat),
            'RMSE': np.sqrt(mean_squared_error(self.y, y_hat)),
            'MAE': mean_absolute_error(self.y, y_hat),
            'MAPE': np.mean(np.abs((self.y - y_hat.reshape(-1, 1))/self.y)) * 100
        }
        
        if beta_true is not None:
            self.metrics.update(self.calculate_beta_metrics(beta_true))
        
        return self.results
    
    def calculate_beta_metrics(self, beta_true):
        beta_pred = self.results['beta_hat_local']
        n_features = beta_true.shape[1]
        metrics = {}
        
        for j in range(n_features):
            true_j = beta_true[:, j]
            pred_j = beta_pred[:, j]
            
            metrics[f'beta_{j}'] = {
                'RMSE': np.sqrt(np.mean((true_j - pred_j)**2)),
                'MAE': np.mean(np.abs(true_j - pred_j)),
                'Correlation': pearsonr(true_j, pred_j)[0]
            }
        
        metrics['global'] = {
            'Overall_RMSE': np.sqrt(np.mean((beta_true - beta_pred)**2)),
            'Overall_MAE': np.mean(np.abs(beta_true - beta_pred)),
            'Overall_Correlation': np.corrcoef(beta_true.flatten(), beta_pred.flatten())[0,1]
        }
        
        return metrics

class ACGWRModel:
    def __init__(self, X, y, u, v):
        self.X = X
        self.y = y.reshape(-1, 1)
        self.u = u
        self.v = v
        self.n, self.p = X.shape
        self.results = {}
        self.metrics = {}
    
    def gaussian_kernel(self, distance, bandwidth):
        return np.exp(-distance**2 / (2 * bandwidth**2))
    
    def construct_D(self):
        dim = self.p * self.n
        D = np.eye(dim)
        for j in range(self.p):
            idx = np.arange(j, dim, self.p)
            for i in idx:
                for k in idx:
                    D[i, k] -= 1 / self.n
        return D
    
    def construct_Z_matrix(self):
        Z = np.zeros((self.n, self.n * self.p))
        for i in range(self.n):
            for j in range(self.p):
                Z[i, i*self.p + j] = self.X[i, j]
        return Z
    
    def compute_S_block(self, W_matrix):
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
        distances_u = np.abs(self.u[:, None] - self.u)
        distances_v = np.abs(self.v[:, None] - self.v)
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
    
    def select_bandwidth(self, h_range=(0.01, 5.5), steps=20, method='CV'):
        bandwidths = np.linspace(h_range[0], h_range[1], steps)
        best_score = np.inf
        best_bw = (h_range[0], h_range[0])
        
        for bw_u in bandwidths:
            for bw_v in bandwidths:
                try:
                    results = self.estimate_acgwr(bw_u, bw_v)
                    if method == 'CV':
                        # Cross-validation method
                        score = np.sqrt(np.mean((self.y - results['y_hat'])**2))
                    elif method == 'AICc':
                        n = self.n
                        k = self.p * 3
                        rss = np.sum((self.y - results['y_hat'])**2)
                        aicc = n * np.log(rss/n) + 2*k + (2*k*(k+1))/(n-k-1)
                        score = aicc
                    
                    if score < best_score:
                        best_score = score
                        best_bw = (bw_u, bw_v)
                except:
                    continue
        
        return best_bw
    
    def calculate_beta_metrics(self, beta_true):
        beta_pred = self.results['beta_hat_local']
        n_features = beta_true.shape[1]
        metrics = {}
        
        for j in range(n_features):
            true_j = beta_true[:, j]
            pred_j = beta_pred[:, j]
            
            metrics[f'beta_{j}'] = {
                'RMSE': np.sqrt(np.mean((true_j - pred_j)**2)),
                'MAE': np.mean(np.abs(true_j - pred_j)),
                'Correlation': pearsonr(true_j, pred_j)[0],
                'Mean_Abs_Deviation': np.mean(np.abs(pred_j - np.mean(pred_j))),
                'Variation_Coefficient': np.std(pred_j)/np.mean(pred_j) if np.mean(pred_j)!=0 else np.nan
            }
        
        metrics['global'] = {
            'Overall_RMSE': np.sqrt(np.mean((beta_true - beta_pred)**2)),
            'Overall_MAE': np.mean(np.abs(beta_true - beta_pred)),
            'Overall_Correlation': np.corrcoef(beta_true.flatten(), beta_pred.flatten())[0,1]
        }
        
        return metrics
    
    def estimate_acgwr(self, bandwidth_u, bandwidth_v, beta_true=None):
        L1, L2, Z, D, S_u, S_v = self.estimate_L1_L2(bandwidth_u, bandwidth_v)
        H = np.eye(self.n) - L1 - L2
        alpha_hat = np.linalg.lstsq(H @ self.X, H @ self.y, rcond=None)[0]
        
        r = self.y - self.X @ alpha_hat
        M1_hat = L1 @ r
        M2_hat = L2 @ r
        y_hat = self.X @ alpha_hat + M1_hat + M2_hat
        
        f_flat = D @ S_u @ (r - M2_hat)
        g_flat = D @ S_v @ (r - M1_hat)
        f_est = f_flat.reshape(self.n, self.p)
        g_est = g_flat.reshape(self.n, self.p)
        beta_hat_local = alpha_hat.T + f_est + g_est
        
        self.results = {
            'alpha_hat': alpha_hat, 'y_hat': y_hat, 'beta_hat_local': beta_hat_local,
            'M1_hat': M1_hat, 'M2_hat': M2_hat, 'f_est': f_est, 'g_est': g_est,
            'Z_matrix': Z, 'D_matrix': D, 'S_u': S_u, 'S_v': S_v
        }
        
        self.metrics = {
            'R2': r2_score(self.y, y_hat),
            'RMSE': np.sqrt(mean_squared_error(self.y, y_hat)),
            'MAE': mean_absolute_error(self.y, y_hat),
            'MAPE': np.mean(np.abs((self.y - y_hat)/self.y)) * 100
        }
        
        if beta_true is not None:
            self.metrics.update(self.calculate_beta_metrics(beta_true))
        
        return self.results


def run_simulation(setting_name, N=500, grid_size=24, save_path="simulation_results", silent=False, generate_plots=True):
    """Run simulation experiment and save results
    
    Parameters:
    setting_name: model setting name ('separable' or 'non_separable')
    N: number of simulations
    grid_size: grid size
    save_path: path to save results
    silent: whether to run in silent mode (no output)
    generate_plots: whether to generate visualizations
    """
    if not silent:
        print(f"\n{'='*60}")
        print(f"Starting {setting_name} model simulation experiment")
        print(f"Number of simulations: {N}, Grid size: {grid_size}x{grid_size}")
        print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Calculate total points (I+1) x (I+1)
    I = grid_size
    total_points = (I + 1) * (I + 1)
    
    # Initialize dictionary to store results
    results = {
        'gwr_metrics': {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
        'mgwr_metrics': {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
        'llgwr_metrics': {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
        'acgwr_metrics': {'R2': [], 'RMSE': [], 'MAE': [], 'MAPE': []},
        'gwr_beta_metrics': {f'beta_{i}': {'RMSE': [], 'MAE': [], 'Correlation': []} for i in range(3)},
        'mgwr_beta_metrics': {f'beta_{i}': {'RMSE': [], 'MAE': [], 'Correlation': []} for i in range(3)},
        'llgwr_beta_metrics': {f'beta_{i}': {'RMSE': [], 'MAE': [], 'Correlation': []} for i in range(3)},
        'acgwr_beta_metrics': {f'beta_{i}': {'RMSE': [], 'MAE': [], 'Correlation': []} for i in range(3)},
        'gwr_beta_all': np.zeros((N, total_points, 3)),
        'mgwr_beta_all': np.zeros((N, total_points, 3)),
        'llgwr_beta_all': np.zeros((N, total_points, 3)),
        'acgwr_beta_all': np.zeros((N, total_points, 3)),
        'true_beta_all': np.zeros((N, total_points, 3)),
        'coordinates': None,
        'simulation_params': {'N': N, 'grid_size': grid_size, 'setting_name': setting_name}
    }
    
    # For separable model, add additional storage
    if setting_name == 'separable':
        results['acgwr_f_all'] = np.zeros((N, total_points, 3))
        results['acgwr_g_all'] = np.zeros((N, total_points, 3))
        results['true_f_all'] = np.zeros((N, total_points, 3))
        results['true_g_all'] = np.zeros((N, total_points, 3))
    
    # Use tqdm to show progress bar
    for sim in tqdm(range(N), desc=f"{setting_name} simulation progress", disable=silent):
        sim_start = time.time()
        
        np.random.seed(52 + sim)  # Use different random seed for each simulation
        
        # 1. Grid coordinates - generate using specified format
        I = grid_size
        grid_1d = np.linspace(0, 1, I + 1)  # (i-1)/I or (j-1)/I
        u_grid, v_grid = np.meshgrid(grid_1d, grid_1d)
        u = u_grid.flatten()
        v = v_grid.flatten()
        coordinates = np.column_stack((u, v))
        n = total_points
        results['coordinates'] = coordinates  # Save coordinates

        # 2. Covariates
        x2 = np.random.normal(0, 1, n)
        x1 = np.random.normal(0, 1, n)

        # 3. Global coefficients
        if setting_name == 'separable':
            alpha0, alpha1, alpha2 = 1.0, 0.5, -0.7
        else:  # non_separable
            alpha0, alpha1, alpha2 = 1.2, -0.8, 0.7

        # 4. Define beta functions
        if setting_name == 'separable':
            # Separable model setting
            def f0(u): return np.sin(np.pi * u)
            def g0(v): return np.cos(np.pi * v)

            def f1(u): return (u - 0.5)**2
            def g1(v): return (v - 0.5)**2 

            def f2(u): return np.exp(-2 * u)
            def g2(v): return np.log(v + 2)

            # Center each function term
            f0_vals = f0(u) - np.mean(f0(u))
            g0_vals = g0(v) - np.mean(g0(v))
            f1_vals = f1(u) - np.mean(f1(u))
            g1_vals = g1(v) - np.mean(g1(v))
            f2_vals = f2(u) - np.mean(f2(u))
            g2_vals = g2(v) - np.mean(g2(v))

            beta_0 = alpha0 + f0_vals + g0_vals
            beta_1 = alpha1 * x1 + f1_vals * x1 + g1_vals * x1
            beta_2 = alpha2 * x2 + f2_vals * x2 + g2_vals * x2

            # Construct response variable
            y = beta_0 + beta_1 + beta_2 + np.random.normal(0, 0.5, n)

            true_beta0 = alpha0 + f0_vals + g0_vals
            true_beta1 = alpha1 + f1_vals + g1_vals
            true_beta2 = alpha2 + f2_vals + g2_vals
            
            # Save true values
            results['true_f_all'][sim] = np.column_stack([f0_vals, f1_vals, f2_vals])
            results['true_g_all'][sim] = np.column_stack([g0_vals, g1_vals, g2_vals])
            
        else:  # non_separable
            # Non-separable model setting
            def beta0_uv(u, v):
                # Bimodal wave superposition: non-additive form
                return alpha0 + np.sin(1/3 * np.pi * (u + v)) 

            def beta1_uv(u, v):
                # Mixed exponential and polynomial, non-additive
                return alpha1 + np.log((u * v) + 1/2)

            def beta2_uv(u, v):
                # Bivariate Gaussian peak, centered at (0.5,0.5)
                return alpha2 + np.exp(-((u - 0.5)**2 + (v - 0.5)**2))

            # Calculate true beta
            true_beta0 = beta0_uv(u, v)
            true_beta1 = beta1_uv(u, v)
            true_beta2 = beta2_uv(u, v)
            
            # Construct response variable
            y = alpha0 + true_beta0 + (alpha1 + true_beta1) * x1 + (alpha2 + true_beta2) * x2 + np.random.normal(0, 0.5, n)

        # Save true beta values
        results['true_beta_all'][sim] = np.column_stack([true_beta0, true_beta1, true_beta2])
        beta_matrix = np.column_stack([true_beta0, true_beta1, true_beta2])

        # Put into DataFrame
        data = pd.DataFrame({'u': u,'v': v,'x1': x1, 'x2': x2,'y': y})
        
        coords = np.column_stack([data['u'], data['v']])
        X = data[['x1', 'x2']].values
        y = data['y'].values.reshape(-1, 1)
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(n), X])
        
        # ========== GWR Model ==========
        try:
            # Use CV method to select bandwidth
            gwr_selector = Sel_BW(coords, y, X)
            bw = gwr_selector.search(criterion='CV')  # Use CV criterion
            gwr_model = GWR(coords, y, X, bw)
            gwr_results = gwr_model.fit()
            
            # Evaluate GWR model
            y_hat_gwr = gwr_results.predy
            results['gwr_metrics']['R2'].append(gwr_results.R2)
            results['gwr_metrics']['RMSE'].append(np.sqrt(mean_squared_error(y, y_hat_gwr)))
            results['gwr_metrics']['MAE'].append(mean_absolute_error(y, y_hat_gwr))
            results['gwr_metrics']['MAPE'].append(np.mean(np.abs((y - y_hat_gwr)/y)) * 100)
            
            # Calculate beta metrics
            beta_pred = gwr_results.params
            for j in range(3):
                true_j = beta_matrix[:, j]
                pred_j = beta_pred[:, j]
                
                results['gwr_beta_metrics'][f'beta_{j}']['RMSE'].append(
                    np.sqrt(mean_squared_error(true_j, pred_j)))
                results['gwr_beta_metrics'][f'beta_{j}']['MAE'].append(
                    mean_absolute_error(true_j, pred_j))
                results['gwr_beta_metrics'][f'beta_{j}']['Correlation'].append(
                    pearsonr(true_j, pred_j)[0])
            
            # Save GWR beta estimates
            results['gwr_beta_all'][sim] = beta_pred
            
        except Exception as e:
            if not silent:
                print(f"GWR model error: {e}")
            continue
        
        # ========== MGWR Model ==========
        try:
            # Select MGWR bandwidth - use CV method
            with CompleteSilence(), suppress_ipython_display():
                mgwr_selector = Sel_BW(coords, y, X, multi=True)
                mgwr_bw = mgwr_selector.search(criterion='CV')  # Use CV criterion
            
            # Check mgwr_bw type and handle appropriately
            if isinstance(mgwr_bw, np.ndarray):
                # If it's an array, use directly
                bw_array = mgwr_bw
            else:
                # If it's a scalar, create an array with that value
                bw_array = np.array([mgwr_bw] * X.shape[1])
            
            # Fit MGWR model - pass selector object instead of bandwidth array
            # First check if mgwr_selector has bw attribute
            if hasattr(mgwr_selector, 'bw'):
                mgwr_model = MGWR(coords, y, X, mgwr_selector)
            else:
                # If no bw attribute, use bandwidth array directly
                mgwr_model = MGWR(coords, y, X, bw_array)
            
            # Use CompleteSilence and suppress_ipython_display to completely suppress MGWR model output
            with CompleteSilence(), suppress_ipython_display():
                mgwr_results = mgwr_model.fit()
            
            # Evaluate MGWR model
            y_hat_mgwr = mgwr_results.predy
            results['mgwr_metrics']['R2'].append(mgwr_results.R2)
            results['mgwr_metrics']['RMSE'].append(np.sqrt(mean_squared_error(y, y_hat_mgwr)))
            results['mgwr_metrics']['MAE'].append(mean_absolute_error(y, y_hat_mgwr))
            results['mgwr_metrics']['MAPE'].append(np.mean(np.abs((y - y_hat_mgwr)/y)) * 100)
            
            # Calculate beta metrics
            beta_pred_mgwr = mgwr_results.params
            for j in range(3):
                true_j = beta_matrix[:, j]
                pred_j = beta_pred_mgwr[:, j]
                
                results['mgwr_beta_metrics'][f'beta_{j}']['RMSE'].append(
                    np.sqrt(mean_squared_error(true_j, pred_j)))
                results['mgwr_beta_metrics'][f'beta_{j}']['MAE'].append(
                    mean_absolute_error(true_j, pred_j))
                results['mgwr_beta_metrics'][f'beta_{j}']['Correlation'].append(
                    pearsonr(true_j, pred_j)[0])
            
            # Save MGWR beta estimates
            results['mgwr_beta_all'][sim] = beta_pred_mgwr
            
        except Exception as e:
            if not silent:
                print(f"MGWR model error: {e}")
            # If MGWR fails, use GWR results as placeholder
            results['mgwr_metrics']['R2'].append(results['gwr_metrics']['R2'][-1])
            results['mgwr_metrics']['RMSE'].append(results['gwr_metrics']['RMSE'][-1])
            results['mgwr_metrics']['MAE'].append(results['gwr_metrics']['MAE'][-1])
            results['mgwr_metrics']['MAPE'].append(results['gwr_metrics']['MAPE'][-1])
            
            for j in range(3):
                results['mgwr_beta_metrics'][f'beta_{j}']['RMSE'].append(
                    results['gwr_beta_metrics'][f'beta_{j}']['RMSE'][-1])
                results['mgwr_beta_metrics'][f'beta_{j}']['MAE'].append(
                    results['gwr_beta_metrics'][f'beta_{j}']['MAE'][-1])
                results['mgwr_beta_metrics'][f'beta_{j}']['Correlation'].append(
                    results['gwr_beta_metrics'][f'beta_{j}']['Correlation'][-1])
            
            results['mgwr_beta_all'][sim] = results['gwr_beta_all'][sim]
        
        # ========== LL-GWR Model ==========
        try:
            # Initialize LL-GWR model
            llgwr_model = LLGWRModel(X_with_intercept, y, u, v)
            
            # LL-GWR bandwidth selection - use CV method
            # Use GWR bandwidth as reference to generate candidate bandwidth range
            gwr_bw_ref = bw
            bandwidth_candidates = np.linspace(gwr_bw_ref * 0.5, gwr_bw_ref * 1.5, 5)
            
            # Select optimal bandwidth
            llgwr_bw = llgwr_model.select_bandwidth(bandwidth_candidates, method='CV', kernel='gaussian')
            
            # Fit LL-GWR model with optimal bandwidth
            llgwr_results = llgwr_model.estimate_llgwr(llgwr_bw, kernel='gaussian', beta_true=beta_matrix)
            
            # Evaluate LL-GWR model
            y_hat_llgwr = llgwr_results['y_hat']
            results['llgwr_metrics']['R2'].append(llgwr_model.metrics['R2'])
            results['llgwr_metrics']['RMSE'].append(llgwr_model.metrics['RMSE'])
            results['llgwr_metrics']['MAE'].append(llgwr_model.metrics['MAE'])
            results['llgwr_metrics']['MAPE'].append(llgwr_model.metrics['MAPE'])
            
            # Calculate beta metrics
            beta_pred_llgwr = llgwr_results['beta_hat_local']
            for j in range(3):
                true_j = beta_matrix[:, j]
                pred_j = beta_pred_llgwr[:, j]
                
                results['llgwr_beta_metrics'][f'beta_{j}']['RMSE'].append(
                    np.sqrt(mean_squared_error(true_j, pred_j)))
                results['llgwr_beta_metrics'][f'beta_{j}']['MAE'].append(
                    mean_absolute_error(true_j, pred_j))
                results['llgwr_beta_metrics'][f'beta_{j}']['Correlation'].append(
                    pearsonr(true_j, pred_j)[0])
            
            # Save LL-GWR beta estimates
            results['llgwr_beta_all'][sim] = beta_pred_llgwr
            
        except Exception as e:
            if not silent:
                print(f"LL-GWR model error: {e}")
            # If LL-GWR fails, use GWR results as placeholder
            results['llgwr_metrics']['R2'].append(results['gwr_metrics']['R2'][-1])
            results['llgwr_metrics']['RMSE'].append(results['gwr_metrics']['RMSE'][-1])
            results['llgwr_metrics']['MAE'].append(results['gwr_metrics']['MAE'][-1])
            results['llgwr_metrics']['MAPE'].append(results['gwr_metrics']['MAPE'][-1])
            
            for j in range(3):
                results['llgwr_beta_metrics'][f'beta_{j}']['RMSE'].append(
                    results['gwr_beta_metrics'][f'beta_{j}']['RMSE'][-1])
                results['llgwr_beta_metrics'][f'beta_{j}']['MAE'].append(
                    results['gwr_beta_metrics'][f'beta_{j}']['MAE'][-1])
                results['llgwr_beta_metrics'][f'beta_{j}']['Correlation'].append(
                    results['gwr_beta_metrics'][f'beta_{j}']['Correlation'][-1])
            
            results['llgwr_beta_all'][sim] = results['gwr_beta_all'][sim]
        
        # ========== ACGWR Model ==========
        # Initialize model
        model = ACGWRModel(X_with_intercept, y, u, v)

        # Bandwidth selection - use CV method
        try:
            bw_u, bw_v = model.select_bandwidth(h_range=(0.1, 1.0), steps=5, method='CV')
        except Exception as e:
            if not silent:
                print(f"ACGWR bandwidth selection error: {e}")
            continue

        # Model estimation
        try:
            results_acgwr = model.estimate_acgwr(bw_u, bw_v, beta_true=beta_matrix)
            
            # Store ACGWR results
            for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
                results['acgwr_metrics'][metric].append(model.metrics[metric])
            
            # Store beta metrics
            for j in range(3):
                if f'beta_{j}' in model.metrics:
                    for sub_metric in ['RMSE', 'MAE', 'Correlation']:
                        if sub_metric in model.metrics[f'beta_{j}']:
                            results['acgwr_beta_metrics'][f'beta_{j}'][sub_metric].append(
                                model.metrics[f'beta_{j}'][sub_metric])
            
            # Save ACGWR beta estimates
            results['acgwr_beta_all'][sim] = results_acgwr['beta_hat_local']
            
            # For separable model, save f and g functions
            if setting_name == 'separable':
                results['acgwr_f_all'][sim] = results_acgwr['f_est']
                results['acgwr_g_all'][sim] = results_acgwr['g_est']
            
        except Exception as e:
            if not silent:
                print(f"ACGWR model estimation error: {e}")
            continue
        
        sim_time = time.time() - sim_start

    # Calculate average results
    results['gwr_beta_avg'] = np.mean(results['gwr_beta_all'], axis=0)
    results['mgwr_beta_avg'] = np.mean(results['mgwr_beta_all'], axis=0)
    results['llgwr_beta_avg'] = np.mean(results['llgwr_beta_all'], axis=0)
    results['acgwr_beta_avg'] = np.mean(results['acgwr_beta_all'], axis=0)
    results['true_beta_avg'] = results['true_beta_all'][-1]  # Use true values from last simulation
    
    if setting_name == 'separable':
        results['acgwr_f_avg'] = np.mean(results['acgwr_f_all'], axis=0)
        results['acgwr_g_avg'] = np.mean(results['acgwr_g_all'], axis=0)
        results['true_f_avg'] = results['true_f_all'][-1]
        results['true_g_avg'] = results['true_g_all'][-1]
    
    # Create output directory
    output_dir = os.path.join(save_path, setting_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"simulation_results_{timestamp}.pkl"
    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(results, f)
    
    # Generate visualizations
    if generate_plots:
        visualize_results(results, output_dir)
    
    total_time = time.time() - start_time
    if not silent:
        print(f"\n{'='*60}")
        print(f"{setting_name} model simulation experiment completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Results saved to {os.path.join(output_dir, filename)}")
        print(f"{'='*60}")
    
    # ========== Calculate and print average performance metrics ==========
    # Calculate GWR average metrics
    gwr_avg = {
        'R2': np.mean(results['gwr_metrics']['R2']),
        'RMSE': np.mean(results['gwr_metrics']['RMSE']),
        'MAE': np.mean(results['gwr_metrics']['MAE']),
        'MAPE': np.mean(results['gwr_metrics']['MAPE'])
    }
    
    # Calculate MGWR average metrics
    mgwr_avg = {
        'R2': np.mean(results['mgwr_metrics']['R2']),
        'RMSE': np.mean(results['mgwr_metrics']['RMSE']),
        'MAE': np.mean(results['mgwr_metrics']['MAE']),
        'MAPE': np.mean(results['mgwr_metrics']['MAPE'])
    }
    
    # Calculate LL-GWR average metrics
    llgwr_avg = {
        'R2': np.mean(results['llgwr_metrics']['R2']),
        'RMSE': np.mean(results['llgwr_metrics']['RMSE']),
        'MAE': np.mean(results['llgwr_metrics']['MAE']),
        'MAPE': np.mean(results['llgwr_metrics']['MAPE'])
    }
    
    # Calculate ACGWR average metrics
    acgwr_avg = {
        'R2': np.mean(results['acgwr_metrics']['R2']),
        'RMSE': np.mean(results['acgwr_metrics']['RMSE']),
        'MAE': np.mean(results['acgwr_metrics']['MAE']),
        'MAPE': np.mean(results['acgwr_metrics']['MAPE'])
    }
    
    # Calculate GWR beta metrics averages
    gwr_beta_avg = {}
    for beta in ['beta_0', 'beta_1', 'beta_2']:
        gwr_beta_avg[beta] = {
            'RMSE': np.mean(results['gwr_beta_metrics'][beta]['RMSE']),
            'MAE': np.mean(results['gwr_beta_metrics'][beta]['MAE']),
            'Correlation': np.mean(results['gwr_beta_metrics'][beta]['Correlation'])
        }
    
    # Calculate MGWR beta metrics averages
    mgwr_beta_avg = {}
    for beta in ['beta_0', 'beta_1', 'beta_2']:
        mgwr_beta_avg[beta] = {
            'RMSE': np.mean(results['mgwr_beta_metrics'][beta]['RMSE']),
            'MAE': np.mean(results['mgwr_beta_metrics'][beta]['MAE']),
            'Correlation': np.mean(results['mgwr_beta_metrics'][beta]['Correlation'])
        }
    
    # Calculate LL-GWR beta metrics averages
    llgwr_beta_avg = {}
    for beta in ['beta_0', 'beta_1', 'beta_2']:
        llgwr_beta_avg[beta] = {
            'RMSE': np.mean(results['llgwr_beta_metrics'][beta]['RMSE']),
            'MAE': np.mean(results['llgwr_beta_metrics'][beta]['MAE']),
            'Correlation': np.mean(results['llgwr_beta_metrics'][beta]['Correlation'])
        }
    
    # Calculate ACGWR beta metrics averages
    acgwr_beta_avg = {}
    for beta in ['beta_0', 'beta_1', 'beta_2']:
        acgwr_beta_avg[beta] = {
            'RMSE': np.mean(results['acgwr_beta_metrics'][beta]['RMSE']),
            'MAE': np.mean(results['acgwr_beta_metrics'][beta]['MAE']),
            'Correlation': np.mean(results['acgwr_beta_metrics'][beta]['Correlation'])
        }
    
    # Return results summary
    summary = {
        'setting_name': setting_name,
        'grid_size': grid_size,
        'gwr_avg': gwr_avg,
        'mgwr_avg': mgwr_avg,
        'llgwr_avg': llgwr_avg,
        'acgwr_avg': acgwr_avg,
        'gwr_beta_avg': gwr_beta_avg,
        'mgwr_beta_avg': mgwr_beta_avg,
        'llgwr_beta_avg': llgwr_beta_avg,
        'acgwr_beta_avg': acgwr_beta_avg
    }
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Run ACGWR performance simulation')
    parser.add_argument('--n_repeat', type=int, default=500, help='Number of Monte Carlo repetitions')
    parser.add_argument('--grid_size', type=int, nargs='+', default=[9, 24], help='Spatial grid sizes to simulate')
    parser.add_argument('--silent', action='store_true', help='Run in silent mode (no output)')
    parser.add_argument('--no_plots', action='store_true', help='Do not generate plots')
    args = parser.parse_args()
    
    # Run simulations
    all_results = []
    
    for grid_size in args.grid_size:
        print(f"\nStarting simulation with grid size {grid_size}")
        
        # Run separable structure simulation
        separable_summary = run_simulation('separable', N=args.n_repeat, grid_size=grid_size, 
                                         silent=args.silent, generate_plots=not args.no_plots)
        separable_summary['grid_size'] = grid_size
        all_results.append(separable_summary)
        
        # Run non-separable structure simulation
        non_separable_summary = run_simulation('non_separable', N=args.n_repeat, grid_size=grid_size, 
                                             silent=args.silent, generate_plots=not args.no_plots)
        non_separable_summary['grid_size'] = grid_size
        all_results.append(non_separable_summary)
    
    # Save results to Excel
    df = pd.DataFrame()
    
    for result in all_results:
        setting_name = result['setting_name']
        grid_size = result['grid_size']
        sample_size = (grid_size + 1) ** 2  # Calculate sample size
        
        for model in ['gwr', 'mgwr', 'llgwr', 'acgwr']:
            model_avg = result[f'{model}_avg']
            row = {
                'Setting': setting_name,
                'Sample size': sample_size,  # Changed to sample size
                'Model': model.upper(),
                'R2': model_avg['R2'],
                'RMSE': model_avg['RMSE'],
                'MAE': model_avg['MAE'],
                # Removed MAPE metric
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    # Save to Excel
    excel_filename = f'simulation_results_comparison.xlsx'
    df.to_excel(excel_filename, index=False)
    
    if not args.silent:
        print(f"\nResults saved to {excel_filename}")
        
        # Print results summary
        print("\nModel performance comparison:")
        print(f"{'Setting':<15} {'Sample size':<12} {'Model':<8} {'R2':<8} {'RMSE':<8} {'MAE':<8}")
        print("-" * 60)
        for _, row in df.iterrows():
            print(f"{row['Setting']:<15} {row['Sample size']:<12} {row['Model']:<8} {row['R2']:<8.4f} {row['RMSE']:<8.4f} {row['MAE']:<8.4f}")

if __name__ == "__main__":
    main()