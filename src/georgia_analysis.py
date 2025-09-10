import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.linalg import inv
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR, MGWR
import mapclassify
import warnings
import os  # Add this import
warnings.filterwarnings("ignore")

# Create directory for images if it doesn't exist
os.makedirs('georgia_plots', exist_ok=True)

# ======= Model Class Definitions =======

class ACGWRModel:  # Changed from AGWRModel to ACGWRModel
    """Adaptive Geographically Weighted Regression Model"""  # English comment
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
                XtWX_inv = np.linalg.inv(XtWX + 1e-8 * np.eye(self.p))
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

    def select_bandwidth(self, h_range=(0.01, 1.5), steps=10, method='AICc'):
        bandwidths = np.linspace(h_range[0], h_range[1], steps)
        best_score = np.inf
        best_bw = (bandwidths[0], bandwidths[0])
        
        for bw_u in bandwidths:
            for bw_v in bandwidths:
                try:
                    results = self.estimate_acgwr(bw_u, bw_v)  # Changed from estimate_agwr
                    if method == 'CV':
                        score = -results['metrics']['R2']
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

    def estimate_acgwr(self, bandwidth_u, bandwidth_v, beta_true=None):  # Changed from estimate_agwr
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
            'alpha_hat': alpha_hat,
            'y_hat': y_hat,
            'beta_hat_local': beta_hat_local,
            'M1_hat': M1_hat,
            'M2_hat': M2_hat,
            'f_est': f_est,
            'g_est': g_est,
            'Z_matrix': Z,
            'D_matrix': D,
            'S_u': S_u,
            'S_v': S_v
        }

        self.metrics = {
            'R2': r2_score(self.y, y_hat),
            'RMSE': np.sqrt(mean_squared_error(self.y, y_hat)),
            'MAE': mean_absolute_error(self.y, y_hat),
            'MAPE': np.mean(np.abs((self.y - y_hat) / self.y)) * 100
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
                'RMSE': np.sqrt(np.mean((true_j - pred_j) ** 2)),
                'MAE': np.mean(np.abs(true_j - pred_j)),
                'Correlation': pearsonr(true_j, pred_j)[0]
            }

        metrics['global'] = {
            'Overall_RMSE': np.sqrt(np.mean((beta_true - beta_pred) ** 2)),
            'Overall_MAE': np.mean(np.abs(beta_true - beta_pred)),
            'Overall_Correlation': np.corrcoef(beta_true.flatten(), beta_pred.flatten())[0, 1]
        }

        return metrics

class LLGWRModel:
    """Locally Linear Geographically Weighted Regression Model"""  # English comment
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
    
    def select_bandwidth(self, bandwidths, method='CV', kernel='gaussian'):
        best_score = np.inf
        best_bw = bandwidths[0]
        
        kernel_func = self.gaussian_kernel
        
        for bw in bandwidths:
            try:
                if method == 'CV':
                    cv_score = 0
                    for i in range(self.n):
                        distances = np.sqrt((self.u - self.u[i])**2 + (self.v - self.v[i])**2)
                        weights = kernel_func(distances, bw)
                        weights[i] = 0
                        
                        delta_u = self.u - self.u[i]
                        delta_v = self.v - self.v[i]
                        
                        X_local = np.zeros((self.n, 3*self.p))
                        for j in range(self.p):
                            X_local[:, j] = self.X[:, j]
                            X_local[:, self.p + j] = self.X[:, j] * delta_u
                            X_local[:, 2*self.p + j] = self.X[:, j] * delta_v
                        
                        W = np.diag(weights)
                        try:
                            XWX = X_local.T @ W @ X_local
                            XWy = X_local.T @ W @ self.y
                            beta_local = inv(XWX + 1e-8*np.eye(3*self.p)) @ XWy
                            
                            x0 = np.zeros(3*self.p)
                            x0[:self.p] = self.X[i, :]
                            y_pred = x0 @ beta_local
                            
                            cv_score += (self.y[i] - y_pred)**2
                        except:
                            cv_score += (self.y[i] - np.mean(self.y))**2
                    
                    score = cv_score / self.n
                    
                elif method == 'AICc':
                    results = self.estimate_llgwr(bw, kernel=kernel)
                    y_hat = results['y_hat']
                    rss = np.sum((self.y - y_hat)**2)
                    
                    eff_params = 3 * self.p * 0.7
                    
                    n = self.n
                    aicc = n * np.log(rss/n) + 2*eff_params + (2*eff_params*(eff_params+1))/(n-eff_params-1)
                    score = aicc
                
                if score < best_score:
                    best_score = score
                    best_bw = bw
                    
            except Exception as e:
                continue
        
        return best_bw
    
    def estimate_llgwr(self, bandwidth, kernel='gaussian', beta_true=None):
        kernel_func = self.gaussian_kernel
        
        y_hat = np.zeros(self.n)
        beta_hat_local = np.zeros((self.n, self.p))
        
        for i in range(self.n):
            distances = np.sqrt((self.u - self.u[i])**2 + (self.v - self.v[i])**2)
            
            weights = kernel_func(distances, bandwidth)
            W = np.diag(weights)
            
            delta_u = self.u - self.u[i]
            delta_v = self.v - self.v[i]
            
            X_local = np.zeros((self.n, 3*self.p))
            for j in range(self.p):
                X_local[:, j] = self.X[:, j]
                X_local[:, self.p + j] = self.X[:, j] * delta_u
                X_local[:, 2*self.p + j] = self.X[:, j] * delta_v
            
            try:
                XWX = X_local.T @ W @ X_local
                XWy = X_local.T @ W @ self.y
                beta_local = inv(XWX + 1e-8*np.eye(3*self.p)) @ XWy
                
                beta_hat_local[i, :] = beta_local[:self.p].flatten()
                
                x0 = np.zeros(3*self.p)
                x0[:self.p] = self.X[i, :]
                y_hat[i] = x0 @ beta_local
            except:
                beta_hat_local[i, :] = np.mean(self.X, axis=0)
                y_hat[i] = np.mean(self.y)
        
        self.results = {
            'y_hat': y_hat.reshape(-1, 1),
            'beta_hat_local': beta_hat_local
        }
        
        self.metrics = {
            'R2': r2_score(self.y, y_hat),
            'RMSE': np.sqrt(mean_squared_error(self.y, y_hat)),
            'MAE': mean_absolute_error(self.y, y_hat)
        }
        
        return self.results

# ======= Helper Functions =======

def add_scalebar(ax, x0, y0, length_km, units_per_km, ticks_km=None, tick_labels=None, 
                 tick_height_km=1.0, label_offset_km=0.002, fontsize=12, color='black'):
    length_data = length_km * units_per_km
    if ticks_km is not None:
        ticks_data = [x0 + t * units_per_km for t in ticks_km]
    else:
        ticks_data = None
    
    tick_height_data = tick_height_km * units_per_km
    label_offset_data = label_offset_km * units_per_km

    ax.plot([x0, x0 + length_data], [y0, y0], color=color, linewidth=2, zorder=10)
    
    if ticks_data is not None:
        for i, tick in enumerate(ticks_data):
            ax.plot([tick, tick], [y0+3000, y0 + tick_height_data + 8000], color=color, linewidth=1, zorder=10)
            
            if tick_labels is not None and i < len(tick_labels):
                label = tick_labels[i]
                ax.text(tick, y0 + 20000, label, 
                        ha='center', va='bottom', fontsize=fontsize, color=color, zorder=10)
    
    ax.text(x0 + length_data + 30000, y0 + 20000, 'km', 
            ha='left', va='bottom', fontsize=fontsize, color=color, zorder=10)

def add_north_arrow(ax, x_pos=0.88, y_pos=0.85, labelsize=18, width=0.06, height=0.09, pad=0.14):
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    xlen = maxx - minx
    ylen = maxy - miny

    left = [minx + xlen * (x_pos - width * 0.5), miny + ylen * (y_pos - pad)]
    right = [minx + xlen * (x_pos + width * 0.5), miny + ylen * (y_pos - pad)]
    top = [minx + xlen * x_pos, miny + ylen * (y_pos - pad + height)]
    center = [minx + xlen * x_pos, left[1] + (top[1] - left[1]) * 0.4]

    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.add_patch(triangle)

    ax.text(s='N',
            x=minx + xlen * x_pos,
            y=miny + ylen * (y_pos - pad + height),
            fontsize=labelsize,
            ha='center',
            va='bottom')

def plot_f_g_beta(f_est, g_est, beta_gdf, coord_u, coord_v, 
                  feature_names1, feature_names2, beta_varnames, legend_titles,
                  output_prefix):
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 14
    })
    n_features = f_est.shape[1]

    for i in range(n_features):
        fig = plt.figure(figsize=(18, 5))
        
        ax_f = fig.add_axes([0.05, 0.15, 0.2, 0.65])
        ax_g = fig.add_axes([0.3, 0.15, 0.2, 0.65])
        ax_beta = fig.add_axes([0.5, 0.15, 0.3, 0.65])

        pos_beta = ax_beta.get_position()
        ax_beta.set_position([pos_beta.x0 - 0.03, pos_beta.y0, pos_beta.width, pos_beta.height])

        # Plot f_i(u)
        idx_u = np.argsort(coord_u)
        u_sorted = coord_u[idx_u]
        f_sorted = f_est[idx_u, i]
        ax_f.plot(u_sorted, f_sorted, color='blue')
        ax_f.set_xlabel('u')
        ax_f.set_ylabel(feature_names1[i])

        # Plot g_i(v)
        idx_v = np.argsort(coord_v)
        v_sorted = coord_v[idx_v]
        g_sorted = g_est[idx_v, i]
        ax_g.plot(v_sorted, g_sorted, color='green')
        ax_g.set_xlabel('v')
        ax_g.set_ylabel(feature_names2[i])

        # Synchronize y-axis
        y_min = min(ax_f.get_ylim()[0], ax_g.get_ylim()[0])
        y_max = max(ax_f.get_ylim()[1], ax_g.get_ylim()[1])
        ax_f.set_ylim(y_min, y_max)
        ax_g.set_ylim(y_min, y_max)

        # Plot beta spatial distribution
        varname = beta_varnames[i]

        n_classes = 6
        classifier = mapclassify.classify(beta_gdf[varname].dropna(), scheme='Quantiles', k=n_classes)
        bins = np.append(classifier.bins, beta_gdf[varname].max())

        cmap = plt.cm.coolwarm
        norm = BoundaryNorm(boundaries=bins, ncolors=cmap.N)

        beta_gdf.plot(
            column=varname,
            ax=ax_beta,
            cmap=cmap,
            norm=norm,
            edgecolor='white',
            legend=False
        )

        # Legend
        legend_labels = [f"{bins[j]:.2f} - {bins[j+1]:.2f}" for j in range(n_classes)]
        patches = [mpatches.Patch(color=cmap(norm(bins[j] + 1e-8)), label=legend_labels[j]) for j in range(n_classes)]

        xmin, xmax = ax_beta.get_xlim()
        ymin, ymax = ax_beta.get_ylim()

        legend = ax_beta.legend(
            handles=patches,
            title=legend_titles[i],
            loc='lower right',
            bbox_to_anchor=(0.98, 0.13),
            handlelength=1,
            borderpad=0.5,
            labelspacing=0.5,
            handletextpad=0.5,
            fontsize=9.5,
            title_fontsize=9.5,
            frameon=True,
            framealpha=1
        )
        
        frame = legend.get_frame()
        frame.set_edgecolor('black')
        frame.set_linewidth(0.8)
        frame.set_facecolor('white')
        frame.set_alpha(1)

        # Adjust axis range
        xpad_left = (xmax - xmin) * 0.05
        xpad_right = (xmax - xmin) * 0.45
        ypad = (ymax - ymin) * 0.1
        ax_beta.set_xlim(xmin - xpad_left, xmax + xpad_right)
        ax_beta.set_ylim(ymin - ypad, ymax + ypad)

        # Add border
        rect = Rectangle(
            (xmin - xpad_left, ymin - ypad),
            (xmax - xmin) + xpad_left + xpad_right,
            (ymax - ymin) + 2 * ypad,
            linewidth=0.8,
            edgecolor='black',
            facecolor='none',
            zorder=20
        )
        ax_beta.add_patch(rect)

        # Turn off axis
        ax_beta.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Add scalebar and north arrow
        add_scalebar(
            ax_beta,
            x0=xmin + (xmax - xmin) * 0.1,
            y0=ymin - (ymax - ymin) * 0.05,
            length_km=200,
            units_per_km=1000,
            ticks_km=[0, 100, 200],
            tick_labels=['0', '100', '200'],
            fontsize=12
        )
        add_north_arrow(ax_beta, x_pos=0.518, y_pos=0.88)

        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.1, wspace=1.1)
        plt.savefig(f'georgia_plots/{output_prefix}_feature_{i}.jpg', dpi=300, bbox_inches='tight')  # Changed path
        plt.close()

# ======= Main Function =======

def run_georgia_analysis():
    """Run Georgia dataset analysis"""  # English comment
    print("Loading data...")
    
    # 1. Load data
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data', 'georgia')
    
    georgia_data = pd.read_csv(os.path.join(data_path, "GData_utm.csv"))
    georgia_shp = gpd.read_file(os.path.join(data_path, 'G_utm.shp'))

    # 2. Define variables
    g_y = georgia_data['PctBach'].values.reshape(-1, 1)
    g_X = georgia_data[['TotPop90', 'PctRural', 'PctEld', 'PctFB', 'PctPov', 'PctBlack']].values

    # 3. Read coordinates
    u = georgia_data['Longitud'].values
    v = georgia_data['Latitude'].values
    coords = np.column_stack((u, v))

    # 4. Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    g_X_std = scaler_X.fit_transform(g_X)
    g_y_std = scaler_y.fit_transform(g_y)

    # 5. Add intercept
    X_with_intercept = np.hstack([np.ones((g_X_std.shape[0], 1)), g_X_std])

    # ======= Run four models =======
    results = {}
    
    print("Running ACGWR model...")  # Changed from AGWR
    # ACGWR model
    acgwr_model = ACGWRModel(X_with_intercept, g_y_std, u, v)  # Changed from AGWRModel
    bw_u, bw_v = acgwr_model.select_bandwidth(h_range=(0.1, 15), steps=20, method='AICc')
    acgwr_results = acgwr_model.estimate_acgwr(bw_u, bw_v)  # Changed from estimate_agwr
    results['ACGWR'] = acgwr_model.metrics  # Changed from AGWR
    print(f"ACGWR completed: R²={acgwr_model.metrics['R2']:.4f}")  # Changed from AGWR

    print("Running GWR model...")
    # GWR model
    gwr_selector = Sel_BW(coords, g_y_std, g_X_std)
    gwr_bw = gwr_selector.search()
    gwr_model = GWR(coords, g_y_std, g_X_std, gwr_bw)
    gwr_results = gwr_model.fit()
    y_pred_gwr = gwr_results.predy
    results['GWR'] = {
        'R2': gwr_results.R2,
        'RMSE': np.sqrt(mean_squared_error(g_y_std, y_pred_gwr)),
        'MAE': mean_absolute_error(g_y_std, y_pred_gwr)
    }
    print(f"GWR completed: R²={gwr_results.R2:.4f}")

    print("Running LL-GWR model...")
    # LL-GWR model
    llgwr_model = LLGWRModel(X_with_intercept, g_y_std, u, v)
    llgwr_bandwidths = np.linspace(gwr_bw * 0.5, gwr_bw * 1.5, 5)
    llgwr_bw = llgwr_model.select_bandwidth(llgwr_bandwidths, method='CV', kernel='gaussian')
    llgwr_results = llgwr_model.estimate_llgwr(llgwr_bw, kernel='gaussian')
    results['LL-GWR'] = llgwr_model.metrics
    print(f"LL-GWR completed: R²={llgwr_model.metrics['R2']:.4f}")

    print("Running MGWR model...")
    # MGWR model
    mgwr_selector = Sel_BW(coords, g_y_std, g_X_std, multi=True)
    mgwr_bw = mgwr_selector.search()
    mgwr_model = MGWR(coords, g_y_std, g_X_std, mgwr_selector)
    mgwr_results = mgwr_model.fit()
    y_pred_mgwr = mgwr_results.predy
    results['MGWR'] = {
        'R2': mgwr_results.R2,
        'RMSE': np.sqrt(mean_squared_error(g_y_std, y_pred_mgwr)),
        'MAE': mean_absolute_error(g_y_std, y_pred_mgwr)
    }
    print(f"MGWR completed: R²={mgwr_results.R2:.4f}")

    # ======= Save results to Excel =======
    df_results = pd.DataFrame(results).T
    df_results = df_results[['R2', 'RMSE', 'MAE']]  # Keep only three metrics
    df_results.to_excel('georgia_model_comparison.xlsx')
    print("Model performance comparison saved to: georgia_model_comparison.xlsx")

    # ======= Generate visualizations =======
    print("Generating visualizations...")
    
    # Prepare ACGWR result data
    f_est = acgwr_results['f_est']  # Changed from agwr_results
    g_est = acgwr_results['g_est']  # Changed from agwr_results
    beta_hat_local = acgwr_results['beta_hat_local']  # Changed from agwr_results
    
    # Save coefficients to CSV
    beta_df = pd.DataFrame(beta_hat_local, columns=['Intercept', 'TotPop90', 'PctRural', 'PctEld', 'PctFB', 'PctPov', 'PctBlack'])
        
    # Link with geographic data
    sam_coefficent = pd.DataFrame(beta_hat_local, 
                                 columns=['cof_Intercept', 'cof_TotPop90', 'cof_PctRural',
                                          'cof_PctEld', 'cof_PctFB', 'cof_PctPov', 'cof_PctBlack'])
    
    georgia_data_geo = gpd.GeoDataFrame(
        georgia_data,
        geometry=gpd.points_from_xy(georgia_data.X, georgia_data.Y)
    ).join(sam_coefficent)

    beta_gdf = gpd.sjoin(
        georgia_shp, 
        georgia_data_geo, 
        how="inner", 
        predicate='intersects'
    ).reset_index()

    # Define plot parameters
    legend_titles = [
        r'$β_0(u,v)$', r'$β_1(u,v)$', r'$β_2(u,v)$',
        r'$β_3(u,v)$', r'$β_4(u,v)$', r'$β_5(u,v)$', r'$β_6(u,v)$'
    ]
    
    feature_names1 = ["$f_{0}(u)$", "$f_{1}(u)$", "$f_{2}(u)$", 
                     "$f_{3}(u)$", "$f_{4}(u)$", "$f_{5}(u)$", "$f_{6}(u)$"]
    
    feature_names2 = ["$g_{0}(v)$", "$g_{1}(v)$", "$g_{2}(v)$", 
                     "$g_{3}(v)$", "$g_{4}(v)$", "$g_{5}(v)$", "$g_{6}(v)$"]
    
    beta_varnames = ['cof_Intercept', 'cof_TotPop90', 'cof_PctRural',
                    'cof_PctEld', 'cof_PctFB', 'cof_PctPov', 'cof_PctBlack']

    # Generate 7 images (6 variables + 1 summary)
    plot_f_g_beta(
        f_est=f_est,
        g_est=g_est,
        beta_gdf=beta_gdf,
        coord_u=coords[:, 0],
        coord_v=coords[:, 1],
        feature_names1=feature_names1,
        feature_names2=feature_names2,
        beta_varnames=beta_varnames,
        legend_titles=legend_titles,
        output_prefix='georgia_acgwr'  # Changed from agwr to acgwr
    )
    
    print("Visualizations generated!")
    print("Output files:")
    print("1. georgia_model_comparison.xlsx - Performance comparison of four models")
    print("2. georgia_acgwr_coefficients.csv - Local coefficient estimates of ACGWR model")  # Changed from AGWR
    print("3. georgia_plots/georgia_acgwr_feature_0.jpg to georgia_acgwr_feature_6.jpg - 7 visualization charts")  # Changed path and name
    
    return results

# ======= Usage Instructions =======
if __name__ == "__main__":
    """
    Usage:
    1. Ensure data file paths are correct
    2. Run this script directly: python georgia_analysis.py
    3. The program will automatically run four models and generate results
    
    Output:
    - georgia_model_comparison.xlsx: Contains R², RMSE, MAE metrics for four models
    - georgia_acgwr_coefficients.csv: Local coefficient estimates of ACGWR model
    - 7 JPG images: f function, g function and beta spatial distribution for each variable
    """
    
    # Run analysis
    results = run_georgia_analysis()
    
    # Print result summary
    print("\n========= Model Performance Comparison =========")
    print(f"{'Model':<10}{'R²':>10}{'RMSE':>10}{'MAE':>10}")
    for model, metrics in results.items():
        print(f"{model:<10}{metrics['R2']:>10.4f}{metrics['RMSE']:>10.4f}{metrics['MAE']:>10.4f}")