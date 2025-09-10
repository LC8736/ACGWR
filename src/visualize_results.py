import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import pickle
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

def visualize_results(beta_true, acgwr_mean, gwr_mean, f_avg=None, g_avg=None, u=None, v=None, output_dir='simulation_visualizations'):
    """
    Visualize results, generate 3x3 coefficient comparison plots, and for separable structure, also plot f and g functions
    
    Parameters:
    beta_true: True coefficient matrix (n_points, n_features)
    acgwr_mean: ACGWR estimated coefficient matrix (n_points, n_features)
    gwr_mean: GWR estimated coefficient matrix (n_points, n_features)
    f_avg: f function values in separable structure (optional)
    g_avg: g function values in separable structure (optional)
    u: Spatial coordinate u (optional)
    v: Spatial coordinate v (optional)
    output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # If u and v are not provided, create default coordinates
    if u is None or v is None:
        n_points = beta_true.shape[0]
        grid_size = int(np.sqrt(n_points))
        u, v = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
        u = u.flatten()
        v = v.flatten()
    
    # Ensure u and v are 1D arrays
    u = np.asarray(u).flatten()
    v = np.asarray(v).flatten()
    
    # Create results dictionary
    results = {
        'true_beta_avg': beta_true,
        'agwr_beta_avg': acgwr_mean,
        'gwr_beta_avg': gwr_mean,
        'coordinates': np.column_stack((u, v)),
        'simulation_params': {
            'setting_name': 'separable' if f_avg is not None and g_avg is not None else 'non_separable'
        }
    }
    
    # Add f and g data (if provided)
    if f_avg is not None and g_avg is not None:
        results['true_f_avg'] = f_avg[0] if isinstance(f_avg, (list, tuple)) and len(f_avg) > 0 else f_avg
        results['true_g_avg'] = g_avg[0] if isinstance(g_avg, (list, tuple)) and len(g_avg) > 0 else g_avg
        results['agwr_f_avg'] = f_avg[1] if isinstance(f_avg, (list, tuple)) and len(f_avg) > 1 else f_avg
        results['agwr_g_avg'] = g_avg[1] if isinstance(g_avg, (list, tuple)) and len(g_avg) > 1 else g_avg
    
    # Call internal visualization function
    _visualize_results(results, output_dir)

def _visualize_results(results, output_dir):
    """Internal visualization function, handles the actual visualization work"""
    setting_name = results['simulation_params']['setting_name']
    u = results['coordinates'][:, 0]
    v = results['coordinates'][:, 1]

    # Create 3x3 coefficient comparison plot
    fig = plt.figure(figsize=(15, 12), dpi=1500)
    fig.set_facecolor('white')

    titles = [
        "$β_0(u,v)$", "$β_0(u,v)$", "$β_0(u,v)$",
        "$β_1(u,v)$", "$β_1(u,v)$", "$β_1(u,v)$",
        "$β_2(u,v)$", "$β_2(u,v)$", "$β_2(u,v)$"
    ]
    
    beta_data = [
        results['true_beta_avg'][:, 0],
        results['agwr_beta_avg'][:, 0],
        results['gwr_beta_avg'][:, 0],
        results['true_beta_avg'][:, 1],
        results['agwr_beta_avg'][:, 1],
        results['gwr_beta_avg'][:, 1],
        results['true_beta_avg'][:, 2],
        results['agwr_beta_avg'][:, 2],
        results['gwr_beta_avg'][:, 2]
    ]

    # Fixed y-coordinate for each row and offset for each column
    y_title_by_row = {0: 0.94, 1: 0.63, 2: 0.32}
    offset_by_col = {0: 0.015, 1: -0.010, 2: -0.025}

    # Labels for the last row (i)(ii)(iii)
    col_labels = [r"(i)True surfaces ", r"(ii)Estimated surfaces of ACGWR", r"(iii)Estimated surfaces of GWR"]
    col_label_y = 0.01  # Last row position
    col_label_offsets = {0: 0.015, 1: -0.010, 2: -0.025}

    for i in range(9):
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        plot_surface(ax, u, v, beta_data[i])

        bbox = ax.get_position()
        x_center = (bbox.x0 + bbox.x1) / 2 + offset_by_col[i % 3]
        y_title = y_title_by_row[i // 3]
        fig.text(x_center, y_title, titles[i],
                 ha='center', fontsize=14, fontname='Times New Roman')

        # Add (i) (ii) (iii) to the last row
        if i // 3 == 2:
            col_x = (bbox.x0 + bbox.x1) / 2 + col_label_offsets[i % 3]
            fig.text(col_x, col_label_y, col_labels[i % 3],
                     ha='center', fontsize=16, fontname='Times New Roman')

    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05,
                        wspace=-0.4, hspace=0.1)

    plt.savefig(os.path.join(output_dir, f"beta_comparison_{setting_name}.jpg"),
                format='jpg', dpi=100, bbox_inches='tight')
    plt.close()

    # If separable structure, plot f and g functions
    if setting_name == 'separable' and 'true_f_avg' in results and 'true_g_avg' in results:
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        axs = axs.flatten()

        plot_function_curve(axs[0], u, results['true_f_avg'][:, 0], results['agwr_f_avg'][:, 0], 
                            "", "u", "$f_0(u)$")
        plot_function_curve(axs[1], u, results['true_f_avg'][:, 1], results['agwr_f_avg'][:, 1], 
                            "", "u", "$f_1(u)$")
        plot_function_curve(axs[2], u, results['true_f_avg'][:, 2], results['agwr_f_avg'][:, 2], 
                            "", "u", "$f_2(u)$")

        plot_function_curve(axs[3], v, results['true_g_avg'][:, 0], results['agwr_g_avg'][:, 0], 
                            "", "v", "$g_0(v)$")
        plot_function_curve(axs[4], v, results['true_g_avg'][:, 1], results['agwr_g_avg'][:, 1], 
                            "", "v", "$g_1(v)$")
        plot_function_curve(axs[5], v, results['true_g_avg'][:, 2], results['agwr_g_avg'][:, 2], 
                            "", "v", "$g_2(v)$")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"f_g_comparison_{setting_name}.jpg"),
                    format='jpg', dpi=100, bbox_inches='tight')
        plt.close()

def load_and_visualize(setting_name, results_dir="simulation_results"):
    """Load data from saved result files and visualize"""
    setting_dir = os.path.join(results_dir, setting_name)
    files = [f for f in os.listdir(setting_dir) if f.endswith('.pkl')]
    if not files:
        print(f"No result files found in {setting_dir}")
        return
    
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(setting_dir, x)))
    with open(os.path.join(setting_dir, latest_file), 'rb') as f:
        results = pickle.load(f)
    
    vis_dir = os.path.join(setting_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    _visualize_results(results, vis_dir)
    print(f"Visualization results saved to: {vis_dir}")

if __name__ == "__main__":
    # If running this script directly, visualize saved results
    load_and_visualize('separable')
    load_and_visualize('non_separable')