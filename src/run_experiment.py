import numpy as np
import pandas as pd
import os
from data_generation import generate_grid_data, generate_covariates, generate_separable_beta, generate_nonseparable_beta
from acgwr_model import ACGWRModel
from llgwr_model import LLGWRModel
from visualize_results import visualize_results
import argparse

# -----------------------------
# Monte Carlo Simulation
# -----------------------------
def run_experiment(n_repeat=500, grid_size=24, separable=True):
    results_acgwr = []
    results_llgwr = []

    if separable:
        all_f = []
        all_g = []

    # Generate coordinates outside the loop to ensure consistent coordinates across experiments
    u, v, coords = generate_grid_data(grid_size)
    
    for i in range(n_repeat):
        x1, x2 = generate_covariates(len(u))
        X = np.column_stack((x1, x2))

        if separable:
            y, beta0, beta1, beta2, f0, f1, f2, g0, g1, g2 = generate_separable_beta(u, v, x1, x2)
            all_f.append(np.column_stack((f0, f1, f2)))
            all_g.append(np.column_stack((g0, g1, g2)))
        else:
            y, beta0, beta1, beta2 = generate_nonseparable_beta(u, v, x1, x2)

        acgwr = ACGWRModel(coords, X, y)
        acgwr.fit()
        results_acgwr.append(acgwr.get_local_coefs())

        llgwr = LLGWRModel(coords, X, y)
        llgwr.fit()
        results_llgwr.append(llgwr.get_local_coefs())

    results_acgwr = np.array(results_acgwr)
    results_llgwr = np.array(results_llgwr)
    beta_true = np.column_stack((beta0, beta1, beta2))

    if separable:
        f_avg = np.mean(np.array(all_f), axis=0)
        g_avg = np.mean(np.array(all_g), axis=0)
        return results_acgwr, results_llgwr, beta_true, f_avg, g_avg, u, v
    else:
        return results_acgwr, results_llgwr, beta_true, u, v  # Return u, v

# -----------------------------
# Performance Evaluation
# -----------------------------
def evaluate_performance(results, beta_true):
    mean_est = np.mean(results, axis=0)
    bias = mean_est - beta_true
    rmse = np.sqrt(np.mean((results - beta_true)**2, axis=0))
    return mean_est, bias, rmse

# -----------------------------
# Main Program
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_repeat", type=int, default=500, help="Number of simulation repetitions")
    parser.add_argument("--grid_size", type=int, default=24, help="Grid size")
    args = parser.parse_args()

    n_repeat = args.n_repeat
    grid_size = args.grid_size

    output_dir = "simulation_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------ Separable Structure ------------------
    results_acgwr, results_llgwr, beta_true, f_avg, g_avg, u, v = run_experiment(
        n_repeat=n_repeat, grid_size=grid_size, separable=True
    )
    acgwr_mean, _, acgwr_rmse = evaluate_performance(results_acgwr, beta_true)
    gwr_mean, _, gwr_rmse = evaluate_performance(results_llgwr, beta_true)

    visualize_results(beta_true, acgwr_mean, gwr_mean, f_avg, g_avg, u, v, output_dir=output_dir)
    print("Separable structure visualization completed")

    # ------------------ Non-separable Structure ------------------
    # Modified here: receive returned u, v
    results_acgwr_ns, results_llgwr_ns, beta_true_ns, u_ns, v_ns = run_experiment(
        n_repeat=n_repeat, grid_size=grid_size, separable=False
    )
    acgwr_mean_ns, _, acgwr_rmse_ns = evaluate_performance(results_acgwr_ns, beta_true_ns)
    gwr_mean_ns, _, gwr_rmse_ns = evaluate_performance(results_llgwr_ns, beta_true_ns)

    # Modified here: pass correct u, v coordinates
    visualize_results(beta_true_ns, acgwr_mean_ns, gwr_mean_ns, None, None, u_ns, v_ns, output_dir=output_dir)
    print("Non-separable structure visualization completed")