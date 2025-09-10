# main.py
import numpy as np
from data_generation import generate_grid_data, generate_covariates, generate_separable_beta
from acgwr_model import ACGWRModel

def main():
    """Main function to demonstrate ACGWR model usage."""
    # 1. Generate data
    u, v, coords = generate_grid_data(grid_size=24)  # 25x25 grid = 625 points
    x1, x2 = generate_covariates(len(u))
    y, beta0, beta1, beta2, *_ = generate_separable_beta(u, v, x1, x2)

    # 2. Construct independent variable matrix
    X = np.column_stack((x1, x2))

    # 3. Run ACGWR model
    model = ACGWRModel(coords, X, y)
    results = model.fit()

    # 4. Output results - FIXED: use correct key from results dictionary
    print("Local coefficients shape:", results['beta_hat_local'].shape)
    print("Sample size:", len(u))
    print("A few local coefficients:\n", results['beta_hat_local'][:5])
    
    # Additional metrics
    print("Global alpha:", results['alpha_hat'].flatten())
    print("R2 score:", np.corrcoef(y, results['y_hat'].flatten())[0, 1]**2)

if __name__ == "__main__":
    main()