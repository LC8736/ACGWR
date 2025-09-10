"""
Evaluation metrics for model performance comparison.
Contains functions to calculate various performance metrics.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def calculate_performance_metrics(y_true, y_pred):
    """
    Calculate performance metrics for model predictions.
    
    Parameters:
    y_true: array-like, true values
    y_pred: array-like, predicted values
    
    Returns:
    dict: Dictionary containing performance metrics
    """
    metrics = {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
    }
    
    # Calculate MAPE if there are no zero values
    if np.all(y_true != 0):
        metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        metrics['MAPE'] = np.nan
        
    return metrics

def calculate_beta_metrics(beta_true, beta_pred):
    """
    Calculate performance metrics for beta coefficient estimates.
    
    Parameters:
    beta_true: array-like, true beta coefficients
    beta_pred: array-like, predicted beta coefficients
    
    Returns:
    dict: Dictionary containing beta performance metrics
    """
    n_features = beta_true.shape[1]
    metrics = {}
    
    # Calculate metrics for each beta coefficient
    for j in range(n_features):
        true_j = beta_true[:, j]
        pred_j = beta_pred[:, j]
        
        metrics[f'beta_{j}'] = {
            'RMSE': np.sqrt(mean_squared_error(true_j, pred_j)),
            'MAE': mean_absolute_error(true_j, pred_j),
            'Correlation': pearsonr(true_j, pred_j)[0] if len(true_j) > 2 else np.nan
        }
    
    # Calculate overall metrics
    metrics['global'] = {
        'Overall_RMSE': np.sqrt(mean_squared_error(beta_true, beta_pred)),
        'Overall_MAE': mean_absolute_error(beta_true, beta_pred),
        'Overall_Correlation': pearsonr(beta_true.flatten(), beta_pred.flatten())[0] 
                              if beta_true.size > 2 else np.nan
    }
    
    return metrics

def calculate_model_comparison(metrics_dict):
    """
    Calculate comparison metrics between different models.
    
    Parameters:
    metrics_dict: dict, dictionary containing metrics for multiple models
    
    Returns:
    dict: Dictionary containing comparison metrics
    """
    comparison = {}
    models = list(metrics_dict.keys())
    
    # For each metric, calculate the best model and differences
    for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
        if metric in metrics_dict[models[0]]:
            values = [metrics_dict[model][metric] for model in models]
            best_idx = np.argmax(values) if metric == 'R2' else np.argmin(values)
            best_model = models[best_idx]
            best_value = values[best_idx]
            
            comparison[metric] = {
                'best_model': best_model,
                'best_value': best_value,
                'all_values': {model: metrics_dict[model][metric] for model in models}
            }
    
    return comparison

def print_performance_table(metrics_dict, title="Performance Metrics"):
    """
    Print performance metrics in a formatted table.
    
    Parameters:
    metrics_dict: dict, dictionary containing metrics for multiple models
    title: str, title of the table
    """
    print(f"\n{title}")
    print("=" * 60)
    
    # Get all models and metrics
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    
    # Print header
    header = f"{'Metric':<15}" + "".join([f"{model:<12}" for model in models])
    print(header)
    print("-" * 60)
    
    # Print each metric
    for metric in metrics:
        row = f"{metric:<15}"
        for model in models:
            value = metrics_dict[model][metric]
            if isinstance(value, float):
                row += f"{value:<12.4f}"
            else:
                row += f"{value:<12}"
        print(row)

def save_results_to_file(results, filename):
    """
    Save results to a file.
    
    Parameters:
    results: dict, results to save
    filename: str, name of the file
    """
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_results_from_file(filename):
    """
    Load results from a file.
    
    Parameters:
    filename: str, name of the file
    
    Returns:
    dict: Loaded results
    """
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)