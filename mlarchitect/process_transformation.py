import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy import linalg
from typing import Dict, List, Union, Optional, Tuple, Callable
import warnings
from time import time

# Constants for numerical stability
EPSILON = np.finfo(float).eps

def diagnose_extreme_values(df: pd.DataFrame, threshold: float = 1e10, n_examples: int = 5) -> Dict:
    """
    Diagnose extreme values in a DataFrame to understand their source and distribution.
    
    Args:
        df: DataFrame to analyze
        threshold: Value above which to consider a value extreme
        n_examples: Number of example extreme values to return per column
        
    Returns:
        Dictionary with diagnostic information about extreme values
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Initialize results dictionary
    diagnostics = {
        "extreme_columns": [],
        "inf_columns": [],
        "statistics": {},
        "examples": {},
        "summary": {"total_extreme_values": 0, "total_inf_values": 0}
    }
    
    # Check each column
    for col in numeric_cols:
        # Get absolute values
        abs_vals = df[col].abs()
        
        # Check for infinities
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            diagnostics["inf_columns"].append(col)
            diagnostics["summary"]["total_inf_values"] += inf_count
            # Get example rows with inf values
            inf_examples = df.loc[np.isinf(df[col]), col].head(n_examples).to_dict()
            diagnostics["examples"][f"{col}_inf"] = inf_examples
        
        # Check for extreme but finite values
        extreme_mask = (abs_vals > threshold) & (~np.isinf(df[col]))
        extreme_count = extreme_mask.sum()
        
        if extreme_count > 0:
            diagnostics["extreme_columns"].append(col)
            diagnostics["summary"]["total_extreme_values"] += extreme_count
            
            # Get statistics
            extreme_values = df.loc[extreme_mask, col]
            stats = {
                "min": extreme_values.min(),
                "max": extreme_values.max(),
                "mean": extreme_values.mean(),
                "median": extreme_values.median(),
                "count": extreme_count,
                "percentage": (extreme_count / len(df)) * 100
            }
            diagnostics["statistics"][col] = stats
            
            # Get a few example extreme values
            examples = df.loc[extreme_mask, col].head(n_examples).to_dict()
            diagnostics["examples"][col] = examples
            
            # If time series data with dates, get timestamps of extreme values
            date_cols = [c for c in df.columns if any(date_term in c.lower() for date_term in ['date', 'time', 'day'])]
            if date_cols and extreme_mask.any():
                date_examples = df.loc[extreme_mask, date_cols].head(n_examples).to_dict()
                if date_examples:  # Only add if there are date columns
                    diagnostics["examples"][f"{col}_dates"] = date_examples
    
    # Add overall summary
    diagnostics["summary"]["total_extreme_columns"] = len(diagnostics["extreme_columns"])
    diagnostics["summary"]["total_inf_columns"] = len(diagnostics["inf_columns"])
    diagnostics["summary"]["all_affected_columns"] = diagnostics["extreme_columns"] + [
        col for col in diagnostics["inf_columns"] if col not in diagnostics["extreme_columns"]
    ]
    
    return diagnostics

def normalize_data(df_train: pd.DataFrame, 
                  df_test: pd.DataFrame, 
                  columns: Optional[List[str]] = None,
                  method: str = "standard",
                  target_name: Optional[str] = None,
                  id_columns: Optional[List[str]] = None,
                  handle_inf: str = "clip",
                  inf_threshold: float = 1e10,
                  diagnose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize specified columns using various scaling methods.
    
    Args:
        df_train: Training DataFrame
        df_test: Testing DataFrame
        columns: List of columns to normalize (if None, all numeric columns except target and ID columns)
        method: Normalization method ('standard', 'minmax', or 'robust')
        target_name: Name of the target column to exclude
        id_columns: List of ID columns to exclude
        handle_inf: How to handle infinity values ('clip', 'remove', or 'nan')
        inf_threshold: Maximum absolute value to use when clipping
        diagnose: Whether to run diagnostics on extreme values
        
    Returns:
        Tuple of normalized (df_train, df_test)
    """
    start_time = time()
    
    # If diagnose is True, run diagnostics on the training data
    if diagnose:
        print("Running diagnostics on extreme values in training data...")
        diag_results = diagnose_extreme_values(df_train, inf_threshold)
        
        if diag_results["extreme_columns"] or diag_results["inf_columns"]:
            print(f"\nDiagnostics Summary:")
            print(f"- Found {diag_results['summary']['total_extreme_columns']} columns with extreme values")
            print(f"- Found {diag_results['summary']['total_inf_columns']} columns with infinity values")
            print(f"- Total extreme values: {diag_results['summary']['total_extreme_values']}")
            print(f"- Total infinity values: {diag_results['summary']['total_inf_values']}")
            
            # Print some examples of the most problematic columns
            if diag_results["extreme_columns"]:
                worst_cols = sorted(
                    diag_results["extreme_columns"],
                    key=lambda col: diag_results["statistics"][col]["count"],
                    reverse=True
                )[:3]
                
                print("\nMost problematic columns with extreme values:")
                for col in worst_cols:
                    stats = diag_results["statistics"][col]
                    print(f"  {col}:")
                    print(f"    - {stats['count']} extreme values ({stats['percentage']:.2f}%)")
                    print(f"    - Range: {stats['min']} to {stats['max']}")
                    examples = list(diag_results["examples"][col].values())[:3]
                    print(f"    - Example values: {examples}")
        else:
            print("No extreme or infinity values found in the data.")
    
    # Identify columns to exclude
    exclude_cols = []
    if target_name:
        exclude_cols.append(target_name)
    if id_columns:
        exclude_cols.extend(id_columns)
    
    # If no columns specified, use all numeric columns except excluded ones
    if not columns:
        numeric_cols = df_train.select_dtypes(include=['number']).columns.tolist()
        columns = [col for col in numeric_cols if col not in exclude_cols]
    else:
        # Filter out excluded columns
        columns = [col for col in columns if col not in exclude_cols]
    
    # Check if columns exist in both dataframes
    valid_columns = [col for col in columns if col in df_train.columns and col in df_test.columns]
    if not valid_columns:
        print("Warning: No valid columns found for normalization")
        return df_train.copy(), df_test.copy()
    
    # Create copies to avoid modifying originals
    df_train_norm = df_train.copy()
    df_test_norm = df_test.copy()
    
    # Handle infinity values and extremely large numbers - vectorized operations
    train_data = df_train_norm[valid_columns]
    test_data = df_test_norm[valid_columns]
    
    # Check for inf and extreme values
    inf_cols_train = np.any(np.isinf(train_data), axis=0) | np.any(np.abs(train_data) > inf_threshold, axis=0)
    inf_cols_test = np.any(np.isinf(test_data), axis=0) | np.any(np.abs(test_data) > inf_threshold, axis=0)
    inf_cols = inf_cols_train | inf_cols_test
    
    if np.any(inf_cols):
        inf_col_names = [valid_columns[i] for i, is_inf in enumerate(inf_cols) if is_inf]
        print(f"Found {len(inf_col_names)} columns with inf/extreme values: {inf_col_names[:5]}{'...' if len(inf_col_names)>5 else ''}")
        
        if handle_inf == 'clip':
            # Clip values to threshold - vectorized operation
            df_train_norm[valid_columns] = df_train_norm[valid_columns].clip(lower=-inf_threshold, upper=inf_threshold)
            df_test_norm[valid_columns] = df_test_norm[valid_columns].clip(lower=-inf_threshold, upper=inf_threshold)
            print(f"Clipped extreme values to +/- {inf_threshold}")
            
        elif handle_inf == 'nan':
            # Convert infinities to NaN - vectorized operation
            df_train_norm[valid_columns] = df_train_norm[valid_columns].replace([np.inf, -np.inf], np.nan)
            df_test_norm[valid_columns] = df_test_norm[valid_columns].replace([np.inf, -np.inf], np.nan)
            
            # Convert extremely large values to NaN - vectorized operation
            mask_train = np.abs(df_train_norm[valid_columns]) > inf_threshold
            mask_test = np.abs(df_test_norm[valid_columns]) > inf_threshold
            df_train_norm[valid_columns] = df_train_norm[valid_columns].mask(mask_train)
            df_test_norm[valid_columns] = df_test_norm[valid_columns].mask(mask_test)
            print(f"Converted inf/extreme values to NaN")
    
    # Handle NaN values efficiently
    has_nan_train = df_train_norm[valid_columns].isna().any().any()
    has_nan_test = df_test_norm[valid_columns].isna().any().any()
    
    if has_nan_train or has_nan_test:
        # Compute means once for all columns - vectorized
        col_means = df_train_norm[valid_columns].mean()
        
        # For columns where mean is NaN, use median
        nan_mean_cols = col_means.isna()
        if nan_mean_cols.any():
            col_medians = df_train_norm[valid_columns].median()
            col_means[nan_mean_cols] = col_medians[nan_mean_cols]
            
            # For columns where both mean and median are NaN, use 0
            still_nan = col_means.isna()
            if still_nan.any():
                print(f"Warning: {still_nan.sum()} columns contain all NaNs. Using 0.")
                col_means[still_nan] = 0
        
        # Fill NaNs with the computed values - vectorized operation
        df_train_norm[valid_columns] = df_train_norm[valid_columns].fillna(col_means)
        df_test_norm[valid_columns] = df_test_norm[valid_columns].fillna(col_means)
        print(f"Filled NaN values in {(df_train_norm[valid_columns].isna().any()).sum()} columns")
    
    # Select appropriate scaler
    if method.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif method.lower() == 'robust':
        scaler = RobustScaler()
    else:  # default to standard scaling
        scaler = StandardScaler()
    
    try:
        # Fit scaler on training data and transform both train and test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            
            # Use bulk transformation for better performance
            df_train_norm[valid_columns] = scaler.fit_transform(df_train_norm[valid_columns])
            df_test_norm[valid_columns] = scaler.transform(df_test_norm[valid_columns])
            
    except Exception as e:
        print(f"Error during scaling: {e}")
        print("Falling back to column-by-column scaling...")
        
        # Fall back to column-by-column scaling in case of errors
        for col in valid_columns:
            try:
                # Create a new scaler for each column
                if method.lower() == 'minmax':
                    col_scaler = MinMaxScaler()
                elif method.lower() == 'robust':
                    col_scaler = RobustScaler()
                else:
                    col_scaler = StandardScaler()
                
                # Reshape for 1D case and transform
                train_values = df_train_norm[col].values.reshape(-1, 1)
                test_values = df_test_norm[col].values.reshape(-1, 1)
                
                df_train_norm[col] = col_scaler.fit_transform(train_values).flatten()
                df_test_norm[col] = col_scaler.transform(test_values).flatten()
            except Exception as col_err:
                print(f"Could not scale column {col}, error: {col_err}")
                print(f"Skipping normalization for column {col}")
    
    print(f"Applied {method} normalization to {len(valid_columns)} columns in {time()-start_time:.2f}s")
    return df_train_norm, df_test_norm

def apply_pca(df_train: pd.DataFrame, 
             df_test: pd.DataFrame,
             n_components: int = 5,
             columns: Optional[List[str]] = None,
             target_name: Optional[str] = None,
             id_columns: Optional[List[str]] = None,
             variance_threshold: Optional[float] = None,
             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply PCA to reduce dimensionality of the data.
    """
    start_time = time()
    
    # Identify columns to exclude
    exclude_cols = []
    if target_name:
        exclude_cols.append(target_name)
    if id_columns:
        exclude_cols.extend(id_columns)
    
    # If no columns specified, use all numeric columns except excluded ones
    if not columns:
        numeric_cols = df_train.select_dtypes(include=['number']).columns.tolist()
        columns = [col for col in numeric_cols if col not in exclude_cols]
    else:
        # Filter out excluded columns
        columns = [col for col in columns if col not in exclude_cols]
    
    # Check if columns exist in both dataframes
    valid_columns = [col for col in columns if col in df_train.columns and col in df_test.columns]
    if not valid_columns:
        print("Warning: No valid columns found for PCA")
        return df_train.copy(), df_test.copy()
    
    # Check if we have enough valid columns
    if len(valid_columns) < 2:
        print("Warning: Need at least 2 columns for PCA")
        return df_train.copy(), df_test.copy()
    
    # Handle NaN and infinity values before PCA
    train_data = df_train[valid_columns].copy()
    test_data = df_test[valid_columns].copy()
    
    # Replace inf values with NaN first
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaN with means efficiently
    if train_data.isna().any().any() or test_data.isna().any().any():
        # Calculate means once for all columns - vectorized
        col_means = train_data.mean()
        
        # For columns where mean is NaN, use median
        nan_mean_cols = col_means.isna()
        if nan_mean_cols.any():
            col_medians = train_data.median()
            col_means[nan_mean_cols] = col_medians[nan_mean_cols]
            
            # For columns where both mean and median are NaN, use 0
            still_nan = col_means.isna()
            if still_nan.any():
                col_means[still_nan] = 0
        
        # Fill NaNs with the computed values - vectorized operation
        train_data = train_data.fillna(col_means)
        test_data = test_data.fillna(col_means)
    
    # Create copies to avoid modifying originals
    df_train_pca = df_train.copy()
    df_test_pca = df_test.copy()
    
    # Initialize PCA
    if variance_threshold is not None:
        # If variance threshold is specified, start with a larger number of components
        max_components = min(len(valid_columns), len(train_data))
        pca = PCA(n_components=max_components, random_state=random_state)
    else:
        # Otherwise use the specified number of components
        pca = PCA(n_components=min(n_components, len(valid_columns)), random_state=random_state)
    
    # Extract features for PCA - use numpy arrays for better performance
    X_train = train_data.values
    X_test = test_data.values
    
    # Fit and transform training data
    X_train_pca = pca.fit_transform(X_train)
    
    # If using variance threshold, determine how many components to keep
    if variance_threshold is not None:
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        n_components = max(1, min(n_components, len(pca.components_)))
        print(f"Selected {n_components} components explaining {cumulative_variance[n_components-1]:.2%} of variance")
        
        # Create a new PCA with the selected number of components
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train_pca = pca.fit_transform(X_train)
    
    # Transform test data
    X_test_pca = pca.transform(X_test)
    
    # Add PCA components as new columns - vectorized approach
    pca_columns = [f'PCA_{i+1}' for i in range(X_train_pca.shape[1])]
    
    # Create DataFrames with PCA results and join with original data efficiently
    train_pca_df = pd.DataFrame(X_train_pca, index=df_train.index, columns=pca_columns)
    test_pca_df = pd.DataFrame(X_test_pca, index=df_test.index, columns=pca_columns)
    
    df_train_pca = pd.concat([df_train_pca, train_pca_df], axis=1)
    df_test_pca = pd.concat([df_test_pca, test_pca_df], axis=1)
    
    # Display explained variance info
    print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    print(f"PCA completed in {time()-start_time:.2f}s")
    return df_train_pca, df_test_pca

def apply_shrinkage(df_train: pd.DataFrame, 
                   df_test: pd.DataFrame,
                   columns: Optional[List[str]] = None,
                   target_name: Optional[str] = None,
                   id_columns: Optional[List[str]] = None,
                   suffix: str = "_shrink") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply covariance shrinkage using Ledoit-Wolf method to stabilize feature relationships.
    """
    start_time = time()
    
    # Identify columns to exclude
    exclude_cols = []
    if target_name:
        exclude_cols.append(target_name)
    if id_columns:
        exclude_cols.extend(id_columns)
    
    # If no columns specified, use all numeric columns except excluded ones
    if not columns:
        numeric_cols = df_train.select_dtypes(include=['number']).columns.tolist()
        columns = [col for col in numeric_cols if col not in exclude_cols]
    else:
        # Filter out excluded columns
        columns = [col for col in columns if col not in exclude_cols]
    
    # Check if columns exist in both dataframes
    valid_columns = [col for col in columns if col in df_train.columns and col in df_test.columns]
    if len(valid_columns) < 2:
        print("Warning: Need at least 2 columns for shrinkage")
        return df_train.copy(), df_test.copy()
    
    # Create copies to avoid modifying originals
    df_train_shrink = df_train.copy()
    df_test_shrink = df_test.copy()
    
    try:
        # Handle NaN and infinity values before shrinkage
        train_data = df_train[valid_columns].copy()
        test_data = df_test[valid_columns].copy()
        
        # Replace inf values with NaN first
        train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN with means efficiently
        if train_data.isna().any().any() or test_data.isna().any().any():
            # Calculate means once for all columns - vectorized
            col_means = train_data.mean()
            
            # For columns where mean is NaN, use median
            nan_mean_cols = col_means.isna()
            if nan_mean_cols.any():
                col_medians = train_data.median()
                col_means[nan_mean_cols] = col_medians[nan_mean_cols]
                
                # For columns where both mean and median are NaN, use 0
                still_nan = col_means.isna()
                if still_nan.any():
                    col_means[still_nan] = 0
            
            # Fill NaNs with computed values - vectorized operation
            train_data = train_data.fillna(col_means)
            test_data = test_data.fillna(col_means)
        
        # Apply Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        
        # Fit and transform training data
        X_train = train_data.values
        X_test = test_data.values
        
        lw.fit(X_train)
        
        # Get the shrunk covariance matrix
        cov_shrink = lw.covariance_
        
        # Calculate the transformed data (whitening using the shrunk covariance)
        # Compute the Cholesky decomposition of the covariance matrix for numerical stability
        chol_factor = linalg.cholesky(cov_shrink + np.eye(cov_shrink.shape[0]) * EPSILON, lower=True)
        
        # Apply the whitening transformation
        inv_chol = linalg.inv(chol_factor.T)
        X_train_whitened = np.dot(X_train, inv_chol)
        X_test_whitened = np.dot(X_test, inv_chol)
        
        # Create new column names for the whitened features
        whitened_cols = [f"{col}{suffix}" for col in valid_columns]
        
        # Add the whitened data as new columns efficiently
        train_whitened_df = pd.DataFrame(X_train_whitened, index=df_train.index, columns=whitened_cols)
        test_whitened_df = pd.DataFrame(X_test_whitened, index=df_test.index, columns=whitened_cols)
        
        df_train_shrink = pd.concat([df_train_shrink, train_whitened_df], axis=1)
        df_test_shrink = pd.concat([df_test_shrink, test_whitened_df], axis=1)
        
        print(f"Applied Ledoit-Wolf shrinkage to {len(valid_columns)} features in {time()-start_time:.2f}s")
        return df_train_shrink, df_test_shrink
        
    except Exception as e:
        print(f"Error applying shrinkage: {e}")
        return df_train.copy(), df_test.copy()

def remove_outliers(df: pd.DataFrame,
                   method: str = 'zscore',
                   threshold: float = 3.0,
                   columns: Optional[List[str]] = None,
                   target_name: Optional[str] = None,
                   id_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove outliers from the data using various methods.
    """
    start_time = time()
    
    # Identify columns to exclude
    exclude_cols = []
    if target_name:
        exclude_cols.append(target_name)
    if id_columns:
        exclude_cols.extend(id_columns)
    
    # If no columns specified, use all numeric columns except excluded ones
    if not columns:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        columns = [col for col in numeric_cols if col not in exclude_cols]
    else:
        # Filter out excluded columns
        columns = [col for col in columns if col not in exclude_cols]
    
    # Check if columns exist
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        print("Warning: No valid columns found for outlier removal")
        return df.copy()
    
    # Handle NaN and infinity values
    data = df[valid_columns].copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create a copy to avoid modifying the original
    df_filtered = df.copy()
    
    # Keep track of rows to keep (start with all rows)
    mask = pd.Series(True, index=df.index)
    
    if method.lower() == 'zscore':
        # Z-score method: remove points with z-score beyond threshold - vectorized
        means = data.mean()
        stds = data.std()
        
        # Avoid division by zero by filtering out zero std columns
        zero_std_cols = stds == 0
        if zero_std_cols.any():
            nonzero_std_cols = [col for col, is_zero in zip(valid_columns, zero_std_cols) if not is_zero]
            print(f"Skipping {zero_std_cols.sum()} columns with zero standard deviation")
        else:
            nonzero_std_cols = valid_columns
        
        if nonzero_std_cols:
            # Calculate z-scores for all columns at once - vectorized
            z_scores = data[nonzero_std_cols].sub(means[nonzero_std_cols]).div(stds[nonzero_std_cols])
            
            # Create mask for rows to keep (those without extreme z-scores in any column)
            outlier_mask = (z_scores.abs() <= threshold).all(axis=1)
            mask = mask & outlier_mask
    
    elif method.lower() == 'iqr':
        # IQR method: vectorized approach
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        # Avoid division by zero by filtering out zero IQR columns
        zero_iqr_cols = iqr == 0
        if zero_iqr_cols.any():
            nonzero_iqr_cols = [col for col, is_zero in zip(valid_columns, zero_iqr_cols) if not is_zero]
            print(f"Skipping {zero_iqr_cols.sum()} columns with zero IQR")
        else:
            nonzero_iqr_cols = valid_columns
        
        if nonzero_iqr_cols:
            # Calculate lower and upper bounds
            lower_bounds = q1[nonzero_iqr_cols] - threshold * iqr[nonzero_iqr_cols]
            upper_bounds = q3[nonzero_iqr_cols] + threshold * iqr[nonzero_iqr_cols]
            
            # Check if each value is within bounds - vectorized
            within_bounds = (data[nonzero_iqr_cols] >= lower_bounds) & (data[nonzero_iqr_cols] <= upper_bounds)
            
            # Keep rows where all columns are within bounds
            outlier_mask = within_bounds.all(axis=1)
            mask = mask & outlier_mask
    
    elif method.lower() == 'isolation_forest':
        # Isolation forest: machine learning method for outlier detection
        from sklearn.ensemble import IsolationForest
        
        # Prepare data - fill NaN with means
        if data.isna().any().any():
            col_means = data.mean()
            data = data.fillna(col_means)
        
        # Apply isolation forest to all columns at once
        X = data.values
        iso_forest = IsolationForest(contamination=threshold/100, random_state=42, n_jobs=-1)
        predictions = iso_forest.fit_predict(X)
        mask = mask & (predictions == 1)  # 1 for inliers, -1 for outliers
    
    else:
        print(f"Warning: Unknown outlier method '{method}'. No outliers removed.")
        return df.copy()
    
    # Apply the mask to filter out outliers
    df_filtered = df[mask].copy()
    
    removed_count = len(df) - len(df_filtered)
    removal_percentage = (removed_count / len(df)) if len(df) > 0 else 0
    print(f"Removed {removed_count} outliers ({removal_percentage:.1%}) using {method} method in {time()-start_time:.2f}s")
    return df_filtered

def select_features(df_train: pd.DataFrame, 
                   df_test: pd.DataFrame,
                   method: str = 'correlation',
                   threshold: float = 0.1,
                   k: int = 20,
                   target_name: Optional[str] = None,
                   id_columns: Optional[List[str]] = None,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform feature selection using various methods.
    """
    start_time = time()
    
    # Validate inputs
    if method in ['correlation', 'mutual_info', 'chi2', 'rfe'] and not target_name:
        print(f"Warning: Target column required for {method} feature selection")
        return df_train.copy(), df_test.copy()
    
    # Identify columns to exclude
    exclude_cols = []
    if target_name:
        exclude_cols.append(target_name)
    if id_columns:
        exclude_cols.extend(id_columns)
    
    # Get numeric columns excluding target and ID columns
    numeric_cols = df_train.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if not feature_cols:
        print("Warning: No valid feature columns found")
        return df_train.copy(), df_test.copy()
    
    # Always keep target and ID columns
    cols_to_keep = exclude_cols.copy()
    
    # Handle NaN and infinity values
    train_data = df_train[feature_cols].copy()
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if train_data.isna().any().any():
        col_means = train_data.mean()
        train_data = train_data.fillna(col_means)
    
    # Select features based on method
    if method == 'correlation':
        # Correlation with target - vectorized operation
        if target_name:
            target_data = df_train[target_name]
            
            # Calculate correlations for all features at once
            correlations = train_data.apply(lambda x: x.corr(target_data) if not np.isnan(x.corr(target_data)) else 0)
            correlations = correlations.abs()
            
            # Sort by absolute correlation
            correlations = correlations.sort_values(ascending=False)
            
            # Select top k features or those above threshold
            if k:
                selected_features = correlations.head(k).index.tolist()
            else:
                selected_features = correlations[correlations >= threshold].index.tolist()
                
            cols_to_keep.extend(selected_features)
        
    elif method == 'variance':
        # Variance threshold - vectorized operation
        variances = train_data.var()
        
        # Sort by variance
        variances = variances.sort_values(ascending=False)
        
        # Select top k features or those above threshold
        if k:
            selected_features = variances.head(k).index.tolist()
        else:
            selected_features = variances[variances >= threshold].index.tolist()
            
        cols_to_keep.extend(selected_features)
        
    elif method == 'mutual_info':
        # Mutual information - select features with highest MI with target
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        # Check if regression or classification
        if df_train[target_name].nunique() < 10:  # Simple heuristic for classification
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression
            
        # Calculate mutual information
        mi = mi_func(train_data, df_train[target_name], random_state=random_state, n_neighbors=3)
        
        # Sort by MI
        mi_series = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
        
        # Select top k features or those above threshold
        if k:
            selected_features = mi_series.head(k).index.tolist()
        else:
            selected_features = mi_series[mi_series >= threshold].index.tolist()
            
        cols_to_keep.extend(selected_features)
        
    elif method == 'chi2':
        # Chi-squared test for classification tasks
        from sklearn.feature_selection import chi2, SelectKBest
        
        # Ensure target is categorical
        if df_train[target_name].nunique() > 10:
            print("Warning: Chi2 is for classification tasks. Target has many unique values.")
            
        # Make sure data is non-negative for chi2
        min_vals = train_data.min()
        if (min_vals < 0).any():
            # Shift data to be non-negative
            train_data = train_data - min_vals + 0.1
            
        # Calculate chi2 scores efficiently
        selector = SelectKBest(chi2, k=k or 'all')
        selector.fit(train_data, df_train[target_name])
        
        # Get selected features
        mask = selector.get_support()
        selected_features = np.array(feature_cols)[mask].tolist()
        
        cols_to_keep.extend(selected_features)
        
    elif method == 'rfe':
        # Recursive Feature Elimination
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        # Choose estimator based on target type
        if df_train[target_name].nunique() < 10:  # Classification
            estimator = LogisticRegression(max_iter=1000, random_state=random_state, 
                                          n_jobs=-1, solver='liblinear')
        else:  # Regression
            estimator = LinearRegression(n_jobs=-1)
            
        # RFE with specified number of features
        n_features = min(k, len(feature_cols))
        step = max(1, len(feature_cols) // 20)  # Faster elimination with larger steps
        
        rfe = RFE(estimator, n_features_to_select=n_features, step=step)
        rfe.fit(train_data, df_train[target_name])
        
        # Get selected features
        mask = rfe.support_
        selected_features = np.array(feature_cols)[mask].tolist()
        
        cols_to_keep.extend(selected_features)
        
    else:
        print(f"Warning: Unknown feature selection method '{method}'. Using all features.")
        cols_to_keep.extend(feature_cols)
    
    # Ensure all columns exist in the dataframes
    valid_cols = [col for col in cols_to_keep if col in df_train.columns]
    
    # Filter DataFrames to keep only selected columns - efficient selection
    df_train_selected = df_train[valid_cols].copy()
    df_test_selected = df_test[valid_cols].copy()
    
    num_features = len(valid_cols) - len(exclude_cols)
    print(f"Selected {num_features} features using {method} method in {time()-start_time:.2f}s")
    
    return df_train_selected, df_test_selected

def process_data(df_train: pd.DataFrame, 
                df_test: pd.DataFrame,
                transformations: Dict[str, Dict],
                target_name: Optional[str] = None,
                id_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a series of data transformations to the train and test sets.
    
    Args:
        df_train: Training DataFrame
        df_test: Testing DataFrame
        transformations: Dictionary of transformations to apply
            Example: {
                'normalize': {'method': 'standard'},
                'pca': {'n_components': 10},
                'remove_outliers': {'method': 'zscore', 'threshold': 3.0},
                'select_features': {'method': 'correlation', 'k': 20}
            }
        target_name: Name of the target column
        id_columns: List of ID column names
        
    Returns:
        Tuple of transformed (df_train, df_test)
    """
    # Create copies of input DataFrames to avoid modifying originals
    df_train_processed = df_train.copy()
    df_test_processed = df_test.copy()
    
    overall_start_time = time()
    
    # Process each transformation in the order specified
    for transform_name, transform_config in transformations.items():
        print(f"Applying transformation: {transform_name}")
        
        # Apply appropriate transformation based on name
        if transform_name == 'normalize':
            # Add diagnose parameter for normalize_data
            diagnose = transform_config.get('diagnose', False)
            
            df_train_processed, df_test_processed = normalize_data(
                df_train_processed, df_test_processed,
                columns=transform_config.get('columns'),
                method=transform_config.get('method', 'standard'),
                target_name=target_name,
                id_columns=id_columns,
                handle_inf=transform_config.get('handle_inf', 'clip'),
                inf_threshold=transform_config.get('inf_threshold', 1e10),
                diagnose=diagnose
            )
            
        elif transform_name == 'pca':
            df_train_processed, df_test_processed = apply_pca(
                df_train_processed, df_test_processed,
                n_components=transform_config.get('n_components', 5),
                columns=transform_config.get('columns'),
                target_name=target_name,
                id_columns=id_columns,
                variance_threshold=transform_config.get('variance_threshold'),
                random_state=transform_config.get('random_state', 42)
            )
            
        elif transform_name == 'shrinkage':
            df_train_processed, df_test_processed = apply_shrinkage(
                df_train_processed, df_test_processed,
                columns=transform_config.get('columns'),
                target_name=target_name,
                id_columns=id_columns,
                suffix=transform_config.get('suffix', '_shrink')
            )
            
        elif transform_name == 'remove_outliers':
            # Only apply to training data
            df_train_processed = remove_outliers(
                df_train_processed,
                method=transform_config.get('method', 'zscore'),
                threshold=transform_config.get('threshold', 3.0),
                columns=transform_config.get('columns'),
                target_name=target_name,
                id_columns=id_columns
            )
            
        elif transform_name == 'select_features':
            df_train_processed, df_test_processed = select_features(
                df_train_processed, df_test_processed,
                method=transform_config.get('method', 'correlation'),
                threshold=transform_config.get('threshold', 0.1),
                k=transform_config.get('k', 20),
                target_name=target_name,
                id_columns=id_columns,
                random_state=transform_config.get('random_state', 42)
            )
            
        elif transform_name == 'custom':
            # Apply a custom transformation function
            custom_func = transform_config.get('function')
            if custom_func and callable(custom_func):
                trans_start = time()
                df_train_processed, df_test_processed = custom_func(
                    df_train_processed, df_test_processed, **transform_config.get('params', {})
                )
                print(f"Applied custom transformation in {time()-trans_start:.2f}s")
            else:
                print(f"Warning: Custom function not callable for transformation '{transform_name}'")
                
        else:
            print(f"Warning: Unknown transformation '{transform_name}' - skipping")
    
    print(f"All data transformations completed in {time()-overall_start_time:.2f}s")
    return df_train_processed, df_test_processed