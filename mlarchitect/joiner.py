import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import mlarchitect
from typing import Dict, List, Union, Optional, Callable
import logging

# Configure logger for DataJoiner
logger = logging.getLogger('mlarchitect.joiner')

class DataJoiner:
    def __init__(self, data_dir, output_dir, sources, processed_dir=None, log_level=logging.INFO):
        """
        Initialize the DataJoiner.

        Args:
            data_dir (str): Directory containing the parquet files.
            output_dir (str): Directory where output files will be saved.
            sources (list): List of source names to merge.
            processed_dir (str, optional): Directory where processed data will be saved. 
                                         If None, defaults to "processed/v1/"
            log_level (int, optional): Logging level (default: logging.INFO)
        """
        # Configure logger
        self._setup_logger(log_level)
        logger.info("Initializing DataJoiner")
        
        self.data_dir = os.path.join(mlarchitect.mlarchitect_config['PATH'], data_dir)
        self.output_dir = os.path.join(mlarchitect.mlarchitect_config['PATH'], output_dir)
        self.sources = sources
        self.seed = mlarchitect.mlarchitect_config.get('SEED', 42)
        self.processed_dir = processed_dir or "processed/v1/"
        
        # Create output directory if it doesn't exist.
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")

    def _setup_logger(self, log_level):
        """Set up logging with the specified log level."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(log_level)

    def join_data(self, transformations=None):
        """
        Merge features from multiple parquet files for both training and testing data.
        Optionally apply transformations to the joined data.

        Args:
            transformations (dict, optional): Dictionary of transformation functions to apply.
                Example: {
                    'normalize': {'method': 'standard', 'columns': ['col1', 'col2']},
                    'pca': {'n_components': 5, 'columns': ['col1', 'col2', 'col3']},
                    'shrinkage': {'suffix': '_shrink'},
                    'remove_outliers': {'method': 'zscore', 'threshold': 3.0},
                    'select_features': {'method': 'correlation', 'k': 50},
                    'custom': {'function': my_custom_function, 'params': {...}}
                }

        Returns:
            tuple: (df_train_joined, df_test_joined) joined pandas DataFrames.
        """
        logger.info("Starting data joining process")
        target_name = mlarchitect.mlarchitect_config['TARGET_NAME']
        fit_type = mlarchitect.mlarchitect_config['FIT_TYPE']
        balance_flag = mlarchitect.mlarchitect_config.get('BALANCE', False)
        
        train_id_column = mlarchitect.mlarchitect_config['TRAIN_ID']
        test_id_column = mlarchitect.mlarchitect_config['TEST_ID']
        
        train_dataframes = []
        test_dataframes = []
        
        for idx, source in enumerate(self.sources):
            train_file = os.path.join(self.data_dir, f"{source}_train.parquet")
            test_file = os.path.join(self.data_dir, f"{source}_test.parquet")
            
            logger.info(f"Processing source: {source}")
            df_train = pd.read_parquet(train_file)
            df_test = pd.read_parquet(test_file)
            
            # Sort by 'id' if the column exists.
            if train_id_column in df_train.columns:
                df_train = df_train.sort_values(by=train_id_column).reset_index(drop=True)
            if test_id_column in df_test.columns:
                df_test = df_test.sort_values(by=test_id_column).reset_index(drop=True)
            
            # Prepare DataFrames for this source
            train_source_data = {}
            test_source_data = {}
            
            # Handle ID columns for the first source
            if idx == 0:
                if train_id_column in df_train.columns:
                    train_source_data[train_id_column] = df_train[train_id_column]
                if test_id_column in df_test.columns:
                    test_source_data[test_id_column] = df_test[test_id_column]
                
                # For the target column, take it from the first source only
                if target_name in df_train.columns:
                    train_source_data[target_name] = df_train[target_name]
                else:
                    error_msg = f"Target column '{target_name}' not found in source '{source}'"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Add feature columns with appropriate naming
            for col in df_train.columns:
                if col in [train_id_column, target_name]:
                    continue
                
                if idx == 0:
                    # First source - keep original column names
                    new_col = col
                else:
                    # Other sources - add prefix
                    new_col = f"{source}:{col}"
                    
                train_source_data[new_col] = df_train[col]
                test_source_data[new_col] = df_test[col]
            
            # Add to the list of DataFrames
            train_dataframes.append(pd.DataFrame(train_source_data))
            test_dataframes.append(pd.DataFrame(test_source_data))
        
        # Combine all DataFrames at once
        df_train_joined = pd.concat(train_dataframes, axis=1)
        df_test_joined = pd.concat(test_dataframes, axis=1)
        
        logger.info(f"Combined data shapes - Train: {df_train_joined.shape}, Test: {df_test_joined.shape}")
        
        # Store the original joined data
        df_train_original = df_train_joined.copy()
        df_test_original = df_test_joined.copy()
        
        # Apply transformations if specified
        df_train_processed = df_train_joined
        df_test_processed = df_test_joined
        
        if transformations:
            try:
                # Use the mlarchitect library's process_transfo module
                from mlarchitect.process_transformation import process_data
                
                logger.info("Using transformation functions from mlarchitect library")
                df_train_processed, df_test_processed = process_data(
                    df_train=df_train_joined,
                    df_test=df_test_joined,
                    transformations=transformations,
                    target_name=target_name,
                    id_columns=[train_id_column, test_id_column]
                )
            except ImportError:
                # Fall back to basic transformations if process_transfo.py not available
                logger.warning("process_transfo.py not found. Using basic transformations only.")
                df_train_processed, df_test_processed = self._apply_basic_transformations(
                    df_train_joined, df_test_joined, transformations, target_name
                )
            
            # Log processing results
            logger.info(f"Processed data shapes - Train: {df_train_processed.shape}, Test: {df_test_processed.shape}")
            
            # Save the processed data
            processed_output_dir = os.path.join(mlarchitect.mlarchitect_config['PATH'], self.processed_dir)
            os.makedirs(processed_output_dir, exist_ok=True)
            
            train_output_path = os.path.join(processed_output_dir, "train.parquet")
            test_output_path = os.path.join(processed_output_dir, "test.parquet")
            
            df_train_processed.to_parquet(train_output_path)
            df_test_processed.to_parquet(test_output_path)
            
            logger.info(f"Saved processed train data to {train_output_path}")
            logger.info(f"Saved processed test data to {test_output_path}")
        
        # If balancing is required (and task is classification), oversample the minority class.
        if balance_flag and fit_type.upper() == 'CLASSIFICATION':
            logger.info("Balancing training data with oversampling")
            df_train_processed = self.balance_data(df_train_processed, target_name)
        
        # Save the original joined data
        self.save_data(df_train_original, df_test_original)
        
        logger.info("Data joining process completed")
        
        # Return the processed data
        return df_train_processed, df_test_processed

    def _apply_basic_transformations(self, df_train, df_test, transformations, target_name):
        """
        Apply basic transformations when the advanced process_transfo.py module is not available.
        This is a simplified version for backward compatibility.
        
        Args:
            df_train: Training DataFrame
            df_test: Testing DataFrame
            transformations: Dictionary of transformations
            target_name: Name of the target column
            
        Returns:
            Tuple of transformed (df_train, df_test)
        """
        # Keep a copy of important columns that should not be transformed
        reserved_cols = {col: df_train[col] for col in df_train.columns 
                        if col == target_name or col == mlarchitect.mlarchitect_config['TRAIN_ID']}
        
        test_reserved_cols = {col: df_test[col] for col in df_test.columns 
                             if col == mlarchitect.mlarchitect_config['TEST_ID']}
        
        # Process each transformation in order
        for transform_name, transform_config in transformations.items():
            logger.info(f"Applying basic transformation: {transform_name}")
            
            # Only implement a few basic transformations for backward compatibility
            if transform_name == 'normalize':
                # Handle normalization
                from sklearn.preprocessing import StandardScaler, MinMaxScaler
                
                # Get columns to normalize
                columns = transform_config.get('columns', [])
                method = transform_config.get('method', 'standard')
                
                # If no columns specified, normalize all numerical columns except target and ID
                if not columns:
                    columns = df_train.select_dtypes(include=['number']).columns.tolist()
                    columns = [col for col in columns if col != target_name 
                              and col != mlarchitect.mlarchitect_config['TRAIN_ID']
                              and col != mlarchitect.mlarchitect_config['TEST_ID']]
                
                # Check if columns exist in both dataframes
                valid_columns = [col for col in columns if col in df_train.columns and col in df_test.columns]
                
                # Select appropriate scaler
                if method.lower() == 'minmax':
                    scaler = MinMaxScaler()
                else:  # default to standard scaling
                    scaler = StandardScaler()
                
                # Fit scaler on training data and transform both train and test
                if valid_columns:
                    df_train[valid_columns] = scaler.fit_transform(df_train[valid_columns])
                    df_test[valid_columns] = scaler.transform(df_test[valid_columns])
                    logger.info(f"Applied {method} normalization to {len(valid_columns)} columns")
            
            elif transform_name == 'pca':
                # Handle PCA
                from sklearn.decomposition import PCA
                
                # Get configuration
                n_components = transform_config.get('n_components', 5)
                columns = transform_config.get('columns', [])
                
                # If no columns specified, use all numerical columns except target and ID
                if not columns:
                    columns = df_train.select_dtypes(include=['number']).columns.tolist()
                    columns = [col for col in columns if col != target_name 
                              and col != mlarchitect.mlarchitect_config['TRAIN_ID']
                              and col != mlarchitect.mlarchitect_config['TEST_ID']]
                
                # Check if columns exist in both dataframes
                valid_columns = [col for col in columns if col in df_train.columns and col in df_test.columns]
                
                if valid_columns:
                    # Initialize PCA
                    pca = PCA(n_components=min(n_components, len(valid_columns)), random_state=self.seed)
                    
                    # Extract features for PCA
                    X_train = df_train[valid_columns].values
                    X_test = df_test[valid_columns].values
                    
                    # Fit and transform
                    X_train_pca = pca.fit_transform(X_train)
                    X_test_pca = pca.transform(X_test)
                    
                    # Add PCA components as new columns
                    for i in range(X_train_pca.shape[1]):
                        df_train[f'PCA_{i+1}'] = X_train_pca[:, i]
                        df_test[f'PCA_{i+1}'] = X_test_pca[:, i]
                    
                    logger.info(f"Applied PCA, created {X_train_pca.shape[1]} new components")
            
            elif transform_name == 'custom':
                # Handle custom transformation function
                custom_func = transform_config.get('function')
                params = transform_config.get('params', {})
                
                if custom_func and callable(custom_func):
                    try:
                        df_train, df_test = custom_func(df_train, df_test, **params)
                        logger.info(f"Applied custom transformation: {custom_func.__name__}")
                    except Exception as e:
                        logger.error(f"Error applying custom transformation: {e}")
        
        # Restore reserved columns
        for col, values in reserved_cols.items():
            df_train[col] = values
            
        for col, values in test_reserved_cols.items():
            df_test[col] = values
        
        logger.info(f"Basic transformations completed, final shapes - Train: {df_train.shape}, Test: {df_test.shape}")
        return df_train, df_test

    def balance_data(self, df, target_name):
        """
        Oversample the minority class to balance the dataset.
        This is a simple implementation for classification tasks.

        Args:
            df (DataFrame): Training DataFrame.
            target_name (str): Name of the target variable.
        
        Returns:
            DataFrame: A balanced DataFrame.
        """
        logger.info("Starting data balancing")
        counts = df[target_name].value_counts()
        max_count = counts.max()
        
        df_balanced = df.copy()
        classes = counts.index.tolist()
        for cls in classes:
            cls_rows = df[df[target_name] == cls]
            if len(cls_rows) < max_count:
                additional = max_count - len(cls_rows)
                sampled = cls_rows.sample(n=additional, replace=True, random_state=self.seed)
                df_balanced = pd.concat([df_balanced, sampled], axis=0)
                logger.info(f"Class {cls}: Added {additional} samples to balance")
        
        df_balanced = df_balanced.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        logger.info(f"Balancing completed, final data shape: {df_balanced.shape}")
        return df_balanced

    def save_data(self, df_train, df_test):
        """
        Save the joined DataFrames to the output directory as parquet files.

        Args:
            df_train (DataFrame): The joined training DataFrame.
            df_test (DataFrame): The joined testing DataFrame.
        """
        train_output_path = os.path.join(self.output_dir, "train.parquet")
        test_output_path = os.path.join(self.output_dir, "test.parquet")
        
        df_train.to_parquet(train_output_path)
        df_test.to_parquet(test_output_path)
        logger.info(f"Saved joined train data to {train_output_path}")
        logger.info(f"Saved joined test data to {test_output_path}")
        
    def save_processed_data(self, df_train_processed, df_test_processed, version=None):
        """
        Save the processed DataFrames to the processed directory as parquet files.
        
        Args:
            df_train_processed (DataFrame): The processed training DataFrame.
            df_test_processed (DataFrame): The processed testing DataFrame.
            version (str, optional): Version string to override the default processed directory.
        """
        # Use provided version or default to the instance's processed_dir
        processed_dir = version or self.processed_dir
        processed_output_dir = os.path.join(mlarchitect.mlarchitect_config['PATH'], processed_dir)
        os.makedirs(processed_output_dir, exist_ok=True)
        
        train_output_path = os.path.join(processed_output_dir, "train.parquet")
        test_output_path = os.path.join(processed_output_dir, "test.parquet")
        
        df_train_processed.to_parquet(train_output_path)
        df_test_processed.to_parquet(test_output_path)
        
        logger.info(f"Saved processed train data to {train_output_path}")
        logger.info(f"Saved processed test data to {test_output_path}")
        
        return train_output_path, test_output_path
