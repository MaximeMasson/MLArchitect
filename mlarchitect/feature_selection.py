import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import shap
import warnings
import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any

# Local imports
import mlarchitect
from mlarchitect.model_config_class import configs
from mlarchitect.model_manager import ModelManager

# Configure logging
logger = logging.getLogger(__name__)

class FeatureSelection:
    """
    A comprehensive feature selection framework for machine learning pipelines.
    
    This class implements various feature selection methods including:
    - Correlation-based feature elimination
    - Noise detection and removal
    - Model-based feature importance
    - SHAP-based feature analysis
    """
    def __init__(self, input_path=None, output_path=None, name='feature_selection', 
                 correlation_threshold=0.9, subsample_ratio=0.3, top_features_ratio=0.4,
                 noise_std_threshold=3.0, noise_outlier_ratio=0.1, model_name='xgboost',
                 log_level=logging.INFO):
        """
        Initialize the FeatureSelection class with paths similar to ModelManager.
        
        Parameters:
        -----------
        input_path : str, optional
            Path to input data directory
        output_path : str, optional
            Path to output directory for results
        name : str, default='feature_selection'
            Name of this feature selection run
        correlation_threshold : float, default=0.9
            Threshold for identifying highly correlated features (0.0 to 1.0)
        subsample_ratio : float, default=0.3
            Ratio of data to use for quick training (0.0 to 1.0)
        top_features_ratio : float, default=0.4
            Ratio of top features to retain based on importance (0.0 to 1.0)
        noise_std_threshold : float, default=3.0
            Number of standard deviations to use for noise detection
        noise_outlier_ratio : float, default=0.1
            Maximum ratio of outliers allowed before considering a feature noisy
        model_name : str, default='xgboost'
            Name of the model to use for feature importance calculation
        log_level : int, default=logging.INFO
            Logging level to use
        """
        # Configure logger
        self._setup_logger(log_level)
        logger.info(f"Initializing FeatureSelection with name: {name}")
        
        # Validate parameters
        self._validate_parameters(
            correlation_threshold, subsample_ratio, top_features_ratio, 
            noise_std_threshold, noise_outlier_ratio
        )
        
        # Set paths
        PATH = mlarchitect.mlarchitect_config.get('PATH', '')
        self.input_path = os.path.join(PATH, input_path) if input_path and PATH else input_path
        self.output_path = os.path.join(PATH, output_path) if output_path and PATH else output_path
        self.name = name
        
        # Create output directory if it doesn't exist
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
            logger.info(f"Output directory set to: {self.output_path}")
        
        # Feature selection parameters
        self.correlation_threshold = correlation_threshold
        self.subsample_ratio = subsample_ratio
        self.top_features_ratio = top_features_ratio
        self.noise_std_threshold = noise_std_threshold
        self.noise_outlier_ratio = noise_outlier_ratio
        self.model_name = model_name
        
        # Initialize state variables
        self.data = None
        self.removed_correlated_features = []
        self.removed_noisy_features = []
        self.top_features = []
        self.feature_importance = None
        self.shap_values = None
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.excluded_features = []  # New attribute to store excluded features
        
        # Load data if input path is provided
        if self.input_path:
            try:
                input_file = os.path.join(self.input_path, "train.parquet")
                if os.path.exists(input_file):
                    self.data = pd.read_parquet(input_file)
                    logger.info(f"Data loaded during initialization: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
                else:
                    logger.warning(f"Input file not found during initialization: {input_file}")
            except Exception as e:
                logger.warning(f"Could not load data during initialization: {str(e)}")
    
    def _setup_logger(self, log_level):
        """Set up logging with the specified log level."""
        # Remove all existing handlers to prevent duplicate logging
        while logger.handlers:
            logger.removeHandler(logger.handlers[0])
            
        # Add a new handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(log_level)
        
    def _validate_parameters(self, correlation_threshold, subsample_ratio, 
                            top_features_ratio, noise_std_threshold, noise_outlier_ratio):
        """Validate that parameters are within acceptable ranges."""
        if not 0 <= correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
        if not 0 < subsample_ratio <= 1:
            raise ValueError("subsample_ratio must be between 0 and 1")
        if not 0 < top_features_ratio <= 1:
            raise ValueError("top_features_ratio must be between 0 and 1")
        if noise_std_threshold <= 0:
            raise ValueError("noise_std_threshold must be greater than 0")
        if not 0 < noise_outlier_ratio < 1:
            raise ValueError("noise_outlier_ratio must be between 0 and 1")
    
    def exclude_features(self, features):
        """
        Exclude specific features from testing without removing them from the dataset.
        These features will be kept in the final selected features list.
        
        Parameters:
        -----------
        features : list or str
            List of feature names to exclude from testing, or a single feature name.
            
        Returns:
        --------
        self : FeatureSelection
            The current instance for method chaining
        """
        # Convert single string to list for uniformity
        if isinstance(features, str):
            features = [features]
        elif not isinstance(features, list):
            raise ValueError("Features must be provided as a list or a string")
        
        # Verify features exist in the data if data is loaded
        if self.data is not None:
            non_existent = [f for f in features if f not in self.data.columns]
            if non_existent:
                logger.warning(f"Some features to exclude do not exist in the data: {non_existent}")
                features = [f for f in features if f in self.data.columns]
        
        # Add to excluded features list, removing duplicates
        self.excluded_features.extend(features)
        self.excluded_features = list(dict.fromkeys(self.excluded_features))  # Remove duplicates while preserving order
        
        logger.info(f"Added {len(features)} features to excluded list. Total excluded: {len(self.excluded_features)}")
        logger.debug(f"Excluded features: {self.excluded_features}")
        
        return self
    
    def run(self, model_name=None, cv_column=None, cv_folds=None):
        """
        Run the complete feature selection pipeline.
        
        Parameters:
        -----------
        model_name : str, optional
            Name of the model to use for importance calculation.
            If None, uses the model_name provided during initialization.
        cv_column : str, optional
            Column name to use for grouping CV splits (e.g., "DATE").
            If provided, prevents data leakage in time series data.
        cv_folds : int, optional
            Number of cross-validation folds to use.
            If None, defaults to 3.
            
        Returns:
        --------
        list
            List of selected feature names
        """
        logger.info(f"Starting feature selection pipeline: {self.name}")
        
        if not self.input_path or not self.output_path:
            logger.error("Input and output paths must be set")
            raise ValueError("Input and output paths must be set")
        
        # Use provided model_name or fall back to instance attribute
        if model_name is not None:
            self.model_name = model_name
        
        # Set default cv_folds if not provided
        if cv_folds is None:
            cv_folds = 3
            
        # Log time-series handling if cv_column is provided
        if cv_column:
            logger.info(f"Using time-series handling with cv_column='{cv_column}'")
            
        # Load data
        logger.info("Loading data...")
        try:
            input_file = os.path.join(self.input_path, "train.parquet")
            if not os.path.exists(input_file):
                logger.error(f"Input file not found: {input_file}")
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            self.data = pd.read_parquet(input_file)
            logger.info(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            
            # Check if cv_column exists in the data
            if cv_column and cv_column not in self.data.columns:
                logger.warning(f"cv_column '{cv_column}' not found in data. Continuing without time-series handling.")
                cv_column = None
                
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
        
        # Identify target column
        target = mlarchitect.mlarchitect_config.get('TARGET_NAME')
        if not target or target not in self.data.columns:
            logger.error(f"Target column '{target}' not found in data")
            raise ValueError(f"Target column '{target}' not found in data")
        
        # Store the original data for later use
        original_data = self.data.copy()
        
        # Create a working copy for testing that excludes specified features
        if self.excluded_features:
            logger.info(f"Excluding {len(self.excluded_features)} features from testing")
            logger.debug(f"Excluded features: {self.excluded_features}")
            
            # Create a list of features to exclude, but ensure cv_column is kept if needed
            features_to_exclude = self.excluded_features.copy()
            if cv_column and cv_column in features_to_exclude:
                logger.info(f"Keeping {cv_column} in data for cross-validation although it's excluded from features")
                features_to_exclude.remove(cv_column)
            
            # Remove excluded features from testing data
            testing_features = [col for col in self.data.columns if col != target and col not in features_to_exclude]
            
            # Ensure cv_column is in the testing features if needed
            if cv_column and cv_column not in testing_features and cv_column in self.data.columns:
                testing_features.append(cv_column)
                
            working_data = self.data[testing_features + [target]]
        else:
            working_data = self.data
            
        # Use the filtered data for feature selection
        X = working_data.drop(columns=[target])
        y = working_data[target]
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Temporarily set self.data to working_data for feature selection process
        self.data = working_data
        
        # Remove highly correlated features
        logger.info("Removing highly correlated features...")
        self.remove_highly_correlated_features()
        
        # Remove noisy features
        logger.info("Removing noisy features...")
        self.remove_noisy_features()
        
        # Calculate feature importance with proper time-series handling
        logger.info(f"Calculating feature importance using {self.model_name} with {cv_folds}-fold CV...")
        
        # CRITICAL FIX: Prepare data for model training properly
        # For quick_feature_evaluation, we need:
        # 1. Include all features in X for model training
        # 2. Make sure cv_column is present for time-series CV but not used in model
        X_for_eval = X.copy()  # Start with X containing ALL features, including DATE column if it exists
        
        # Check if we need to exclude cv_column from model features
        if cv_column and cv_column in X_for_eval.columns:
            logger.info(f"Using {cv_column} for CV splits but excluding it from model features")
            model_features = [col for col in X_for_eval.columns if col != cv_column]
        else:
            # If cv_column is not in X, we can't use it for CV and must warn the user
            if cv_column:
                logger.warning(f"{cv_column} specified for CV but not found in data after filtering - cannot use for CV")
                cv_column = None
            model_features = X_for_eval.columns.tolist()
        
        self.feature_importance = self.quick_feature_evaluation(
            X=X_for_eval,  # Pass ALL features including DATE if present
            y=y,
            model_name=self.model_name,
            cv_folds=cv_folds,
            cv_column=cv_column  # Pass cv_column name (or None if not available)
        )
        
        # Restore the original data including excluded features
        testing_data = self.data.copy()  # Save the testing data with removed features
        self.data = original_data
        
        # Check if we need to add excluded features back to selected features
        if self.excluded_features:
            logger.info(f"Adding {len(self.excluded_features)} excluded features to the selected features")
            self.top_features.extend([f for f in self.excluded_features if f in original_data.columns and f != cv_column])
            # Remove duplicates while preserving order
            self.top_features = list(dict.fromkeys(self.top_features))
        
        # Save results
        logger.info("Saving feature selection results...")
        self._save_results()
        
        logger.info(f"Feature selection complete. Selected {len(self.top_features)} features.")
        return self.top_features
    
    def _save_results(self):
        """Save feature selection results to output directory."""
        if not self.output_path:
            logger.warning("Output path not set. Results will not be saved.")
            return
        
        try:    
            # Create a timestamped results directory
            results_dir = os.path.join(self.output_path, f"{self.name}_{self.run_timestamp}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save selected features and metadata
            results_file = os.path.join(results_dir, f"results.json")
            results = {
                "selected_features": self.top_features,
                "removed_correlated": self.removed_correlated_features,
                "removed_noisy": self.removed_noisy_features,
                "excluded_features": self.excluded_features,  # Include excluded features in results
                "timestamp": self.run_timestamp,
                "parameters": {
                    "correlation_threshold": self.correlation_threshold,
                    "subsample_ratio": self.subsample_ratio,
                    "top_features_ratio": self.top_features_ratio,
                    "noise_std_threshold": self.noise_std_threshold,
                    "noise_outlier_ratio": self.noise_outlier_ratio,
                    "model_name": self.model_name
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Save feature importance if available
            if isinstance(self.feature_importance, pd.DataFrame):
                importance_file = os.path.join(results_dir, f"feature_importance.csv")
                self.feature_importance.to_csv(importance_file, index=False)
                
                # Generate importance plot
                plt.figure(figsize=(12, max(6, len(self.feature_importance) // 10)))
                top_n = min(50, len(self.feature_importance))  # Limit to top 50 features
                sns.barplot(
                    x='Importance', 
                    y='Feature', 
                    data=self.feature_importance.head(top_n)
                )
                plt.title('Top Features by Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'importance_plot.png'))
                plt.close()
                
            logger.info(f"Results saved to {results_dir}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise
    
    def set_data(self, data):
        """Set the data for feature selection."""
        self.data = data
        return self
    
    def set_params(self, correlation_threshold=0.9, subsample_ratio=0.3, top_features_ratio=0.4):
        """
        Set parameters for feature selection process.
        
        Parameters:
        -----------
        correlation_threshold : float
            Threshold for identifying highly correlated features (0.0 to 1.0)
        subsample_ratio : float
            Ratio of data to use for quick training (0.0 to 1.0)
        top_features_ratio : float
            Ratio of top features to retain based on importance (0.0 to 1.0)
        """
        self.correlation_threshold = correlation_threshold
        self.subsample_ratio = subsample_ratio
        self.top_features_ratio = top_features_ratio
        return self
    
    def remove_highly_correlated_features(self, threshold=None):
        """
        Remove highly correlated features, keeping only one from each correlated group.
        
        Parameters:
        -----------
        threshold : float, optional
            Correlation threshold. If not provided, uses the instance's threshold.
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with highly correlated features removed.
        """
        if threshold is not None:
            self.correlation_threshold = threshold
            
        if self.data is None:
            logger.error("Data not loaded. Set data using set_data() first.")
            raise ValueError("Data not loaded. Set data using set_data() first.")
            
        # Calculate the correlation matrix
        numerical_data = self.data.select_dtypes(include=['float64', 'int64'])
        logger.info(f"Calculating correlation matrix for {numerical_data.shape[1]} numerical features")
        
        try:
            corr_matrix = numerical_data.corr().abs()
            
            # Create a mask for the upper triangle
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation greater than threshold
            to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
            
            # Store the removed features
            self.removed_correlated_features = to_drop
            
            # Remove correlated features
            if to_drop:
                self.data = self.data.drop(columns=to_drop)
                logger.info(f"Removed {len(to_drop)} highly correlated features with threshold {self.correlation_threshold}")
                logger.info(f"Removed correlated features: {to_drop}")
            else:
                logger.info(f"No highly correlated features found with threshold {self.correlation_threshold}")
                
            return self.data
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            raise
    
    def remove_noisy_features(self):
        """
        Remove features exhibiting irregular distributions or suspicious values.
        This method identifies and removes:
        1. Features with extremely high variance
        2. Features with too many outliers
        
        Returns:
        --------
        pandas.DataFrame
            Dataset with noisy features removed.
        """
        if self.data is None:
            logger.error("Data not loaded. Set data using set_data() first.")
            raise ValueError("Data not loaded. Set data using set_data() first.")
            
        logger.info("Identifying noisy features...")
            
        try:
            numerical_data = self.data.select_dtypes(include=['float64', 'int64'])
            logger.info(f"Checking {numerical_data.shape[1]} numerical features for noise")
            
            noisy_features = []
            extreme_value_metrics = {}
            
            # Check for features with extremely high variance
            std_values = numerical_data.std()
            mean_std = std_values.mean()
            std_of_std = std_values.std()
            
            threshold = mean_std + self.noise_std_threshold * std_of_std
            logger.debug(f"Variance threshold: {threshold:.6f} (mean std: {mean_std:.6f}, std of std: {std_of_std:.6f})")
            
            high_var_features = std_values[std_values > threshold].index.tolist()
            if high_var_features:
                logger.debug(f"Found {len(high_var_features)} features with high variance")
                noisy_features.extend(high_var_features)
            
            # Check for features with too many extreme values
            logger.debug(f"Checking for outliers using threshold: {self.noise_outlier_ratio:.2f}")
            for col in numerical_data.columns:
                mean = numerical_data[col].mean()
                std = numerical_data[col].std()
                
                if std == 0:  # Skip constant features
                    logger.debug(f"Skipping constant feature: {col}")
                    continue
                    
                upper_bound = mean + self.noise_std_threshold * std
                lower_bound = mean - self.noise_std_threshold * std
                
                extreme_ratio = ((numerical_data[col] > upper_bound) | 
                                (numerical_data[col] < lower_bound)).mean()
                
                extreme_value_metrics[col] = extreme_ratio
                
                if extreme_ratio > self.noise_outlier_ratio:
                    logger.debug(f"Feature {col} has {extreme_ratio:.4f} outlier ratio")
                    noisy_features.append(col)
            
            # Remove duplicates
            noisy_features = list(set(noisy_features))
            
            # Store the removed features
            self.removed_noisy_features = noisy_features
            
            # Remove noisy features
            if noisy_features:
                self.data = self.data.drop(columns=noisy_features)
                logger.info(f"Removed {len(noisy_features)} noisy features")
                logger.debug(f"Removed noisy features: {noisy_features}")
            else:
                logger.info("No noisy features found")
                
            return self.data
            
        except Exception as e:
            logger.error(f"Error in noise analysis: {str(e)}")
            raise
    
    def quick_feature_evaluation(self, X, y, model_name=None, target_name=None, save_model=False, save_dir='feature_selection_models', cv_folds=3, cv_column=None):
        """
        Train a quick model to evaluate feature importance with enhanced functionality.
        Uses ModelManager's cross-validation capabilities to get more robust feature importance scores.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series or numpy.ndarray
            Target variable
        model_name : str, optional
            Name of the model to use (from model_config_class)
            If None, uses the model_name provided during initialization
        target_name : str, optional
            Name of the target variable. If not provided, uses the default in mlarchitect_config
        save_model : bool, default=False
            Whether to save the trained model
        save_dir : str, default='feature_selection_models'
            Directory to save the model and results
        cv_folds : int, default=3
            Number of cross-validation folds to use (lower for speed in feature selection)
        cv_column : str, optional
            Column name to use for grouping CV splits (e.g., "DATE").
            If provided, prevents data leakage in time series data.
            
        Returns:
        --------
        pandas.DataFrame
            Feature importance DataFrame sorted in descending order with stability metrics
        """
        # Set model name if not provided
        if model_name is None:
            model_name = self.model_name
            
        # Set target name if not provided
        target_name = target_name or mlarchitect.mlarchitect_config.get('TARGET_NAME', 'target')
        logger.info(f"Starting feature evaluation using {model_name} model with {cv_folds}-fold CV")
        
        # Validate input data
        if X is None or y is None:
            logger.error("Feature matrix X or target variable y is None")
            raise ValueError("Feature matrix X or target variable y is None")
            
        if X.shape[0] != y.shape[0]:
            logger.error(f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}")
            raise ValueError(f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}")
            
        try:
            # Get seed from config to use consistently
            seed = mlarchitect.mlarchitect_config.get('SEED', 42)
            
            # Take a subsample for quick training if needed
            if self.subsample_ratio < 1.0:
                # Ensure stratified sampling if binary classification
                fit_type = mlarchitect.mlarchitect_config.get('FIT_TYPE', 'regression')
                stratify_arg = y if fit_type == 'bclass' else None
                
                X_sub, X_val, y_sub, y_val = train_test_split(
                    X, y, 
                    train_size=self.subsample_ratio,
                    random_state=seed,
                    stratify=stratify_arg
                )
                logger.info(f"Using {len(X_sub)} samples ({self.subsample_ratio*100:.1f}% of data) for feature evaluation")
            else:
                # Use all data but still create a validation set
                fit_type = mlarchitect.mlarchitect_config.get('FIT_TYPE', 'regression')
                stratify_arg = y if fit_type == 'bclass' else None
                
                X_sub, X_val, y_sub, y_val = train_test_split(
                    X, y, 
                    test_size=0.2,
                    random_state=seed,
                    stratify=stratify_arg
                )
                logger.info(f"Using {len(X_sub)} samples for feature evaluation")
            
            # Get model configuration from config list
            model_keys = list(configs.keys())
            if model_name not in configs:
                logger.warning(f"Model {model_name} not found in configs. Using default model.")
                model_name = 'gradient_boosting' if 'gradient_boosting' in configs else model_keys[0]
            
            model_config = configs[model_name]
            logger.info(f"Using {model_name} model for feature evaluation")
            
            # -----------------------------------------------------
            # Method 1: Cross-validation based feature importances
            # -----------------------------------------------------
            # Use ModelManager with cross-validation to get more robust importances
            logger.info("Starting cross-validated feature importance calculation")
            
            # Create a ModelManager instance specifically for feature selection
            from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
            metrics_list = [r2_score] if mlarchitect.mlarchitect_config.get('FIT_TYPE', 'regression') == 'regression' else [accuracy_score]
            
            # Initialize ModelManager with minimal configuration
            model_manager = ModelManager(
                model_config=model_config,
                use_time_cv=False,
                tuning=False,  # No tuning for speed
                metrics=metrics_list,
                test_size=0.2
            )
            
            # Set up data in ModelManager
            model_manager.X_train = X_sub
            model_manager.y_train = y_sub
            model_manager.X_val = X_val
            model_manager.y_val = y_val
            model_manager.X_test = X_val.copy()  # Use validation set as test for simplicity
            
            # Create list of features to exclude from training (but keep for CV)
            excluded_features = []
            
            # CRITICAL FIX: Properly handle DATE column for CV
            # Make sure cv_column is present in the data for CV splits
            if cv_column:
                # Check if cv_column exists in the data
                if cv_column not in X_sub.columns:
                    logger.error(f"CV column '{cv_column}' not found in feature matrix. Available columns: {X_sub.columns.tolist()}")
                    raise KeyError(f"CV column '{cv_column}' not found in feature matrix")
                
                # Always exclude cv_column from model features but keep it for CV
                excluded_features.append(cv_column)
                logger.info(f"Excluding {cv_column} from model features but using it for CV splits")
                
            # Perform cross-validation with specified excluded features
            logger.info(f"Performing {cv_folds}-fold cross-validation for feature evaluation")
            model_manager.perform_cv(cv_folds=cv_folds, cv_column=cv_column, excluded_features=excluded_features)
            
            # Train a single model for feature importances
            logger.info("Training final model for feature importance extraction")
            model_manager.train_final_model(final_tuning=False, save_dir=save_dir, model_name=f"{model_name}_feature_selection")
            
            # Get all feature names except excluded ones
            # CRITICAL FIX: Exclude cv_column from feature_names to match model feature dimensions
            feature_names = [col for col in X_sub.columns if col not in (excluded_features or [])]
            
            if hasattr(model_manager, 'best_model'):
                best_model = model_manager.best_model
                
                # Extract feature importances
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    logger.info("Feature importances extracted directly from best model")
                elif hasattr(best_model, 'coef_'):
                    importances = np.abs(best_model.coef_).mean(axis=0) if best_model.coef_.ndim > 1 else np.abs(best_model.coef_)
                    logger.info("Feature importances extracted from model coefficients")
                elif hasattr(best_model, 'named_steps') and 'model' in best_model.named_steps:
                    # Handle pipeline case
                    model_in_pipeline = best_model.named_steps['model']
                    if hasattr(model_in_pipeline, 'feature_importances_'):
                        importances = model_in_pipeline.feature_importances_
                    elif hasattr(model_in_pipeline, 'coef_'):
                        importances = np.abs(model_in_pipeline.coef_).mean(axis=0) if model_in_pipeline.coef_.ndim > 1 else np.abs(model_in_pipeline.coef_)
                    else:
                        logger.warning("No feature_importances_ or coef_ found in model. Using random importance.")
                        importances = np.random.rand(len(feature_names))
                else:
                    logger.warning("Model doesn't provide native importance. Using random importance.")
                    # Use permutation importance as fallback
                    try:
                        from sklearn.inspection import permutation_importance
                        X_val_features = X_val[feature_names]  # Only use actual model features
                        result = permutation_importance(
                            best_model, X_val_features, y_val, 
                            n_repeats=10, random_state=seed
                        )
                        importances = result.importances_mean
                    except Exception as e:
                        logger.error(f"Failed to compute permutation importance: {e}")
                        logger.warning("Using random ranking as last resort")
                        # Use config seed for random ranking too
                        np.random.seed(seed)
                        importances = np.random.rand(len(feature_names))
                
                # Create feature importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances,
                })
                
                # Add normalized importance (percentage)
                if importances.sum() > 0:
                    importance_df['Importance_Normalized'] = importance_df['Importance'] / importances.sum() * 100
                else:
                    importance_df['Importance_Normalized'] = 100 / len(importances)
                    
                # Sort by importance
                importance_df = importance_df.sort_values(by='Importance', ascending=False)
                
                # Add importance rank
                importance_df['Rank'] = np.arange(1, len(importance_df) + 1)
                
                # Select top features based on top_features_ratio
                num_top_features = max(1, int(len(feature_names) * self.top_features_ratio))
                self.top_features = importance_df.head(num_top_features)['Feature'].tolist()
                
                logger.info(f"Selected top {num_top_features} features ({self.top_features_ratio*100:.1f}% of all features)")
                
                return importance_df
            else:
                logger.error("No best model available from ModelManager")
                raise ValueError("Feature importance calculation failed: no best model available")
                
        except Exception as e:
            logger.error(f"Error in feature evaluation: {str(e)}")
            raise
    
    def load_datasets(self, input_data_path, output_data_path):
        """
        Load and integrate input and output datasets for feature selection.

        Parameters:
        -----------
        input_data_path : str
            Path to the input dataset (features).
        output_data_path : str
            Path to the output dataset (target variable).
            
        Returns:
        --------
        tuple
            (input_data, output_data) - The loaded datasets
        """
        logger.info(f"Loading datasets from {input_data_path} and {output_data_path}")
        
        if not os.path.exists(input_data_path):
            logger.error(f"Input data file not found: {input_data_path}")
            raise FileNotFoundError(f"Input data file not found: {input_data_path}")
            
        if not os.path.exists(output_data_path):
            logger.error(f"Output data file not found: {output_data_path}")
            raise FileNotFoundError(f"Output data file not found: {output_data_path}")
            
        try:
            # Determine file type and load accordingly
            if input_data_path.endswith('.csv'):
                self.input_data = pd.read_csv(input_data_path)
                logger.info(f"Loaded CSV input data: {self.input_data.shape}")
            elif input_data_path.endswith('.parquet'):
                self.input_data = pd.read_parquet(input_data_path)
                logger.info(f"Loaded Parquet input data: {self.input_data.shape}")
            else:
                logger.error(f"Unsupported input file format: {input_data_path}")
                raise ValueError(f"Unsupported input file format: {input_data_path}")
                
            if output_data_path.endswith('.csv'):
                self.output_data = pd.read_csv(output_data_path)
                logger.info(f"Loaded CSV output data: {self.output_data.shape}")
            elif output_data_path.endswith('.parquet'):
                self.output_data = pd.read_parquet(output_data_path)
                logger.info(f"Loaded Parquet output data: {self.output_data.shape}")
            else:
                logger.error(f"Unsupported output file format: {output_data_path}")
                raise ValueError(f"Unsupported output file format: {output_data_path}")
                
            return self.input_data, self.output_data
            
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise

    def save_feature_selection_results(self, save_dir='feature_selection_results'):
        """
        Save selected features and additional information to a designated folder.

        Parameters:
        -----------
        save_dir : str, default='feature_selection_results'
            Directory to save the feature selection results.
            
        Returns:
        --------
        str
            Path to the saved results file
        """
        logger.info(f"Saving feature selection results to {save_dir}")
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            selected_features_path = os.path.join(save_dir, 'selected_features.json')
            
            # Prepare results dictionary with metadata
            results = {
                "selected_features": self.top_features,
                "removed_features": {
                    "correlated": self.removed_correlated_features,
                    "noisy": self.removed_noisy_features
                },
                "excluded_features": self.excluded_features,  # Include excluded features in results
                "parameters": {
                    "correlation_threshold": self.correlation_threshold,
                    "subsample_ratio": self.subsample_ratio,
                    "top_features_ratio": self.top_features_ratio,
                    "noise_std_threshold": self.noise_std_threshold,
                    "noise_outlier_ratio": self.noise_outlier_ratio
                },
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name
            }
            
            with open(selected_features_path, 'w') as f:
                json.dump(results, f, indent=4)
                
            logger.info(f"Selected features saved to {selected_features_path}")
            return selected_features_path
            
        except Exception as e:
            logger.error(f"Error saving feature selection results: {str(e)}")
            raise

    def load_feature_selection_results(self, save_dir='feature_selection_results'):
        """
        Load previously saved feature selection results.

        Parameters:
        -----------
        save_dir : str, default='feature_selection_results'
            Directory containing the saved feature selection results.
            
        Returns:
        --------
        dict
            The loaded feature selection results
        """
        logger.info(f"Loading feature selection results from {save_dir}")
        
        selected_features_path = os.path.join(save_dir, 'selected_features.json')
        
        if not os.path.exists(selected_features_path):
            logger.warning(f"No saved feature selection results found in {save_dir}")
            return None
            
        try:
            with open(selected_features_path, 'r') as f:
                results = json.load(f)
                
            # Update instance attributes from loaded results
            self.top_features = results.get("selected_features", [])
            
            if "removed_features" in results:
                self.removed_correlated_features = results["removed_features"].get("correlated", [])
                self.removed_noisy_features = results["removed_features"].get("noisy", [])
                
            if "parameters" in results:
                params = results["parameters"]
                self.correlation_threshold = params.get("correlation_threshold", self.correlation_threshold)
                self.subsample_ratio = params.get("subsample_ratio", self.subsample_ratio)
                self.top_features_ratio = params.get("top_features_ratio", self.top_features_ratio)
                self.noise_std_threshold = params.get("noise_std_threshold", self.noise_std_threshold)
                self.noise_outlier_ratio = params.get("noise_outlier_ratio", self.noise_outlier_ratio)
                
            self.model_name = results.get("model_name", self.model_name)
            self.excluded_features = results.get("excluded_features", [])  # Load excluded features
            
            logger.info(f"Loaded {len(self.top_features)} selected features")
            return results
            
        except Exception as e:
            logger.error(f"Error loading feature selection results: {str(e)}")
            raise

    def generate_shap_analysis(self, X, y, model_name=None, n_samples=None, plot=True, save_plots=True):
        """
        Generate SHAP values for the given data and model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix (should contain only selected features)
        y : pandas.Series or numpy.ndarray
            Target variable
        model_name : str, optional
            Name of the model to use. If None, uses the instance's model_name.
        n_samples : int, optional
            Number of samples to use for SHAP analysis. If None, uses all data.
        plot : bool, default=True
            Whether to generate and display SHAP plots
        save_plots : bool, default=True
            Whether to save SHAP plots to disk (only if output_path is set)
            
        Returns:
        --------
        tuple
            (shap_values, explainer) - SHAP values and explainer object
        """
        if model_name is None:
            model_name = self.model_name
            
        logger.info(f"Starting SHAP analysis using {model_name} model")
        
        # Validate input data
        if X is None or y is None:
            logger.error("Feature matrix X or target variable y is None")
            raise ValueError("Feature matrix X or target variable y is None")
            
        if X.shape[0] != y.shape[0]:
            logger.error(f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}")
            raise ValueError(f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}")
        
        try:
            # Take a subset of the data if specified
            if n_samples is not None and n_samples < len(X):
                # Get seed from config
                seed = mlarchitect.mlarchitect_config.get('SEED', 42)
                X_sample, _, y_sample, _ = train_test_split(
                    X, y, 
                    train_size=min(n_samples, len(X)) / len(X),
                    random_state=seed
                )
                logger.info(f"Using {len(X_sample)} samples for SHAP analysis")
            else:
                X_sample, y_sample = X, y
                logger.info(f"Using all {len(X_sample)} samples for SHAP analysis")
            
            # Get model configuration
            model_keys = list(configs.keys())
            if model_name not in configs:
                logger.warning(f"Model {model_name} not found in configs. Using default model.")
                model_name = 'gradient_boosting' if 'gradient_boosting' in configs else model_keys[0]
                
            model_config = configs[model_name]
            model = clone(model_config.model)
            
            # Train the model
            logger.info("Training model for SHAP analysis...")
            model.fit(X_sample, y_sample)
            
            # Extract model from pipeline if needed
            if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                model_in_pipeline = model.named_steps['model']
                logger.debug("Extracted model from pipeline for SHAP analysis")
            else:
                model_in_pipeline = model
            
            # Create SHAP explainer
            logger.info("Creating SHAP explainer...")
            try:
                # Try TreeExplainer first (for tree-based models)
                explainer = shap.TreeExplainer(model_in_pipeline)
                shap_values = explainer.shap_values(X_sample)
                logger.info("Created TreeExplainer successfully")
            except Exception as e:
                # Fall back to KernelExplainer for other model types
                logger.warning(f"TreeExplainer failed: {str(e)}. Falling back to KernelExplainer.")
                try:
                    explainer = shap.KernelExplainer(
                        model_in_pipeline.predict_proba if hasattr(model_in_pipeline, 'predict_proba') 
                        else model_in_pipeline.predict, 
                        shap.kmeans(X_sample, min(50, len(X_sample)))
                    )
                    shap_values = explainer.shap_values(X_sample)
                    logger.info("Created KernelExplainer successfully")
                except Exception as e2:
                    logger.error(f"SHAP analysis failed: {str(e2)}")
                    return None, None
            
            # Handle multiple outputs (e.g., XGBoost for binary classification returns two sets)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # For binary classification, we often use the second class's SHAP values
                logger.debug("Using second class SHAP values for binary classification")
                shap_values = shap_values[1]
                
            self.shap_values = shap_values
            
            # Create output directory for plots if saving is enabled
            plots_dir = None
            if save_plots and self.output_path:
                plots_dir = os.path.join(self.output_path, f"{self.name}_{self.run_timestamp}", "shap_plots")
                os.makedirs(plots_dir, exist_ok=True)
                logger.info(f"SHAP plots will be saved to {plots_dir}")
            
            # Generate plots if requested
            if plot or save_plots:
                # Summary bar plot
                logger.info("Generating SHAP summary bar plot")
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=plot)
                plt.title("SHAP Feature Importance")
                plt.tight_layout()
                
                if save_plots and plots_dir:
                    plt.savefig(os.path.join(plots_dir, 'shap_importance_bar.png'))
                    logger.debug("Saved SHAP importance bar plot")
                
                if not plot:
                    plt.close()
                
                # Summary dot plot
                logger.info("Generating SHAP summary dot plot")
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, show=plot)
                plt.title("SHAP Summary Plot")
                plt.tight_layout()
                
                if save_plots and plots_dir:
                    plt.savefig(os.path.join(plots_dir, 'shap_summary_dot.png'))
                    logger.debug("Saved SHAP summary dot plot")
                    
                if not plot:
                    plt.close()
                
            logger.info("SHAP analysis completed successfully")
            return shap_values, explainer
            
        except Exception as e:
            logger.error(f"Error in SHAP analysis: {str(e)}")
            return None, None
    
    def generate_shap_dependence_plots(self, X, top_n=5, save_plots=True):
        """
        Generate SHAP dependence plots for the top N features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix used for SHAP analysis
        top_n : int, default=5
            Number of top features to analyze
        save_plots : bool, default=True
            Whether to save plots to disk (only if output_path is set)
            
        Returns:
        --------
        list
            List of top features analyzed
        """
        logger.info(f"Generating SHAP dependence plots for top {top_n} features")
        
        if self.shap_values is None:
            logger.error("No SHAP values available. Run generate_shap_analysis first.")
            return []
        
        try:
            # Create output directory for plots if saving is enabled
            plots_dir = None
            if save_plots and self.output_path:
                plots_dir = os.path.join(self.output_path, f"{self.name}_{self.run_timestamp}", "shap_dependence")
                os.makedirs(plots_dir, exist_ok=True)
                logger.info(f"SHAP dependence plots will be saved to {plots_dir}")
            
            # Get top features by mean absolute SHAP value
            shap_sum = np.abs(self.shap_values).mean(axis=0)
            top_indices = shap_sum.argsort()[-top_n:]
            top_features = X.columns[top_indices].tolist()
            
            logger.info(f"Top {top_n} features by SHAP importance: {top_features}")
            
            # Create dependence plots for top features
            for feature in top_features:
                logger.debug(f"Creating dependence plot for feature: {feature}")
                plt.figure(figsize=(10, 7))
                shap.dependence_plot(feature, self.shap_values, X, show=True)
                plt.title(f"SHAP Dependence Plot for {feature}")
                plt.tight_layout()
                
                if save_plots and plots_dir:
                    # Clean feature name for filename
                    safe_feature = ''.join(c if c.isalnum() else '_' for c in feature)
                    plt.savefig(os.path.join(plots_dir, f'dependence_{safe_feature}.png'))
                    logger.debug(f"Saved dependence plot for {feature}")
                
                plt.close()
                
            logger.info("SHAP dependence plots generated successfully")
            return top_features
            
        except Exception as e:
            logger.error(f"Error in generating SHAP dependence plots: {str(e)}")
            return []
    
    def get_feature_interaction_strength(self, interaction_features=None, max_interactions=10):
        """
        Analyze feature interactions based on SHAP values.
        
        Parameters:
        -----------
        interaction_features : list, optional
            Specific features to analyze interactions for
        max_interactions : int
            Maximum number of interactions to return
            
        Returns:
        --------
        pandas.DataFrame
            Ranked feature interactions
        """
        if self.shap_values is None:
            warnings.warn("No SHAP values available. Run generate_shap_analysis first.")
            return pd.DataFrame()
        
        # For now, return a placeholder. Full interaction analysis requires more complex SHAP calculations.
        print("Feature interaction analysis via SHAP is a placeholder in this version.")
        
        # Create a dummy DataFrame for interactions
        interactions = pd.DataFrame({
            'Feature1': ['placeholder'],
            'Feature2': ['placeholder'],
            'Strength': [0.0]
        })
        
        return interactions
    
    def get_selected_features(self):
        """
        Get the list of selected features after all selection steps.
        
        Returns:
        --------
        list
            List of selected feature names
        """
        return self.top_features
    
    def get_removed_features(self):
        """
        Get the list of all removed features.
        
        Returns:
        --------
        dict
            Dictionary with lists of removed features by category
        """
        return {
            'correlated': self.removed_correlated_features,
            'noisy': self.removed_noisy_features
        }
    
    def custom_transform(self, transform_func=None, features_to_transform=None, new_feature_prefix='tf_', 
                         save_transform=True, transform_name=None):
        """
        Apply a custom transformation to create new features, using and updating the internal data.
        
        Parameters:
        -----------
        transform_func : callable
            Transformation function that takes a DataFrame as input and returns a DataFrame with new features.
            If None, you will be prompted to provide a function interactively.
        features_to_transform : list, optional
            List of specific columns to transform. If None, all numerical columns are used.
        new_feature_prefix : str, default='tf_'
            Prefix to use for new feature names
        save_transform : bool, default=True
            If True, saves the transformation for later reuse
        transform_name : str, optional
            Name of the transformation for saving. If None, a timestamp is used.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with new features added (also updates self.data)
        """
        if self.data is None:
            logger.error("No data loaded. Load data first or initialize the class with proper input_path.")
            raise ValueError("No data loaded")
            
        logger.info("Starting custom feature creation")
            
        # Use all numeric columns if none specified
        if features_to_transform is None:
            features_to_transform = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            logger.info(f"Auto-selecting {len(features_to_transform)} numerical columns")
        else:
            # Filter out non-existent columns
            features_to_transform = [col for col in features_to_transform if col in self.data.columns]
            
        # Interactive mode if no function provided
        if transform_func is None:
            logger.info("No transform function provided, using interactive mode")
            try:
                transform_code = input("""
                Please enter your transformation function code.
                Example:
                def my_transform(df):
                    result = pd.DataFrame()
                    result['feature_1'] = df['col1'] * df['col2']
                    result['feature_2'] = np.log1p(df['col3'])
                    return result
                
                Your function: """)
                
                # Create function from input code
                local_vars = {}
                exec(transform_code, globals(), local_vars)
                transform_func = list(local_vars.values())[0]
            except Exception as e:
                logger.error(f"Error creating transformation function: {str(e)}")
                raise
        
        try:
            # Apply transformation
            df_subset = self.data[features_to_transform]
            new_features = transform_func(df_subset)
            
            if not isinstance(new_features, pd.DataFrame):
                raise TypeError("Transform function must return a DataFrame")
                
            # Add prefix to column names
            if new_feature_prefix:
                new_features = new_features.rename(
                    columns={col: f"{new_feature_prefix}{col}" for col in new_features.columns}
                )
                
            # Check for duplicate column names
            for col in new_features.columns:
                if col in self.data.columns:
                    new_name = f"{col}_new"
                    new_features = new_features.rename(columns={col: new_name})
                    logger.info(f"Renamed column: {col} -> {new_name}")
            
            # Add new features to internal data
            for col in new_features.columns:
                self.data[col] = new_features[col]
            
            # Save transformation metadata
            if save_transform:
                transform_name = transform_name or f"transform_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if not hasattr(self, 'transforms'):
                    self.transforms = {}
                
                self.transforms[transform_name] = {
                    'function': transform_func,
                    'features_transformed': features_to_transform,
                    'new_features': new_features.columns.tolist(),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save metadata to file if output_path is set
                if hasattr(self, 'output_path') and self.output_path:
                    try:
                        transform_dir = os.path.join(self.output_path, 'transformations')
                        os.makedirs(transform_dir, exist_ok=True)
                        
                        transform_meta = {
                            'name': transform_name,
                            'features_transformed': features_to_transform,
                            'new_features': new_features.columns.tolist(),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        with open(os.path.join(transform_dir, f"{transform_name}.json"), 'w') as f:
                            json.dump(transform_meta, f, indent=4)
                    except Exception as e:
                        logger.warning(f"Unable to save transform metadata: {str(e)}")
            
            logger.info(f"Feature creation complete. {len(new_features.columns)} new features added.")
            return self.data
            
        except Exception as e:
            logger.error(f"Error during feature creation: {str(e)}")
            raise

    def select_features(self, df, features_to_keep=None, features_to_drop=None, inplace=False, save_selection=True, selection_name=None):
        """
        Select features to keep or remove from the DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing data to filter
        features_to_keep : list, optional
            List of columns to explicitly keep. If specified, all others will be dropped.
        features_to_drop : list, optional
            List of columns to explicitly drop. Ignored if features_to_keep is specified.
        inplace : bool, default=False
            If True, modifies the input DataFrame. If False, returns a copy.
        save_selection : bool, default=True
            If True, saves the selection for later reuse
        selection_name : str, optional
            Name of the selection for saving. If None, a timestamp is used.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with only the selected features
        """
        logger.info("Starting feature selection")
        
        # Work on a copy if needed
        result_df = df if inplace else df.copy()
            
        # Determine columns to drop
        columns_to_drop = []
        
        if features_to_keep is not None:
            # If features_to_keep is specified, drop all other columns
            features_to_keep = [col for col in features_to_keep if col in df.columns]
            columns_to_drop = [col for col in df.columns if col not in features_to_keep]
            logger.info(f"Keeping {len(features_to_keep)} specified columns")
        
        elif features_to_drop is not None:
            # If features_to_drop is specified, drop these columns
            columns_to_drop = [col for col in features_to_drop if col in df.columns]
            logger.info(f"Dropping {len(columns_to_drop)} specified columns")
            
        # Interactive mode if no specification provided
        if not columns_to_drop and features_to_keep is None:
            logger.info("No columns specified. Activating interactive mode.")
            
            try:
                print(f"DataFrame contains {len(df.columns)} columns:")
                for i, col in enumerate(sorted(df.columns)):
                    print(f"{i+1}. {col}")
                
                selection_input = input("""
                Please specify columns to keep or drop:
                1. To keep: type "keep:" followed by column numbers or names separated by commas
                2. To drop: type "drop:" followed by column numbers or names separated by commas
                
                Example: "keep: 1, 3, 5-10" or "drop: col1, col2, col3"
                Your selection: """)
                
                if selection_input.strip().lower().startswith("keep:"):
                    # Keep mode
                    to_keep = selection_input.split(":", 1)[1].strip()
                    keep_cols = []
                    
                    # Handle numeric and range inputs
                    if any(c.isdigit() for c in to_keep):
                        parts = [p.strip() for p in to_keep.split(",")]
                        for part in parts:
                            if "-" in part:
                                start, end = map(int, part.split("-"))
                                keep_cols.extend([sorted(df.columns)[i-1] for i in range(start, end+1)])
                            elif part.isdigit():
                                keep_cols.append(sorted(df.columns)[int(part)-1])
                    else:
                        # Handle column names directly
                        keep_cols = [c.strip() for c in to_keep.split(",") if c.strip() in df.columns]
                    
                    columns_to_drop = [col for col in df.columns if col not in keep_cols]
                    
                elif selection_input.strip().lower().startswith("drop:"):
                    # Drop mode
                    to_drop = selection_input.split(":", 1)[1].strip()
                    drop_cols = []
                    
                    # Handle numeric and range inputs
                    if any(c.isdigit() for c in to_drop):
                        parts = [p.strip() for p in to_drop.split(",")]
                        for part in parts:
                            if "-" in part:
                                start, end = map(int, part.split("-"))
                                drop_cols.extend([sorted(df.columns)[i-1] for i in range(start, end+1)])
                            elif part.isdigit():
                                drop_cols.append(sorted(df.columns)[int(part)-1])
                    else:
                        # Handle column names directly
                        drop_cols = [c.strip() for c in to_drop.split(",") if c.strip() in df.columns]
                    
                    columns_to_drop = drop_cols
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
        
        # Perform column dropping
        if columns_to_drop:
            before_cols = len(result_df.columns)
            result_df = result_df.drop(columns=columns_to_drop, errors='ignore')
            after_cols = len(result_df.columns)
            logger.info(f"Feature selection complete: {before_cols - after_cols} columns dropped, {after_cols} columns kept")
        else:
            logger.info(f"No columns to drop. All {len(result_df.columns)} columns kept.")
            
        # Save selection if requested
        if save_selection:
            selection_name = selection_name or f"selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if not hasattr(self, 'selections'):
                self.selections = {}
                
            self.selections[selection_name] = {
                'kept_features': result_df.columns.tolist(),
                'dropped_features': columns_to_drop,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save metadata to file if output_path is set
            if hasattr(self, 'output_path') and self.output_path:
                try:
                    selection_dir = os.path.join(self.output_path, 'selections')
                    os.makedirs(selection_dir, exist_ok=True)
                    
                    selection_meta = {
                        'name': selection_name,
                        'kept_features': result_df.columns.tolist(),
                        'dropped_features': columns_to_drop,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with open(os.path.join(selection_dir, f"{selection_name}.json"), 'w') as f:
                        json.dump(selection_meta, f, indent=4)
                except Exception as e:
                    logger.warning(f"Unable to save selection metadata: {str(e)}")
        
        return result_df