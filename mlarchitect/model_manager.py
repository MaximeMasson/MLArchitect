from mlarchitect.model_config_class import configs
import mlarchitect

import os
import time
import pickle
import json
import zipfile
import numpy as np
import pandas as pd
import optuna
import logging
from sklearn.base import clone

from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    KFold, 
    TimeSeriesSplit
)
from sklearn.metrics import accuracy_score

# Configure logger for ModelManager
logger = logging.getLogger('mlarchitect.model_manager')

class ModelManager:
    def __init__(self, 
                 model_config, 
                 use_time_cv: bool = False,
                 custom_params: dict = None,
                 metrics: list = None,
                 pred_transform_func=None,
                 test_size: float = 0.2,
                 train_file_name: str = 'train_joined.parquet',
                 test_file_name: str = 'test_joined.parquet',
                 log_level=logging.INFO):
        # Configure logger
        self._setup_logger(log_level)
        logger.info("Initializing ModelManager...")
        
        self.model_config = model_config
        self.use_time_cv = use_time_cv
        self.custom_params = custom_params
        self.metrics = metrics if metrics is not None else [accuracy_score]
        self.optimize_metric = self.metrics[0]
        self.pred_transform_func = pred_transform_func
        self.test_size = test_size
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        
        # Keep references for OOS & test IDs
        self.train_id_col = None
        self.test_id_col = None
        self.target_col = None

        # Data holders
        self.df_train = None
        self.df_test = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_test = None

        # Predictions holders
        self.oos_predictions = None   # Store OOS predictions
        self.test_predictions = None  # Store test set predictions

        # CV info
        self.cv_scores = None
        self.cv_mean = None
        self.cv_std = None

        # Metadata about final model
        self.best_params = {}
        self.best_model = None
        self.cv_results = None
        self.training_time = None
            
    def _setup_logger(self, log_level):
        """Set up logging with the specified log level."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(log_level)

    def _load_data(self, joined_folder):
        """Loads train/test data from the specified folder, sets up train/val split."""
        from sklearn.model_selection import train_test_split
        train_file = os.path.join(mlarchitect.mlarchitect_config['PATH'], joined_folder, self.train_file_name)
        test_file = os.path.join(mlarchitect.mlarchitect_config['PATH'], joined_folder, self.test_file_name)
        
        logger.info(f"Loading training data from: {train_file}")
        logger.info(f"Loading test data from: {test_file}")
        
        self.df_train = pd.read_parquet(train_file)
        self.df_test = pd.read_parquet(test_file)

        # Read config
        self.train_id_col = mlarchitect.mlarchitect_config['TRAIN_ID']   
        self.test_id_col = mlarchitect.mlarchitect_config['TEST_ID']      
        self.target_col = mlarchitect.mlarchitect_config['TARGET_NAME']  

        # Split the training DataFrame into features and target.
        self.X = self.df_train.drop(columns=[self.target_col])
        self.y = self.df_train[self.target_col]

        # Use stratification if binary classification (FIT_TYPE=='bclass')
        stratify = self.y if mlarchitect.mlarchitect_config['FIT_TYPE'] == 'bclass' else None
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, 
            test_size=self.test_size,
            random_state=mlarchitect.mlarchitect_config['SEED'],
            stratify=stratify
        )
        
        logger.info(f"Data loaded and split - Train: {self.X_train.shape}, Validation: {self.X_val.shape}")

        # For test set:
        self.X_test = self.df_test.copy()

    def load_from_folder(self, joined_folder: str):
        """
        Loads train/test data from the specified folder, sets up train/val split.
        
        Parameters:
        -----------
        joined_folder : str
            Path to the folder containing the training and test data files
        """
        train_file = os.path.join(mlarchitect.mlarchitect_config['PATH'], joined_folder, self.train_file_name)
        test_file = os.path.join(mlarchitect.mlarchitect_config['PATH'], joined_folder, self.test_file_name)
        
        logger.info(f"Loading training data from: {train_file}")
        logger.info(f"Loading test data from: {test_file}")
        
        self.df_train = pd.read_parquet(train_file)
        self.df_test = pd.read_parquet(test_file)

        # Read config
        self.train_id_col = mlarchitect.mlarchitect_config['TRAIN_ID']   
        self.test_id_col = mlarchitect.mlarchitect_config['TEST_ID']      
        self.target_col = mlarchitect.mlarchitect_config['TARGET_NAME']  

        # Split the training DataFrame into features and target.
        self._prepare_train_val_split()
        
        # For test set:
        self.X_test = self.df_test.copy()
        logger.info(f"Data loaded from folder and split - Train: {self.X_train.shape}, Validation: {self.X_val.shape}")
        
    def load_from_dataframes(self, df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str,
                          train_id_col: str = None, test_id_col: str = None):
        """
        Loads train/test data directly from provided DataFrames, sets up train/val split.
        
        Parameters:
        -----------
        df_train : pandas.DataFrame
            Training DataFrame containing features and target column
        df_test : pandas.DataFrame
            Test DataFrame containing features
        target_col : str
            Name of the target column in df_train
        train_id_col : str, optional
            Name of the ID column in df_train
        test_id_col : str, optional  
            Name of the ID column in df_test
        """
        logger.info(f"Loading data from provided DataFrames - Train: {df_train.shape}, Test: {df_test.shape}")
        
        self.df_train = df_train
        self.df_test = df_test
        self.train_id_col = train_id_col
        self.test_id_col = test_id_col
        self.target_col = target_col

        # Split the training DataFrame into features and target
        self._prepare_train_val_split()
        
        # For test set:
        self.X_test = self.df_test.copy()
        logger.info(f"Data loaded from DataFrames and split - Train: {self.X_train.shape}, Validation: {self.X_val.shape}")
        
    def _prepare_train_val_split(self):
        """Helper method to prepare train/validation split from loaded data"""
        # Split the training DataFrame into features and target
        self.X = self.df_train.drop(columns=[self.target_col])
        self.y = self.df_train[self.target_col]

        # Use stratification if binary classification (FIT_TYPE=='bclass')
        stratify = self.y if mlarchitect.mlarchitect_config.get('FIT_TYPE', '') == 'bclass' else None
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, 
            test_size=self.test_size,
            random_state=mlarchitect.mlarchitect_config.get('SEED', 42),
            stratify=stratify
        )

    def perform_cv(self, cv_folds: int = None, cv_column: str = None, n_trials: int = 10, excluded_features=None, subsample_ratio=1.0, feature_selection_metric=None, tuning: bool = False):
        """
        Perform cross-validation with out-of-fold predictions.

        This method supports two modes:
        1. Tuning mode (if tuning is True):
            Uses Optuna to optimize hyperparameters for each fold based on the
            parameter grid in self.model_config.params_search. For each fold, an objective
            function is defined and an Optuna study is run with up to n_trials. The best
            hyperparameters are used to train the model for that fold.
        2. No tuning (if tuning is False):
            Applies self.custom_params (if provided) and fits the model directly.

        When cv_column is provided (e.g., 'DATE'), the splits are created based on unique groups
        in that column. Otherwise, a standard (or time-based) row split is applied.

        All metrics in self.metrics are computed for each fold and stored in a dictionary,
        while summary statistics (mean, std) are computed based on the primary metric
        (self.optimize_metric). When tuning, the best hyperparameters for each fold are also stored.

        Parameters:
        -----------
        cv_folds : int, optional
            Number of folds for cross-validation (defaults to config value).
        cv_column : str, optional
            Column name to use for grouping splits. If provided, CV is done on unique groups.
        n_trials : int, default=10
            Maximum number of Optuna trials per fold (only used if tuning is enabled).
        excluded_features : list, optional
            Optional list of features to exclude from training but keep for CV.
        subsample_ratio : float, default=1.0
            Ratio of data to use for training. Values < 1.0 enable subsampling for faster feature selection.
        feature_selection_metric : callable, optional
            Metric to optimize for feature selection. Overrides self.optimize_metric if provided.

        Returns:
        --------
        tuple
            (cv_mean, cv_std) - Mean and standard deviation of cross-validation scores
        """
        logger.info("Starting cross-validation with out-of-fold predictions" + 
                  (" using Optuna for tuning" if tuning else ""))
        
        # Check if data has been loaded
        if self.X is None or self.y is None:
            logger.error("No data loaded. Please call load_from_folder() or load_from_dataframes() first.")
            raise ValueError("No data loaded. Please call load_from_folder() or load_from_dataframes() first.")
        
        # If feature selection metric is provided, update metrics list and optimize_metric
        if feature_selection_metric is not None:
            original_metrics = self.metrics
            original_optimize_metric = self.optimize_metric
            self.metrics = [feature_selection_metric]
            self.optimize_metric = feature_selection_metric
            logger.info(f"Using custom metric for optimization: {feature_selection_metric.__name__}")
        
        # Initialize excluded_features if not provided
        if excluded_features is None:
            excluded_features = []
            
        # Ensure cv_column is not in excluded_features if it's needed
        if cv_column is not None and cv_column in excluded_features:
            logger.info(f"Note: {cv_column} is excluded from features but will be used for CV splits")
        
        # Determine the number of folds from config if not provided.
        cv_folds = cv_folds if cv_folds is not None else mlarchitect.mlarchitect_config.get('FOLDS', 5)
        logger.info(f"Using {cv_folds} cross-validation folds" + 
                  (f" grouped by {cv_column}" if cv_column else ""))
        
        # Working with the loaded data
        X_train_array = self.X_train.reset_index(drop=True)
        y_train_array = self.y_train.reset_index(drop=True)
        
        # Apply subsampling if requested
        if subsample_ratio < 1.0:
            logger.info(f"Subsampling existing data to {subsample_ratio*100:.1f}% of original size")
            from sklearn.model_selection import train_test_split
            fit_type = mlarchitect.mlarchitect_config.get('FIT_TYPE', 'regression')
            stratify = y_train_array if fit_type == 'bclass' else None
            
            X_train_array, _, y_train_array, _ = train_test_split(
                X_train_array, y_train_array,
                train_size=subsample_ratio,
                random_state=mlarchitect.mlarchitect_config.get('SEED', 42),
                stratify=stratify
            )
            logger.info(f"Subsampled to {len(X_train_array)} samples ({subsample_ratio*100:.1f}% of data)")
        
        # Determine train IDs.
        if self.train_id_col and (self.train_id_col in X_train_array.columns):
            train_ids = X_train_array[self.train_id_col].values
        else:
            train_ids = X_train_array.index.values

        # ---------------------------
        # Helper function: get_cv_indices
        # ---------------------------
        def get_cv_indices(X, y, cv_column, cv_folds, seed):
            """
            Yields (train_idx, val_idx) pairs.
            If cv_column is provided, groups are split based on unique values in that column.
            Otherwise, standard (or time-based) splitting is used.
            """
            if cv_column is not None:
                unique_groups = X[cv_column].unique()
                group_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
                for train_group_idx, val_group_idx in group_splitter.split(unique_groups):
                    train_groups = unique_groups[train_group_idx]
                    val_groups = unique_groups[val_group_idx]
                    train_mask = X[cv_column].isin(train_groups)
                    val_mask = X[cv_column].isin(val_groups)
                    yield X.index[train_mask], X.index[val_mask]
            else:
                if self.use_time_cv:
                    cv_splitter = TimeSeriesSplit(n_splits=cv_folds)
                else:
                    if mlarchitect.mlarchitect_config.get('FIT_TYPE') == 'bclass':
                        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                                    random_state=mlarchitect.mlarchitect_config.get('SEED', 42))
                    else:
                        cv_splitter = KFold(n_splits=cv_folds, shuffle=True,
                                            random_state=mlarchitect.mlarchitect_config.get('SEED', 42))
                for train_idx, val_idx in cv_splitter.split(X, y):
                    yield train_idx, val_idx

        # ---------------------------
        # Helper function: run_fold
        # ---------------------------
        def run_fold(X_tr, y_tr, X_val, y_val, fold_idx, tuning=False):
            """
            Train the model on the training fold and predict on the validation fold.
            Returns the predictions and (if tuning) appends the best parameters.
            """
            # Remove excluded features from training data (but keep them in dataframe for CV)
            if excluded_features:
                features_to_use = [col for col in X_tr.columns if col not in excluded_features]
                X_tr_features = X_tr[features_to_use]
                X_val_features = X_val[features_to_use]
            else:
                X_tr_features = X_tr
                X_val_features = X_val
            
            if tuning:
                # Tuning mode: define the Optuna objective function.
                def objective(trial):
                    model = clone(self.model_config.model)
                    trial_params = {
                        param: trial.suggest_categorical(param, values)
                        for param, values in self.model_config.params_search.items()
                    }
                    model.set_params(**trial_params)
                    model.fit(X_tr_features, y_tr)
                    preds = model.predict(X_val_features)
                    if self.pred_transform_func:
                        # If val_idx is needed for the transform function
                        val_indices = X_val.index if hasattr(X_val, 'index') else None
                        preds = self.pred_transform_func(preds, fold_idx=fold_idx, val_idx=val_indices)
                    return self.optimize_metric(y_val, preds)
                
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials)
                best_params = study.best_params
                fold_best_params.append(best_params)
                logger.info(f"Fold {fold_idx} best hyperparameters: {best_params}")

                best_model = clone(self.model_config.model)
                best_model.set_params(**best_params)
                best_model.fit(X_tr_features, y_tr)
                preds_val = best_model.predict(X_val_features)

                if self.pred_transform_func:
                    # If val_idx is needed for the transform function
                    val_indices = X_val.index if hasattr(X_val, 'index') else None
                    preds_val = self.pred_transform_func(preds_val, fold_idx=fold_idx, val_idx=val_indices)
            else:
                # No tuning mode: apply custom parameters if provided.
                if self.custom_params:
                    logger.info(f"Applying custom parameters before CV: {self.custom_params}")
                    self.model_config.model.set_params(**self.custom_params)
                    
                # Fit model and make predictions
                logger.debug(f"Fitting model for fold {fold_idx}")
                model = clone(self.model_config.model)
                model.fit(X_tr_features, y_tr)
                preds_val = model.predict(X_val_features)
                
                # Apply prediction transform if provided
                if self.pred_transform_func:
                    # If val_idx is needed for the transform function
                    val_indices = X_val.index if hasattr(X_val, 'index') else None
                    preds_val = self.pred_transform_func(preds_val, fold_idx=fold_idx, val_idx=val_indices)
                    
            return preds_val

        # ---------------------------
        # Main CV loop
        # ---------------------------
        # Initialize out-of-fold predictions and storage lists.
        oof_preds = np.zeros(len(X_train_array), dtype=float)
        fold_scores = []
        fold_metrics_all = []
        fold_best_params = []  # Only used if tuning is enabled.

        # Loop over the CV splits (either grouped or standard).
        for fold_idx, (tr_idx, val_idx) in enumerate(get_cv_indices(X_train_array, y_train_array,
                                                                   cv_column, cv_folds,
                                                                   mlarchitect.mlarchitect_config.get('SEED', 42)), start=1):
            logger.info(f"---- Fold {fold_idx}" + (f" (grouped by {cv_column})" if cv_column else "") + " ----")
            X_tr_cv = X_train_array.loc[tr_idx]
            y_tr_cv = y_train_array.loc[tr_idx]
            X_val = X_train_array.loc[val_idx]
            y_val = y_train_array.loc[val_idx]
            
            logger.debug(f"Fold {fold_idx} sizes - Train: {X_tr_cv.shape}, Validation: {X_val.shape}")

            # Run training and prediction for this fold.
            preds_val = run_fold(X_tr_cv, y_tr_cv, X_val, y_val, fold_idx, tuning=tuning)

            # Store predictions in the OOF array.
            oof_preds[val_idx] = preds_val

            # Compute metrics for this fold.
            fold_metrics = {
                metric.__name__: metric(y_val, preds_val)
                for metric in self.metrics
            }
            fold_metrics_all.append(fold_metrics)
            primary_score = fold_metrics[self.optimize_metric.__name__]
            fold_scores.append(primary_score)
            logger.info(f"Fold {fold_idx} metrics: {fold_metrics}")

        # ---------------------------
        # Final CV summary
        # ---------------------------
        fold_scores = np.array(fold_scores)
        self.cv_scores = fold_scores
        self.cv_mean = fold_scores.mean()
        self.cv_std = fold_scores.std()
        self.fold_metrics = fold_metrics_all
        if tuning:
            self.fold_best_params = fold_best_params

        self._print_cv_summary(fold_scores)

        # Create a DataFrame with out-of-fold predictions along with IDs and actual values.
        self.oos_predictions = pd.DataFrame({
            self.train_id_col if self.train_id_col else 'train_index': train_ids,
            "oos_pred": oof_preds,
            "actual": y_train_array.values
        })

        overall_oos_score = self.optimize_metric(self.oos_predictions["actual"], self.oos_predictions["oos_pred"])
        logger.info(f"Overall OOS score (entire training set): {overall_oos_score:.4f}")
            
        # If feature selection metric was provided, restore original metrics
        if feature_selection_metric is not None:
            self.metrics = original_metrics
            self.optimize_metric = original_optimize_metric
            logger.debug("Restored original metrics after CV")
            
        return self.cv_mean, self.cv_std

    def _print_cv_summary(self, scores):
        mean = np.mean(scores) * 100
        std = np.std(scores) * 100
        u = mean + std
        l = mean - std
        self.cv_scores = scores
        self.cv_mean = np.mean(scores)
        self.cv_std = np.std(scores)
        logger.info(f'CV {self.optimize_metric.__name__}: {mean:.2f}% [{l:.2f} ; {u:.2f}] (+- {std:.2f})')

    def train_final_model(self, final_tuning: bool = True, n_trials_final: int = 30, save_dir: str = "models", model_name: str = "model", excluded_features=None, transform_metadata=None):
        """
        Trains the final model on the entire X_train+X_val dataset, with an optional final tuning
        step using Optuna on all available training data.

        Parameters:
        -----------
        final_tuning : bool, default=True
            If True, perform final hyperparameter tuning on the combined training data.
        n_trials_final : int, default=30
            Number of Optuna trials to run during the final tuning step.
        save_dir : str, default="models"
            Directory to save model files (not used, kept for backward compatibility).
        model_name : str, default="model"
            Base name for the model files (not used, kept for backward compatibility).
        excluded_features : list, optional
            Features to exclude from training but keep in dataset for potential use in pred_transform_func.
        transform_metadata : list or str, optional
            Specific column names to pass to the prediction transformation function.
            If str: passes a single column (e.g. 'DATE')
            If list: passes multiple columns (e.g. ['DATE', 'INDUSTRY'])
            If None: passes the entire DataFrame with excluded_features
        """
        logger.info("Training final model on (X_train + X_val)...")
        start_time = time.time()

        # Combine training and validation data
        X_full = pd.concat([self.X_train, self.X_val])
        y_full = pd.concat([self.y_train, self.y_val])

        # Remove excluded features for training (but keep them in the dataset for transform function)
        if excluded_features:
            features_to_use = [col for col in X_full.columns if col not in excluded_features]
            X_full_for_training = X_full[features_to_use]
            logger.info(f"Training with {len(features_to_use)} features (excluded {len(excluded_features)} metadata features)")
        else:
            X_full_for_training = X_full

        if final_tuning:
            logger.info("Performing final tuning with Optuna on the full training data...")
            import optuna
            from sklearn.base import clone

            # Define the objective function for Optuna.
            def objective(trial):
                # Clone the current model to avoid contamination between trials.
                model = clone(self.model_config.model)
                # Retrieve the hyperparameter grid from the model configuration.
                param_grid = self.model_config.params_search  
                trial_params = {}
                for param, values in param_grid.items():
                    trial_params[param] = trial.suggest_categorical(param, values)
                # Set the sampled hyperparameters.
                model.set_params(**trial_params)
                # Fit the model on the full training data.
                model.fit(X_full_for_training, y_full)
                # Predict on the full training set.
                preds = model.predict(X_full_for_training)
                if self.pred_transform_func:
                    # Préparer les métadonnées pour la transformation
                    metadata = self._prepare_transform_metadata(X_full, transform_metadata)
                    preds = self.pred_transform_func(preds, metadata)
                # Compute and return the primary metric (assumed to be maximized).
                score = self.optimize_metric(y_full, preds)
                return score

            # Create an Optuna study and optimize the objective.
            optuna.logging.set_verbosity(optuna.logging.INFO)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials_final)
            best_params = study.best_params
            self.best_params = best_params
            logger.info(f"Final tuning best parameters: {best_params}")
            # Update the model configuration with the best parameters.
            self.model_config.model.set_params(**best_params)

        # Train the final model on the full training data.
        logger.info("Fitting final model on complete training dataset...")
        self.best_model = self.model_config.model.fit(X_full_for_training, y_full)

        # Evaluate on X_val for a simple hold-out estimate.
        if excluded_features:
            features_to_use = [col for col in self.X_val.columns if col not in excluded_features]
            X_val_for_eval = self.X_val[features_to_use]
        else:
            X_val_for_eval = self.X_val
            
        preds_val = self.best_model.predict(X_val_for_eval)
        if self.pred_transform_func:
            # Préparer les métadonnées pour la transformation
            metadata = self._prepare_transform_metadata(self.X_val, transform_metadata)
            preds_val = self.pred_transform_func(preds_val, metadata)

        self.training_time = time.time() - start_time
        logger.info(f"Training time: {self.training_time:.2f} seconds")
        
        return self.best_model

    def predict_test(self, excluded_features=None, test_path=None, transform_metadata=None):
        """
        Generate predictions on the test set using the best model.
        
        Parameters:
        -----------
        excluded_features : list, optional
            Optional list of features to exclude from prediction but keep for transform_func.
        test_path : str, optional
            Path to a test file to load for prediction. If provided, this will override the current test data.
        transform_metadata : list or str, optional
            Specific column names to pass to the prediction transformation function.
            If str: passes a single column (e.g. 'DATE')
            If list: passes multiple columns (e.g. ['DATE', 'INDUSTRY'])
            If None: passes the entire DataFrame with excluded_features
        
        Returns:
        --------
        numpy.ndarray
            Predictions for the test set.
        """
        logger.info("Generating predictions on the test set...")
        
        # Load test data from path if provided
        if test_path:
            logger.info(f"Loading test data from: {test_path}")
            try:
                self.df_test = pd.read_parquet(test_path)
                self.X_test = self.df_test.copy()
                logger.info(f"Loaded test data with shape: {self.X_test.shape}")
            except Exception as e:
                logger.error(f"Error loading test data from {test_path}: {str(e)}")
                raise
        
        # Ensure test data has the same features as the trained model
        if hasattr(self, 'best_model') and self.best_model is not None:
            # Get the features used for training
            if hasattr(self.X_train, 'columns'):
                trained_features = [col for col in self.X_train.columns if col not in (excluded_features or [])]
                
                # Check if X_test has all required features
                missing_features = [f for f in trained_features if f not in self.X_test.columns]
                if missing_features:
                    logger.warning(f"Test data is missing {len(missing_features)} features used in training: {missing_features[:5]}...")
        
        # Remove excluded features for prediction (but keep them available for transform function)
        if excluded_features:
            features_to_use = [col for col in self.X_test.columns if col not in excluded_features]
            X_test_for_pred = self.X_test[features_to_use]
            logger.info(f"Predicting with {len(features_to_use)} features (excluded {len(excluded_features)} metadata features)")
        else:
            X_test_for_pred = self.X_test
        
        preds = self.best_model.predict(X_test_for_pred)
        
        if self.pred_transform_func:
            # Préparer les métadonnées pour la transformation
            metadata = self._prepare_transform_metadata(self.X_test, transform_metadata)
            preds = self.pred_transform_func(preds, metadata)
        
        self.test_predictions = preds
        return preds

    def generate_submission(self, filename_prefix: str = "submit", additional_info: dict = None):
        """
        Generates a submission file from self.test_predictions. 
        By default, it is called from save_model(), 
        but you can also call it directly if needed.
        """
        logger.info("Generating submission file...")
        output_path = os.path.join(self.save_dir)

        time_str = time.strftime("%Y%m%d_%H%M%S")
        metric_str = f"{self.cv_mean:.4f}" if self.cv_mean is not None else "NA"
        filename = f"{filename_prefix}_{metric_str}_{time_str}"
        csv_file = os.path.join(output_path, filename + ".csv")
        zip_file = os.path.join(output_path, filename + ".zip")

        submission = pd.DataFrame()
        
        # Ensure test_id_col exists in df_test
        if self.test_id_col in self.df_test.columns:
            submission[self.test_id_col] = self.df_test[self.test_id_col]
        else:
            # fallback if the ID is the index or not found
            submission[self.test_id_col] = self.df_test.index
        
        submission[mlarchitect.mlarchitect_config['TARGET_NAME']] = self.test_predictions
        submission.to_csv(csv_file, index=False, encoding=mlarchitect.mlarchitect_config['ENCODING'])

        # Create a ZIP archive
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(csv_file, os.path.basename(csv_file))

        logger.info(f"Submission files generated: {csv_file} and {zip_file}")

    def save_model(self, save_dir: str = "models", model_name: str = "model_test"):
        """
        Saves:
          1) The final model (pkl)
          2) A meta.json with run details
          3) OOS predictions CSV (if self.oos_predictions is not None)
          4) Submission file (CSV & ZIP) by calling generate_submission
        """
        logger.info("Saving the final model, OOS predictions, and submission...")

        # Make sure we have predictions for test
        if self.test_predictions is None:
            logger.info("Test predictions not found; generating now...")
            self.predict_test()

        # Ensure we have a directory to store everything
        save_dir = os.path.join(mlarchitect.mlarchitect_config['PATH'], save_dir, model_name)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 1) Save the model as a pickle
        model_path = os.path.join(save_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        logger.info(f"Model saved to {model_path}")

        # 2) Save metadata as meta.json
        meta = {
            "model_config": str(self.model_config),
            "best_params": self.best_params,
            "cv_scores": self.cv_scores.tolist() if (
                isinstance(self.cv_scores, np.ndarray) or isinstance(self.cv_scores, list)
            ) else None,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "folds_metrics": self.fold_metrics,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "training_time": self.training_time
        }
        meta_path = os.path.join(save_dir, "meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)
        logger.info(f"Model metadata saved to {meta_path}")
        
        # 3) Save OOS predictions (if available)
        if self.oos_predictions is not None:
            oos_file = os.path.join(save_dir, "oos_predictions.csv")
            self.oos_predictions.to_csv(oos_file, index=False)
            logger.info(f"OOS predictions saved to {oos_file}")

        logger.info(f"Model and metadata saved to {save_dir}")

        # 4) Generate submission file (CSV & ZIP)
        self.generate_submission(filename_prefix=model_name, additional_info=meta)
        
        return save_dir

    def save_cv_results(self, save_dir: str = "models", model_name: str = "model_cv"):
        """
        Saves the cross-validation results to disk before final model training:
          1) A cv_meta.json with detailed CV metrics and parameters
          2) OOS predictions CSV from the CV process
        """
        logger.info("Saving cross-validation results...")

        # Check if we have OOS predictions
        if self.oos_predictions is None:
            logger.warning("No OOS predictions found. Run perform_cv() first.")
            return

        # Ensure we have a directory to store everything
        save_dir = os.path.join(mlarchitect.mlarchitect_config['PATH'], save_dir, model_name)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 1) Save CV metadata as cv_meta.json
        cv_meta = {
            "model_config": str(self.model_config),
            "cv_scores": self.cv_scores.tolist() if (
                isinstance(self.cv_scores, np.ndarray) or isinstance(self.cv_scores, list)
            ) else None,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "folds_metrics": self.fold_metrics,
            "timestamp": time.strftime("%Y%m%d_%H%M%S")
        }
        
        # Save fold-specific parameters if tuning was enabled
        if hasattr(self, 'fold_best_params') and self.fold_best_params:
            cv_meta["fold_best_params"] = self.fold_best_params
            
        cv_meta_path = os.path.join(save_dir, "cv_meta.json")
        with open(cv_meta_path, 'w') as f:
            json.dump(cv_meta, f, indent=4)
        
        # 2) Save OOS predictions
        oos_file = os.path.join(save_dir, "cv_oos_predictions.csv")
        self.oos_predictions.to_csv(oos_file, index=False)
        
        logger.info(f"Cross-validation results saved to {save_dir}")
        logger.info(f"CV {self.optimize_metric.__name__}: {self.cv_mean:.4f} (± {self.cv_std:.4f})")

    def get_feature_importances(self, top_n=None, normalize=True, excluded_features=None):
        """
        Extract feature importances from the best model.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to return. If None, returns all features
        normalize : bool, default=True
            Whether to normalize importances to sum to 100
        excluded_features : list, optional
            List of feature names that were excluded during model training
            If not provided, the method will try to infer which features were used
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature names and importance scores
        """
        if not hasattr(self, 'best_model') or self.best_model is None:
            logger.warning("No best model available. Make sure to call train_final_model() first.")
            return None
            
        model = self.best_model
        
        # Check if we're dealing with a pipeline
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            model_in_pipeline = model.named_steps['model']
        else:
            model_in_pipeline = model
            
        # Get all available feature names
        all_feature_names = None
        if hasattr(self, 'X_train') and self.X_train is not None:
            all_feature_names = self.X_train.columns.tolist()
        elif hasattr(self, 'X') and self.X is not None:
            all_feature_names = self.X.columns.tolist()
            
        if all_feature_names is None:
            logger.error("Could not determine feature names")
            return None
            
        # Filter out excluded features to get the actual features used in the model
        if excluded_features is not None:
            used_feature_names = [f for f in all_feature_names if f not in excluded_features]
            logger.info(f"Using {len(used_feature_names)} features from training set (excluded {len(excluded_features)} features)")
        else:
            # Try to infer which features were actually used by looking at the shape of feature importances
            used_feature_names = all_feature_names
            
        # Extract feature importances based on what's available
        try:
            if hasattr(model_in_pipeline, 'feature_importances_'):
                # Tree-based models
                importances = model_in_pipeline.feature_importances_
                logger.info("Extracted feature importances from model's feature_importances_ attribute")
            elif hasattr(model_in_pipeline, 'coef_'):
                # Linear models
                importances = np.abs(model_in_pipeline.coef_).mean(axis=0) if model_in_pipeline.coef_.ndim > 1 else np.abs(model_in_pipeline.coef_)
                logger.info("Extracted feature importances from model's coefficient values")
            else:
                # Fall back to permutation importance
                try:
                    from sklearn.inspection import permutation_importance
                    if not hasattr(self, 'X_val') or not hasattr(self, 'y_val'):
                        logger.warning("Cannot compute permutation importance: validation data not available")
                        return None
                    
                    # Filter validation data to only include used features
                    if excluded_features:
                        X_val_for_importance = self.X_val[[f for f in self.X_val.columns if f not in excluded_features]]
                    else:
                        X_val_for_importance = self.X_val
                        
                    result = permutation_importance(
                        model, X_val_for_importance, self.y_val,
                        n_repeats=10,
                        random_state=mlarchitect.mlarchitect_config.get('SEED', 42)
                    )
                    importances = result.importances_mean
                    logger.info("Computed permutation importance as fallback")
                except Exception as e:
                    logger.error(f"Failed to compute permutation importance: {str(e)}")
                    return None
        except Exception as e:
            logger.error(f"Error extracting feature importances: {str(e)}")
            return None
            
        # Check if the lengths match and handle appropriately
        if len(used_feature_names) != len(importances):
            logger.warning(f"Length mismatch: {len(used_feature_names)} feature names vs {len(importances)} importance values")
            
            # If we didn't get excluded_features as input, try to infer which features were actually used
            if excluded_features is None:
                # Try to infer which features were used based on importance array length
                inferred_features = all_feature_names[:len(importances)]
                logger.info(f"Inferred that only {len(inferred_features)} features were used during training")
                used_feature_names = inferred_features
            
            # If still mismatched after inference, truncate to shorter length
            if len(used_feature_names) != len(importances):
                min_length = min(len(used_feature_names), len(importances))
                used_feature_names = used_feature_names[:min_length]
                importances = importances[:min_length]
                logger.info(f"Truncated to {min_length} features/importances")
                
        # Create DataFrame with feature names and importances
        importance_df = pd.DataFrame({
            'Feature': used_feature_names,
            'Importance': importances
        })
        
        # Normalize if requested
        if normalize and importances.sum() > 0:
            importance_df['Importance_Normalized'] = importance_df['Importance'] / importances.sum() * 100
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Add rank
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        
        # Filter to top N if specified
        if top_n is not None and top_n < len(importance_df):
            importance_df = importance_df.head(top_n)
            logger.info(f"Returning top {top_n} features by importance")
            
        return importance_df

    def _prepare_transform_metadata(self, df, transform_metadata=None):
        """
        Prépare les métadonnées à passer à la fonction de transformation des prédictions.
        Cette fonction permet une approche modulaire qui fonctionne avec différentes colonnes de métadonnées.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame contenant les données et métadonnées
        transform_metadata : str, list, or None
            - Si str: passe une seule colonne (ex: 'DATE')
            - Si list: passe plusieurs colonnes (ex: ['DATE', 'INDUSTRY'])
            - Si None: passe le DataFrame entier

        Returns:
        --------
        object
            Métadonnées préparées pour la fonction de transformation:
            - DataFrame avec les colonnes sélectionnées
            - Series pour une seule colonne
            - Le DataFrame original si transform_metadata est None
        """
        if transform_metadata is None:
            # Par défaut, retourne le DataFrame complet
            return df
        elif isinstance(transform_metadata, str):
            # Une seule colonne demandée
            if transform_metadata in df.columns:
                logger.debug(f"Passing single column {transform_metadata} to transform function")
                return df[transform_metadata]
            else:
                logger.warning(f"Requested metadata column {transform_metadata} not found, returning full DataFrame")
                return df
        elif isinstance(transform_metadata, list):
            # Multiple colonnes demandées
            available_cols = [col for col in transform_metadata if col in df.columns]
            if len(available_cols) < len(transform_metadata):
                missing = [col for col in transform_metadata if col not in df.columns]
                logger.warning(f"Some requested metadata columns not found: {missing}")
            
            if not available_cols:
                logger.warning("None of the requested metadata columns found, returning full DataFrame")
                return df
                
            logger.debug(f"Passing {len(available_cols)} metadata columns to transform function")
            return df[available_cols]
        else:
            # Type non supporté, retourne le DataFrame complet
            logger.warning(f"Unsupported metadata type {type(transform_metadata)}, returning full DataFrame")
            return df
