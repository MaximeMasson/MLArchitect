import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

import config
import mlarchitect
from mlarchitect.model_config_class import configs
from mlarchitect.model_manager import ModelManager

# Instantiate the ModelManager.
# Here, accuracy_score is the primary metric (first in the list) for optimization.
manager = ModelManager(
    model_config=configs['xgboost'],  
    joined_folder="processed/v1",  # Folder containing the joined data files.
    use_time_cv=False, # Set to True if you require time-series cross-validation.
    tuning=True, # 'grid' uses GridSearchCV
    metrics=[accuracy_score, recall_score, precision_score, f1_score, roc_auc_score],
    train_file_name='train_processed.parquet',
    test_file_name='test_processed.parquet'
)

# Perform cross-validation with hyperparameter tuning.
manager.perform_cv(cv_folds=4, cv_column="DATE")

# Train the final model on the full training data (combining train and validation splits).
manager.train_final_model()

# Generate predictions for test data
manager.predict_test()

# Save the final trained model and its metadata.
manager.save_model(save_dir="models/v1", model_name="xgboost")

# # Load the saved model and its metadata.
# loaded_manager = ModelManager.load_model(model_dir="models/v1/logistic_model")
