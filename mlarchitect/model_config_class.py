from sklearn.pipeline import Pipeline

# Importing all classifiers, ensemble methods, and related components
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor

class CreatePipeline:
    def __init__(self, model, params_search):
        # Initialize the attributes
        self.model_class = model 
        self.params_search = params_search
        
        # Construct the pipeline
        pipeline_steps = [
            # ('scaler', StandardScaler()), 
            ('model', model())
        ]
        
        # Create the pipeline
        self.model = Pipeline(pipeline_steps)
        
    def set_params(self, **params):
        self.model = self.model.set_params(**params)
        

configs = {
    'logistic_regression': CreatePipeline(
        LogisticRegression,
        {
            'model__penalty': ['l2', 'l1'],
            'model__C': [0.1, 0.5, 1, 5],
            'model__solver': ['liblinear'],
            'model__random_state': [42],
        }
    ),
    'xgboost_class': CreatePipeline(
        XGBClassifier,
        {
            'model__n_estimators': [50, 100, 200, 300, 400, 500, 600],
            'model__max_depth': [1, 3, 5, 7, 9, 11, 13, 15],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
            'model__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'model__gamma': [0, 1, 2, 3, 4, 5],
            'model__min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'model__random_state': [42],
        }
    ),
    'xgboost': CreatePipeline(
        XGBRegressor,
        {
            'model__n_estimators': [50, 100, 200, 300, 400, 500, 600],
            'model__max_depth': [1, 3, 5, 7, 9, 11, 13, 15],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
            'model__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'model__gamma': [0, 1, 2, 3, 4, 5],
            'model__min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'model__random_state': [42],
        }
    ),
    'random_forest_class': CreatePipeline(
        RandomForestClassifier,
        {
            'model__n_estimators': [50, 100, 150, 200, 250, 300],
            'model__max_depth': [10, 20, 30, 40, 50],
            'model__min_samples_split': [2, 4, 6, 8, 10],
            'model__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'model__max_features': ['sqrt', 'log2', None],
            'model__random_state': [42],
        }
    ),
    'random_forest': CreatePipeline(
        RandomForestClassifier,
        {
            'model__n_estimators': [50, 100, 150, 200, 250, 300],
            'model__max_depth': [10, 20, 30, 40, 50],
            'model__min_samples_split': [2, 4, 6, 8, 10],
            'model__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'model__max_features': ['sqrt', 'log2', None],
            'model__random_state': [42],
        }
    ),
    'svc': CreatePipeline(
        SVC,
        {
            'model__C': [0.1, 0.5, 1, 1.5, 2],
            'model__kernel': ['linear'],
            'model__gamma': ['scale', 'auto'],
            'model__random_state': [42],
        }
    ),
    'knn': CreatePipeline(
        KNeighborsClassifier,
        {
            'model__n_neighbors': [1, 5, 10, 15, 20],
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'model__leaf_size': [10, 20, 30, 40, 50]
        }
    ),
    'gradient_boosting': CreatePipeline(
        GradientBoostingClassifier,
        {
            'model__n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            'model__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
            'model__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'model__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'model__random_state': [42],
        }
    ),
    'adaboost': CreatePipeline(
        AdaBoostClassifier,
        {
            'model__n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            'model__learning_rate': [0.01, 0.1, 0.5, 1],
            'model__random_state': [42],
        }
    ),
    'gaussian_nb': CreatePipeline(
        GaussianNB,
        {}
    ),
}