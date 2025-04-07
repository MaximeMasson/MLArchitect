import json
import math
import os
import pickle
import shutil
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import auc, log_loss, roc_curve, r2_score
from sklearn.utils import check_array

import mlarchitect.const
from mlarchitect.feature_engineering import FeatureEngineering

NA_VALUES = ['NA', '?', '-inf', '+inf', 'inf', '', 'nan']

def save_pandas(df,filename):
    path = mlarchitect.mlarchitect_config['PATH']
    if filename.endswith(".csv"):
        df.to_csv(os.path.join(path, filename),index=False)
    elif filename.endswith(".pkl"):
        df.to_pickle(os.path.join(path, filename))

def load_pandas(filename):
    path = mlarchitect.mlarchitect_config['PATH']
    if filename.endswith(".csv"):
        return pd.read_csv(os.path.join(path, filename), na_values=NA_VALUES)
    elif filename.endswith(".pkl"):
        return pd.read_pickle(os.path.join(path, filename))


def balance(df_train):
    y_train = df_train['target'].values
    pos_train = df_train[y_train == 1]
    neg_train = df_train[y_train == 0]

    orig_ser = pos_train['id'].append(neg_train['id'])

    max_id = int(max(df_train['id']))
    balance_id = math.ceil(max_id / 100000) * 100000
    print("First balanced ID: {}".format(balance_id))

    print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
    print("Before oversample: pos={},neg={}".format(len(pos_train), len(neg_train)))
    p = 0.17426  # 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    current_neg_len = len(neg_train)
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -= 1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    added = len(neg_train) - current_neg_len

    print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
    df_train = pd.concat([pos_train, neg_train])
    print("After oversample: pos={},neg={}".format(len(pos_train), len(neg_train)))

    # Fill in id's for new data
    added_ser = pd.Series(range(balance_id, balance_id + added))
    new_id = orig_ser.append(added_ser)
    df_train['id'] = new_id.values

    del pos_train, neg_train

    df_train.sort_values(by=["id"], inplace=True)
    df_train.reset_index(inplace=True, drop=True)
    return df_train


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# Encode text values to a single dummy variable.  The new columns (which do not replace the old) will have a 1
# at every location where the original column (name) matches each of the target_values.  One column is added for
# each target value.
def encode_text_single_dummy(df, name, target_values):
    for tv in target_values:
        l = list(df[name].astype(str))
        l = [1 if str(x) == str(tv) else 0 for x in l]
        name2 = "{}-{}".format(name, tv)
        df[name2] = l


# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)


# Convert all missing values in the specified column to the default
def missing_default(df, name, default_value):
    df[name] = df[name].fillna(default_value)


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type

    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.int32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)

# Regression chart, we will see more of this chart in the next class.
def chart_regression(pred, y):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    t.sort_values(by=['y'], inplace=True)
    a = plt.plot(t['y'].tolist(), label='expected')
    b = plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()


# Get a new directory to hold checkpoints from a neural network.  This allows the neural network to be
# loaded later.  If the erase param is set to true, the contents of the directory will be cleared.
def get_model_dir(name, erase):
    path = mlarchitect.mlarchitect_config['PATH']
    model_dir = os.path.join(path, name)
    os.makedirs(model_dir, exist_ok=True)
    if erase and len(model_dir) > 4 and os.path.isdir(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)  # be careful, this deletes everything below the specified path
    return model_dir


# Remove all rows where the specified column is +/- sd standard deviations
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean()) >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)


# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
               * (normalized_high - normalized_low) + normalized_low


def create_submit_package(name, score):
    score_str = str(round(float(score), 6)).replace('.', 'p')
    time_str = time.strftime("%Y%m%d-%H%M%S")
    filename = name + "-" + score_str + "_" + time_str
    path = os.path.join(mlarchitect.mlarchitect_config['PATH'], filename)
    if not os.path.exists(path):
        os.makedirs(path)
    return path, score_str, time_str


def stretch(y):
    return (y - y.min()) / (y.max() - y.min())

def model_score(y_pred,y_valid):
    final_eval = mlarchitect.mlarchitect_config['FINAL_EVAL']
    if final_eval == mlarchitect.const.EVAL_R2:
        return r2_score(y_valid, y_pred)
    elif final_eval == mlarchitect.const.EVAL_LOGLOSS:
        return log_loss(y_valid, y_pred)
    elif final_eval == mlarchitect.const.EVAL_AUC:
        fpr, tpr, thresholds = roc_curve(y_valid, y_pred, pos_label=1)
        return auc(fpr, tpr)
    else:
        raise Exception(f"Unknown FINAL_EVAL: {final_eval}")

class TrainModel:
    def __init__(self, data_source, run_single_fold):
        self.data_source = data_source
        self.run_single_fold = run_single_fold
        self.num_folds = None
        self.zscore = False
        self.steps = None # How many steps to the best model
        self.cv_steps = [] # How many steps at each CV fold
        self.rounds = None # How many rounds are desired (if supported by model)
        self.pred_denom = 1

    def _run_startup(self):
        self.start_time = time.time()
        self.x_train = load_pandas("train-joined-{}.pkl".format(self.data_source))
        self.x_submit = load_pandas("test-joined-{}.pkl".format(self.data_source))

        self.input_columns = list(self.x_train.columns.values)

        # Grab what columns we need, but are not used for training
        self.train_ids = self.x_train['id']
        self.y_train = self.x_train['target']
        self.submit_ids = self.x_submit['id']
        self.folds = self.x_train['fold']
        self.num_folds = self.folds.nunique()
        print("Found {} folds in dataset.".format(self.num_folds))

        # Drop what is not used for training
        self.x_train.drop('id', axis=1, inplace=True)
        self.x_train.drop('fold', axis=1, inplace=True)
        self.x_train.drop('target', axis=1, inplace=True)
        self.x_submit.drop('id', axis=1, inplace=True)

        self.input_columns2 = list(self.x_train.columns.values)
        self.final_preds_train = np.zeros(self.x_train.shape[0])
        self.final_preds_submit = np.zeros(self.x_submit.shape[0])

        for i in range(len(self.x_train.dtypes)):
            dt = self.x_train.dtypes[i]
            name = self.x_train.columns[i]

            if dt not in [np.float64, np.float32, np.int32, np.int64]:
                print("Bad type: {}:{}".format(name,name.dtype))

            elif self.x_train[name].isnull().any():
                print("Null values: {}".format(name))

        if self.zscore:
            self.x_train = scipy.stats.zscore(self.x_train)
            self.x_submit = scipy.stats.zscore(self.x_submit)

    def _run_cv(self):
        folds2run = self.num_folds if not self.run_single_fold else 1

        for fold_idx in range(folds2run):
            fold_no = fold_idx + 1
            print("*** Fold #{} ***".format(fold_no))

            # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
            mask_train = np.array(self.folds != fold_no)
            mask_test = np.array(self.folds == fold_no)
            fold_x_train = self.x_train[mask_train]
            fold_x_valid = self.x_train[mask_test]
            fold_y_train = self.y_train[mask_train]
            fold_y_valid = self.y_train[mask_test]

            print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(fold_x_train.shape, len(fold_y_train),
                                                                               self.x_submit.shape))
            self.model = self.train_model(fold_x_train, fold_y_train, fold_x_valid, fold_y_valid)
            preds_valid = self.predict_model(self.model, fold_x_valid)

            score = model_score(preds_valid,fold_y_valid)

            preds_submit = self.predict_model(self.model, self.x_submit)

            self.final_preds_train[mask_test] = preds_valid
            self.final_preds_submit += preds_submit
            self.denom += 1
            self.pred_denom +=1

            if self.steps is not None:
                self.cv_steps.append(self.steps)

            self.scores.append(score)
            print("Fold score: {}".format(score))

            if fold_no==1:
                self.model_fold1 = self.model
        self.score = np.mean(self.scores)

        if len(self.cv_steps)>0:
            self.rounds = max(self.cv_steps) # Choose how many rounds to use after all CV steps

    def _run_single(self):
        print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(self.x_train.shape, len(self.y_train),
                                                                               self.x_submit.shape))
        self.model = self.train_model(self.x_train, self.y_train, None, None)

#        if not self.run_single_fold:
#            self.preds_oos = self.predict_model(self.model, self.x_train)

        #score = 0 #log_loss(fold_y_valid, self.preds_oos)

        #self.final_preds_train = self.preds_oos
        self.final_preds_submit = self.predict_model(self.model, self.x_submit)
        self.pred_denom = 1

    def _run_assemble(self):
        target_name = mlarchitect.mlarchitect_config['TARGET_NAME']
        test_id = mlarchitect.mlarchitect_config['TEST_ID']

        print("Training done, generating submission file")

        if len(self.scores)==0:
            self.denom = 1
            self.scores.append(-1)
            self.score = -1
            print("Warning, could not produce a validation score.")


        # create folder
        path, score_str, time_str = create_submit_package(self.name, self.score)

        filename = "submit-" + score_str + "_" + time_str
        filename_csv = os.path.join(path, filename) + ".csv"
        filename_zip = os.path.join(path, filename) + ".zip"
        filename_txt = os.path.join(path, filename) + ".txt"

        sub = pd.DataFrame()
        sub[test_id] = self.submit_ids
        sub[target_name] = self.final_preds_submit / self.pred_denom
        print("Pred denom: {}".format(self.pred_denom))
        sub.to_csv(filename_csv, index=False)

        z = zipfile.ZipFile(filename_zip, 'w', zipfile.ZIP_DEFLATED)
        z.write(filename_csv, filename + ".csv")
        output = ""
        # Generate training OOS file
        if not self.run_single_fold:
            filename = "oos-" + score_str + "_" + time_str + ".csv"
            filename = os.path.join(path, filename)
            sub = pd.DataFrame()
            sub['id'] = self.train_ids
            sub['expected'] = self.y_train
            sub['predicted'] = self.final_preds_train
            sub.to_csv(filename, index=False)
            output+="OOS Score: {}".format(model_score(self.final_preds_train,self.y_train))
            self.save_model(path, 'model-submit')
            if self.model_fold1:
                t = self.model
                self.model = self.model_fold1
                self.save_model(path, 'model-fold1')
                self.model = t

        print("Generated: {}".format(path))
        elapsed_time = time.time() - self.start_time

        output += "Elapsed time: {}\n".format(hms_string(elapsed_time))
        output += "Mean score: {}\n".format(self.score)
        output += "Fold scores: {}\n".format(self.scores)
        output += "Params: {}\n".format(self.params)
        output += "Columns: {}\n".format(self.input_columns)
        output += "Columns Used: {}\n".format(self.input_columns2)
        output += "Steps: {}\n".format(self.steps)

        output += "*** Model Specific Feature Importance ***\n"
        output = self.feature_rank(output)

        print(output)

        with open(filename_txt, "w") as text_file:
            text_file.write(output)

    def feature_rank(self,output):
        return output

    def run(self):
        self.denom = 0
        self.scores = []

        self._run_startup()
        self._run_cv()
        print("Fitting single model for entire training set.")
        self._run_single()
        self._run_assemble()

    def save_model(self, path, name):
        print("Saving Model")
        with open(os.path.join(path, name + ".pkl"), 'wb') as fp:  
            pickle.dump(self.model, fp)

        meta = {
            'name': self.__class__.__name__,
            'data_source': self.data_source,
            'params': self.params
        }
        
        with open(os.path.join(path,"meta.json"), 'w') as outfile:
            json.dump(meta, outfile)

    @classmethod
    def load_model(cls,path,name):
        root = mlarchitect.mlarchitect_config['PATH']
        model_path = os.path.join(root,path)
        meta_filename = os.path.join(model_path,"meta.json")
        with open(meta_filename, 'r') as fp:
            meta = json.load(fp)

        result = cls(meta['data_source'],meta['params'],False)
        with open(os.path.join(root, name + ".pkl"), 'rb') as fp:  
            result.model = pickle.load(fp)
        return result
    
class GenerateDataFile:
    """
    Manages train and test datasets in memory,
    then applies multiple transformations that each:
        - Fix up NaNs however they like
        - Generate new columns
        - Output those columns as separate Parquet files for train and test
    """

    def __init__(
        self,
        data_dir="/raw",
        train_csv="train.csv",
        test_csv="test.csv",
        output_dir="/transformed",
        overwrite=False,
        drop_train_na_threshold=None
    ):
        self.data_dir = mlarchitect.mlarchitect_config['PATH'] + data_dir
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.output_dir = mlarchitect.mlarchitect_config['PATH'] + output_dir
        self.overwrite = overwrite
        self.drop_train_na_threshold = drop_train_na_threshold

        self._original_train = None
        self._original_test = None
        self._merged_data = None

        self.transformations = []
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_original(self):
        if self._original_train is None or self._original_test is None:
            train_path = os.path.join(self.data_dir, self.train_csv)
            test_path = os.path.join(self.data_dir, self.test_csv)

            if os.path.exists(train_path) and os.path.exists(test_path):
                if train_path.lower().endswith('.csv') and test_path.lower().endswith('.csv'):
                    self._original_train = pd.read_csv(train_path)
                    self._original_test = pd.read_csv(test_path)
                elif train_path.lower().endswith('.parquet') and test_path.lower().endswith('.parquet'):
                    self._original_train = pd.read_parquet(train_path, engine='pyarrow')
                    self._original_test = pd.read_parquet(test_path, engine='pyarrow')
                    
                else:
                    raise ValueError(f"Unsupported file format for: {train_path} or {test_path}")
            else:
                fe_train = FeatureEngineering(data_dir=self.data_dir)
                fe_train.load_data()
                self._original_train = fe_train.data.copy(deep=True)

                fe_test = FeatureEngineering(data_dir=self.data_dir)
                fe_test.load_data()
                self._original_test = fe_test.data.copy(deep=True)

            # Set ID column as index if it exists
            train_id_column = mlarchitect.mlarchitect_config['TRAIN_ID']
            test_id_column = mlarchitect.mlarchitect_config['TEST_ID']
            if train_id_column in self._original_train.columns:
                self._original_train.set_index(train_id_column, inplace=True)
            if test_id_column in self._original_test.columns:
                self._original_test.set_index(test_id_column, inplace=True)
                
            # Apply NaN dropping BEFORE transformation ONLY on train set
            if self.drop_train_na_threshold is not None:
                fraction_missing = self._original_train.isnull().mean(axis=1)
                
                # Find rows to drop
                before_rows = len(self._original_train)
                self._original_train = self._original_train.loc[fraction_missing <= self.drop_train_na_threshold]
                after_rows = len(self._original_train)

                print(f"[DROP] {before_rows - after_rows} rows dropped due to NA threshold {self.drop_train_na_threshold}")

            # Add marker for train/test split
            self._original_train['_is_train'] = True
            self._original_test['_is_train'] = False
            
            # Concatenate with indexes preserved
            self._merged_data = pd.concat([self._original_train, self._original_test], axis=0)

        return self._merged_data, self._original_train, self._original_test

    def add_transformation(self, name, transform_fn, keep_original_columns=False):
        self.transformations.append((name, transform_fn, keep_original_columns))

    def run_all(self):
        if not self.transformations:
            print("No transformations to run.")
            return

        merged_data, original_train, original_test = self._load_original()

        for (name, transform_fn, keep_original) in self.transformations:
            train_path = os.path.join(self.output_dir, f"{name}_train.parquet")
            test_path = os.path.join(self.output_dir, f"{name}_test.parquet")

            if not self.overwrite and os.path.exists(train_path) and os.path.exists(test_path):
                print(f"[SKIP] {name} (files already exist)")
                continue

            print(f"[RUN] {name} -> {train_path} and {test_path}")
            start_time = time.time()

            merged_copy = merged_data.copy(deep=True)
            transformed_data = transform_fn(merged_copy)

            # Ensure columns are updated without chained assignment issues
            transformed_data.loc[:, '_is_train'] = merged_copy['_is_train'].values

            if len(transformed_data) != len(merged_copy):
                raise ValueError(
                    f"Transformation altered row count! Expected {len(merged_copy)}, "
                    f"got {len(transformed_data)}."
                )

            if not keep_original:
                original_cols = set(merged_data.columns)
                new_cols = [col for col in transformed_data.columns if col not in original_cols or col == '_is_train']
                transformed_data = transformed_data[new_cols]

            train_data = transformed_data[transformed_data['_is_train']].drop('_is_train', axis=1)
            test_data = transformed_data[~transformed_data['_is_train']].drop('_is_train', axis=1)

            train_data.to_parquet(train_path)
            test_data.to_parquet(test_path)

            elapsed_time = time.time() - start_time
            print(f"[DONE] Saved {train_path} and {test_path} in {elapsed_time:.2f} seconds")
            print(f"       Train shape: {train_data.shape}, Test shape: {test_data.shape}")

class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

def load_model(model_name):
    idx = model_name.find('-')
    suffix = model_name[idx:]
    path = os.path.join( mlarchitect.mlarchitect_config['PATH'], model_name)

    filename_oos = "oos" + suffix + ".csv"
    path_oos = os.path.join(path, filename_oos)
    df_oos = pd.read_csv(path_oos)

    filename_submit = "submit" + suffix + ".csv"
    path_submit = os.path.join(path, filename_submit)
    df_submit = pd.read_csv(path_submit)

    return df_oos, df_submit

def save_importance_report(model,imp):
  root_path = mlarchitect.mlarchitect_config['PATH']
  model_path = os.path.join(root_path,model)
  imp.to_csv(os.path.join(model_path,'peturb.csv'),index=False)

def remove_highly_correlated_features(df, threshold=0.95, excluded_columns=None):
    """
    Remove highly correlated features from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing features.
    threshold : float, default=0.95
        Correlation threshold above which features are considered highly correlated.
    excluded_columns : list, optional
        List of column names to exclude from correlation analysis.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with highly correlated features removed.
    list
        List of removed feature names.
    """
    print(f"Checking for highly correlated features with threshold: {threshold}")
    
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Default empty list if None
    if excluded_columns is None:
        excluded_columns = []
    
    # Get columns that should be analyzed (exclude non-numeric and specified columns)
    columns_to_analyze = [col for col in df_copy.columns if col not in excluded_columns 
                          and pd.api.types.is_numeric_dtype(df_copy[col])]
    
    # Calculate the correlation matrix
    corr_matrix = df_copy[columns_to_analyze].corr().abs()
    
    # Create an upper triangular mask
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Found {len(to_drop)} highly correlated features to remove")
    
    # Remove the correlated features
    df_result = df_copy.drop(columns=to_drop)
    
    return df_result, to_drop

def remove_noisy_features(df, noise_std_threshold=3.0, noise_outlier_ratio=0.1, excluded_columns=None):
    """
    Remove features that appear to be noisy based on outlier analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing features.
    noise_std_threshold : float, default=3.0
        Number of standard deviations beyond which values are considered outliers.
    noise_outlier_ratio : float, default=0.1
        If a feature has more than this fraction of outliers, it's considered noisy.
    excluded_columns : list, optional
        List of column names to exclude from noise analysis.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with noisy features removed.
    list
        List of removed feature names.
    """
    print(f"Checking for noisy features (std threshold: {noise_std_threshold}, outlier ratio: {noise_outlier_ratio})")
    
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Default empty list if None
    if excluded_columns is None:
        excluded_columns = []
    
    # Get columns that should be analyzed (exclude non-numeric and specified columns)
    columns_to_analyze = [col for col in df_copy.columns if col not in excluded_columns 
                          and pd.api.types.is_numeric_dtype(df_copy[col])]
    
    noisy_features = []
    
    # Analyze each feature for noise
    for col in columns_to_analyze:
        # Calculate mean and standard deviation
        mean = df_copy[col].mean()
        std = df_copy[col].std()
        
        if std == 0:  # Skip constant features
            continue
            
        # Calculate how many values are outliers (beyond threshold standard deviations)
        outliers = np.abs(df_copy[col] - mean) > (noise_std_threshold * std)
        outlier_ratio = outliers.mean()
        
        # If more than specified ratio of values are outliers, consider the feature noisy
        if outlier_ratio > noise_outlier_ratio:
            noisy_features.append(col)
    
    print(f"Found {len(noisy_features)} noisy features to remove")
    
    # Remove the noisy features
    df_result = df_copy.drop(columns=noisy_features)
    
    return df_result, noisy_features

