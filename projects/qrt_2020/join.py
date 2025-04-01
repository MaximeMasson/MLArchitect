import os
import sys

# Add the framework_ml root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import mlarchitect
mlarchitect.load_config('qrt-2020')

from mlarchitect.joiner import DataJoiner

# Define source datasets
sources = [
    'nan_handling', 
    'basic_statistical_features', 
    'date_comparison_features', 
    'stock_comparison_features', 
    'temporal_features'
]

# Define data directories
data_dir = "transformed/v1/"        # folder containing the parquet files
output_dir = "joined/v1/"    # folder where joined files will be saved
processed_dir = "processed/v1/"  # Folder for processed data

# Define the transformations to apply with added diagnostics
transformations = {
    # # Normalize features with diagnostics enabled
    # 'normalize': {
    #     'method': 'standard',     # Options: 'standard', 'minmax', 'robust'
    #     'diagnose': True,         # Enable diagnostics to examine extreme values
    #     'handle_inf': 'clip',     # Options: 'clip', 'nan'
    #     'inf_threshold': 1e6      # Lower threshold to see more extreme values
    # },    
    # # Remove outliers from training data
    'remove_outliers': {
        'method': 'zscore',       # Options: 'zscore', 'iqr', 'isolation_forest'
        'threshold': 4.0          # Use higher threshold to keep more data (was 3.0)
    }
}

# Create DataJoiner with custom processed directory
joiner = DataJoiner(
    data_dir=data_dir, 
    output_dir=output_dir, 
    sources=sources,
    processed_dir=processed_dir)

# Join the data and apply transformations with diagnostic output
print("Joining and processing data...")
df_train, df_test = joiner.join_data(transformations=transformations)

print("Data processing pipeline complete!")
