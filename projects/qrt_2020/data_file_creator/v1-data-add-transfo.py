import os
import sys

# Add the framework_ml root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import mlarchitect
mlarchitect.load_config('qrt-2020')

from mlarchitect.util import GenerateDataFile
import projects.qrt_2020.transformation as tf

manager = GenerateDataFile(    
        data_dir="/transformed/v1",
        train_csv="nan_handling_train.parquet",
        test_csv="nan_handling_test.parquet",
        output_dir="/transformed/simple",
        overwrite=True,
        drop_train_na_threshold=None
        )
    
# Add transformations
# manager.add_transformation("basic_statistical_features", tf.basic_statistical_features, keep_original_columns=False)
# manager.add_transformation("temporal_features", tf.temporal_features, keep_original_columns=False)
# manager.add_transformation("stock_comparison_features", tf.stock_comparison_features, keep_original_columns=False)
manager.add_transformation("mean_std_group", tf.mean_std_group, keep_original_columns=False)
manager.run_all()
