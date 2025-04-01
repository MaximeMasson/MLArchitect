import os
import sys

# Add the framework_ml root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import mlarchitect
mlarchitect.load_config('qrt-2020')

from mlarchitect.util import GenerateDataFile
import projects.qrt_2020.transformation as tf

manager = GenerateDataFile(    
        data_dir="/raw",
        train_csv="train.csv",
        test_csv="test.csv",
        output_dir="/transformed/v1_test",
        overwrite=True,
        drop_train_na_threshold=0.1 # Drop rows with more than 30% of missing values
        )

# Add transformations
manager.add_transformation("nan_handling", tf.nan_handling, keep_original_columns=True)
manager.run_all()
