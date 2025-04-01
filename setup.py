from setuptools import setup, find_packages

setup(
    name="mlarchitect",
    version="1.2",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'optuna',
        'pyarrow',
        'matplotlib',
        'shap',
        'jupyter',
        'ipywidgets',
        'seaborn',
        'plotly',
        'pytest',
        'hyperopt',
        'catboost',
        'dask',
        'statsmodels'
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={
        'mlarchitect': ['projects/projectsConfig.json'],
    }
)
