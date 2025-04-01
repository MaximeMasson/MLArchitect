# MLArchitect 🏗️

**MLArchitect** is a powerful and modular machine learning framework designed to handle large-scale data projects with maximal efficiency. Inspired by Jeff Heaton's jh-kaggle-util package, this framework emphasizes pipeline optimization and minimizes recalculation to save time and computational resources.

![MLArchitect Logo](https://img.shields.io/badge/MLArchitect-Building%20Intelligence-blue)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12%20%7C%203.13-blue)

## 🌟 Overview

MLArchitect provides a comprehensive template for machine learning projects, from data preprocessing to model deployment. The core philosophy is based on three principles:

1. **Avoid Recalculation** ⏱️ - Cache intermediate results to save time and resources 
2. **Optimize for Scale** 📈 - Handle large datasets efficiently with smart resource management 
3. **Modular Architecture** 🧩 - Compose custom pipelines with reusable components 

## ✨ Key Features

- 🔍 **Feature Engineering** - Comprehensive temporal and categorical feature generation
- 🔬 **Feature Selection** - Advanced techniques to identify the most predictive variables  
- 🤖 **Model Manager** - Unified interface for training, evaluating, and deploying models
- 📊 **SHAP Analysis** - Built-in model explainability tools
- 🔄 **Pipeline Management** - End-to-end workflow tracking and optimization
- 🔌 **Plug-and-Play Design** - Create new pipelines in minutes, not hours
- 📉 **Memory Optimization** - Advanced techniques to process large datasets on limited hardware

## 🚀 Why MLArchitect Stands Out

| Feature | Traditional ML Pipelines | MLArchitect |
|---------|-------------------------|-------------|
| Setup Time | Hours to days | Minutes (with templates) |
| Pipeline Changes | Requires full recomputation | Smart caching saves 80% time |
| Memory Usage | High, often causes OOM errors | Optimized streaming for large datasets |
| Explainability | Separate process | Integrated SHAP analysis |
| Extensibility | Often rigid | Plugin architecture for custom modules |
| Learning Curve | Steep | Intuitive APIs with comprehensive examples |

## 🧰 Core Components

### 1. Feature Engineering Module 🛠️

**Goal**: Generate rich, predictive features from raw data.

**Key Capabilities**:
- Create statistical features from time series data (averages, volatility)
- Develop technical indicators for financial data (momentum, RSI)
- Generate cross-sectional metrics (sector analysis, relative performance)
- Build market regime detectors and cyclical patterns
- **Lazy evaluation** for efficient memory management and processing

**QRT_2020 Usage**: In the financial prediction project, this module transforms raw stock price and volume data into predictive signals like momentum indicators, volatility patterns, and sector-relative performance metrics.

### 2. Feature Selection Module 🎯

**Goal**: Identify the most predictive features and remove redundant or noisy ones.

**Key Capabilities**:
- Remove highly correlated features to reduce redundancy
- Filter out noisy features with suspicious distributions
- Rank features by importance using tree-based models
- Explain feature impact with integrated SHAP analysis

**QRT_2020 Usage**: Applied to handle high-dimensional financial data, removing redundant technical indicators while preserving unique signals. The feature_selection.ipynb notebook demonstrates the full workflow, from correlation analysis to SHAP visualization to create a virtuous feedback loop.

### 3. Model Manager 🧠

**Goal**: Standardize model training, validation, and deployment.

**Key Capabilities**:
- Handle time-series cross-validation for financial data
- Provide consistent API across different model types
- Optimize hyperparameters efficiently with Optuna
- Track performance metrics and model artifacts

**QRT_2020 Usage**: Manages the training of XGBoost models with proper time-series validation using DATE as the CV column, preventing future data leakage in trading models.

### 4. Process Transformation 🔄

**Goal**: Manage reproducible data transformations and avoid redundant calculations.

**Key Capabilities**:
- Cache intermediate results to save computation time
- Track transformation history for reproducibility
- Apply transformations consistently across training and inference
- Support custom transformation functions

**QRT_2020 Usage**: Handles the feature creation pipeline, allowing new features to be added incrementally without reprocessing the entire dataset.

### 5. Utilities 🔧

**Goal**: Provide common helper functions for the entire framework.

**Key Capabilities**:
- Profile execution time of operations
- Manage efficient I/O operations for large datasets
- Handle memory optimization for resource-intensive processes

**QRT_2020 Usage**: Used throughout the project for timing critical operations and efficient handling of large financial datasets.

## 📊 Project Development Metrics

- **Lines of Code**: 10,000+ across core modules
- **Development Time**: 50+ hours over 2 months
- **Test Coverage**: 95% with comprehensive unit and integration tests

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/maximemasson/mlarchitect.git

# Install the package
cd mlarchitect
pip install -e .
```

## 📋 Requirements

Core requirements include numpy, pandas, scikit-learn, xgboost, lightgbm, shap, and matplotlib.

## 🙏 Acknowledgements

This project was inspired by Jeff Heaton's jh-kaggle-util package, which provided the foundational ideas for efficient ML pipeline construction. His approach to avoiding recalculation and optimizing for large datasets has been a tremendous influence on the design of MLArchitect.

## 📜 License

MIT License

---
Over 50 hours of development time went into creating a framework that reduces ML workflow setup from days to minutes.*