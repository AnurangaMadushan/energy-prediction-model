# energy-prediction-model
# Appliance Energy Prediction Using Deep Learning

This project implements a comprehensive multivariate time-series prediction system to forecast appliance energy consumption using deep learning techniques. The solution includes data preprocessing, feature engineering, model development, and performance evaluation.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## 🎯 Project Overview

This project addresses the challenge of predicting appliance energy consumption using environmental and temporal features. The solution employs:

- **Exploratory Data Analysis (EDA)** for understanding data patterns
- **Feature Engineering** including time-based, rolling window, and lagged features
- **Deep Learning Models** (LSTM networks) for time-series forecasting
- **Model Optimization** through hyperparameter tuning and regularization
- **Comprehensive Evaluation** using MAE, RMSE, and R² metrics

## 📊 Dataset

The project uses the **Appliance Energy Prediction Dataset** containing:
- **Size**: ~20,000 records
- **Interval**: 10-minute observations
- **Duration**: Several months of data
- **Features**: 
  - Environmental: Indoor/outdoor temperature and humidity
  - Temporal: Date, time-based indicators
  - Target: Appliance energy consumption (Wh)

### Key Features:
- `Appliances`: Energy consumption (target variable)
- `T1-T6`: Indoor temperature readings
- `RH_1-RH_6`: Indoor humidity readings
- `T_out`, `RH_out`: Outdoor conditions
- `Windspeed`, `Visibility`, `Press_mm_hg`: Weather data
- `Date`: Timestamp information

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Original dataset files
│   ├── processed/
│   │   ├── preprocessed/       # Cleaned and preprocessed data
│   │   └── final/             # Final feature-engineered dataset
├── notebooks/
│   └── EDA_Analysis.ipynb     # Exploratory Data Analysis notebook
├── src/                       # Source code modules
│   ├── data_preprocessing.py  # Data cleaning and preprocessing functions
│   ├── feature_engineering.py # Feature creation and selection functions
│   └── model_training_and_evaluation.py # Model development and evaluation
├── models/                    # Trained model files
├── reports/                   # Generated reports and visualizations
├── requirements.txt          # Project dependencies
└── README.md                 # Project overview and setup instructions
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Google Colab (recommended) or Jupyter Notebook
- Internet connection for downloading dependencies

### Setup Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd appliance-energy-prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **For Google Colab users:**
   - Upload the project folder to Google Drive
   - Open the notebook in Google Colab
   - Install required packages using the provided cells

## 📂 Data Setup

### Important: Data Path Configuration

Since this project was developed using Google Colab with Google Drive integration, you need to configure the data paths according to your environment:

1. **Place your dataset files in the following structure:**
   - Raw data: `data/raw/`
   - Preprocessed data: `data/processed/preprocessed/`
   - Final data: `data/processed/final/`

2. **Update data paths in the notebook:**
   ```python
   # Example path configuration
   RAW_DATA_PATH = "your/path/to/data/raw/"
   PROCESSED_DATA_PATH = "your/path/to/data/processed/"
   FINAL_DATA_PATH = "your/path/to/data/processed/final/"
   ```

3. **For Google Colab users:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Update these paths to match your Google Drive structure
   DATA_PATH = "/content/drive/MyDrive/your-project-folder/data/"
   ```

## 💻 Usage

### How to Run the Code

This project is structured with modular components that can be executed in sequence or independently:

#### Option 1: Complete Pipeline Execution
1. **Start with Exploratory Data Analysis:**
   ```bash
   # Open and run the EDA notebook
   jupyter notebook notebooks/EDA_Analysis.ipynb
   ```
   Or in Google Colab:
   - Upload `notebooks/EDA_Analysis.ipynb` to Colab
   - Run all cells to understand data patterns and insights

2. **Execute the Pipeline Modules:**
   ```bash
   # Run data preprocessing
   python src/data_preprocessing.py
   
   # Run feature engineering
   python src/feature_engineering.py
   
   # Run model training and evaluation
   python src/model_training_and_evaluation.py
   ```

#### Option 2: Step-by-Step Execution

1. **Data Preprocessing:**
   ```python
   # Execute data cleaning and preprocessing
   from src.data_preprocessing import preprocess_data
   processed_data = preprocess_data(raw_data_path)
   ```

2. **Feature Engineering:**
   ```python
   # Create engineered features
   from src.feature_engineering import engineer_features
   final_data = engineer_features(processed_data)
   ```

3. **Model Training and Evaluation:**
   ```python
   # Train and evaluate models
   from src.model_training_and_evaluation import train_evaluate_models
   results = train_evaluate_models(final_data)
   ```

### Running the Complete Pipeline

1. **Data Preprocessing:**
   - Load and explore the raw dataset using `notebooks/EDA_Analysis.ipynb`
   - Clean data and handle missing values using `src/data_preprocessing.py`
   - Apply data scaling and normalization

2. **Feature Engineering:**
   - Create time-based features using `src/feature_engineering.py`
   - Generate rolling window statistics
   - Implement lagged features
   - Perform feature selection

3. **Model Development:**
   - Train baseline models (Linear Regression, Random Forest)
   - Develop LSTM deep learning model using `src/model_training_and_evaluation.py`
   - Apply regularization techniques

4. **Model Evaluation:**
   - Evaluate using MAE, RMSE, R² metrics
   - Generate prediction visualizations
   - Compare model performances

### Execution Order:
1. `notebooks/EDA_Analysis.ipynb` - Understand the data
2. `src/data_preprocessing.py` - Clean and prepare data
3. `src/feature_engineering.py` - Create features
4. `src/model_training_and_evaluation.py` - Train and evaluate models

## 🏗️ Model Architecture

### Deep Learning Model (LSTM)
- **Architecture**: Sequential LSTM network
- **Layers**: 
  - LSTM layers with dropout regularization
  - Dense output layer
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error
- **Regularization**: Dropout layers, Early Stopping

### Baseline Models
- **Linear Regression**: For performance comparison
- **Random Forest**: Tree-based ensemble method

## 📈 Results

The project achieves competitive performance in energy consumption prediction:

- **Evaluation Metrics**: MAE, RMSE, R²
- **Model Comparison**: Deep learning vs. baseline models
- **Visualizations**: Predicted vs. actual values, residual plots
- **Performance Analysis**: Detailed in the project report

## 📦 File Descriptions

### Core Files:

- **`notebooks/EDA_Analysis.ipynb`**: Comprehensive exploratory data analysis including visualizations, statistical summaries, and data insights
- **`src/data_preprocessing.py`**: Data cleaning functions, missing value handling, outlier detection, and data scaling operations
- **`src/feature_engineering.py`**: Time-based feature creation, rolling window calculations, lagged features, and feature selection methods
- **`src/model_training_and_evaluation.py`**: Model implementation, training procedures, hyperparameter tuning, and evaluation metrics calculation
- **`requirements.txt`**: Complete list of Python dependencies with version specifications
- **`README.md`**: Project overview, setup instructions, and usage guidelines

## 📦 Dependencies

### Core Libraries:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
```

### Installation via requirements.txt:
```bash
pip install -r requirements.txt
```

## 🛠️ Technical Specifications

- **Development Environment**: Google Colab
- **Python Version**: 3.7+
- **Deep Learning Framework**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn

## 📝 Notes

- **Data Paths**: Remember to update data file paths according to your local setup
- **Google Colab**: Mount Google Drive and adjust paths accordingly
- **Memory Usage**: Large datasets may require memory management techniques
- **Training Time**: LSTM model training may take considerable time depending on hardware

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📧 Contact

For questions or issues regarding this project, please open an issue in the repository or contact the project maintainer.

---

**Note**: This project was developed as part of a technical assessment for multivariate time-series prediction using deep learning techniques. The implementation demonstrates end-to-end machine learning pipeline development with emphasis on time-series forecasting applications.