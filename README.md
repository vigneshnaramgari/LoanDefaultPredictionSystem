# Loan Default Prediction ML Project

A comprehensive machine learning project for predicting loan defaults using multiple classification algorithms.

## Project Overview

This project implements a complete ML pipeline to predict loan defaults using three different models:
- **Logistic Regression** - Linear baseline model
- **Random Forest** - Ensemble tree-based model
- **XGBoost** - Advanced gradient boosting model

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Dataset

- **File**: `Loan_default.csv`
- **Target Variable**: `Default` (1 = default, 0 = non-default)
- The dataset contains various features including:
  - Numerical: Age, Income, LoanAmount, CreditScore, etc.
  - Categorical: Education, EmploymentType, MaritalStatus, etc.

## Project Structure

```
├── loan_default_prediction.py  # Main ML pipeline script
├── app.py                      # Streamlit web application
├── requirements.txt             # Python dependencies
├── Loan_default.csv            # Dataset
└── README.md                   # This file
```

## Running the Project

### Option 1: Command Line Script

Simply run the main script:

```bash
python loan_default_prediction.py
```

### Option 2: Streamlit Web Application (Recommended)

Run the interactive web UI:

```bash
streamlit run app.py
```

The web application will open in your default browser at `http://localhost:8501`

#### Web App Features:
- **🏠 Home**: Project overview and dataset statistics
- **📈 Data Overview**: Explore dataset structure and statistics
- **🔍 Exploratory Analysis**: Interactive visualizations (distributions, correlations)
- **🤖 Model Training**: View model information and training status
- **📊 Model Evaluation**: Compare model performance with metrics and confusion matrices
- **🎯 Make Prediction**: Interactive form to predict loan defaults for new applicants
- **⭐ Feature Importance**: Analyze which features matter most for predictions

## Output

The script generates several outputs:

1. **Console Output**: 
   - Dataset information
   - Preprocessing steps
   - Model training progress
   - Evaluation metrics for each model
   - Feature importance analysis
   - Final conclusions

2. **Visualization Files**:
   - `eda_analysis.png` - Exploratory data analysis plots
   - `feature_correlation.png` - Top features correlated with default
   - `confusion_matrix_logistic_regression.png` - LR confusion matrix
   - `confusion_matrix_random_forest.png` - RF confusion matrix
   - `confusion_matrix_xgboost.png` - XGBoost confusion matrix
   - `model_comparison.png` - Side-by-side model performance comparison
   - `feature_importance.png` - Feature importance from RF and XGBoost

## Project Steps

1. **Data Loading**: Loads the loan default dataset
2. **Data Preprocessing**: 
   - Handles missing values
   - Label encoding for categorical variables
   - Standard scaling for numerical features
   - Train/test split (80/20)
3. **Exploratory Data Analysis**:
   - Target variable distribution
   - Correlation heatmap
   - Feature distribution plots
4. **Model Training**: Trains all three models
5. **Model Evaluation**: 
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrices
   - Model comparison table
6. **Feature Importance**: Analysis of important features
7. **Final Conclusion**: Best model identification and recommendations

## Model Evaluation Metrics

Each model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Web Application Usage

The Streamlit app provides an interactive interface where you can:

1. **Explore Data**: Navigate through different sections to understand the dataset
2. **View Model Performance**: See how each model performs with detailed metrics
3. **Make Predictions**: Enter borrower information in the prediction form to get:
   - Individual model predictions (Logistic Regression, Random Forest, XGBoost)
   - Default probability scores
   - Consensus prediction from all models
4. **Analyze Features**: Understand which features are most important for predictions

## Notes

- The script uses stratified train-test split to maintain class distribution
- All models use random_state=42 for reproducibility
- Missing values are handled automatically
- Feature importance is extracted from tree-based models (RF and XGBoost)
- The Streamlit app uses caching for faster performance on repeated runs

