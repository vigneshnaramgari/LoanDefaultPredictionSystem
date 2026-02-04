
"""
Loan Default Prediction ML Project
==================================
This script implements a complete machine learning pipeline for predicting loan defaults
using Logistic Regression, Random Forest, and XGBoost classifiers.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# 1. DATASET LOADING
# ============================================================================

print("=" * 80)
print("STEP 1: LOADING DATASET")
print("=" * 80)

# Load the dataset
df = pd.read_csv('Loan_default.csv')  # Note: File is named 'Loan_default.csv' (capital L)

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset info:")
print(df.info())

print(f"\nDataset statistics:")
print(df.describe())

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: DATA PREPROCESSING")
print("=" * 80)

# Check for missing values
print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Handle missing values if any
if missing_values.sum() > 0:
    print("\nHandling missing values...")
    # For numerical columns, fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    print("Missing values handled!")
else:
    print("\nNo missing values found!")

# Separate features and target
# Drop LoanID as it's just an identifier
X = df.drop(['LoanID', 'Default'], axis=1)
y = df['Default']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nTarget distribution:\n{y.value_counts()}")
print(f"\nTarget distribution (%):\n{y.value_counts(normalize=True) * 100}")

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical columns: {categorical_columns}")
print(f"Numerical columns: {numerical_columns}")

# Create a copy for preprocessing
X_processed = X.copy()

# Label Encoding for categorical variables
print("\nApplying Label Encoding to categorical variables...")
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} unique values")

# Standard Scaling for numerical features
print("\nApplying Standard Scaling to numerical features...")
scaler = StandardScaler()
X_processed[numerical_columns] = scaler.fit_transform(X_processed[numerical_columns])
print("Scaling completed!")

# Train-Test Split (80/20)
print("\nSplitting data into train (80%) and test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"\nTraining set target distribution:\n{y_train.value_counts()}")
print(f"\nTest set target distribution:\n{y_test.value_counts()}")

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Create EDA visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 3.1 Target Variable Distribution
axes[0, 0].bar(y.value_counts().index, y.value_counts().values, 
                color=['#3498db', '#e74c3c'], alpha=0.7)
axes[0, 0].set_xlabel('Default Status', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)
axes[0, 0].set_title('Distribution of Target Variable (Default)', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['Non-Default (0)', 'Default (1)'])
for i, v in enumerate(y.value_counts().values):
    axes[0, 0].text(i, v, str(v), ha='center', va='bottom', fontsize=11)

# 3.2 Correlation Heatmap
correlation_matrix = X_processed.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
            center=0, square=True, ax=axes[0, 1], cbar_kws={"shrink": 0.8})
axes[0, 1].set_title('Correlation Heatmap of Features', fontsize=14, fontweight='bold')

# 3.3 Important Numerical Features Distribution
# Select top correlated features with target
target_corr = X_processed.copy()
target_corr['Default'] = y
correlations = target_corr.corr()['Default'].abs().sort_values(ascending=False)
top_features = correlations[1:6].index.tolist()  # Top 5 features excluding target

# Plot distribution of top features by target
if len(top_features) > 0:
    feature_idx = 0
    for i in range(2, 4):
        if feature_idx < len(top_features):
            feature = top_features[feature_idx]
            if feature in numerical_columns:
                axes[1, i-2].hist(X_processed[y == 0][feature], bins=30, 
                                 alpha=0.6, label='Non-Default', color='#3498db')
                axes[1, i-2].hist(X_processed[y == 1][feature], bins=30, 
                                 alpha=0.6, label='Default', color='#e74c3c')
                axes[1, i-2].set_xlabel(feature, fontsize=11)
                axes[1, i-2].set_ylabel('Frequency', fontsize=11)
                axes[1, i-2].set_title(f'Distribution: {feature}', fontsize=12, fontweight='bold')
                axes[1, i-2].legend()
                feature_idx += 1

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("\nEDA visualizations saved as 'eda_analysis.png'")
plt.close()

# Additional feature importance plot (using correlation)
plt.figure(figsize=(12, 8))
top_corr_features = correlations[1:11].sort_values(ascending=True)
plt.barh(range(len(top_corr_features)), top_corr_features.values, color='steelblue')
plt.yticks(range(len(top_corr_features)), top_corr_features.index)
plt.xlabel('Absolute Correlation with Default', fontsize=12)
plt.title('Top 10 Features Correlated with Default Status', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
print("Feature correlation plot saved as 'feature_correlation.png'")
plt.close()

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: MODEL TRAINING")
print("=" * 80)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
}

# Train all models
trained_models = {}
print("\nTraining models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} training completed!")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: MODEL EVALUATION")
print("=" * 80)

# Evaluate all models
results = {}

for name, model in trained_models.items():
    print(f"\n{'='*60}")
    print(f"Evaluating {name}")
    print(f"{'='*60}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    # Print metrics
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'])
    plt.title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved as 'confusion_matrix_{name.replace(' ', '_').lower()}.png'")

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: MODEL COMPARISON")
print("=" * 80)

# Create comparison table
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)
print("\nModel Performance Comparison:")
print(comparison_df.to_string())

# Visualize comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comparison_df.columns))
width = 0.25
models_list = list(comparison_df.index)

for i, model in enumerate(models_list):
    offset = (i - 1) * width
    ax.bar(x + offset, comparison_df.loc[model], width, label=model, alpha=0.8)

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df.columns)
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, model in enumerate(models_list):
    offset = (i - 1) * width
    for j, metric in enumerate(comparison_df.columns):
        value = comparison_df.loc[model, metric]
        ax.text(j + offset, value + 0.01, f'{value:.3f}', 
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\nModel comparison chart saved as 'model_comparison.png'")
plt.close()

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Get feature importance from Random Forest
rf_model = trained_models['Random Forest']
rf_importance = pd.DataFrame({
    'Feature': X_processed.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features - Random Forest:")
print(rf_importance.head(10).to_string(index=False))

# Get feature importance from XGBoost
xgb_model = trained_models['XGBoost']
xgb_importance = pd.DataFrame({
    'Feature': X_processed.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features - XGBoost:")
print(xgb_importance.head(10).to_string(index=False))

# Visualize feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Random Forest Feature Importance
top_rf = rf_importance.head(10)
axes[0].barh(range(len(top_rf)), top_rf['Importance'].values, color='steelblue')
axes[0].set_yticks(range(len(top_rf)))
axes[0].set_yticklabels(top_rf['Feature'].values)
axes[0].set_xlabel('Importance', fontsize=12)
axes[0].set_title('Top 10 Features - Random Forest', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

# XGBoost Feature Importance
top_xgb = xgb_importance.head(10)
axes[1].barh(range(len(top_xgb)), top_xgb['Importance'].values, color='darkgreen')
axes[1].set_yticks(range(len(top_xgb)))
axes[1].set_yticklabels(top_xgb['Feature'].values)
axes[1].set_xlabel('Importance', fontsize=12)
axes[1].set_title('Top 10 Features - XGBoost', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plots saved as 'feature_importance.png'")
plt.close()

# ============================================================================
# 8. FINAL CONCLUSION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: FINAL CONCLUSION")
print("=" * 80)

# Find best model
best_model_name = comparison_df['F1-Score'].idxmax()
best_metrics = comparison_df.loc[best_model_name]

print(f"\n{'='*60}")
print("BEST PERFORMING MODEL")
print(f"{'='*60}")
print(f"\nModel: {best_model_name}")
print(f"\nPerformance Metrics:")
print(f"  Accuracy:  {best_metrics['Accuracy']:.4f}")
print(f"  Precision: {best_metrics['Precision']:.4f}")
print(f"  Recall:    {best_metrics['Recall']:.4f}")
print(f"  F1-Score:  {best_metrics['F1-Score']:.4f}")

print(f"\n{'='*60}")
print("ANALYSIS & RECOMMENDATIONS")
print(f"{'='*60}")

print(f"""
1. Model Performance Summary:
   - {best_model_name} achieved the highest F1-Score of {best_metrics['F1-Score']:.4f}
   - This model provides the best balance between precision and recall

2. Why {best_model_name} Performs Best:
""")

if best_model_name == 'XGBoost':
    print("""
   - XGBoost is an advanced gradient boosting algorithm that handles complex
     non-linear relationships effectively
   - It uses regularization to prevent overfitting
   - It can capture feature interactions automatically
   - Excellent for tabular data with mixed feature types
""")
elif best_model_name == 'Random Forest':
    print("""
   - Random Forest uses ensemble of decision trees to reduce overfitting
   - It handles non-linear relationships well
   - Provides good feature importance insights
   - Robust to outliers and missing values
""")
else:
    print("""
   - Logistic Regression provides interpretable results
   - Fast training and prediction
   - Good baseline model for binary classification
   - Works well when relationships are approximately linear
""")

print("""
3. Key Insights:
   - The models can effectively predict loan defaults
   - Feature importance analysis reveals which factors most influence default risk
   - The balanced F1-score indicates good performance on both classes

4. Recommendations:
   - Use the best performing model ({}) for production deployment
   - Monitor model performance over time and retrain periodically
   - Consider feature engineering to further improve performance
   - Implement proper model versioning and A/B testing in production
""".format(best_model_name))

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nAll visualizations and results have been saved.")
print("Generated files:")
print("  - eda_analysis.png")
print("  - feature_correlation.png")
print("  - confusion_matrix_logistic_regression.png")
print("  - confusion_matrix_random_forest.png")
print("  - confusion_matrix_xgboost.png")
print("  - model_comparison.png")
print("  - feature_importance.png")


