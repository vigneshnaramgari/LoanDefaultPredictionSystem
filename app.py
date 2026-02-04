"""
Loan Default Prediction - Streamlit Web Application
===================================================
Interactive web UI for loan default prediction using ML models
"""

import streamlit as st
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

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💰 Loan Default Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📈 Data Overview", "🔍 Exploratory Analysis", "🤖 Model Training", "📊 Model Evaluation", "🎯 Make Prediction", "⭐ Feature Importance"]
)

# Cache data loading
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('Loan_default.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file 'Loan_default.csv' not found!")
        return None

# Cache preprocessing
@st.cache_data
def preprocess_data(df):
    """Preprocess the data"""
    # Handle missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Separate features and target
    X = df.drop(['LoanID', 'Default'], axis=1)
    y = df['Default']
    
    # Identify columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Label encoding
    X_processed = X.copy()
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        label_encoders[col] = le
    
    # Scaling
    scaler = StandardScaler()
    X_processed[numerical_columns] = scaler.fit_transform(X_processed[numerical_columns])
    
    return X, X_processed, y, categorical_columns, numerical_columns, label_encoders, scaler

# Cache model training
@st.cache_resource
def train_models(X_train, y_train):
    """Train all models and cache them"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

# Load data
df = load_data()

if df is not None:
    # Preprocess data
    X, X_processed, y, categorical_columns, numerical_columns, label_encoders, scaler = preprocess_data(df.copy())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

# Lazy model loading to avoid long blocking on initial load
@st.cache_resource
def get_trained_models(X_train, y_train):
    """Train all models and cache them lazily"""
    return train_models(X_train, y_train)

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.header("Welcome to Loan Default Prediction System")
    st.markdown("""
    This application uses machine learning to predict loan defaults based on various borrower characteristics.
    
    ### Features:
    - 📊 **Data Overview**: Explore the dataset structure and statistics
    - 🔍 **Exploratory Analysis**: Visualize data distributions and correlations
    - 🤖 **Model Training**: Train three ML models (Logistic Regression, Random Forest, XGBoost)
    - 📊 **Model Evaluation**: Compare model performance metrics
    - 🎯 **Make Prediction**: Predict loan default for new applicants
    - ⭐ **Feature Importance**: Understand which features matter most
    
    ### Models Used:
    1. **Logistic Regression** - Linear baseline model
    2. **Random Forest** - Ensemble tree-based model
    3. **XGBoost** - Advanced gradient boosting model
    
    ### How to Use:
    1. Navigate through different sections using the sidebar
    2. Explore the data and model performance
    3. Use the "Make Prediction" page to predict loan defaults for new applicants
    """)
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(X.columns))
        with col3:
            st.metric("Default Rate", f"{(y.sum() / len(y) * 100):.2f}%")
        with col4:
            st.metric("Non-Default Rate", f"{((len(y) - y.sum()) / len(y) * 100):.2f}%")

# ============================================================================
# DATA OVERVIEW PAGE
# ============================================================================
elif page == "📈 Data Overview":
    st.header("📈 Dataset Overview")
    
    if df is not None:
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Target Variable:** Default (1 = Default, 0 = Non-Default)")
        
        with col2:
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
            st.write(f"**Duplicate Rows:** {df.duplicated().sum()}")
        
        st.subheader("First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Target Variable Distribution")
        col1, col2 = st.columns(2)
        with col1:
            target_counts = y.value_counts()
            st.bar_chart(target_counts)
        with col2:
            st.dataframe(pd.DataFrame({
                'Status': ['Non-Default (0)', 'Default (1)'],
                'Count': [target_counts[0], target_counts[1]],
                'Percentage': [
                    f"{(target_counts[0] / len(y) * 100):.2f}%",
                    f"{(target_counts[1] / len(y) * 100):.2f}%"
                ]
            }))

# ============================================================================
# EXPLORATORY ANALYSIS PAGE
# ============================================================================
elif page == "🔍 Exploratory Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    if df is not None:
        # Target Distribution
        st.subheader("Target Variable Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        target_counts = y.value_counts()
        ax.bar(target_counts.index, target_counts.values, color=['#3498db', '#e74c3c'], alpha=0.7)
        ax.set_xlabel('Default Status')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Target Variable')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-Default (0)', 'Default (1)'])
        st.pyplot(fig)
        plt.close()
        
        # Correlation Heatmap
        st.subheader("Feature Correlation Heatmap")
        correlation_matrix = X_processed.corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                    center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Heatmap of Features')
        st.pyplot(fig)
        plt.close()
        
        # Feature Distributions
        st.subheader("Feature Distributions by Default Status")
        selected_feature = st.selectbox("Select a feature to visualize:", numerical_columns)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(X_processed[y == 0][selected_feature], bins=30, 
                alpha=0.6, label='Non-Default', color='#3498db')
        ax.hist(X_processed[y == 1][selected_feature], bins=30, 
                alpha=0.6, label='Default', color='#e74c3c')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {selected_feature} by Default Status')
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
        # Top Correlated Features
        st.subheader("Top Features Correlated with Default")
        target_corr = X_processed.copy()
        target_corr['Default'] = y
        correlations = target_corr.corr()['Default'].abs().sort_values(ascending=False)
        top_corr = correlations[1:11].sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_corr)), top_corr.values, color='steelblue')
        ax.set_yticks(range(len(top_corr)))
        ax.set_yticklabels(top_corr.index)
        ax.set_xlabel('Absolute Correlation with Default')
        ax.set_title('Top 10 Features Correlated with Default Status')
        st.pyplot(fig)
        plt.close()

# ============================================================================
# MODEL TRAINING PAGE
# ============================================================================
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if df is not None:
        with st.spinner("Training models (first run may take a few minutes)..."):
            models = get_trained_models(X_train, y_train)
        st.info("Models are automatically trained when you first load the application. They are cached for faster performance.")
        
        st.subheader("Model Information")
        model_info = {
            'Logistic Regression': {
                'Type': 'Linear Classifier',
                'Description': 'Baseline model using logistic function for binary classification',
                'Advantages': 'Fast, interpretable, good baseline'
            },
            'Random Forest': {
                'Type': 'Ensemble Tree-based',
                'Description': 'Uses multiple decision trees with bagging',
                'Advantages': 'Handles non-linearity, robust to overfitting'
            },
            'XGBoost': {
                'Type': 'Gradient Boosting',
                'Description': 'Advanced boosting algorithm with regularization',
                'Advantages': 'High performance, handles complex patterns'
            }
        }
        
        for model_name, info in model_info.items():
            with st.expander(f"📌 {model_name}"):
                st.write(f"**Type:** {info['Type']}")
                st.write(f"**Description:** {info['Description']}")
                st.write(f"**Advantages:** {info['Advantages']}")
        
        st.success("✅ All models have been trained successfully!")
        st.write(f"**Training Set Size:** {len(X_train):,} samples")
        st.write(f"**Test Set Size:** {len(X_test):,} samples")

# ============================================================================
# MODEL EVALUATION PAGE
# ============================================================================
elif page == "📊 Model Evaluation":
    st.header("📊 Model Evaluation")
    
    if df is not None:
        with st.spinner("Training models (first run may take a few minutes)..."):
            models = get_trained_models(X_train, y_train)
        # Calculate metrics for all models
        results = {}
        predictions = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred)
            }
        
        # Display metrics table
        st.subheader("Model Performance Metrics")
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(4)
        st.dataframe(results_df, use_container_width=True)
        
        # Visualize comparison
        st.subheader("Model Comparison Chart")
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(results_df.columns))
        width = 0.25
        models_list = list(results_df.index)
        
        for i, model in enumerate(models_list):
            offset = (i - 1) * width
            ax.bar(x + offset, results_df.loc[model], width, label=model, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df.columns)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Confusion Matrices
        st.subheader("Confusion Matrices")
        selected_model = st.selectbox("Select a model to view confusion matrix:", list(models.keys()))
        
        cm = confusion_matrix(y_test, predictions[selected_model])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Default', 'Default'],
                    yticklabels=['Non-Default', 'Default'], ax=ax)
        ax.set_title(f'Confusion Matrix - {selected_model}')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)
        plt.close()
        
        # Best Model
        best_model = results_df['F1-Score'].idxmax()
        st.success(f"🏆 **Best Model:** {best_model} (F1-Score: {results_df.loc[best_model, 'F1-Score']:.4f})")

# ============================================================================
# MAKE PREDICTION PAGE
# ============================================================================
elif page == "🎯 Make Prediction":
    st.header("🎯 Predict Loan Default")
    
    if df is not None:
        with st.spinner("Training models (first run may take a few minutes)..."):
            models = get_trained_models(X_train, y_train)
        st.write("Enter borrower information to predict loan default probability:")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                income = st.number_input("Income", min_value=0, value=50000)
                loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
                credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
                months_employed = st.number_input("Months Employed", min_value=0, value=24)
                num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=2)
            
            with col2:
                interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0)
                loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=120, value=36)
                dti_ratio = st.slider("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
                education = st.selectbox("Education", ['High School', "Bachelor's", "Master's", 'PhD'])
                employment_type = st.selectbox("Employment Type", ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'])
                marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
            
            col3, col4 = st.columns(2)
            with col3:
                has_mortgage = st.selectbox("Has Mortgage", ['Yes', 'No'])
                has_dependents = st.selectbox("Has Dependents", ['Yes', 'No'])
            with col4:
                loan_purpose = st.selectbox("Loan Purpose", ['Home', 'Auto', 'Education', 'Business', 'Other'])
                has_cosigner = st.selectbox("Has Co-Signer", ['Yes', 'No'])
            
            submitted = st.form_submit_button("🔮 Predict Default", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                'Age': age,
                'Income': income,
                'LoanAmount': loan_amount,
                'CreditScore': credit_score,
                'MonthsEmployed': months_employed,
                'NumCreditLines': num_credit_lines,
                'InterestRate': interest_rate,
                'LoanTerm': loan_term,
                'DTIRatio': dti_ratio,
                'Education': education,
                'EmploymentType': employment_type,
                'MaritalStatus': marital_status,
                'HasMortgage': has_mortgage,
                'HasDependents': has_dependents,
                'LoanPurpose': loan_purpose,
                'HasCoSigner': has_cosigner
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Preprocess
            input_processed = input_df.copy()
            for col in categorical_columns:
                if col in input_processed.columns:
                    le = label_encoders[col]
                    # Handle unseen categories
                    if input_processed[col].iloc[0] in le.classes_:
                        input_processed[col] = le.transform([input_processed[col].iloc[0]])
                    else:
                        input_processed[col] = 0  # Default to first class
            
            # Scale numerical features
            input_processed[numerical_columns] = scaler.transform(input_processed[numerical_columns])
            
            # Make predictions
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            predictions_dict = {}
            probabilities_dict = {}
            
            for name, model in models.items():
                pred = model.predict(input_processed)[0]
                prob = model.predict_proba(input_processed)[0]
                predictions_dict[name] = pred
                probabilities_dict[name] = prob
            
            # Display results
            for name, model in models.items():
                with st.expander(f"📊 {name} Prediction"):
                    pred = predictions_dict[name]
                    prob = probabilities_dict[name]
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if pred == 1:
                            st.error(f"**Prediction: DEFAULT** ⚠️")
                        else:
                            st.success(f"**Prediction: NO DEFAULT** ✅")
                    
                    with col_b:
                        st.metric("Default Probability", f"{prob[1]*100:.2f}%")
                        st.metric("Non-Default Probability", f"{prob[0]*100:.2f}%")
                    
                    # Progress bar
                    st.progress(float(prob[1]))
                    st.caption(f"Risk Level: {'High' if prob[1] > 0.5 else 'Low'}")
            
            # Consensus prediction
            st.subheader("🎯 Consensus Prediction")
            avg_prob = float(np.mean([probabilities_dict[m][1] for m in models.keys()]))
            consensus = 1 if avg_prob > 0.5 else 0
            
            if consensus == 1:
                st.error(f"**Overall Prediction: DEFAULT** (Average Probability: {avg_prob*100:.2f}%)")
            else:
                st.success(f"**Overall Prediction: NO DEFAULT** (Average Probability: {(1-avg_prob)*100:.2f}%)")
            
            st.progress(avg_prob)

# ============================================================================
# FEATURE IMPORTANCE PAGE
# ============================================================================
elif page == "⭐ Feature Importance":
    st.header("⭐ Feature Importance Analysis")
    
    if df is not None:
        with st.spinner("Training models (first run may take a few minutes)..."):
            models = get_trained_models(X_train, y_train)
        # Random Forest Feature Importance
        st.subheader("Random Forest - Feature Importance")
        rf_model = models['Random Forest']
        rf_importance = pd.DataFrame({
            'Feature': X_processed.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(rf_importance.head(10), use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            top_rf = rf_importance.head(10)
            ax.barh(range(len(top_rf)), top_rf['Importance'].values, color='steelblue')
            ax.set_yticks(range(len(top_rf)))
            ax.set_yticklabels(top_rf['Feature'].values)
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Features - Random Forest')
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
        
        # XGBoost Feature Importance
        st.subheader("XGBoost - Feature Importance")
        xgb_model = models['XGBoost']
        xgb_importance = pd.DataFrame({
            'Feature': X_processed.columns,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(xgb_importance.head(10), use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            top_xgb = xgb_importance.head(10)
            ax.barh(range(len(top_xgb)), top_xgb['Importance'].values, color='darkgreen')
            ax.set_yticks(range(len(top_xgb)))
            ax.set_yticklabels(top_xgb['Feature'].values)
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Features - XGBoost')
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
        
        # Comparison
        st.subheader("Top Features Comparison")
        # Merge top features from both models
        comparison_df = pd.DataFrame({
            'Feature': X_processed.columns,
            'RF_Importance': rf_model.feature_importances_,
            'XGB_Importance': xgb_model.feature_importances_
        })
        comparison_df['Average_Importance'] = (comparison_df['RF_Importance'] + comparison_df['XGB_Importance']) / 2
        comparison_df = comparison_df.sort_values('Average_Importance', ascending=False)
        
        st.dataframe(comparison_df.head(15), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Loan Default Prediction System | Built with Streamlit 🤖</p>
</div>
""", unsafe_allow_html=True)

