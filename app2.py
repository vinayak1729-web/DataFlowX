import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import base64
import uuid
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
from sklearn.model_selection import learning_curve
from scipy.stats import ks_2samp
import joblib
import xgboost as xgb
import lightgbm as lgb

# LaTeX for PDF generation
from pylatex import Document, Section, Subsection, Command, Package, Figure, Tabular
from pylatex.utils import italic, NoEscape

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Enhanced Data Science Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    .step-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
        animation: slideInLeft 0.5s ease-in;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: scale(1.05);
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff7f0e;
        transform: translateY(-2px);
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    @keyframes slideInLeft {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(0); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'figures' not in st.session_state:
    st.session_state.figures = []

def detect_problem_type(df, target_col):
    """Detect problem type based on target variable characteristics."""
    if target_col is None or target_col not in df.columns:
        return "Unknown"
    
    if df[target_col].dtype == 'object' or df[target_col].nunique() <= 20:
        return "Classification"
    elif df[target_col].dtype in ['int64', 'float64']:
        return "Regression"
    else:
        return "Clustering"  # Default for unsupervised tasks

def main():
    st.markdown('<h1 class="main-header">🚀 Enhanced Data Science Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    steps = [
        "🏠 Dashboard",
        "🎯 Problem Definition", 
        "📥 Data Collection",
        "🧹 Data Cleaning",
        "🔍 Exploratory Data Analysis",
        "🔧 Feature Engineering",
        "🤖 Model Selection & Training",
        "📊 Model Evaluation",
        "⚡ Model Optimization",
        "🚀 Model Deployment",
        "📈 Monitoring & Results"
    ]
    
    selected_step = st.sidebar.selectbox("Select Step:", steps)
    
    # Main content
    if selected_step == "🏠 Dashboard":
        show_dashboard()
    elif selected_step == "🎯 Problem Definition":
        problem_definition()
    elif selected_step == "📥 Data Collection":
        data_collection()
    elif selected_step == "🧹 Data Cleaning":
        data_cleaning()
    elif selected_step == "🔍 Exploratory Data Analysis":
        exploratory_data_analysis()
    elif selected_step == "🔧 Feature Engineering":
        feature_engineering()
    elif selected_step == "🤖 Model Selection & Training":
        model_selection_training()
    elif selected_step == "📊 Model Evaluation":
        model_evaluation()
    elif selected_step == "⚡ Model Optimization":
        model_optimization()
    elif selected_step == "🚀 Model Deployment":
        model_deployment()
    elif selected_step == "📈 Monitoring & Results":
        monitoring_results()

def show_dashboard():
    st.markdown('<h2 class="step-header">📊 Project Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Status", "✅ Loaded" if st.session_state.data is not None else "❌ Not Loaded", delta=None, delta_color="normal")
    
    with col2:
        if st.session_state.data is not None:
            st.metric("Records", len(st.session_state.data))
        else:
            st.metric("Records", "N/A")
    
    with col3:
        if st.session_state.data is not None:
            st.metric("Features", len(st.session_state.data.columns))
        else:
            st.metric("Features", "N/A")
    
    with col4:
        st.metric("Model Status", "✅ Trained" if st.session_state.model is not None else "❌ Not Trained")
    
    # Project overview
    st.subheader("📋 Project Overview")
    
    if 'project_info' not in st.session_state:
        st.info("Complete the Problem Definition step to see project overview here.")
    else:
        info = st.session_state.project_info
        st.write(f"**Problem Type:** {info.get('problem_type', 'Not defined')}")
        st.write(f"**Objective:** {info.get('objective', 'Not defined')}")
        st.write(f"**Success Metrics:** {info.get('metrics', 'Not defined')}")
    
    # Quick data preview
    if st.session_state.data is not None:
        st.subheader("📊 Data Preview")
        st.dataframe(st.session_state.data.head(), use_container_width=True)
        
        # Quick stats
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Data Types")
            dtype_counts = st.session_state.data.dtypes.value_counts()
            fig = px.pie(values=dtype_counts.values, names=dtype_counts.index.astype(str), title="Distribution of Data Types")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.figures.append(('Data Types Distribution', fig))
        
        with col2:
            st.subheader("🔍 Missing Values")
            missing_data = st.session_state.data.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                fig = px.bar(x=missing_data.index, y=missing_data.values, title="Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.figures.append(('Missing Values', fig))
            else:
                st.success("No missing values found!")

def problem_definition():
    st.markdown('<h2 class="step-header">🎯 Step 1: Problem Definition</h2>', unsafe_allow_html=True)
    
    st.write("Define your data science problem clearly to guide the entire workflow.")
    
    with st.form("problem_definition_form"):
        problem_type = st.selectbox(
            "Problem Type:",
            ["Classification", "Regression", "Clustering", "Time Series", "NLP", "PCA", "Other"]
        )
        
        objective = st.text_area(
            "Project Objective:",
            placeholder="Describe what you want to achieve (e.g., Predict customer churn to improve retention)"
        )
        
        business_questions = st.text_area(
            "Key Business Questions:",
            placeholder="List the main questions you want to answer"
        )
        
        success_metrics = st.text_area(
            "Success Metrics:",
            placeholder="How will you measure success? (e.g., Accuracy > 85%, ROI improvement)"
        )
        
        timeline = st.text_input("Project Timeline:", placeholder="e.g., 4 weeks")
        
        stakeholders = st.text_input("Key Stakeholders:", placeholder="Who will use the results?")
        
        submitted = st.form_submit_button("💾 Save Problem Definition")
        
        if submitted:
            st.session_state.project_info = {
                'problem_type': problem_type,
                'objective': objective,
                'business_questions': business_questions,
                'metrics': success_metrics,
                'timeline': timeline,
                'stakeholders': stakeholders
            }
            st.session_state.problem_type = problem_type
            st.success("✅ Problem definition saved successfully!")
            
    if 'project_info' in st.session_state:
        st.subheader("📋 Current Problem Definition")
        st.json(st.session_state.project_info)

def data_collection():
    st.markdown('<h2 class="step-header">📥 Step 2: Data Collection</h2>', unsafe_allow_html=True)
    
    st.write("Upload your dataset or use sample data to get started.")
    
    data_source = st.radio(
        "Choose data source:",
        ["Upload CSV/Excel file", "Use sample dataset", "Connect to database (simulation)"]
    )
    
    if data_source == "Upload CSV/Excel file":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset in CSV or Excel format"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    # Try different encodings and delimiters
                    encodings = ['utf-8', 'latin1', 'iso-8859-1']
                    delimiters = [',', ';', '\t']
                    df = None
                    for encoding in encodings:
                        for delimiter in delimiters:
                            try:
                                df = pd.read_csv(uploaded_file, encoding=encoding, sep=delimiter)
                                if len(df.columns) > 1:  # Check if multiple columns detected
                                    break
                            except:
                                continue
                        if df is not None and len(df.columns) > 1:
                            break
                    if df is None or len(df.columns) <= 1:
                        raise ValueError("Could not parse CSV file correctly. Please check the file format.")
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.data = df
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Suggest problem type
                target_col = st.selectbox("Select target column (optional):", ["None"] + list(df.columns))
                if target_col != "None":
                    suggested_type = detect_problem_type(df, target_col)
                    st.info(f"Suggested problem type: **{suggested_type}**")
                
                # Show data preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    elif data_source == "Use sample dataset":
        sample_choice = st.selectbox(
            "Choose sample dataset:",
            ["Iris (Classification)", "Boston Housing (Regression)", "Titanic (Classification)", "Mall Customers (Clustering)"]
        )
        
        if st.button("Load Sample Dataset"):
            if sample_choice == "Iris (Classification)":
                from sklearn.datasets import load_iris
                iris = load_iris()
                df = pd.DataFrame(iris.data, columns=iris.feature_names)
                df['target'] = iris.target
                df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
                st.session_state.problem_type = "Classification"
            
            elif sample_choice == "Titanic (Classification)":
                np.random.seed(42)
                n_samples = 891
                df = pd.DataFrame({
                    'age': np.random.normal(29, 14, n_samples),
                    'fare': np.random.exponential(32, n_samples),
                    'sex': np.random.choice(['male', 'female'], n_samples),
                    'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
                    'embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09]),
                    'survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38])
                })
                st.session_state.problem_type = "Classification"
            
            elif sample_choice == "Boston Housing (Regression)":
                from sklearn.datasets import load_boston
                boston = load_boston()
                df = pd.DataFrame(boston.data, columns=boston.feature_names)
                df['target'] = boston.target
                st.session_state.problem_type = "Regression"
            
            else:  # Mall Customers
                np.random.seed(42)
                df = pd.DataFrame({
                    'customer_id': range(1, 201),
                    'age': np.random.randint(18, 70, 200),
                    'annual_income': np.random.randint(15, 150, 200),
                    'spending_score': np.random.randint(1, 100, 200)
                })
                st.session_state.problem_type = "Clustering"
            
            st.session_state.data = df
            st.success(f"✅ Sample dataset loaded! Shape: {df.shape}")
            st.dataframe(df.head(), use_container_width=True)
    
    else:  # Database simulation
        st.info("🔗 Database connection simulation")
        st.code("""
import sqlite3
import pandas as pd

conn = sqlite3.connect('database.db')
query = "SELECT * FROM your_table"
df = pd.read_sql_query(query, conn)
conn.close()
        """)
        
        if st.button("Simulate Database Connection"):
            np.random.seed(42)
            df = pd.DataFrame({
                'customer_id': range(1, 1001),
                'age': np.random.randint(18, 80, 1000),
                'income': np.random.normal(50000, 20000, 1000),
                'spending_score': np.random.randint(1, 100, 1000),
                'churn': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
            })
            st.session_state.data = df
            st.session_state.problem_type = "Classification"
            st.success("✅ Simulated database data loaded!")
            st.dataframe(df.head(), use_container_width=True)

def data_cleaning():
    st.markdown('<h2 class="step-header">🧹 Step 3: Data Cleaning & Preprocessing</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Collection step.")
        return
    
    df = st.session_state.data.copy()
    
    # Data quality overview
    st.subheader("🔍 Data Quality Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Missing Values", df.isnull().sum().sum())
    with col2:
        st.metric("Duplicate Rows", df.duplicated().sum())
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
    
    # Missing values handling
    st.subheader("🔧 Handle Missing Values")
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if missing_cols:
        st.write("Columns with missing values:")
        missing_info = pd.DataFrame({
            'Column': missing_cols,
            'Missing Count': [df[col].isnull().sum() for col in missing_cols],
            'Missing %': [df[col].isnull().sum() / len(df) * 100 for col in missing_cols]
        })
        st.dataframe(missing_info, use_container_width=True)
        
        for col in missing_cols:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{col}:**")
            with col2:
                if df[col].dtype in ['int64', 'float64']:
                    method = st.selectbox(
                        f"Method for {col}:",
                        ["Keep as is", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"],
                        key=f"missing_{col}"
                    )
                else:
                    method = st.selectbox(
                        f"Method for {col}:",
                        ["Keep as is", "Drop rows", "Fill with mode", "Fill with 'Unknown'"],
                        key=f"missing_{col}"
                    )
                
                if method == "Drop rows":
                    df = df.dropna(subset=[col])
                elif method == "Fill with mean" and df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == "Fill with median" and df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                elif method == "Fill with mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif method == "Fill with 'Unknown'":
                    df[col].fillna('Unknown', inplace=True)
    else:
        st.success("✅ No missing values found!")
    
    # Remove duplicates
    st.subheader("🔄 Handle Duplicates")
    if df.duplicated().sum() > 0:
        if st.button("Remove Duplicate Rows"):
            df = df.drop_duplicates()
            st.success(f"✅ Removed {st.session_state.data.duplicated().sum()} duplicate rows")
    else:
        st.success("✅ No duplicate rows found!")
    
    # Outlier detection
    st.subheader("📊 Outlier Detection")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols)
        
        if selected_col:
            Q1 = df[selected_col].quantile(0.25)
            Q3 = df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.figures.append((f'Box Plot of {selected_col}', fig))
            
            with col2:
                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.figures.append((f'Distribution of {selected_col}', fig))
            
            st.write(f"Found {len(outliers)} outliers using IQR method")
            
            if len(outliers) > 0 and st.button(f"Remove outliers from {selected_col}"):
                df = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]
                st.success(f"✅ Removed {len(outliers)} outliers")
    
    # Data type conversion with validation
    st.subheader("🔧 Data Type Conversion")
    for col in df.columns:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{col}:**")
        with col2:
            st.write(f"Current type: {df[col].dtype}")
        with col3:
            if st.button(f"Convert", key=f"convert_{col}"):
                allowed_types = ["int64", "float64", "object", "datetime64", "category"]
                if st.session_state.problem_type == "Classification" and col == st.session_state.get('target_column'):
                    allowed_types = ["category", "object"]  # Restrict target to categorical
                elif st.session_state.problem_type == "Regression" and col == st.session_state.get('target_column'):
                    allowed_types = ["float64", "int64"]  # Restrict target to numeric
                
                new_type = st.selectbox(
                    f"New type for {col}:",
                    allowed_types,
                    key=f"type_{col}"
                )
                try:
                    if new_type == "datetime64":
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    else:
                        df[col] = df[col].astype(new_type)
                    st.success(f"✅ Converted {col} to {new_type}")
                except Exception as e:
                    st.error(f"❌ Could not convert {col}: {str(e)}")
    
    # Save cleaned data
    if st.button("💾 Save Cleaned Data"):
        st.session_state.processed_data = df
        st.success("✅ Cleaned data saved successfully!")
        
        # Show comparison
        st.subheader("📊 Before vs After Cleaning")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Data:**")
            st.write(f"Shape: {st.session_state.data.shape}")
            st.write(f"Missing values: {st.session_state.data.isnull().sum().sum()}")
            st.write(f"Duplicates: {st.session_state.data.duplicated().sum()}")
        
        with col2:
            st.write("**Cleaned Data:**")
            st.write(f"Shape: {df.shape}")
            st.write(f"Missing values: {df.isnull().sum().sum()}")
            st.write(f"Duplicates: {df.duplicated().sum()}")

def exploratory_data_analysis():
    st.markdown('<h2 class="step-header">🔍 Step 4: Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Collection step.")
        return
    
    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
    
    # Dataset overview
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Basic statistics
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Data types and info
    st.subheader("🔍 Data Types & Info")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types:**")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.write("**Data Type Distribution:**")
        dtype_counts = df.dtypes.value_counts()
        dtype_counts.index = dtype_counts.index.astype(str)
        fig = px.pie(values=dtype_counts.values, names=dtype_counts.index, title="Data Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.session_state.figures.append(('Data Type Distribution', fig))
    
    # Univariate Analysis
    st.subheader("📊 Univariate Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    analysis_type = st.radio("Select analysis type:", ["Numeric Variables", "Categorical Variables"])
    
    if analysis_type == "Numeric Variables" and numeric_cols:
        selected_numeric = st.selectbox("Select numeric column:", numeric_cols)
        
        if selected_numeric:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x=selected_numeric, title=f"Distribution of {selected_numeric}")
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.figures.append((f'Distribution of {selected_numeric}', fig))
            
            with col2:
                fig = px.box(df, y=selected_numeric, title=f"Box Plot of {selected_numeric}")
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.figures.append((f'Box Plot of {selected_numeric}', fig))
            
            # Statistics
            st.write(f"**Statistics for {selected_numeric}:**")
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Mode', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                'Value': [
                    df[selected_numeric].mean(),
                    df[selected_numeric].median(),
                    df[selected_numeric].mode().iloc[0] if not df[selected_numeric].mode().empty else 'N/A',
                    df[selected_numeric].std(),
                    df[selected_numeric].min(),
                    df[selected_numeric].max(),
                    df[selected_numeric].skew(),
                    df[selected_numeric].kurtosis()
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
    
    elif analysis_type == "Categorical Variables" and categorical_cols:
        selected_categorical = st.selectbox("Select categorical column:", categorical_cols)
        
        if selected_categorical:
            value_counts = df[selected_categorical].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f"Distribution of {selected_categorical}")
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.figures.append((f'Distribution of {selected_categorical}', fig))
            
            with col2:
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Pie Chart of {selected_categorical}")
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.figures.append((f'Pie Chart of {selected_categorical}', fig))
            
            st.write(f"**Value Counts for {selected_categorical}:**")
            st.dataframe(value_counts.reset_index(), use_container_width=True)
    
    # Bivariate Analysis
    st.subheader("📈 Bivariate Analysis")
    
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Select X variable:", numeric_cols, key="x_var")
        with col2:
            y_var = st.selectbox("Select Y variable:", numeric_cols, key="y_var")
        
        if x_var and y_var and x_var != y_var:
            fig = px.scatter(df, x=x_var, y=y_var, title=f"{x_var} vs {y_var}")
            
            if categorical_cols:
                color_var = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
                if color_var != "None":
                    fig = px.scatter(df, x=x_var, y=y_var, color=color_var, 
                                   title=f"{x_var} vs {y_var} (colored by {color_var})")
            
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.figures.append((f'{x_var} vs {y_var}', fig))
            
            correlation = df[x_var].corr(df[y_var])
            st.metric("Correlation Coefficient", f"{correlation:.3f}")
    
    # Correlation Matrix
    st.subheader("🔗 Correlation Matrix")
    
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       title="Correlation Matrix",
                       color_continuous_scale="RdBu_r",
                       aspect="auto")
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.session_state.figures.append(('Correlation Matrix', fig))
        
        st.subheader("🔝 Strongest Correlations")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
        st.dataframe(corr_df.head(10), use_container_width=True)

def feature_engineering():
    st.markdown('<h2 class="step-header">🔧 Step 5: Feature Engineering</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Collection step.")
        return
    
    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data.copy()
    
    st.subheader("🛠️ Feature Engineering Tools")
    
    # Feature scaling
    st.subheader("📏 Feature Scaling")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        scaling_method = st.selectbox(
            "Select scaling method:",
            ["None", "StandardScaler (Z-score)", "MinMaxScaler (0-1)", "RobustScaler"]
        )
        
        columns_to_scale = st.multiselect("Select columns to scale:", numeric_cols)
        
        if scaling_method != "None" and columns_to_scale and st.button("Apply Scaling"):
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            
            if scaling_method == "StandardScaler (Z-score)":
                scaler = StandardScaler()
            elif scaling_method == "MinMaxScaler (0-1)":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()
            
            try:
                df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
                st.success(f"✅ Applied {scaling_method} to selected columns")
                st.dataframe(df[columns_to_scale].describe(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error applying scaling: {str(e)}")
    
    # Encoding categorical variables
    st.subheader("🏷️ Categorical Variable Encoding")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        selected_cat_col = st.selectbox("Select categorical column to encode:", categorical_cols)
        n_unique = df[selected_cat_col].nunique() if selected_cat_col else 0
        encoding_options = ["Label Encoding", "One-Hot Encoding"] if n_unique <= 10 else ["Label Encoding"]
        encoding_method = st.selectbox(
            "Select encoding method:",
            encoding_options,
            help="One-Hot Encoding is disabled for high-cardinality features (>10 unique values)."
        )
        
        if selected_cat_col and st.button("Apply Encoding"):
            try:
                if encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    df[f"{selected_cat_col}_encoded"] = le.fit_transform(df[selected_cat_col].astype(str))
                    st.success(f"✅ Applied Label Encoding to {selected_cat_col}")
                
                elif encoding_method == "One-Hot Encoding":
                    dummies = pd.get_dummies(df[selected_cat_col], prefix=selected_cat_col)
                    df = pd.concat([df, dummies], axis=1)
                    st.success(f"✅ Applied One-Hot Encoding to {selected_cat_col}")
                
                st.write("**Encoding Results:**")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error applying encoding: {str(e)}")
    
    # Feature creation
    st.subheader("➕ Create New Features")
    
    feature_type = st.selectbox(
        "Select feature creation type:",
        ["Mathematical Operations", "Date/Time Features", "Text Features", "Binning"]
    )
    
    if feature_type == "Mathematical Operations":
        st.write("Create new features using mathematical operations:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            var1 = st.selectbox("Select first variable:", numeric_cols, key="math_var1")
        with col2:
            operation = st.selectbox("Operation:", ["+", "-", "*", "/", "**"])
        with col3:
            var2 = st.selectbox("Select second variable:", numeric_cols, key="math_var2")
        
        new_feature_name = st.text_input("New feature name:", f"{var1}_{operation}_{var2}")
        
        if var1 and var2 and new_feature_name and st.button("Create Mathematical Feature"):
            try:
                if operation == "+":
                    df[new_feature_name] = df[var1] + df[var2]
                elif operation == "-":
                    df[new_feature_name] = df[var1] - df[var2]
                elif operation == "*":
                    df[new_feature_name] = df[var1] * df[var2]
                elif operation == "/":
                    df[new_feature_name] = df[var1] / df[var2]
                elif operation == "**":
                    df[new_feature_name] = df[var1] ** df[var2]
                
                st.success(f"✅ Created new feature: {new_feature_name}")
                st.dataframe(df[[var1, var2, new_feature_name]].head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error creating feature: {str(e)}")
    
    elif feature_type == "Date/Time Features":
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if not date_cols:
            st.info("No datetime columns found. Try converting a column to datetime first.")
        else:
            selected_date_col = st.selectbox("Select date column:", date_cols)
            
            date_features = st.multiselect(
                "Select date features to extract:",
                ["Year", "Month", "Day", "Weekday", "Quarter", "Is_Weekend"]
            )
            
            if selected_date_col and date_features and st.button("Create Date Features"):
                for feature in date_features:
                    if feature == "Year":
                        df[f"{selected_date_col}_year"] = df[selected_date_col].dt.year
                    elif feature == "Month":
                        df[f"{selected_date_col}_month"] = df[selected_date_col].dt.month
                    elif feature == "Day":
                        df[f"{selected_date_col}_day"] = df[selected_date_col].dt.day
                    elif feature == "Weekday":
                        df[f"{selected_date_col}_weekday"] = df[selected_date_col].dt.dayofweek
                    elif feature == "Quarter":
                        df[f"{selected_date_col}_quarter"] = df[selected_date_col].dt.quarter
                    elif feature == "Is_Weekend":
                        df[f"{selected_date_col}_is_weekend"] = df[selected_date_col].dt.dayofweek >= 5
                
                st.success(f"✅ Created {len(date_features)} date features")
    
    elif feature_type == "Binning":
        if numeric_cols:
            bin_col = st.selectbox("Select column to bin:", numeric_cols)
            n_bins = st.slider("Number of bins:", 2, 20, 5)
            bin_method = st.selectbox("Binning method:", ["Equal width", "Equal frequency"])
            
            if bin_col and st.button("Create Binned Feature"):
                try:
                    if bin_method == "Equal width":
                        df[f"{bin_col}_binned"] = pd.cut(df[bin_col], bins=n_bins, labels=False)
                    else:
                        df[f"{bin_col}_binned"] = pd.qcut(df[bin_col], q=n_bins, labels=False, duplicates='drop')
                    
                    st.success(f"✅ Created binned feature: {bin_col}_binned")
                    
                    bin_counts = df[f"{bin_col}_binned"].value_counts().sort_index()
                    fig = px.bar(x=bin_counts.index, y=bin_counts.values, 
                               title=f"Distribution of {bin_col}_binned")
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.figures.append((f'Distribution of {bin_col}_binned', fig))
                except Exception as e:
                    st.error(f"❌ Error creating binned feature: {str(e)}")
    
    # Feature selection
    st.subheader("🎯 Feature Selection")
    
    if 'target_column' not in st.session_state:
        target_col = st.selectbox("Select target column for feature selection:", df.columns.tolist())
        if st.button("Set Target Column"):
            st.session_state.target_column = target_col
            st.session_state.problem_type = detect_problem_type(df, target_col)
            st.success(f"✅ Target column set to: {target_col} (Problem Type: {st.session_state.problem_type})")
    else:
        st.info(f"Current target column: {st.session_state.target_column} (Problem Type: {st.session_state.problem_type})")
        if st.button("Change Target Column"):
            del st.session_state.target_column
            st.rerun()
    
    if 'target_column' in st.session_state:
        target_col = st.session_state.target_column
        feature_cols = [col for col in df.columns if col != target_col]
        numeric_features = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        if numeric_features and len(numeric_features) > 1:
            selection_method = st.selectbox(
                "Feature selection method:",
                ["Correlation with target", "Mutual information", "Manual selection"]
            )
            
            if selection_method == "Correlation with target":
                correlations = df[numeric_features + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
                correlations = correlations.drop(target_col)
                
                st.write("**Feature correlations with target:**")
                corr_df = pd.DataFrame({
                    'Feature': correlations.index,
                    'Correlation': correlations.values
                })
                st.dataframe(corr_df, use_container_width=True)
                
                n_features = st.slider("Select top N features:", 1, len(correlations), min(10, len(correlations)))
                selected_features = correlations.head(n_features).index.tolist()
                
                st.write(f"**Selected features:** {selected_features}")
                
            elif selection_method == "Manual selection":
                selected_features = st.multiselect("Select features manually:", feature_cols)
    
    # Save engineered features
    if st.button("💾 Save Engineered Features"):
        st.session_state.processed_data = df
        st.success("✅ Engineered features saved successfully!")
        
        st.subheader("📊 Feature Engineering Summary")
        st.write(f"**Total features:** {len(df.columns)}")
        st.write(f"**Numeric features:** {len(df.select_dtypes(include=[np.number]).columns)}")
        st.write(f"**Categorical features:** {len(df.select_dtypes(include=['object', 'category']).columns)}")
        
        st.dataframe(df.head(), use_container_width=True)

def model_selection_training():
    st.markdown('<h2 class="step-header">🤖 Step 6: Model Selection & Training</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Collection step.")
        return
    
    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data.copy()
    
    # Model configuration
    st.subheader("⚙️ Model Configuration")
    
    target_col = st.selectbox("Select target variable:", df.columns.tolist())
    
    feature_cols = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect("Select features:", feature_cols, default=feature_cols[:min(5, len(feature_cols))])
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        return
    
    # Validate data types
    if st.session_state.problem_type == "Classification":
        if df[target_col].dtype not in ['object', 'category'] and df[target_col].nunique() > 20:
            st.warning("Target variable for classification should be categorical or have low cardinality.")
            return
    elif st.session_state.problem_type == "Regression":
        if df[target_col].dtype not in ['int64', 'float64']:
            st.warning("Target variable for regression should be numeric.")
            return
    
    # Prepare data
    X = df[selected_features]
    y = df[target_col]
    
    # Handle categorical variables
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        st.warning(f"Categorical features detected: {categorical_features}")
        encoding_method = st.selectbox("Encoding method for categorical features:", ["Label Encoding", "One-Hot Encoding"])
        
        try:
            if encoding_method == "Label Encoding":
                for col in categorical_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            else:
                X = pd.get_dummies(X, columns=categorical_features)
                selected_features = X.columns.tolist()
        except Exception as e:
            st.error(f"❌ Error encoding categorical features: {str(e)}")
            return
    
    # Handle categorical target
    if st.session_state.problem_type == "Classification" and y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        st.session_state.target_encoder = le_target
    
    # Train-test split
    st.subheader("📊 Data Splitting")
    test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state:", 0, 1000, 42)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    except Exception as e:
        st.error(f"❌ Error splitting data: {str(e)}")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Set Size", len(X_train))
    with col2:
        st.metric("Test Set Size", len(X_test))
    with col3:
        st.metric("Features", len(selected_features))
    
    # Model selection
    st.subheader("🎯 Model Selection")
    
    if st.session_state.problem_type == "Classification":
        available_models = {
            "Random Forest": RandomForestClassifier(random_state=random_state),
            "Logistic Regression": LogisticRegression(random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "SVM": SVC(random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "XGBoost": xgb.XGBClassifier(random_state=random_state, use_label_encoder=False),
            "LightGBM": lgb.LGBMClassifier(random_state=random_state),
            "KNN": KNeighborsClassifier()
        }
    else:
        available_models = {
            "Random Forest": RandomForestRegressor(random_state=random_state),
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "SVR": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
            "XGBoost": xgb.XGBRegressor(random_state=random_state),
            "LightGBM": lgb.LGBMRegressor(random_state=random_state),
            "KNN": KNeighborsRegressor()
        }
    
    selected_model_name = st.selectbox("Select model:", list(available_models.keys()))
    selected_model = available_models[selected_model_name]
    
    # Hyperparameter tuning
    st.subheader("🔧 Hyperparameter Tuning")
    tune_hyperparams = st.checkbox("Enable hyperparameter tuning (may take longer)")
    
    if tune_hyperparams:
        if selected_model_name == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif selected_model_name == "Logistic Regression":
            param_grid = {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        elif selected_model_name == "Gradient Boosting":
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        elif selected_model_name == "XGBoost":
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        elif selected_model_name == "LightGBM":
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50]
            }
        elif selected_model_name == "KNN":
            param_grid = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        else:
            param_grid = {}
            st.info("Basic hyperparameter tuning not configured for this model.")
    
    # Train model
    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            try:
                if tune_hyperparams and param_grid:
                    grid_search = GridSearchCV(selected_model, param_grid, cv=5, scoring='accuracy' if st.session_state.problem_type == "Classification" else 'r2')
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    
                    st.success("✅ Model trained with hyperparameter tuning!")
                    st.write(f"**Best parameters:** {grid_search.best_params_}")
                else:
                    model = selected_model
                    model.fit(X_train, y_train)
                    st.success("✅ Model trained successfully!")
                
                # Store model and data
                st.session_state.model = model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_names = selected_features
                st.session_state.target_name = target_col
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                if st.session_state.problem_type == "Classification":
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    
                    metrics = {
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'train_predictions': y_train_pred,
                        'test_predictions': y_test_pred
                    }
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Accuracy", f"{train_accuracy:.3f}")
                    with col2:
                        st.metric("Test Accuracy", f"{test_accuracy:.3f}")
                
                else:
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    
                    metrics = {
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_predictions': y_train_pred,
                        'test_predictions': y_test_pred
                    }
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Training R²", f"{train_r2:.3f}")
                    with col2:
                        st.metric("Test R²", f"{test_r2:.3f}")
                    with col3:
                        st.metric("Training RMSE", f"{train_rmse:.3f}")
                    with col4:
                        st.metric("Test RMSE", f"{test_rmse:.3f}")
                
                st.session_state.model_metrics = metrics
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    st.subheader("📊 Feature Importance")
                    
                    importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                               title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.figures.append(('Feature Importance', fig))
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")

def model_evaluation():
    st.markdown('<h2 class="step-header">📊 Step 7: Model Evaluation</h2>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the Model Selection & Training step.")
        return
    
    model = st.session_state.model
    metrics = st.session_state.model_metrics
    
    st.subheader("📈 Model Performance Metrics")
    
    if st.session_state.problem_type == "Classification":
        y_pred = metrics['test_predictions']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['test_accuracy']:.3f}")
        with col2:
            precision = classification_report(st.session_state.y_test, y_pred, output_dict=True)['weighted avg']['precision']
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            recall = classification_report(st.session_state.y_test, y_pred, output_dict=True)['weighted avg']['recall']
            st.metric("Recall", f"{recall:.3f}")
        
        # Confusion Matrix
        st.subheader("🔄 Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        st.session_state.figures.append(('Confusion Matrix', fig))
        
        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # ROC Curve
        if len(np.unique(st.session_state.y_test)) == 2 and hasattr(model, 'predict_proba'):
            from sklearn.metrics import roc_curve, auc
            y_proba = model.predict_proba(st.session_state.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(st.session_state.y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            st.subheader("📈 ROC Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random'))
            fig.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.figures.append(('ROC Curve', fig))
    
    else:
        y_pred = metrics['test_predictions']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{metrics['test_r2']:.3f}")
        with col2:
            st.metric("RMSE", f"{metrics['test_rmse']:.3f}")
        with col3:
            mae = mean_absolute_error(st.session_state.y_test, y_pred)
            st.metric("MAE", f"{mae:.3f}")
        with col4:
            mape = np.mean(np.abs((st.session_state.y_test - y_pred) / st.session_state.y_test)) * 100
            st.metric("MAPE", f"{mape:.1f}%")
        
        # Actual vs Predicted
        st.subheader("📊 Actual vs Predicted")
        fig = px.scatter(x=st.session_state.y_test, y=y_pred, title="Actual vs Predicted Values")
        fig.add_trace(go.Scatter(x=[st.session_state.y_test.min(), st.session_state.y_test.max()], 
                               y=[st.session_state.y_test.min(), st.session_state.y_test.max()], 
                               mode='lines', name='Perfect Prediction'))
        fig.update_layout(xaxis_title='Actual Values', yaxis_title='Predicted Values')
        st.plotly_chart(fig, use_container_width=True)
        st.session_state.figures.append(('Actual vs Predicted', fig))
        
        # Residual Plot
        st.subheader("📈 Residual Plot")
        residuals = st.session_state.y_test - y_pred
        fig = px.scatter(x=y_pred, y=residuals, title="Residual Plot")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(xaxis_title='Predicted Values', yaxis_title='Residuals')
        st.plotly_chart(fig, use_container_width=True)
        st.session_state.figures.append(('Residual Plot', fig))
    
    # Cross-validation
    st.subheader("🔄 Cross-Validation")
    if st.button("Perform Cross-Validation"):
        with st.spinner("Performing cross-validation..."):
            try:
                X_full = pd.concat([st.session_state.X_train, st.session_state.X_test])
                y_full = pd.concat([pd.Series(st.session_state.y_train), pd.Series(st.session_state.y_test)])
                
                scoring = 'accuracy' if st.session_state.problem_type == "Classification" else 'r2'
                cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring=scoring)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CV Mean Score", f"{cv_scores.mean():.3f}")
                with col2:
                    st.metric("CV Std Score", f"{cv_scores.std():.3f}")
                with col3:
                    st.metric("CV Scores Range", f"{cv_scores.min():.3f} - {cv_scores.max():.3f}")
                
                fig = px.bar(x=[f"Fold {i+1}" for i in range(len(cv_scores))], 
                            y=cv_scores, title="Cross-Validation Scores")
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.figures.append(('Cross-Validation Scores', fig))
            except Exception as e:
                st.error(f"❌ Error in cross-validation: {str(e)}")
    
    # Learning Curves
    st.subheader("📈 Learning Curves")
    if st.button("Generate Learning Curves"):
        with st.spinner("Generating learning curves..."):
            try:
                X_full = pd.concat([st.session_state.X_train, st.session_state.X_test])
                y_full = pd.concat([pd.Series(st.session_state.y_train), pd.Series(st.session_state.y_test)])
                
                train_sizes = np.linspace(0.1, 1.0, 10)
                scoring = 'accuracy' if st.session_state.problem_type == "Classification" else 'r2'
                
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X_full, y_full, train_sizes=train_sizes, cv=5, scoring=scoring
                )
                
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                val_scores_mean = np.mean(val_scores, axis=1)
                val_scores_std = np.std(val_scores, axis=1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=train_sizes, y=train_scores_mean,
                    mode='lines+markers', name='Training Score',
                    error_y=dict(type='data', array=train_scores_std)
                ))
                fig.add_trace(go.Scatter(
                    x=train_sizes, y=val_scores_mean,
                    mode='lines+markers', name='Validation Score',
                    error_y=dict(type='data', array=val_scores_std)
                ))
                fig.update_layout(
                    title='Learning Curves',
                    xaxis_title='Training Set Size',
                    yaxis_title=f'{scoring.title()} Score'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.figures.append(('Learning Curves', fig))
            except Exception as e:
                st.error(f"❌ Error generating learning curves: {str(e)}")

def model_optimization():
    st.markdown('<h2 class="step-header">⚡ Step 8: Model Optimization</h2>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the Model Selection & Training step.")
        return
    
    st.subheader("🔧 Optimization Techniques")
    
    optimization_method = st.selectbox(
        "Select optimization method:",
        ["Hyperparameter Tuning", "Feature Selection", "Ensemble Methods", "Regularization"]
    )
    
    if optimization_method == "Hyperparameter Tuning":
        st.write("### 🎛️ Advanced Hyperparameter Tuning")
        
        model_type = type(st.session_state.model).__name__
        st.info(f"Current model: {model_type}")
        
        if "RandomForest" in model_type:
            param_options = {
                'n_estimators': st.multiselect("Number of estimators:", [50, 100, 200, 300], default=[100, 200]),
                'max_depth': st.multiselect("Max depth:", [None, 10, 20, 30], default=[None, 20]),
                'min_samples_split': st.multiselect("Min samples split:", [2, 5, 10], default=[2, 5]),
                'min_samples_leaf': st.multiselect("Min samples leaf:", [1, 2, 4], default=[1, 2])
            }
        elif "LogisticRegression" in model_type:
            param_options = {
                'C': st.multiselect("Regularization strength:", [0.01, 0.1, 1, 10, 100], default=[0.1, 1, 10]),
                'solver': st.multiselect("Solver:", ['liblinear', 'lbfgs', 'saga'], default=['liblinear', 'lbfgs']),
                'max_iter': st.multiselect("Max iterations:", [100, 200, 500], default=[200])
            }
        elif "GradientBoosting" in model_type:
            param_options = {
                'n_estimators': st.multiselect("Number of estimators:", [50, 100, 200], default=[100]),
                'learning_rate': st.multiselect("Learning rate:", [0.01, 0.1, 0.2], default=[0.1]),
                'max_depth': st.multiselect("Max depth:", [3, 5, 7], default=[3])
            }
        elif "XGB" in model_type:
            param_options = {
                'n_estimators': st.multiselect("Number of estimators:", [50, 100, 200], default=[100]),
                'learning_rate': st.multiselect("Learning rate:", [0.01, 0.1, 0.2], default=[0.1]),
                'max_depth': st.multiselect("Max depth:", [3, 5, 7], default=[3])
            }
        elif "LGBM" in model_type:
            param_options = {
                'n_estimators': st.multiselect("Number of estimators:", [50, 100, 200], default=[100]),
                'learning_rate': st.multiselect("Learning rate:", [0.01, 0.1, 0.2], default=[0.1]),
                'num_leaves': st.multiselect("Number of leaves:", [31, 50, 70], default=[31])
            }
        elif "KNN" in model_type:
            param_options = {
                'n_neighbors': st.multiselect("Number of neighbors:", [3, 5, 7, 9], default=[5]),
                'weights': st.multiselect("Weights:", ['uniform', 'distance'], default=['uniform'])
            }
        else:
            param_options = {}
            st.info("Custom hyperparameter tuning not available for this model type.")
        
        if param_options and st.button("🚀 Optimize Hyperparameters"):
            with st.spinner("Optimizing hyperparameters..."):
                try:
                    param_grid = {param: values for param, values in param_options.items() if values}
                    if param_grid:
                        grid_search = GridSearchCV(
                            st.session_state.model, 
                            param_grid, 
                            cv=5, 
                            scoring='accuracy' if st.session_state.problem_type == "Classification" else 'r2',
                            n_jobs=-1
                        )
                        
                        grid_search.fit(st.session_state.X_train, st.session_state.y_train)
                        st.session_state.model = grid_search.best_estimator_
                        
                        st.success("✅ Hyperparameter optimization completed!")
                        st.write(f"**Best score:** {grid_search.best_score_:.4f}")
                        st.write(f"**Best parameters:** {grid_search.best_params_}")
                        
                        results_df = pd.DataFrame(grid_search.cv_results_)
                        st.subheader("📊 Optimization Results")
                        st.dataframe(results_df[['params', 'mean_test_score', 'rank_test_score']].head(10))
                        
                        if 'param_n_estimators' in results_df.columns:
                            fig = px.scatter(
                                results_df,
                                x='param_n_estimators',
                                y='mean_test_score',
                                color='param_max_depth',
                                size='mean_test_score',
                                title="Hyperparameter Tuning Results",
                                labels={'mean_test_score': 'Mean Test Score', 'param_n_estimators': 'Number of Estimators'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.session_state.figures.append(('Hyperparameter Tuning Results', fig))
                except Exception as e:
                    st.error(f"❌ Error during hyperparameter optimization: {str(e)}")
    
    elif optimization_method == "Feature Selection":
        st.write("### 🎯 Feature Selection Optimization")
        
        feature_cols = st.session_state.feature_names
        if not feature_cols:
            st.warning("No features selected. Please train a model first.")
            return
        
        selection_approach = st.radio(
            "Feature Selection Approach:",
            ["Recursive Feature Elimination (RFE)", "Select K Best", "Manual Selection"]
        )
        
        if selection_approach == "Recursive Feature Elimination (RFE)":
            n_features = st.slider("Number of features to select:", 1, len(feature_cols), len(feature_cols)//2)
            if st.button("Run RFE"):
                with st.spinner("Performing Recursive Feature Elimination..."):
                    try:
                        rfe = RFE(estimator=st.session_state.model, n_features_to_select=n_features)
                        rfe.fit(st.session_state.X_train, st.session_state.y_train)
                        
                        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_]
                        st.session_state.feature_names = selected_features
                        
                        st.success(f"✅ Selected {len(selected_features)} features: {selected_features}")
                        
                        X_train_new = st.session_state.X_train[selected_features]
                        X_test_new = st.session_state.X_test[selected_features]
                        st.session_state.model.fit(X_train_new, st.session_state.y_train)
                        
                        y_train_pred = st.session_state.model.predict(X_train_new)
                        y_test_pred = st.session_state.model.predict(X_test_new)
                        
                        if st.session_state.problem_type == "Classification":
                            metrics = {
                                'train_accuracy': accuracy_score(st.session_state.y_train, y_train_pred),
                                'test_accuracy': accuracy_score(st.session_state.y_test, y_test_pred),
                                'train_predictions': y_train_pred,
                                'test_predictions': y_test_pred
                            }
                            st.metric("Updated Test Accuracy", f"{metrics['test_accuracy']:.3f}")
                        else:
                            metrics = {
                                'train_r2': r2_score(st.session_state.y_train, y_train_pred),
                                'test_r2': r2_score(st.session_state.y_test, y_test_pred),
                                'train_rmse': np.sqrt(mean_squared_error(st.session_state.y_train, y_train_pred)),
                                'test_rmse': np.sqrt(mean_squared_error(st.session_state.y_test, y_test_pred)),
                                'train_predictions': y_train_pred,
                                'test_predictions': y_test_pred
                            }
                            st.metric("Updated Test R²", f"{metrics['test_r2']:.3f}")
                        
                        st.session_state.model_metrics = metrics
                        
                        ranking_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Ranking': rfe.ranking_
                        }).sort_values('Ranking')
                        fig = px.bar(ranking_df, x='Feature', y='Ranking', title="Feature Rankings (RFE)")
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.figures.append(('Feature Rankings (RFE)', fig))
                    except Exception as e:
                        st.error(f"❌ Error during RFE: {str(e)}")
        
        elif selection_approach == "Select K Best":
            k = st.slider("Number of features to select:", 1, len(feature_cols), len(feature_cols)//2)
            if st.button("Run Select K Best"):
                with st.spinner("Performing Select K Best..."):
                    try:
                        score_func = f_classif if st.session_state.problem_type == "Classification" else f_regression
                        selector = SelectKBest(score_func=score_func, k=k)
                        selector.fit(st.session_state.X_train, st.session_state.y_train)
                        
                        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()]
                        st.session_state.feature_names = selected_features
                        
                        st.success(f"✅ Selected {len(selected_features)} features: {selected_features}")
                        
                        X_train_new = st.session_state.X_train[selected_features]
                        X_test_new = st.session_state.X_test[selected_features]
                        st.session_state.model.fit(X_train_new, st.session_state.y_train)
                        
                        y_train_pred = st.session_state.model.predict(X_train_new)
                        y_test_pred = st.session_state.model.predict(X_test_new)
                        
                        if st.session_state.problem_type == "Classification":
                            metrics = {
                                'train_accuracy': accuracy_score(st.session_state.y_train, y_train_pred),
                                'test_accuracy': accuracy_score(st.session_state.y_test, y_test_pred),
                                'train_predictions': y_train_pred,
                                'test_predictions': y_test_pred
                            }
                            st.metric("Updated Test Accuracy", f"{metrics['test_accuracy']:.3f}")
                        else:
                            metrics = {
                                'train_r2': r2_score(st.session_state.y_train, y_train_pred),
                                'test_r2': r2_score(st.session_state.y_test, y_test_pred),
                                'train_rmse': np.sqrt(mean_squared_error(st.session_state.y_train, y_train_pred)),
                                'test_rmse': np.sqrt(mean_squared_error(st.session_state.y_test, y_test_pred)),
                                'train_predictions': y_train_pred,
                                'test_predictions': y_test_pred
                            }
                            st.metric("Updated Test R²", f"{metrics['test_r2']:.3f}")
                        
                        st.session_state.model_metrics = metrics
                        
                        scores_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Score': selector.scores_
                        }).sort_values('Score', ascending=False)
                        fig = px.bar(scores_df, x='Feature', y='Score', title="Feature Scores (Select K Best)")
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.figures.append(('Feature Scores (Select K Best)', fig))
                    except Exception as e:
                        st.error(f"❌ Error during Select K Best: {str(e)}")
        
        elif selection_approach == "Manual Selection":
            selected_features = st.multiselect("Select features:", feature_cols, default=feature_cols)
            if st.button("Apply Manual Selection"):
                st.session_state.feature_names = selected_features
                st.success(f"✅ Selected {len(selected_features)} features: {selected_features}")
                
                X_train_new = st.session_state.X_train[selected_features]
                X_test_new = st.session_state.X_test[selected_features]
                st.session_state.model.fit(X_train_new, st.session_state.y_train)
                
                y_train_pred = st.session_state.model.predict(X_train_new)
                y_test_pred = st.session_state.model.predict(X_test_new)
                
                if st.session_state.problem_type == "Classification":
                    metrics = {
                        'train_accuracy': accuracy_score(st.session_state.y_train, y_train_pred),
                        'test_accuracy': accuracy_score(st.session_state.y_test, y_test_pred),
                        'train_predictions': y_train_pred,
                        'test_predictions': y_test_pred
                    }
                    st.metric("Updated Test Accuracy", f"{metrics['test_accuracy']:.3f}")
                else:
                    metrics = {
                        'train_r2': r2_score(st.session_state.y_train, y_train_pred),
                        'test_r2': r2_score(st.session_state.y_test, y_test_pred),
                        'train_rmse': np.sqrt(mean_squared_error(st.session_state.y_train, y_train_pred)),
                        'test_rmse': np.sqrt(mean_squared_error(st.session_state.y_test, y_test_pred)),
                        'train_predictions': y_train_pred,
                        'test_predictions': y_test_pred
                    }
                    st.metric("Updated Test R²", f"{metrics['test_r2']:.3f}")
                
                st.session_state.model_metrics = metrics
    
    elif optimization_method == "Ensemble Methods":
        st.write("### 🤝 Ensemble Methods")
        
        ensemble_type = st.selectbox(
            "Select ensemble method:",
            ["Voting (Classification only)", "Stacking", "Bagging"]
        )
        
        if ensemble_type == "Voting" and st.session_state.problem_type != "Classification":
            st.warning("Voting ensemble is only available for classification problems.")
            return
        
        if st.button("Apply Ensemble Method"):
            with st.spinner("Building ensemble model..."):
                try:
                    if ensemble_type == "Voting":
                        from sklearn.ensemble import VotingClassifier
                        models = [
                            ('rf', RandomForestClassifier(random_state=42)),
                            ('lr', LogisticRegression(random_state=42)),
                            ('svm', SVC(probability=True, random_state=42))
                        ]
                        ensemble = VotingClassifier(estimators=models, voting='soft')
                    
                    elif ensemble_type == "Stacking":
                        from sklearn.ensemble import StackingClassifier, StackingRegressor
                        if st.session_state.problem_type == "Classification":
                            models = [
                                ('rf', RandomForestClassifier(random_state=42)),
                                ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False)),
                                ('lgbm', lgb.LGBMClassifier(random_state=42))
                            ]
                            ensemble = StackingClassifier(
                                estimators=models,
                                final_estimator=LogisticRegression(),
                                cv=5
                            )
                        else:
                            models = [
                                ('rf', RandomForestRegressor(random_state=42)),
                                ('xgb', xgb.XGBRegressor(random_state=42)),
                                                                ('lgbm', lgb.LGBMRegressor(random_state=42))
                            ]
                            ensemble = StackingRegressor(
                                estimators=models,
                                final_estimator=LinearRegression(),
                                cv=5
                            )
                    
                    elif ensemble_type == "Bagging":
                        from sklearn.ensemble import BaggingClassifier, BaggingRegressor
                        if st.session_state.problem_type == "Classification":
                            ensemble = BaggingClassifier(
                                base_estimator=DecisionTreeClassifier(),
                                n_estimators=100,
                                random_state=42
                            )
                        else:
                            ensemble = BaggingRegressor(
                                base_estimator=DecisionTreeRegressor(),
                                n_estimators=100,
                                random_state=42
                            )
                    
                    ensemble.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.model = ensemble
                    
                    y_train_pred = ensemble.predict(st.session_state.X_train)
                    y_test_pred = ensemble.predict(st.session_state.X_test)
                    
                    if st.session_state.problem_type == "Classification":
                        metrics = {
                            'train_accuracy': accuracy_score(st.session_state.y_train, y_train_pred),
                            'test_accuracy': accuracy_score(st.session_state.y_test, y_test_pred),
                            'train_predictions': y_train_pred,
                            'test_predictions': y_test_pred
                        }
                        st.metric("Updated Test Accuracy", f"{metrics['test_accuracy']:.3f}")
                    else:
                        metrics = {
                            'train_r2': r2_score(st.session_state.y_train, y_train_pred),
                            'test_r2': r2_score(st.session_state.y_test, y_test_pred),
                            'train_rmse': np.sqrt(mean_squared_error(st.session_state.y_train, y_train_pred)),
                            'test_rmse': np.sqrt(mean_squared_error(st.session_state.y_test, y_test_pred)),
                            'train_predictions': y_train_pred,
                            'test_predictions': y_test_pred
                        }
                        st.metric("Updated Test R²", f"{metrics['test_r2']:.3f}")
                    
                    st.session_state.model_metrics = metrics
                    st.success(f"✅ {ensemble_type} ensemble model trained successfully!")
                    
                    # Visualize ensemble performance
                    if st.session_state.problem_type == "Classification":
                        cm = confusion_matrix(st.session_state.y_test, y_test_pred)
                        fig = px.imshow(cm, text_auto=True, title=f"{ensemble_type} Confusion Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.figures.append((f'{ensemble_type} Confusion Matrix', fig))
                    else:
                        fig = px.scatter(x=st.session_state.y_test, y=y_test_pred, 
                                       title=f"{ensemble_type} Actual vs Predicted")
                        fig.add_trace(go.Scatter(
                            x=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                            y=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                            mode='lines', name='Perfect Prediction'
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.figures.append((f'{ensemble_type} Actual vs Predicted', fig))
                
                except Exception as e:
                    st.error(f"❌ Error building ensemble model: {str(e)}")
    
    elif optimization_method == "Regularization":
        st.write("### 🛡️ Regularization")
        
        if type(st.session_state.model).__name__ in ["LogisticRegression", "LinearRegression"]:
            reg_strength = st.slider("Regularization strength (C for classification, alpha for regression):", 
                                   0.01, 10.0, 1.0)
            
            if st.button("Apply Regularization"):
                try:
                    if st.session_state.problem_type == "Classification":
                        model = LogisticRegression(C=reg_strength, random_state=42)
                    else:
                        from sklearn.linear_model import Ridge
                        model = Ridge(alpha=reg_strength)
                    
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.model = model
                    
                    y_train_pred = model.predict(st.session_state.X_train)
                    y_test_pred = model.predict(st.session_state.X_test)
                    
                    if st.session_state.problem_type == "Classification":
                        metrics = {
                            'train_accuracy': accuracy_score(st.session_state.y_train, y_train_pred),
                            'test_accuracy': accuracy_score(st.session_state.y_test, y_test_pred),
                            'train_predictions': y_train_pred,
                            'test_predictions': y_test_pred
                        }
                        st.metric("Updated Test Accuracy", f"{metrics['test_accuracy']:.3f}")
                    else:
                        metrics = {
                            'train_r2': r2_score(st.session_state.y_train, y_train_pred),
                            'test_r2': r2_score(st.session_state.y_test, y_test_pred),
                            'train_rmse': np.sqrt(mean_squared_error(st.session_state.y_train, y_train_pred)),
                            'test_rmse': np.sqrt(mean_squared_error(st.session_state.y_test, y_test_pred)),
                            'train_predictions': y_train_pred,
                            'test_predictions': y_test_pred
                        }
                        st.metric("Updated Test R²", f"{metrics['test_r2']:.3f}")
                    
                    st.session_state.model_metrics = metrics
                    st.success("✅ Regularization applied successfully!")
                
                except Exception as e:
                    st.error(f"❌ Error applying regularization: {str(e)}")
        else:
            st.info("Regularization is only available for LogisticRegression or LinearRegression models.")

def model_deployment():
    st.markdown('<h2 class="step-header">🚀 Step 9: Model Deployment</h2>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the Model Selection & Training step.")
        return
    
    st.subheader("💾 Model Export")
    
    if st.button("Export Model"):
        try:
            model_file = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(st.session_state.model, model_file)
            
            with open(model_file, 'rb') as file:
                st.download_button(
                    label="Download Model File",
                    data=file,
                    file_name=model_file,
                    mime="application/octet-stream"
                )
            st.success("✅ Model exported successfully!")
        except Exception as e:
            st.error(f"❌ Error exporting model: {str(e)}")
    
    st.subheader("🔍 Model Prediction Interface")
    
    # Create input fields for features
    st.write("Enter values for prediction:")
    input_data = {}
    for feature in st.session_state.feature_names:
        if st.session_state.processed_data[feature].dtype in ['int64', 'float64']:
            input_data[feature] = st.number_input(f"{feature}:", 
                                               value=float(st.session_state.processed_data[feature].mean()))
        else:
            unique_values = st.session_state.processed_data[feature].unique().tolist()
            input_data[feature] = st.selectbox(f"{feature}:", unique_values)
    
    if st.button("Make Prediction"):
        try:
            input_df = pd.DataFrame([input_data])
            
            # Handle categorical features if needed
            categorical_features = input_df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_features and 'target_encoder' in st.session_state:
                for col in categorical_features:
                    le = LabelEncoder()
                    input_df[col] = le.fit_transform(input_df[col].astype(str))
            
            prediction = st.session_state.model.predict(input_df)
            
            if st.session_state.problem_type == "Classification" and 'target_encoder' in st.session_state:
                prediction = st.session_state.target_encoder.inverse_transform(prediction)
            
            st.success("✅ Prediction made successfully!")
            st.write(f"**Prediction:** {prediction[0]}")
            
            if hasattr(st.session_state.model, 'predict_proba') and st.session_state.problem_type == "Classification":
                probabilities = st.session_state.model.predict_proba(input_df)[0]
                prob_df = pd.DataFrame({
                    'Class': st.session_state.target_encoder.classes_ if 'target_encoder' in st.session_state else range(len(probabilities)),
                    'Probability': probabilities
                })
                fig = px.bar(prob_df, x='Class', y='Probability', title="Prediction Probabilities")
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.figures.append(('Prediction Probabilities', fig))
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    st.subheader("📡 API Deployment Simulation")
    st.code("""
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('model.pkl')

@app.post("/predict")
async def predict(data: dict):
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return {"prediction": prediction.tolist()}
    """, language="python")
    st.info("This is a simulated API endpoint. In a production environment, deploy using a framework like FastAPI or Flask.")

def monitoring_results():
    st.markdown('<h2 class="step-header">📈 Step 10: Monitoring & Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the Model Selection & Training step.")
        return
    
    st.subheader("📊 Model Performance Monitoring")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.problem_type == "Classification":
            st.metric("Current Test Accuracy", f"{st.session_state.model_metrics['test_accuracy']:.3f}")
        else:
            st.metric("Current Test R²", f"{st.session_state.model_metrics['test_r2']:.3f}")
    
    with col2:
        if st.session_state.problem_type == "Classification":
            st.metric("Training Accuracy", f"{st.session_state.model_metrics['train_accuracy']:.3f}")
        else:
            st.metric("Training R²", f"{st.session_state.model_metrics['train_r2']:.3f}")
    
    # Data drift detection
    st.subheader("🔍 Data Drift Detection")
    
    if st.button("Check for Data Drift"):
        try:
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            
            drift_results = []
            for feature in X_train.columns:
                stat, p_value = ks_2samp(X_train[feature], X_test[feature])
                drift_results.append({
                    'Feature': feature,
                    'KS Statistic': stat,
                    'p-value': p_value,
                    'Drift Detected': p_value < 0.05
                })
            
            drift_df = pd.DataFrame(drift_results)
            st.dataframe(drift_df, use_container_width=True)
            
            # Visualize drift
            drifted_features = drift_df[drift_df['Drift Detected'] == True]['Feature'].tolist()
            if drifted_features:
                st.warning(f"⚠️ Data drift detected in features: {drifted_features}")
                for feature in drifted_features[:3]:  # Show up to 3 drifted features
                    fig = make_subplots(rows=1, cols=2, subplot_titles=['Training Distribution', 'Test Distribution'])
                    fig.add_trace(go.Histogram(x=X_train[feature], name='Training'), row=1, col=1)
                    fig.add_trace(go.Histogram(x=X_test[feature], name='Test'), row=1, col=2)
                    fig.update_layout(title=f"Distribution Comparison for {feature}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.figures.append((f'Distribution Comparison for {feature}', fig))
            else:
                st.success("✅ No significant data drift detected!")
        
        except Exception as e:
            st.error(f"❌ Error in data drift detection: {str(e)}")
    
    # Generate comprehensive report
    st.subheader("📑 Generate Project Report")
    
    if st.button("Generate PDF Report"):
        try:
            doc = Document()
            doc.packages.append(Package('geometry', options=['margin=1in']))
            doc.packages.append(Package('graphicx'))
            doc.packages.append(Package('float'))
            
            with doc.create(Section('Data Science Project Report')):
                doc.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                with doc.create(Subsection('Problem Definition')):
                    if 'project_info' in st.session_state:
                        doc.append(f"Problem Type: {st.session_state.project_info['problem_type']}\n")
                        doc.append(f"Objective: {st.session_state.project_info['objective']}\n")
                        doc.append(f"Success Metrics: {st.session_state.project_info['metrics']}\n")
                
                with doc.create(Subsection('Data Summary')):
                    if st.session_state.data is not None:
                        doc.append(f"Dataset Size: {st.session_state.data.shape}\n")
                        doc.append(f"Features: {len(st.session_state.data.columns)}\n")
                        doc.append(f"Missing Values: {st.session_state.data.isnull().sum().sum()}\n")
                
                with doc.create(Subsection('Model Performance')):
                    if st.session_state.problem_type == "Classification":
                        doc.append(f"Test Accuracy: {st.session_state.model_metrics['test_accuracy']:.3f}\n")
                    else:
                        doc.append(f"Test R²: {st.session_state.model_metrics['test_r2']:.3f}\n")
                        doc.append(f"Test RMSE: {st.session_state.model_metrics['test_rmse']:.3f}\n")
                
                with doc.create(Subsection('Figures')):
                    for fig_name, fig in st.session_state.figures:
                        with doc.create(Figure(position='H')) as figure:
                            img_buffer = io.BytesIO()
                            fig.write_image(img_buffer, format='png')
                            img_str = base64.b64encode(img_buffer.getvalue()).decode()
                            figure.add_image(f'data:image/png;base64,{img_str}', width=NoEscape(r'0.8\textwidth'))
                            figure.add_caption(fig_name)
            
            pdf_file = f"project_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            doc.generate_pdf(pdf_file, clean_tex=False)
            
            with open(pdf_file, 'rb') as file:
                st.download_button(
                    label="Download PDF Report",
                    data=file,
                    file_name=pdf_file,
                    mime="application/pdf"
                )
            st.success("✅ PDF report generated successfully!")
        
        except Exception as e:
            st.error(f"❌ Error generating PDF report: {str(e)}")

if __name__ == "__main__":
    main()