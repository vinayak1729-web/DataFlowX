import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="DataFLowX",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
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

def main():
    st.markdown('<h1 class="main-header">🚀 Complete Data Science Platform</h1>', unsafe_allow_html=True)
    
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
    
    # Main content based on selected step
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
        st.metric("Data Status", "✅ Loaded" if st.session_state.data is not None else "❌ Not Loaded")
    
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
            fig = px.pie(values=dtype_counts.values, names=dtype_counts.index, title="Distribution of Data Types")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Missing Values")
            missing_data = st.session_state.data.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                fig = px.bar(x=missing_data.index, y=missing_data.values, title="Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found!")

def problem_definition():
    st.markdown('<h2 class="step-header">🎯 Step 1: Problem Definition</h2>', unsafe_allow_html=True)
    
    st.write("Define your data science problem clearly to guide the entire workflow.")
    
    with st.form("problem_definition_form"):
        problem_type = st.selectbox(
            "Problem Type:",
            ["Classification", "Regression", "Clustering", "Time Series", "NLP", "Other"]
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
            st.success("✅ Problem definition saved successfully!")
            
    if 'project_info' in st.session_state:
        st.subheader("📋 Current Problem Definition")
        info = st.session_state.project_info
        st.json(info)

def data_collection():
    st.markdown('<h2 class="step-header">📥 Step 2: Data Collection</h2>', unsafe_allow_html=True)
    
    st.write("Upload your dataset or use sample data to get started.")
    
    # Data source options
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
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.data = df
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Show data preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    elif data_source == "Use sample dataset":
        sample_choice = st.selectbox(
            "Choose sample dataset:",
            ["Iris (Classification)", "Boston Housing (Regression)", "Titanic (Classification)"]
        )
        
        if st.button("Load Sample Dataset"):
            if sample_choice == "Iris (Classification)":
                from sklearn.datasets import load_iris
                iris = load_iris()
                df = pd.DataFrame(iris.data, columns=iris.feature_names)
                df['target'] = iris.target
                df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
                
            elif sample_choice == "Titanic (Classification)":
                # Create a sample Titanic-like dataset
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
                
            else:  # Boston Housing
                from sklearn.datasets import load_boston
                boston = load_boston()
                df = pd.DataFrame(boston.data, columns=boston.feature_names)
                df['target'] = boston.target
            
            st.session_state.data = df
            st.success(f"✅ Sample dataset loaded! Shape: {df.shape}")
            st.dataframe(df.head(), use_container_width=True)
    
    else:  # Database simulation
        st.info("🔗 Database connection simulation")
        st.code("""
# Example database connection code:
import sqlite3
import pandas as pd

conn = sqlite3.connect('database.db')
query = "SELECT * FROM your_table"
df = pd.read_sql_query(query, conn)
conn.close()
        """)
        
        if st.button("Simulate Database Connection"):
            # Create sample data to simulate database
            np.random.seed(42)
            df = pd.DataFrame({
                'customer_id': range(1, 1001),
                'age': np.random.randint(18, 80, 1000),
                'income': np.random.normal(50000, 20000, 1000),
                'spending_score': np.random.randint(1, 100, 1000),
                'churn': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
            })
            st.session_state.data = df
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
        
        # Handle missing values
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
            # Calculate outliers using IQR method
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
            
            with col2:
                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"Found {len(outliers)} outliers using IQR method")
            
            if len(outliers) > 0 and st.button(f"Remove outliers from {selected_col}"):
                df = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]
                st.success(f"✅ Removed {len(outliers)} outliers")
    
    # Data type conversion
    st.subheader("🔧 Data Type Conversion")
    for col in df.columns:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{col}:**")
        with col2:
            st.write(f"Current type: {df[col].dtype}")
        with col3:
            if st.button(f"Convert", key=f"convert_{col}"):
                new_type = st.selectbox(
                    f"New type for {col}:",
                    ["int64", "float64", "object", "datetime64", "category"],
                    key=f"type_{col}"
                )
                try:
                    if new_type == "datetime64":
                        df[col] = pd.to_datetime(df[col])
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
        # Convert dtype objects to strings for Plotly compatibility
        dtype_counts.index = dtype_counts.index.astype(str)
        fig = px.pie(values=dtype_counts.values, names=dtype_counts.index, title="Data Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
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
            
            with col2:
                fig = px.box(df, y=selected_numeric, title=f"Box Plot of {selected_numeric}")
                st.plotly_chart(fig, use_container_width=True)
            
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
            
            with col2:
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Pie Chart of {selected_categorical}")
                st.plotly_chart(fig, use_container_width=True)
            
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
            # Scatter plot
            fig = px.scatter(df, x=x_var, y=y_var, title=f"{x_var} vs {y_var}")
            
            # Add color by categorical variable if available
            if categorical_cols:
                color_var = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
                if color_var != "None":
                    fig = px.scatter(df, x=x_var, y=y_var, color=color_var, 
                                   title=f"{x_var} vs {y_var} (colored by {color_var})")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation
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
        
        # Top correlations
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
            else:  # RobustScaler
                scaler = RobustScaler()
            
            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
            st.success(f"✅ Applied {scaling_method} to selected columns")
            
            # Show before/after comparison
            st.write("**Scaling Results:**")
            st.dataframe(df[columns_to_scale].describe(), use_container_width=True)
    
    # Encoding categorical variables
    st.subheader("🏷️ Categorical Variable Encoding")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        selected_cat_col = st.selectbox("Select categorical column to encode:", categorical_cols)
        encoding_method = st.selectbox(
            "Select encoding method:",
            ["Label Encoding", "One-Hot Encoding", "Target Encoding"]
        )
        
        if selected_cat_col and st.button("Apply Encoding"):
            if encoding_method == "Label Encoding":
                le = LabelEncoder()
                df[f"{selected_cat_col}_encoded"] = le.fit_transform(df[selected_cat_col].astype(str))
                st.success(f"✅ Applied Label Encoding to {selected_cat_col}")
                
            elif encoding_method == "One-Hot Encoding":
                # Get dummies
                dummies = pd.get_dummies(df[selected_cat_col], prefix=selected_cat_col)
                df = pd.concat([df, dummies], axis=1)
                st.success(f"✅ Applied One-Hot Encoding to {selected_cat_col}")
                
            st.write("**Encoding Results:**")
            st.dataframe(df.head(), use_container_width=True)
    
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
                st.error(f"Error creating feature: {str(e)}")
    
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
        st.write("Create categorical features by binning continuous variables:")
        
        if numeric_cols:
            bin_col = st.selectbox("Select column to bin:", numeric_cols)
            n_bins = st.slider("Number of bins:", 2, 20, 5)
            bin_method = st.selectbox("Binning method:", ["Equal width", "Equal frequency"])
            
            if bin_col and st.button("Create Binned Feature"):
                if bin_method == "Equal width":
                    df[f"{bin_col}_binned"] = pd.cut(df[bin_col], bins=n_bins, labels=False)
                else:
                    df[f"{bin_col}_binned"] = pd.qcut(df[bin_col], q=n_bins, labels=False, duplicates='drop')
                
                st.success(f"✅ Created binned feature: {bin_col}_binned")
                
                # Show binning results
                bin_counts = df[f"{bin_col}_binned"].value_counts().sort_index()
                fig = px.bar(x=bin_counts.index, y=bin_counts.values, 
                           title=f"Distribution of {bin_col}_binned")
                st.plotly_chart(fig, use_container_width=True)
    
    # Feature selection
    st.subheader("🎯 Feature Selection")
    
    if 'target_column' not in st.session_state:
        target_col = st.selectbox("Select target column for feature selection:", df.columns.tolist())
        if st.button("Set Target Column"):
            st.session_state.target_column = target_col
            st.success(f"✅ Target column set to: {target_col}")
    else:
        st.info(f"Current target column: {st.session_state.target_column}")
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
                
                # Select top features
                n_features = st.slider("Select top N features:", 1, len(correlations), min(10, len(correlations)))
                selected_features = correlations.head(n_features).index.tolist()
                
                st.write(f"**Selected features:** {selected_features}")
                
            elif selection_method == "Manual selection":
                selected_features = st.multiselect("Select features manually:", feature_cols)
    
    # Save engineered features
    if st.button("💾 Save Engineered Features"):
        st.session_state.processed_data = df
        st.success("✅ Engineered features saved successfully!")
        
        # Show feature summary
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
    
    # Select target variable
    target_col = st.selectbox("Select target variable:", df.columns.tolist())
    
    # Select features
    feature_cols = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect("Select features:", feature_cols, default=feature_cols[:5])
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        return
    
    # Determine problem type
    if df[target_col].dtype == 'object' or df[target_col].nunique() <= 10:
        problem_type = "Classification"
    else:
        problem_type = "Regression"
    
    st.info(f"Detected problem type: **{problem_type}**")
    
    # Prepare data
    X = df[selected_features]
    y = df[target_col]
    
    # Handle categorical variables in features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        st.warning(f"Categorical features detected: {categorical_features}")
        st.info("Applying label encoding to categorical features...")
        
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle categorical target for classification
    if problem_type == "Classification" and y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        st.session_state.target_encoder = le_target
    
    # Train-test split
    st.subheader("📊 Data Splitting")
    test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state:", 0, 1000, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Set Size", len(X_train))
    with col2:
        st.metric("Test Set Size", len(X_test))
    with col3:
        st.metric("Features", len(selected_features))
    
    # Model selection
    st.subheader("🎯 Model Selection")
    
    if problem_type == "Classification":
        available_models = {
            "Random Forest": RandomForestClassifier(random_state=random_state),
            "Logistic Regression": LogisticRegression(random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "SVM": SVC(random_state=random_state)
        }
    else:
        available_models = {
            "Random Forest": RandomForestRegressor(random_state=random_state),
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "SVR": SVR()
        }
    
    selected_model_name = st.selectbox("Select model:", list(available_models.keys()))
    selected_model = available_models[selected_model_name]
    
    # Hyperparameter tuning option
    st.subheader("🔧 Hyperparameter Tuning")
    tune_hyperparams = st.checkbox("Enable hyperparameter tuning (may take longer)")
    
    if tune_hyperparams:
        if selected_model_name == "Random Forest":
            if problem_type == "Classification":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
        elif selected_model_name in ["Logistic Regression"]:
            param_grid = {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        else:
            param_grid = {}
            st.info("Basic hyperparameter tuning not configured for this model.")
    
    # Train model
    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            try:
                if tune_hyperparams and param_grid:
                    # Grid search
                    grid_search = GridSearchCV(selected_model, param_grid, cv=5, scoring='accuracy' if problem_type == "Classification" else 'r2')
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
                st.session_state.problem_type = problem_type
                st.session_state.feature_names = selected_features
                st.session_state.target_name = target_col
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                if problem_type == "Classification":
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
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("📊 Feature Importance")
                    
                    importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                               title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")

def model_evaluation():
    st.markdown('<h2 class="step-header">📊 Step 7: Model Evaluation</h2>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the Model Selection & Training step.")
        return
    
    model = st.session_state.model
    metrics = st.session_state.model_metrics
    problem_type = st.session_state.problem_type
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    st.subheader("📈 Model Performance Metrics")
    
    if problem_type == "Classification":
        # Classification metrics
        y_pred = metrics['test_predictions']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['test_accuracy']:.3f}")
        with col2:
            precision = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision']
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            recall = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
            st.metric("Recall", f"{recall:.3f}")
        
        # Confusion Matrix
        st.subheader("🔄 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # ROC Curve (for binary classification)
        if len(np.unique(y_test)) == 2:
            from sklearn.metrics import roc_curve, auc
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
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
    
    else:
        # Regression metrics
        y_pred = metrics['test_predictions']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{metrics['test_r2']:.3f}")
        with col2:
            st.metric("RMSE", f"{metrics['test_rmse']:.3f}")
        with col3:
            mae = mean_absolute_error(y_test, y_pred)
            st.metric("MAE", f"{mae:.3f}")
        with col4:
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            st.metric("MAPE", f"{mape:.1f}%")
        
        # Actual vs Predicted
        st.subheader("📊 Actual vs Predicted")
        fig = px.scatter(x=y_test, y=y_pred, title="Actual vs Predicted Values")
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                               y=[y_test.min(), y_test.max()], 
                               mode='lines', name='Perfect Prediction'))
        fig.update_layout(xaxis_title='Actual Values', yaxis_title='Predicted Values')
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual Plot
        st.subheader("📈 Residual Plot")
        residuals = y_test - y_pred
        fig = px.scatter(x=y_pred, y=residuals, title="Residual Plot")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(xaxis_title='Predicted Values', yaxis_title='Residuals')
        st.plotly_chart(fig, use_container_width=True)
    
    # Cross-validation
    st.subheader("🔄 Cross-Validation")
    if st.button("Perform Cross-Validation"):
        with st.spinner("Performing cross-validation..."):
            X_full = pd.concat([st.session_state.X_train, st.session_state.X_test])
            y_full = pd.concat([pd.Series(st.session_state.y_train), pd.Series(st.session_state.y_test)])
            
            scoring = 'accuracy' if problem_type == "Classification" else 'r2'
            cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring=scoring)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CV Mean Score", f"{cv_scores.mean():.3f}")
            with col2:
                st.metric("CV Std Score", f"{cv_scores.std():.3f}")
            with col3:
                st.metric("CV Scores Range", f"{cv_scores.min():.3f} - {cv_scores.max():.3f}")
            
            # Plot CV scores
            fig = px.bar(x=[f"Fold {i+1}" for i in range(len(cv_scores))], 
                        y=cv_scores, title="Cross-Validation Scores")
            st.plotly_chart(fig, use_container_width=True)
    
    # Learning Curves
    st.subheader("📈 Learning Curves")
    if st.button("Generate Learning Curves"):
        from sklearn.model_selection import learning_curve
        
        with st.spinner("Generating learning curves..."):
            X_full = pd.concat([st.session_state.X_train, st.session_state.X_test])
            y_full = pd.concat([pd.Series(st.session_state.y_train), pd.Series(st.session_state.y_test)])
            
            train_sizes = np.linspace(0.1, 1.0, 10)
            scoring = 'accuracy' if problem_type == "Classification" else 'r2'
            
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
        
        # Get current model type
        model_type = type(st.session_state.model).__name__
        st.info(f"Current model: {model_type}")
        
        # Define parameter grids based on model type
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
        
        else:
            st.info("Custom hyperparameter tuning not available for this model type.")
            param_options = {}
        
        if param_options and st.button("🚀 Optimize Hyperparameters"):
            with st.spinner("Optimizing hyperparameters... This may take several minutes."):
                try:
                    # Prepare parameter grid
                    param_grid = {}
                    for param, values in param_options.items():
                        if values:
                            param_grid[param] = values
                    
                    if param_grid:
                        # Grid search with cross-validation
                        scoring = 'accuracy' if st.session_state.problem_type == "Classification" else 'r2'
                        grid_search = GridSearchCV(
                            st.session_state.model, 
                            param_grid, 
                            cv=5, 
                            scoring=scoring,
                            n_jobs=-1
                        )
                        
                        X_train = st.session_state.X_train
                        y_train = st.session_state.y_train
                        
                        grid_search.fit(X_train, y_train)
                        
                        # Update model with best parameters
                        st.session_state.model = grid_search.best_estimator_
                        
                        st.success("✅ Hyperparameter optimization completed!")
                        st.write(f"**Best score:** {grid_search.best_score_:.4f}")
                        st.write(f"**Best parameters:** {grid_search.best_params_}")
                        
                        # Show results comparison
                        results_df = pd.DataFrame(grid_search.cv_results_)
                        st.subheader("📊 Optimization Results")
                        st.dataframe(results_df[['params', 'mean_test_score', 'rank_test_score']].head(10))
                    
                except Exception as e:
                                        st.error(f"❌ Error during hyperparameter optimization: {str(e)}")
        
        # Visualize hyperparameter tuning results if available
        if 'grid_search' in locals() and grid_search.best_params_:
            st.subheader("📈 Hyperparameter Tuning Visualization")
            results_df = pd.DataFrame(grid_search.cv_results_)
            if 'param_n_estimators' in results_df.columns:  # Example for RandomForest
                fig = px.scatter(
                    results_df,
                    x='param_n_estimators',
                    y='mean_test_score',
                    color='param_max_depth',
                    size='mean_test_score',
                    title="Hyperparameter Tuning Results (Random Forest)",
                    labels={'mean_test_score': 'Mean Test Score', 'param_n_estimators': 'Number of Estimators'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
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
            from sklearn.feature_selection import RFE
            
            n_features = st.slider("Number of features to select:", 1, len(feature_cols), len(feature_cols)//2)
            if st.button("Run RFE"):
                with st.spinner("Performing Recursive Feature Elimination..."):
                    try:
                        rfe = RFE(estimator=st.session_state.model, n_features_to_select=n_features)
                        rfe.fit(st.session_state.X_train, st.session_state.y_train)
                        
                        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_]
                        st.session_state.feature_names = selected_features
                        
                        st.success(f"✅ Selected {len(selected_features)} features: {selected_features}")
                        
                        # Update model with selected features
                        X_train_new = st.session_state.X_train[selected_features]
                        X_test_new = st.session_state.X_test[selected_features]
                        st.session_state.model.fit(X_train_new, st.session_state.y_train)
                        
                        # Update metrics
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
                        
                        # Visualize feature ranking
                        ranking_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Ranking': rfe.ranking_
                        }).sort_values('Ranking')
                        fig = px.bar(ranking_df, x='Feature', y='Ranking', title="Feature Rankings (RFE)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error during RFE: {str(e)}")
        
        elif selection_approach == "Select K Best":
            from sklearn.feature_selection import SelectKBest, f_classif, f_regression
            
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
                        
                        # Update model and metrics
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
                        
                        # Visualize feature scores
                        scores_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Score': selector.scores_
                        }).sort_values('Score', ascending=False)
                        fig = px.bar(scores_df, x='Feature', y='Score', title="Feature Scores (Select K Best)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error during Select K Best: {str(e)}")
        
        elif selection_approach == "Manual Selection":
            selected_features = st.multiselect("Select features:", feature_cols, default=feature_cols)
            if st.button("Apply Manual Selection"):
                st.session_state.feature_names = selected_features
                st.success(f"✅ Selected {len(selected_features)} features: {selected_features}")
                
                # Update model and metrics
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
                                ('svm', SVC(random_state=42))
                            ]
                            ensemble = StackingClassifier(
                                estimators=models,
                                final_estimator=LogisticRegression(),
                                cv=5
                            )
                        else:
                            models = [
                                ('rf', RandomForestRegressor(random_state=42)),
                                ('svr', SVR())
                            ]
                            ensemble = StackingRegressor(
                                estimators=models,
                                final_estimator=LinearRegression(),
                                cv=5
                            )
                    
                    else:  # Bagging
                        from sklearn.ensemble import BaggingClassifier, BaggingRegressor
                        
                        if st.session_state.problem_type == "Classification":
                            ensemble = BaggingClassifier(
                                base_estimator=DecisionTreeClassifier(),
                                n_estimators=50,
                                random_state=42
                            )
                        else:
                            ensemble = BaggingRegressor(
                                base_estimator=DecisionTreeRegressor(),
                                n_estimators=50,
                                random_state=42
                            )
                    
                    # Train ensemble
                    ensemble.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.model = ensemble
                    
                    # Update metrics
                    y_train_pred = ensemble.predict(st.session_state.X_train)
                    y_test_pred = ensemble.predict(st.session_state.X_test)
                    
                    if st.session_state.problem_type == "Classification":
                        metrics = {
                            'train_accuracy': accuracy_score(st.session_state.y_train, y_train_pred),
                            'test_accuracy': accuracy_score(st.session_state.y_test, y_test_pred),
                            'train_predictions': y_train_pred,
                            'test_predictions': y_test_pred
                        }
                        st.metric("Ensemble Test Accuracy", f"{metrics['test_accuracy']:.3f}")
                    else:
                        metrics = {
                            'train_r2': r2_score(st.session_state.y_train, y_train_pred),
                            'test_r2': r2_score(st.session_state.y_test, y_test_pred),
                            'train_rmse': np.sqrt(mean_squared_error(st.session_state.y_train, y_train_pred)),
                            'test_rmse': np.sqrt(mean_squared_error(st.session_state.y_test, y_test_pred)),
                            'train_predictions': y_train_pred,
                            'test_predictions': y_test_pred
                        }
                        st.metric("Ensemble Test R²", f"{metrics['test_r2']:.3f}")
                    
                    st.session_state.model_metrics = metrics
                    st.success(f"✅ Ensemble model ({ensemble_type}) trained successfully!")
                
                except Exception as e:
                    st.error(f"❌ Error building ensemble model: {str(e)}")
    
    elif optimization_method == "Regularization":
        st.write("### 🛠️ Regularization")
        
        if not isinstance(st.session_state.model, (LogisticRegression, LinearRegression)):
            st.warning("Regularization optimization is only available for Logistic or Linear Regression models.")
            return
        
        if st.session_state.problem_type == "Classification":
            penalty = st.selectbox("Penalty type:", ['l1', 'l2'])
            C = st.slider("Regularization strength (C):", 0.01, 100.0, 1.0, format="%.2f")
            
            if st.button("Apply Regularization"):
                with st.spinner("Applying regularization..."):
                    try:
                        model = LogisticRegression(
                            penalty=penalty,
                            C=C,
                            solver='liblinear' if penalty == 'l1' else 'lbfgs',
                            random_state=42
                        )
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                        st.session_state.model = model
                        
                        # Update metrics
                        y_train_pred = model.predict(st.session_state.X_train)
                        y_test_pred = model.predict(st.session_state.X_test)
                        metrics = {
                            'train_accuracy': accuracy_score(st.session_state.y_train, y_train_pred),
                            'test_accuracy': accuracy_score(st.session_state.y_test, y_test_pred),
                            'train_predictions': y_train_pred,
                            'test_predictions': y_test_pred
                        }
                        st.session_state.model_metrics = metrics
                        st.metric("Regularized Test Accuracy", f"{metrics['test_accuracy']:.3f}")
                        st.success("✅ Regularization applied successfully!")
                    
                    except Exception as e:
                        st.error(f"❌ Error applying regularization: {str(e)}")
        
        else:
            alpha = st.slider("Regularization strength (alpha):", 0.01, 100.0, 1.0, format="%.2f")
            reg_type = st.selectbox("Regularization type:", ['Lasso', 'Ridge'])
            
            if st.button("Apply Regularization"):
                with st.spinner("Applying regularization..."):
                    try:
                        from sklearn.linear_model import Lasso, Ridge
                        model = Lasso(alpha=alpha) if reg_type == 'Lasso' else Ridge(alpha=alpha)
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                        st.session_state.model = model
                        
                        # Update metrics
                        y_train_pred = model.predict(st.session_state.X_train)
                        y_test_pred = model.predict(st.session_state.X_test)
                        metrics = {
                            'train_r2': r2_score(st.session_state.y_train, y_train_pred),
                            'test_r2': r2_score(st.session_state.y_test, y_test_pred),
                            'train_rmse': np.sqrt(mean_squared_error(st.session_state.y_train, y_train_pred)),
                            'test_rmse': np.sqrt(mean_squared_error(st.session_state.y_test, y_test_pred)),
                            'train_predictions': y_train_pred,
                            'test_predictions': y_test_pred
                        }
                        st.session_state.model_metrics = metrics
                        st.metric("Regularized Test R²", f"{metrics['test_r2']:.3f}")
                        st.success("✅ Regularization applied successfully!")
                    
                    except Exception as e:
                        st.error(f"❌ Error applying regularization: {str(e)}")

def model_deployment():
    st.markdown('<h2 class="step-header">🚀 Step 9: Model Deployment</h2>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the Model Selection & Training step.")
        return
    
    st.subheader("📦 Model Deployment Options")
    
    # Save model
    st.write("### 💾 Save Model")
    model_name = st.text_input("Enter model name:", "my_model")
    
    if st.button("Save Model"):
        try:
            joblib.dump(st.session_state.model, f"{model_name}.pkl")
            st.success(f"✅ Model saved as {model_name}.pkl")
            
            # Provide download link
            with open(f"{model_name}.pkl", "rb") as file:
                b64 = base64.b64encode(file.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{model_name}.pkl">Download Model</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error saving model: {str(e)}")
    
    # API simulation
    st.subheader("🌐 API Simulation")
    st.write("Simulate model predictions through an API-like interface.")
    
    feature_cols = st.session_state.feature_names
    input_data = {}
    
    st.write("**Enter input values for prediction:**")
    for feature in feature_cols:
        input_data[feature] = st.number_input(
            f"{feature}:",
            value=float(st.session_state.X_train[feature].mean()),
            key=f"input_{feature}"
        )
    
    if st.button("Make Prediction"):
        try:
            input_df = pd.DataFrame([input_data])
            prediction = st.session_state.model.predict(input_df)
            
            if st.session_state.problem_type == "Classification" and 'target_encoder' in st.session_state:
                prediction = st.session_state.target_encoder.inverse_transform(prediction)
            
            st.success("✅ Prediction completed!")
            st.write(f"**Prediction:** {prediction[0]}")
            
            # Show prediction probability if available
            if st.session_state.problem_type == "Classification" and hasattr(st.session_state.model, 'predict_proba'):
                proba = st.session_state.model.predict_proba(input_df)[0]
                proba_df = pd.DataFrame({
                    'Class': st.session_state.target_encoder.classes_,
                    'Probability': proba
                })
                fig = px.bar(proba_df, x='Class', y='Probability', title="Prediction Probabilities")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    # Deployment instructions
    st.subheader("📋 Deployment Instructions")
    st.write("Example code to deploy the model using Flask:")
    st.code("""
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('my_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
    """)
    
    st.info("For production deployment, consider using platforms like AWS, GCP, or Azure with proper containerization (e.g., Docker).")

def monitoring_results():
    st.markdown('<h2 class="step-header">📈 Step 10: Monitoring & Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the Model Selection & Training step.")
        return
    
    st.subheader("📊 Model Performance Summary")
    
    metrics = st.session_state.model_metrics
    if st.session_state.problem_type == "Classification":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Accuracy", f"{metrics['train_accuracy']:.3f}")
        with col2:
            st.metric("Test Accuracy", f"{metrics['test_accuracy']:.3f}")
        with col3:
            st.metric("Accuracy Gap", f"{abs(metrics['train_accuracy'] - metrics['test_accuracy']):.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(st.session_state.y_test, metrics['test_predictions'])
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix (Test Set)")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training R²", f"{metrics['train_r2']:.3f}")
        with col2:
            st.metric("Test R²", f"{metrics['test_r2']:.3f}")
        with col3:
            st.metric("Training RMSE", f"{metrics['train_rmse']:.3f}")
        with col4:
            st.metric("Test RMSE", f"{metrics['test_rmse']:.3f}")
        
        # Actual vs Predicted
        fig = px.scatter(
            x=st.session_state.y_test,
            y=metrics['test_predictions'],
            title="Actual vs Predicted (Test Set)"
        )
        fig.add_trace(go.Scatter(
            x=[st.session_state.y_test.min(), st.session_state.y_test.max()],
            y=[st.session_state.y_test.min(), st.session_state.y_test.max()],
            mode='lines',
            name='Perfect Prediction'
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Data drift simulation
    st.subheader("🔍 Data Drift Monitoring")
    st.write("Simulate data drift by comparing training data distribution with new data.")
    
    if st.button("Simulate New Data & Check Drift"):
        try:
            # Simulate new data with slight distribution shift
            new_data = st.session_state.X_train.copy()
            for col in new_data.columns:
                if new_data[col].dtype in ['int64', 'float64']:
                    new_data[col] = new_data[col] * np.random.normal(1.1, 0.1, len(new_data))
            
            # Compare distributions
            drift_report = []
            for col in new_data.columns:
                if new_data[col].dtype in ['int64', 'float64']:
                    from scipy.stats import ks_2samp
                    stat, p_value = ks_2samp(st.session_state.X_train[col], new_data[col])
                    drift_report.append({
                        'Feature': col,
                        'KS Statistic': stat,
                        'P-Value': p_value,
                        'Drift Detected': p_value < 0.05
                    })
            
            drift_df = pd.DataFrame(drift_report)
            st.dataframe(drift_df, use_container_width=True)
            
            # Visualize drift for a selected feature
            drift_feature = st.selectbox("Select feature to visualize drift:", st.session_state.feature_names)
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=st.session_state.X_train[drift_feature],
                name='Training Data',
                opacity=0.5
            ))
            fig.add_trace(go.Histogram(
                x=new_data[drift_feature],
                name='New Data',
                opacity=0.5
            ))
            fig.update_layout(
                title=f"Distribution Comparison: {drift_feature}",
                xaxis_title=drift_feature,
                yaxis_title="Count",
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error in drift analysis: {str(e)}")
    
    # Model retraining simulation
    st.subheader("🔄 Model Retraining")
    st.write("Simulate model retraining with new data.")
    
    if st.button("Simulate Retraining"):
        try:
            # Simulate new data
            new_X = st.session_state.X_train.copy()
            new_y = st.session_state.y_train.copy()
            # Add noise to simulate new patterns
            new_X += np.random.normal(0, 0.05, new_X.shape)
            
            # Retrain model
            model = st.session_state.model
            model.fit(new_X, new_y)
            
            # Evaluate on test set
            y_test_pred = model.predict(st.session_state.X_test)
            if st.session_state.problem_type == "Classification":
                new_accuracy = accuracy_score(st.session_state.y_test, y_test_pred)
                st.metric("Retrained Test Accuracy", f"{new_accuracy:.3f}")
                st.write(f"Previous Accuracy: {metrics['test_accuracy']:.3f}")
            else:
                new_r2 = r2_score(st.session_state.y_test, y_test_pred)
                new_rmse = np.sqrt(mean_squared_error(st.session_state.y_test, y_test_pred))
                st.metric("Retrained Test R²", f"{new_r2:.3f}")
                st.metric("Retrained Test RMSE", f"{new_rmse:.3f}")
                st.write(f"Previous R²: {metrics['test_r2']:.3f}")
                st.write(f"Previous RMSE: {metrics['test_rmse']:.3f}")
            
            st.success("✅ Model retrained successfully!")
        
        except Exception as e:
            st.error(f"❌ Error during retraining: {str(e)}")
    
    # Results Communication
    st.subheader("📢 Results Communication")
    st.write("Generate a summary report for stakeholders.")
    
    if st.button("Generate Report"):
        report = f"""
# Data Science Project Report
## Project Overview
**Problem Type**: {st.session_state.problem_type}
**Target Variable**: {st.session_state.target_name}
**Features Used**: {', '.join(st.session_state.feature_names)}

## Model Performance
"""
        if st.session_state.problem_type == "Classification":
            report += f"""
- **Training Accuracy**: {metrics['train_accuracy']:.3f}
- **Test Accuracy**: {metrics['test_accuracy']:.3f}
"""
        else:
            report += f"""
- **Training R²**: {metrics['train_r2']:.3f}
- **Test R²**: {metrics['test_r2']:.3f}
- **Training RMSE**: {metrics['train_rmse']:.3f}
- **Test RMSE**: {metrics['test_rmse']:.3f}
"""

        if 'project_info' in st.session_state:
            report += f"""
## Business Insights
**Objective**: {st.session_state.project_info.get('objective', 'N/A')}
**Success Metrics**: {st.session_state.project_info.get('metrics', 'N/A')}
**Key Findings**: Model performance meets/exceeds defined metrics.
**Recommendations**: Deploy model and monitor for data drift.
"""
        
        st.markdown(report)
        
        # Provide download option
        st.download_button(
            label="Download Report",
            data=report,
            file_name="data_science_report.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()