import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import base64
from data_cleaningPreprocessing import load_data
from data_cleaningPreprocessing import understand_data, clean_data, transform_data, structure_data, preprocess_by_type, select_features, handle_zeros, generate_visualizations
from data_selection import dynamic_split
from model_training import train_model
from model_evaluation import evaluate_model
from data_modeling import NeuralNetworkRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Function to test model with user query
def test_model_with_user_query(model, query, task='regression', model_type='sklearn', preprocessor=None, device='cpu'):
    """Test a trained model with a user input query."""
    try:
        query = np.asarray(query).reshape(1, -1)
        if preprocessor is not None:
            query = preprocessor.transform(query)
        if model_type == 'sklearn':
            prediction = model.predict(query)[0]
            probabilities = None
        elif model_type == 'pytorch':
            model.eval()
            query_t = torch.tensor(query, dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(query_t).cpu().numpy().flatten()
            prediction = output[0]
            probabilities = None
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        results = {'prediction': prediction}
        return results, None
    except Exception as e:
        return None, f"ERROR: Failed to predict: {str(e)}"

# Streamlit App
st.title("Machine Learning Pipeline")
st.write("Machine Learning Pipeline")
st.write("Upload a CSV file to process, train models, and make predictions.")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'task' not in st.session_state:
    st.session_state.task = 'regression'
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'split_data' not in st.session_state:
    st.session_state.split_data = None
if 'model_configs' not in st.session_state:
    st.session_state.model_configs = None

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read file content
        file_content = uploaded_file.getvalue()
        
        # Initialize progress bar and text
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        progress_bar.progress(0.0)
        progress_text.text("Starting pipeline...")

        # Step 1: Load Data
        if st.session_state.step == 0:
            progress_text.text("Step 1: Loading data...")
            df, error = load_data(file_content)
            if error:
                st.error(error)
                st.stop()
            if 'Salary' not in df.columns:
                st.error("Dataset must contain a 'Salary' column as the target.")
                st.stop()
            st.session_state.data = df
            st.session_state.step = 1
            progress_bar.progress(0.1)
        
        df = st.session_state.data

        # Step 2: Understand Data
        if st.session_state.step == 1:
            progress_text.text("Step 2: Understanding data...")
            info, error = understand_data(df)
            if error:
                st.error(error)
                st.stop()
            st.subheader("Raw Data Understanding")
            st.write("Data Types:", info["dtypes"])
            st.write("Shape:", info["shape"])
            st.write("Null Counts:", info["null_counts"])
            st.write("Duplicates:", info["duplicates"])
            st.write("Basic Stats:", info["stats"])
            st.write("Target Leakage:", info["target_leakage"])
            st.write(df.head())
            st.session_state.step = 2
            progress_bar.progress(0.2)
        
        df = st.session_state.data

        # Step 3: Clean Data
        if st.session_state.step == 2:
            progress_text.text("Step 3: Cleaning data...")
            cleaned_df, error = clean_data(df.copy())
            if error:
                st.error(error)
                st.stop()
            st.session_state.data = cleaned_df
            st.subheader("Cleaned Data")
            st.write(cleaned_df.head())
            st.session_state.step = 3
            progress_bar.progress(0.3)
        
        df = st.session_state.data

        # Step 4: Transform Data
        if st.session_state.step == 3:
            progress_text.text("Step 4: Transforming data...")
            transformed_df, error = transform_data(df.copy())
            if error:
                st.error(error)
                st.stop()
            st.session_state.data = transformed_df
            st.subheader("Transformed Data")
            st.write(transformed_df.head())
            st.session_state.step = 4
            progress_bar.progress(0.4)
        
        df = st.session_state.data

        # Step 5: Structure Data
        if st.session_state.step == 4:
            progress_text.text("Step 5: Structuring data...")
            structured_df, error = structure_data(df.copy())
            if error:
                st.error(error)
                st.stop()
            if 'Salary' in structured_df.columns:
                cols = [col for col in structured_df.columns if col != 'Salary'] + ['Salary']
                structured_df = structured_df[cols]
            st.session_state.data = structured_df
            st.subheader("Structured Data")
            st.write(structured_df.head())
            st.session_state.step = 5
            progress_bar.progress(0.5)
        
        df = st.session_state.data

        # Step 6: Preprocess by Type
        if st.session_state.step == 5:
            progress_text.text("Step 6: Preprocessing by type...")
            preprocessed_df, error = preprocess_by_type(df.copy(), task='regression')
            if error:
                st.error(error)
                st.stop()
            st.session_state.data = preprocessed_df
            st.subheader("Preprocessed Data")
            st.write(preprocessed_df.head())
            st.session_state.step = 6
            progress_bar.progress(0.6)
        
        df = st.session_state.data

        # Step 7: Feature Selection
        if st.session_state.step == 6:
            progress_text.text("Step 7: Selecting features...")
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            selected_df, error = select_features(df.copy())
            if error:
                st.error(error)
                st.stop()
            st.session_state.data = selected_df
            st.subheader("Feature Selected Data")
            st.write(selected_df.head())
            st.session_state.step = 7
            progress_bar.progress(0.7)
        
        df = st.session_state.data

        # Step 8: Handle Zeros
        if st.session_state.step == 7:
            progress_text.text("Step 8: Handling zeros...")
            zero_handled_df, error = handle_zeros(df.copy())
            if error:
                st.error(error)
                st.stop()
            st.session_state.data = zero_handled_df
            st.subheader("Zero-Handled Data")
            st.write(zero_handled_df.head())
            
            csv = zero_handled_df.to_csv(index=False)
            st.download_button(
                label="Download Processed Data",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
            
            st.session_state.step = 8
            progress_bar.progress(0.75)
        
        df = st.session_state.data

        # Step 9: Generate Visualizations
        if st.session_state.step == 8:
            progress_text.text("Step 9: Generating visualizations...")
            vis_files, error = generate_visualizations(df, output_dir='visualizations')
            if error:
                st.error(error)
                st.stop()
            st.subheader("Visualizations")
            for vis_file in vis_files:
                try:
                    with open(vis_file, 'rb') as f:
                        img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode()
                    st.image(f"data:image/png;base64,{img_base64}", caption=vis_file)
                except FileNotFoundError:
                    st.warning(f"Visualization file {vis_file} not found.")
            st.session_state.step = 9
            progress_bar.progress(0.8)
        
        # Step 10: User-Defined Train-Test Split
        if st.session_state.step == 9:
            progress_text.text("Step 10: Configuring train-test split...")
            st.subheader("Configure Train-Test Split")
            train_ratio = st.slider("Training set ratio", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
            val_ratio = st.slider("Validation set ratio", min_value=0.0, max_value=0.3, value=0.15, step=0.05)
            test_ratio = 1.0 - train_ratio - val_ratio
            
            if test_ratio < 0:
                st.error("Invalid split: Train + Validation ratios must leave room for the test set.")
                st.stop()
            if test_ratio < 0.1:
                st.warning("Test set ratio is low (<10%). Consider adjusting for better evaluation.")
            
            st.write(f"Test set ratio: {test_ratio:.2f}")
            
            # Preprocessing pipeline
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns[:-1]
            categorical_cols = df.select_dtypes(include=['category', 'object']).columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_cols),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
                ])
            preprocessor.fit(df.iloc[:, :-1])
            st.session_state.preprocessor = preprocessor
            joblib.dump(preprocessor, 'preprocessor.pkl')
            
            # Split data
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            split_result = dynamic_split(
                X=X,
                y=y,
                strategy='standard',
                task=st.session_state.task,
                train_size=train_ratio,
                val_size=val_ratio,
                test_size=test_ratio,
                random_state=42
            )
            st.session_state.split_data = split_result
            st.subheader("Data Splitting Results")
            st.write("Train Shape:", split_result['X_train'].shape)
            st.write("Validation Shape:", split_result['X_val'].shape if val_ratio > 0 else "No validation set")
            st.write("Test Shape:", split_result['X_test'].shape)
            st.session_state.step = 10
            progress_bar.progress(0.85)
        
        # Step 11: Model Selection and Training
        if st.session_state.step == 10:
            progress_text.text("Step 11: Selecting and training models...")
            X_train = st.session_state.split_data['X_train']
            y_train = st.session_state.split_data['y_train']
            X_val = st.session_state.split_data.get('X_val', None)
            y_val = st.session_state.split_data.get('y_val', None)
            X_test = st.session_state.split_data['X_test']
            y_test = st.session_state.split_data['y_test']
            
            X_train_transformed = st.session_state.preprocessor.transform(pd.DataFrame(X_train, columns=df.columns[:-1]))
            X_test_transformed = st.session_state.preprocessor.transform(pd.DataFrame(X_test, columns=df.columns[:-1]))
            X_val_transformed = st.session_state.preprocessor.transform(pd.DataFrame(X_val, columns=df.columns[:-1])) if X_val is not None else None
            
            model_options = [
                ('Linear Regression', LinearRegression, 'sklearn', {}),
                ('Ridge Regression', Ridge, 'sklearn', {'alpha': 1.0}),
                ('Lasso Regression', Lasso, 'sklearn', {'alpha': 1.0}),
                ('Decision Tree Regressor', DecisionTreeRegressor, 'sklearn', {'max_depth': None, 'random_state': 42}),
                ('Random Forest Regressor', RandomForestRegressor, 'sklearn', {'n_estimators': 100, 'random_state': 42}),
                ('XGBoost Regressor', XGBRegressor, 'sklearn', {'n_estimators': 100, 'random_state': 42}),
                ('LightGBM Regressor', LGBMRegressor, 'sklearn', {'n_estimators': 100, 'random_state': 42}),
                ('Support Vector Regressor', SVR, 'sklearn', {'kernel': 'rbf'}),
                ('Neural Network Regressor', NeuralNetworkRegressor, 'pytorch', {'input_dim': X_train_transformed.shape[1], 'hidden_layers': (64, 32)})
            ]
            
            st.subheader("Select Models to Train")
            selected_models = []
            for model_name, _, _, _ in model_options:
                if st.checkbox(model_name, key=model_name):
                    selected_models.append(model_name)
            
            if not selected_models:
                st.warning("Please select at least one model to train.")
                st.stop()
            
            model_configs = []
            for model_name, model_class, model_type, params in model_options:
                if model_name in selected_models:
                    progress_text.text(f"Training {model_name}...")
                    try:
                        model = model_class(**params)
                        trained_model = train_model(
                            model=model,
                            X_train=X_train_transformed,
                            y_train=y_train,
                            task=st.session_state.task,
                            model_type=model_type,
                            device='cpu',
                            batch_size=32,
                            epochs=50,
                            lr=0.001
                        )
                        eval_data = (X_val_transformed, y_val) if X_val is not None else (X_test_transformed, y_test)
                        eval_results = evaluate_model(
                            model=trained_model,
                            X=eval_data[0],
                            y=eval_data[1],
                            task=st.session_state.task,
                            cv_method='kfold',
                            n_splits=3,
                            random_state=42,
                            model_type=model_type,
                            device='cpu',
                            visualize=False
                        )
                        metric = eval_results.get('test_metrics', {}).get('rmse', np.inf)
                        model_configs.append({
                            'name': model_name,
                            'model': trained_model,
                            'model_type': model_type,
                            'metric': metric
                        })
                        st.write(f"{model_name} trained. {'Validation' if X_val is not None else 'Test'} RMSE: {metric:.4f}")
                    except Exception as e:
                        st.error(f"Error training {model_name}: {str(e)}")
            
            st.session_state.model_configs = model_configs
            st.session_state.step = 11
            progress_bar.progress(0.9)
        
        # Step 12: Final Evaluation
        if st.session_state.step == 11:
            progress_text.text("Step 12: Evaluating models...")
            X_test_transformed = st.session_state.preprocessor.transform(pd.DataFrame(st.session_state.split_data['X_test'], columns=df.columns[:-1]))
            y_test = st.session_state.split_data['y_test']
            
            best_model = None
            best_metric = float('inf')
            best_model_name = None
            best_model_type = None
            
            st.subheader("Model Evaluation Results")
            for config in st.session_state.model_configs:
                model = config['model']
                model_name = config['name']
                model_type = config['model_type']
                
                eval_results = evaluate_model(
                    model=model,
                    X=X_test_transformed,
                    y=y_test,
                    task=st.session_state.task,
                    cv_method='kfold',
                    n_splits=3,
                    random_state=42,
                    model_type=model_type,
                    device='cpu',
                    visualize=True,
                    output_dir='final_plots'
                )
                
                metric = eval_results.get('test_metrics', {}).get('rmse', np.inf)
                st.write(f"{model_name} Test RMSE: {metric:.4f}")
                
                for plot in ['pred_vs_actual.png']:
                    plot_path = os.path.join('final_plots', plot)
                    if os.path.exists(plot_path):
                        with open(plot_path, 'rb') as f:
                            img_data = f.read()
                        img_base64 = base64.b64encode(img_data).decode()
                        st.image(f"data:image/png;base64,{img_base64}", caption=plot)
                
                if metric < best_metric:
                    best_metric = metric
                    best_model = model
                    best_model_name = model_name
                    best_model_type = model_type
            
            st.session_state.best_model = best_model
            st.session_state.model_type = best_model_type
            st.write("Best Model:", best_model_name)
            st.write(f"Best Test RMSE: {best_metric:.4f}")
            
            st.session_state.step = 12
            progress_bar.progress(0.95)
        
        # Step 13: User Query Prediction
        if st.session_state.step == 12:
            progress_text.text("Step 13: Ready for user query prediction...")
            st.subheader("Make a Prediction")
            st.write("Enter feature values to predict Salary.")
            
            feature_names = df.columns[:-1].tolist()
            st.session_state.feature_names = feature_names
            with st.form("prediction_form"):
                query = []
                for feature in feature_names:
                    if feature != 'Unnamed: 0':
                        value = st.number_input(f"Enter {feature}", value=0.0)
                        query.append(value)
                submit = st.form_submit_button("Predict")
                
                if submit:
                    progress_text.text("Processing user query...")
                    result, error = test_model_with_user_query(
                        model=st.session_state.best_model,
                        query=query,
                        task=st.session_state.task,
                        model_type=st.session_state.model_type,
                        preprocessor=st.session_state.preprocessor,
                        device='cpu'
                    )
                    if error:
                        st.error(error)
                    else:
                        st.success("Prediction Results:")
                        st.write("Predicted Salary:", result['prediction'])
                    progress_text.text("Prediction complete")
            
            progress_bar.progress(1.0)
            progress_text.text("Pipeline Complete")
    
    except Exception as e:
        st.error(f"Error in pipeline: {str(e)}")
        progress_bar.progress(0.0)
        progress_text.text("Pipeline Failed")

# Reset button
if st.button("Reset Pipeline"):
    st.session_state.step = 0
    st.session_state.data = None
    st.session_state.preprocessor = None
    st.session_state.best_model = None
    st.session_state.task = None
    st.session_state.model_type = None
    st.session_state.feature_names = None
    st.session_state.split_data = None
    st.session_state.model_configs = None
    st.experimental_rerun()