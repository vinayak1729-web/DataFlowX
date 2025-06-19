# import streamlit as st
# import pandas as pd
# import numpy as np
# import torch
# import joblib
# import os
# import base64
# from data_cleaningPreprocessing import load_data, understand_data, clean_data, transform_data, structure_data, preprocess_by_type, select_features, handle_zeros, generate_visualizations
# from data_selection import dynamic_split
# from model_training import train_model
# from hyperparameter_tuning import tune_hyperparameters
# from model_evaluation import evaluate_model
# from final_model_selection import select_and_train_best_model
# from data_modeling import NeuralNetworkClassifier, NeuralNetworkRegressor
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import matplotlib.pyplot as plt

# # Function to test model with user query (included for completeness)
# def test_model_with_query(model, query, task='classification', model_type='sklearn', preprocessor=None, device='cpu'):
#     """Test a trained model with a user input query."""
#     try:
#         query = np.asarray(query).reshape(1, -1)
#         if preprocessor is not None:
#             query = preprocessor.transform(query)
#         if model_type == 'sklearn':
#             prediction = model.predict(query)[0]
#             probabilities = model.predict_proba(query)[0] if task == 'classification' and hasattr(model, 'predict_proba') else None
#         elif model_type == 'pytorch':
#             model.eval()
#             query_t = torch.tensor(query, dtype=torch.float32).to(device)
#             with torch.no_grad():
#                 output = model(query_t).cpu().numpy().flatten()
#             if task == 'classification':
#                 probabilities = output
#                 prediction = (output > 0.5).astype(int)[0]
#             else:
#                 prediction = output[0]
#                 probabilities = None
#         else:
#             raise ValueError(f"Unsupported model_type: {model_type}")
#         results = {'prediction': prediction}
#         if probabilities is not None:
#             results['probabilities'] = probabilities.tolist()
#         return results, None
#     except Exception as e:
#         return None, f"ERROR: Failed to predict: {str(e)}"

# # Streamlit App
# st.title("Machine Learning Pipeline")
# st.write("Upload a CSV file to process, train models, and make predictions. Follow the step-by-step pipeline with progress tracking.")

# # Initialize session state
# if 'step' not in st.session_state:
#     st.session_state.step = 0
# if 'data' not in st.session_state:
#     st.session_state.data = None
# if 'preprocessor' not in st.session_state:
#     st.session_state.preprocessor = None
# if 'best_model' not in st.session_state:
#     st.session_state.best_model = None
# if 'task' not in st.session_state:
#     st.session_state.task = None
# if 'model_type' not in st.session_state:
#     st.session_state.model_type = None
# if 'feature_names' not in st.session_state:
#     st.session_state.feature_names = None

# # File upload
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     try:
#         # Read file content
#         file_content = uploaded_file.getvalue()
        
#         # Initialize progress bar and text
#         progress_bar = st.progress(0.0)
#         progress_text = st.empty()
#         progress_bar.progress(0.0)
#         progress_text.text("Starting pipeline...")

#         # Step 1: Load Data
#         if st.session_state.step == 0:
#             progress_text.text("Step 1: Loading data...")
#             df, error = load_data(file_content)
#             if error:
#                 st.error(error)
#                 st.stop()
#             st.session_state.data = df
#             st.session_state.step = 1
#             progress_bar.progress(0.1)
        
#         df = st.session_state.data

#         # Step 2: Understand Data
#         if st.session_state.step == 1:
#             progress_text.text("Step 2: Understanding data...")
#             info, error = understand_data(df)
#             if error:
#                 st.error(error)
#                 st.stop()
#             st.subheader("Raw Data Understanding")
#             st.write("Data Types:", info["dtypes"])
#             st.write("Shape:", info["shape"])
#             st.write("Null Counts:", info["null_counts"])
#             st.write("Duplicates:", info["duplicates"])
#             st.write("Basic Stats:", info["stats"])
#             st.write("Target Leakage:", info["target_leakage"])
#             st.write(df.head())
#             st.session_state.step = 2
#             progress_bar.progress(0.2)

#         # Step 3: Clean Data
#         if st.session_state.step == 2:
#             progress_text.text("Step 3: Cleaning data...")
#             cleaned_df, error = clean_data(df.copy())
#             if error:
#                 st.error(error)
#                 st.stop()
#             st.session_state.data = cleaned_df
#             st.subheader("Cleaned Data")
#             st.write(cleaned_df.head())
#             st.session_state.step = 3
#             progress_bar.progress(0.3)

#         df = st.session_state.data

#         # Step 4: Transform Data
#         if st.session_state.step == 3:
#             progress_text.text("Step 4: Transforming data...")
#             transformed_df, error = transform_data(df.copy())
#             if error:
#                 st.error(error)
#                 st.stop()
#             st.session_state.data = transformed_df
#             st.subheader("Transformed Data")
#             st.write(transformed_df.head())
#             st.session_state.step = 4
#             progress_bar.progress(0.4)

#         df = st.session_state.data

#         # Step 5: Structure Data
#         if st.session_state.step == 4:
#             progress_text.text("Step 5: Structuring data...")
#             structured_df, error = structure_data(df.copy())
#             if error:
#                 st.error(error)
#                 st.stop()
#             st.session_state.data = structured_df
#             st.subheader("Structured Data")
#             st.write(structured_df.head())
#             st.session_state.step = 5
#             progress_bar.progress(0.5)

#         df = st.session_state.data

#         # Step 6: Preprocess by Type
#         if st.session_state.step == 5:
#             progress_text.text("Step 6: Preprocessing by type...")
#             preprocessed_df, error = preprocess_by_type(df.copy())
#             if error:
#                 st.error(error)
#                 st.stop()
#             st.session_state.data = preprocessed_df
#             st.subheader("Preprocessed Data")
#             st.write(preprocessed_df.head())
#             st.session_state.step = 6
#             progress_bar.progress(0.6)

#         df = st.session_state.data

#         # Step 7: Feature Selection
#         if st.session_state.step == 6:
#             progress_text.text("Step 7: Selecting features...")
#             selected_df, error = select_features(df.copy())
#             if error:
#                 st.error(error)
#                 st.stop()
#             st.session_state.data = selected_df
#             st.subheader("Feature Selected Data")
#             st.write(selected_df.head())
#             st.session_state.step = 7
#             progress_bar.progress(0.7)

#         df = st.session_state.data

#         # Step 8: Handle Zeros
#         if st.session_state.step == 7:
#             progress_text.text("Step 8: Handling zeros...")
#             zero_handled_df, error = handle_zeros(df.copy())
#             if error:
#                 st.error(error)
#                 st.stop()
#             st.session_state.data = zero_handled_df
#             st.subheader("Zero-Handled Data")
#             st.write(zero_handled_df.head())
            
#             # Save processed data for download
#             csv = zero_handled_df.to_csv(index=False)
#             st.download_button(
#                 label="Download Processed Data",
#                 data=csv,
#                 file_name="processed_data.csv",
#                 mime="text/csv"
#             )
#             st.session_state.step = 8
#             progress_bar.progress(0.75)

#         df = st.session_state.data

#         # Step 9: Generate Visualizations
#         if st.session_state.step == 8:
#             progress_text.text("Step 9: Generating visualizations...")
#             vis_files, error = generate_visualizations(df, output_dir="visualizations")
#             if error:
#                 st.error(error)
#                 st.stop()
#             st.subheader("Visualizations")
#             for vis_file in vis_files:
#                 try:
#                     with open(vis_file, "rb") as f:
#                         img_data = f.read()
#                     img_base64 = base64.b64encode(img_data).decode()
#                     st.image(f"data:image/png;base64,{img_base64}", caption=vis_file)
#                 except FileNotFoundError:
#                     st.warning(f"Visualization file {vis_file} not found.")
#             st.session_state.step = 9
#             progress_bar.progress(0.8)

#         # Step 10: Data Splitting
#         if st.session_state.step == 9:
#             progress_text.text("Step 10: Splitting data...")
#             # Determine task (classification or regression)
#             y = df.iloc[:, -1]
#             is_classification = y.dtype in ['int64', 'category', 'object'] or (y.dtype == 'float64' and y.nunique() < 10 and y.apply(lambda x: x.is_integer()).all())
#             st.session_state.task = 'classification' if is_classification else 'regression'
            
#             # Prepare preprocessing pipeline
#             numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns[:-1]  # Exclude target
#             categorical_cols = df.select_dtypes(include=['category', 'object']).columns
#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ('num', StandardScaler(), numeric_cols),
#                     ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
#                 ])
#             preprocessor.fit(df.iloc[:, :-1])
#             st.session_state.preprocessor = preprocessor
#             joblib.dump(preprocessor, 'preprocessor.pkl')
            
#             # Split data
#             X = df.iloc[:, :-1].values
#             y = df.iloc[:, -1].values
#             split_result = dynamic_split(
#                 X=X,
#                 y=y,
#                 strategy='standard',
#                 task=st.session_state.task,
#                 train_size=0.7,
#                 val_size=0.15,
#                 test_size=0.15,
#                 random_state=42
#             )
#             st.session_state.split_data = split_result
#             st.subheader("Data Splitting")
#             st.write("Train Shape:", split_result['X_train'].shape)
#             st.write("Validation Shape:", split_result['X_val'].shape)
#             st.write("Test Shape:", split_result['X_test'].shape)
#             st.session_state.step = 10
#             progress_bar.progress(0.85)

#         # Step 11: Model Training and Hyperparameter Tuning
#         if st.session_state.step == 10:
#             progress_text.text("Step 11: Training and tuning models...")
#             X_train_val = np.vstack([st.session_state.split_data['X_train'], st.session_state.split_data['X_val']])
#             y_train_val = np.concatenate([st.session_state.split_data['y_train'], st.session_state.split_data['y_val']])
#             X_train_val_transformed = st.session_state.preprocessor.transform(pd.DataFrame(X_train_val, columns=df.columns[:-1]))
            
#             # Define models
#             model_configs = [
#                 {
#                     'name': 'random_forest',
#                     'model': RandomForestClassifier(random_state=42) if st.session_state.task == 'classification' else RandomForestRegressor(random_state=42),
#                     'task': st.session_state.task,
#                     'model_type': 'sklearn',
#                     'param_grid': {
#                         'n_estimators': [50, 100],
#                         'max_depth': [None, 10]
#                     }
#                 },
#                 {
#                     'name': 'neural_network',
#                     'model': NeuralNetworkClassifier(input_dim=X_train_val_transformed.shape[1], hidden_layers=(64, 32)) if st.session_state.task == 'classification' else NeuralNetworkRegressor(input_dim=X_train_val_transformed.shape[1], hidden_layers=(64, 32)),
#                     'task': st.session_state.task,
#                     'model_type': 'pytorch',
#                     'param_grid': {
#                         'lr': (0.0001, 0.01),
#                         'batch_size': [16, 32]
#                     }
#                 }
#             ]
            
#             # Tune hyperparameters
#             tuned_configs = []
#             for config in model_configs:
#                 progress_text.text(f"Tuning {config['name']}...")
#                 results = tune_hyperparameters(
#                     model=config['model'],
#                     X=X_train_val_transformed,
#                     y=y_train_val,
#                     param_grid=config['param_grid'],
#                     method='grid' if config['model_type'] == 'sklearn' else 'bayesian',
#                     task=config['task'],
#                     cv_method='stratified_kfold' if config['task'] == 'classification' else 'kfold',
#                     n_splits=3,
#                     n_iter=5 if config['model_type'] == 'pytorch' else None,
#                     model_type=config['model_type'],
#                     output_file=f"{config['name']}_best_params.json"
#                 )
#                 config['params'] = results['best_params']
#                 config['kwargs'] = {'batch_size': results['best_params'].get('batch_size', 32), 'epochs': 10} if config['model_type'] == 'pytorch' else {}
#                 tuned_configs.append(config)
#                 st.write(f"{config['name']} Best Params:", results['best_params'])
            
#             st.session_state.model_configs = tuned_configs
#             st.session_state.step = 11
#             progress_bar.progress(0.9)

#         # Step 12: Model Selection and Final Training
#         if st.session_state.step == 11:
#             progress_text.text("Step 12: Selecting and training final model...")
#             X_train_val = np.vstack([st.session_state.split_data['X_train'], st.session_state.split_data['X_val']])
#             y_train_val = np.concatenate([st.session_state.split_data['y_train'], st.session_state.split_data['y_val']])
#             X_test = st.session_state.split_data['X_test']
#             y_test = st.session_state.split_data['y_test']
            
#             X_train_val_transformed = st.session_state.preprocessor.transform(pd.DataFrame(X_train_val, columns=df.columns[:-1]))
#             X_test_transformed = st.session_state.preprocessor.transform(pd.DataFrame(X_test, columns=df.columns[:-1]))
            
#             results = select_and_train_best_model(
#                 model_configs=st.session_state.model_configs,
#                 X_train_val=X_train_val_transformed,
#                 y_train_val=y_train_val,
#                 X_test=X_test_transformed,
#                 y_test=y_test,
#                 task=st.session_state.task,
#                 cv_method='stratified_kfold' if st.session_state.task == 'classification' else 'kfold',
#                 n_splits=3,
#                 output_dir='final_plots'
#             )
#             st.session_state.best_model = results['best_model']
#             st.session_state.model_type = next(config['model_type'] for config in st.session_state.model_configs if config['name'] == results['best_model_name'])
#             st.subheader("Model Selection")
#             st.write("Best Model:", results['best_model_name'])
#             st.write("Final Metrics:", results['final_results'])
            
#             # Display final evaluation plots
#             for plot in ['roc_curve.png', 'pr_curve.png', 'confusion_matrix.png', 'pred_vs_actual.png']:
#                 plot_path = os.path.join('final_plots', plot)
#                 if os.path.exists(plot_path):
#                     with open(plot_path, "rb") as f:
#                         img_data = f.read()
#                     img_base64 = base64.b64encode(img_data).decode()
#                     st.image(f"data:image/png;base64,{img_base64}", caption=plot)
            
#             st.session_state.step = 12
#             progress_bar.progress(0.95)

#         # Step 13: User Query Prediction
#         if st.session_state.step == 12:
#             progress_text.text("Step 13: Ready for user query prediction...")
#             st.subheader("Make a Prediction")
#             st.write("Enter feature values to predict the target using the trained model.")
            
#             # Create input form for user query
#             feature_names = df.columns[:-1].tolist()
#             st.session_state.feature_names = feature_names
#             with st.form("prediction_form"):
#                 query = []
#                 for feature in feature_names:
#                     value = st.number_input(f"Enter {feature}", value=0.0)
#                     query.append(value)
#                 submit = st.form_submit_button("Predict")
                
#                 if submit:
#                     progress_text.text("Processing user query...")
#                     result, error = test_model_with_query(
#                         model=st.session_state.best_model,
#                         query=query,
#                         task=st.session_state.task,
#                         model_type=st.session_state.model_type,
#                         preprocessor=st.session_state.preprocessor,
#                         device='cpu'
#                     )
#                     if error:
#                         st.error(error)
#                     else:
#                         st.success("Prediction Results:")
#                         st.write("Prediction:", result['prediction'])
#                         if 'probabilities' in result:
#                             st.write("Probabilities:", result['probabilities'])
#                     progress_text.text("Prediction complete")
            
#             progress_bar.progress(1.0)
#             progress_text.text("Pipeline Complete")

#     except Exception as e:
#         st.error(f"Error in pipeline: {str(e)}")
#         progress_bar.progress(0.0)
#         progress_text.text("Pipeline Failed")

# # Reset button
# if st.button("Reset Pipeline"):
#     st.session_state.step = 0
#     st.session_state.data = None
#     st.session_state.preprocessor = None
#     st.session_state.best_model = None
#     st.session_state.task = None
#     st.session_state.model_type = None
#     st.session_state.feature_names = None
#     st.session_state.split_data = None
#     st.session_state.model_configs = None
#     st.experimental_rerun()


import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import base64
from data_cleaningPreprocessing import load_data, understand_data, clean_data, transform_data, structure_data, preprocess_by_type, select_features, handle_zeros, generate_visualizations
from data_selection import dynamic_split
from model_training import train_model
from model_evaluation import evaluate_model
from final_model_selection import select_and_train_best_model
from data_modeling import (
    train_linear_regression, train_ridge_regression, train_lasso_regression,
    train_decision_tree_regressor, train_random_forest_regressor, train_xgboost_regressor,
    train_lightgbm_regressor, train_svr, train_neural_network_regressor,
    train_logistic_regression, train_knn_classifier, train_decision_tree_classifier,
    train_random_forest_classifier, train_xgboost_classifier, train_lightgbm_classifier,
    train_catboost_classifier, train_svm_classifier, train_naive_bayes_classifier,
    train_neural_network_classifier, evaluate_regression_model, evaluate_classification_model
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Function to test model with user query
def test_model_with_query(model, query, task='classification', model_type='sklearn', preprocessor=None, device='cpu'):
    """Test a trained model with a user input query."""
    try:
        query = np.asarray(query).reshape(1, -1)
        if preprocessor is not None:
            query = preprocessor.transform(query)
        if model_type == 'sklearn':
            prediction = model.predict(query)[0]
            probabilities = model.predict_proba(query)[0] if task == 'classification' and hasattr(model, 'predict_proba') else None
        elif model_type == 'pytorch':
            model.eval()
            query_t = torch.tensor(query, dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(query_t).cpu().numpy().flatten()
            if task == 'classification':
                probabilities = output
                prediction = (output > 0.5).astype(int)[0]
            else:
                prediction = output[0]
                probabilities = None
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        results = {'prediction': prediction}
        if probabilities is not None:
            results['probabilities'] = probabilities.tolist()
        return results, None
    except Exception as e:
        return None, f"ERROR: Failed to predict: {str(e)}"

# Streamlit App
st.title("Machine Learning Pipeline")
st.write("Upload a CSV file to process, train models, and make predictions. Follow the step-by-step pipeline with progress tracking.")

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
    st.session_state.task = None
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
            st.session_state.data = structured_df
            st.subheader("Structured Data")
            st.write(structured_df.head())
            st.session_state.step = 5
            progress_bar.progress(0.5)

        df = st.session_state.data

        # Step 6: Preprocess by Type
        if st.session_state.step == 5:
            progress_text.text("Step 6: Preprocessing by type...")
            preprocessed_df, error = preprocess_by_type(df.copy())
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
            
            # Save processed data for download
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
            vis_files, error = generate_visualizations(df, output_dir="visualizations")
            if error:
                st.error(error)
                st.stop()
            st.subheader("Visualizations")
            for vis_file in vis_files:
                try:
                    with open(vis_file, "rb") as f:
                        img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode()
                    st.image(f"data:image/png;base64,{img_base64}", caption=vis_file)
                except FileNotFoundError:
                    st.warning(f"Visualization file {vis_file} not found.")
            st.session_state.step = 9
            progress_bar.progress(0.8)

        # Step 10: Data Splitting
        if st.session_state.step == 9:
            progress_text.text("Step 10: Splitting data...")
            # Determine task (classification or regression)
            y = df.iloc[:, -1]
            is_classification = y.dtype in ['int64', 'category', 'object'] or (y.dtype == 'float64' and y.nunique() < 10 and y.apply(lambda x: x.is_integer()).all())
            st.session_state.task = 'classification' if is_classification else 'regression'
            
            # Prepare preprocessing pipeline
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns[:-1]  # Exclude target
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
                train_size=0.7,
                val_size=0.15,
                test_size=0.15,
                random_state=42
            )
            st.session_state.split_data = split_result
            st.subheader("Data Splitting")
            st.write("Train Shape:", split_result['X_train'].shape)
            st.write("Validation Shape:", split_result['X_val'].shape)
            st.write("Test Shape:", split_result['X_test'].shape)
            st.session_state.step = 10
            progress_bar.progress(0.85)

        # Step 11: Model Selection and Training
        if st.session_state.step == 10:
            progress_text.text("Step 11: Selecting and training models...")
            X_train = st.session_state.split_data['X_train']
            y_train = st.session_state.split_data['y_train']
            X_val = st.session_state.split_data['X_val']
            y_val = st.session_state.split_data['y_val']
            X_test = st.session_state.split_data['X_test']
            y_test = st.session_state.split_data['y_test']
            
            # Transform data
            X_train_transformed = st.session_state.preprocessor.transform(pd.DataFrame(X_train, columns=df.columns[:-1]))
            X_val_transformed = st.session_state.preprocessor.transform(pd.DataFrame(X_val, columns=df.columns[:-1]))
            X_test_transformed = st.session_state.preprocessor.transform(pd.DataFrame(X_test, columns=df.columns[:-1]))
            
            # Define model checklist based on task
            if st.session_state.task == 'regression':
                model_options = [
                    ('Linear Regression', train_linear_regression, 'sklearn', {}),
                    ('Ridge Regression', train_ridge_regression, 'sklearn', {'alpha': 1.0}),
                    ('Lasso Regression', train_lasso_regression, 'sklearn', {'alpha': 1.0}),
                    ('Decision Tree Regressor', train_decision_tree_regressor, 'sklearn', {'max_depth': None, 'random_state': 42}),
                    ('Random Forest Regressor', train_random_forest_regressor, 'sklearn', {'n_estimators': 100, 'random_state': 42}),
                    ('XGBoost Regressor', train_xgboost_regressor, 'sklearn', {'n_estimators': 100, 'random_state': 42}),
                    ('LightGBM Regressor', train_lightgbm_regressor, 'sklearn', {'n_estimators': 100, 'random_state': 42}),
                    ('Support Vector Regressor', train_svr, 'sklearn', {'kernel': 'rbf'}),
                    ('Neural Network Regressor', train_neural_network_regressor, 'pytorch', {'hidden_layers': (64, 32), 'epochs': 50, 'batch_size': 32, 'device': 'cpu'})
                ]
            else:  # Classification
                model_options = [
                    ('Logistic Regression', train_logistic_regression, 'sklearn', {'random_state': 42}),
                    ('k-Nearest Neighbors', train_knn_classifier, 'sklearn', {'n_neighbors': 5}),
                    ('Decision Tree Classifier', train_decision_tree_classifier, 'sklearn', {'max_depth': None, 'random_state': 42}),
                    ('Random Forest Classifier', train_random_forest_classifier, 'sklearn', {'n_estimators': 100, 'random_state': 42}),
                    ('XGBoost Classifier', train_xgboost_classifier, 'sklearn', {'n_estimators': 100, 'random_state': 42}),
                    ('LightGBM Classifier', train_lightgbm_classifier, 'sklearn', {'n_estimators': 100, 'random_state': 42}),
                    ('CatBoost Classifier', train_catboost_classifier, 'sklearn', {'n_estimators': 100, 'random_state': 42}),
                    ('Support Vector Classifier', train_svm_classifier, 'sklearn', {'kernel': 'rbf', 'random_state': 42}),
                    ('Naive Bayes Classifier', train_naive_bayes_classifier, 'sklearn', {}),
                    ('Neural Network Classifier', train_neural_network_classifier, 'pytorch', {'hidden_layers': (64, 32), 'epochs': 50, 'batch_size': 32, 'device': 'cpu'})
                ]
            
            # Model selection checklist
            st.subheader("Select Models to Train")
            selected_models = []
            for model_name, _, _, _ in model_options:
                if st.checkbox(model_name, key=model_name):
                    selected_models.append(model_name)
            
            if not selected_models:
                st.warning("Please select at least one model to train.")
                st.stop()
            
            # Train selected models
            model_configs = []
            for model_name, train_func, model_type, params in model_options:
                if model_name in selected_models:
                    progress_text.text(f"Training {model_name}...")
                    try:
                        if model_type == 'pytorch':
                            # Pass input_dim for NeuralNetwork models
                            params['input_dim'] = X_train_transformed.shape[1]
                            model = train_func(X_train_transformed, y_train, **params)
                            # Evaluate on validation set
                            if st.session_state.task == 'regression':
                                metric = evaluate_regression_model(model, X_val_transformed, y_val, is_pytorch_model=True, device='cpu')
                            else:
                                metric = evaluate_classification_model(model, X_val_transformed, y_val, is_pytorch_model=True, device='cpu')['accuracy']
                        else:
                            model = train_func(X_train_transformed, y_train, **params)
                            if st.session_state.task == 'regression':
                                metric = evaluate_regression_model(model, X_val_transformed, y_val)
                            else:
                                metric = evaluate_classification_model(model, X_val_transformed, y_val)['accuracy']
                        
                        model_configs.append({
                            'name': model_name,
                            'model': model,
                            'model_type': model_type,
                            'metric': metric
                        })
                        st.write(f"{model_name} trained. Validation {'RMSE' if st.session_state.task == 'regression' else 'Accuracy'}: {metric:.4f}")
                    except Exception as e:
                        st.error(f"Error training {model_name}: {str(e)}")
            
            st.session_state.model_configs = model_configs
            st.session_state.step = 11
            progress_bar.progress(0.9)

        # Step 12: Model Selection and Final Evaluation
        if st.session_state.step == 11:
            progress_text.text("Step 12: Selecting and evaluating final model...")
            X_test_transformed = st.session_state.preprocessor.transform(pd.DataFrame(st.session_state.split_data['X_test'], columns=df.columns[:-1]))
            y_test = st.session_state.split_data['y_test']
            
            # Evaluate all models on test set
            best_model = None
            best_metric = float('inf') if st.session_state.task == 'regression' else -float('inf')
            best_model_name = None
            best_model_type = None
            
            st.subheader("Model Evaluation Results")
            for config in st.session_state.model_configs:
                model = config['model']
                model_name = config['name']
                model_type = config['model_type']
                
                if st.session_state.task == 'regression':
                    metric = evaluate_regression_model(model, X_test_transformed, y_test, is_pytorch_model=(model_type == 'pytorch'), device='cpu')
                    st.write(f"{model_name} Test RMSE: {metric:.4f}")
                    if metric < best_metric:
                        best_metric = metric
                        best_model = model
                        best_model_name = model_name
                        best_model_type = model_type
                else:
                    metrics = evaluate_classification_model(model, X_test_transformed, y_test, is_pytorch_model=(model_type == 'pytorch'), device='cpu')
                    metric = metrics['accuracy']
                    st.write(f"{model_name} Test Accuracy: {metric:.4f}")
                    st.write(f"Classification Report:\n{metrics['classification_report']}")
                    if metric > best_metric:
                        best_metric = metric
                        best_model = model
                        best_model_name = model_name
                        best_model_type = model_type
            
            st.session_state.best_model = best_model
            st.session_state.model_type = best_model_type
            st.write("Best Model:", best_model_name)
            st.write(f"Best Test {'RMSE' if st.session_state.task == 'regression' else 'Accuracy'}: {best_metric:.4f}")
            
            st.session_state.step = 12
            progress_bar.progress(0.95)

        # Step 13: User Query Prediction
        if st.session_state.step == 12:
            progress_text.text("Step 13: Ready for user query prediction...")
            st.subheader("Make a Prediction")
            st.write("Enter feature values to predict the target using the trained model.")
            
            # Create input form for user query
            feature_names = df.columns[:-1].tolist()
            st.session_state.feature_names = feature_names
            with st.form("prediction_form"):
                query = []
                for feature in feature_names:
                    value = st.number_input(f"Enter {feature}", value=0.0)
                    query.append(value)
                submit = st.form_submit_button("Predict")
                
                if submit:
                    progress_text.text("Processing user query...")
                    result, error = test_model_with_query(
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
                        st.write("Prediction:", result['prediction'])
                        if 'probabilities' in result:
                            st.write("Probabilities:", result['probabilities'])
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