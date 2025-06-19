# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# from data_cleaningPreprocessing import load_data, understand_data, clean_data, transform_data, structure_data, preprocess_by_type, select_features, handle_zeros, generate_visualizations
# from data_selection import dynamic_split
# from data_modeling import train_random_forest_classifier, train_xgboost_classifier, train_neural_network_classifier, train_random_forest_regressor, train_xgboost_regressor, train_neural_network_regressor
# from model_training import train_model
# from hyperparameter_tuning import tune_hyperparameters
# from final_model_selection import select_and_train_best_model
# from model_evaluation import evaluate_model
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
# import base64

# # Set up directories
# OUTPUT_DIR = "outputs"
# PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(PLOT_DIR, exist_ok=True)

# # Streamlit app layout
# st.set_page_config(page_title="Data Science Management Service", layout="wide")
# st.title("Data Science Management Service")
# st.markdown("Upload a CSV dataset to perform automated data preprocessing, model training, hyperparameter tuning, and evaluation.")

# # Sidebar for configuration
# st.sidebar.header("Configuration")
# task = st.sidebar.selectbox("Task Type", ["classification", "regression"], help="Select whether the task is classification or regression")
# split_strategy = st.sidebar.selectbox("Data Split Strategy", ["standard", "kfold", "stratified_kfold", "time_series"], help="Choose how to split the dataset")
# cv_method = st.sidebar.selectbox("Cross-Validation Method", ["kfold", "stratified_kfold", "time_series"], help="Choose cross-validation method for model evaluation")
# n_splits = st.sidebar.slider("Number of CV Folds", 2, 10, 5, help="Number of folds for cross-validation")
# random_state = st.sidebar.number_input("Random Seed", value=42, help="Random seed for reproducibility")

# # File upload
# uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], help="Upload a CSV file with features and a target column (last column)")
# if uploaded_file is not None:
#     # Load and validate data
#     with st.spinner("Loading dataset..."):
#         file_content = uploaded_file.read()
#         df, error = load_data(file_content)
#         if error:
#             st.error(error)
#             st.stop()
#         st.success("Dataset loaded successfully!")

#     # Stage 1: Data Understanding
#     st.header("1. Data Understanding")
#     info, error = understand_data(df)
#     if error:
#         st.error(error)
#         st.stop()
#     st.write("**Dataset Info**")
#     st.write(f"Shape: {info['shape']}")
#     st.write(f"Null Counts: {info['null_counts']}")
#     st.write(f"Duplicates: {info['duplicates']}")
#     st.write("**Data Types**")
#     st.write(info['dtypes'])
#     st.write("**Summary Statistics**")
#     st.write(pd.DataFrame(info['stats']))

#     # Generate and display visualizations
#     with st.spinner("Generating visualizations..."):
#         vis_files, error = generate_visualizations(df, output_dir=PLOT_DIR)
#         if error:
#             st.error(error)
#         else:
#             st.subheader("Visualizations")
#             for file in vis_files:
#                 st.image(file, caption=os.path.basename(file), use_column_width=True)

#     # Stage 2: Data Cleaning
#     st.header("2. Data Cleaning")
#     with st.spinner("Cleaning data..."):
#         df_cleaned, error = clean_data(df.copy())
#         if error:
#             st.error(error)
#             st.stop()
#         st.success("Data cleaned successfully!")
#         st.write("**Cleaned Dataset Preview**")
#         st.write(df_cleaned.head())

#     # Stage 3: Data Transformation
#     st.header("3. Data Transformation")
#     with st.spinner("Transforming data..."):
#         df_transformed, error = transform_data(df_cleaned.copy())
#         if error:
#             st.error(error)
#             st.stop()
#         st.success("Data transformed successfully!")
#         st.write("**Transformed Dataset Preview**")
#         st.write(df_transformed.head())

#     # Stage 4: Data Structuring
#     st.header("4. Data Structuring")
#     with st.spinner("Structuring data..."):
#         df_structured, error = structure_data(df_transformed.copy())
#         if error:
#             st.error(error)
#             st.stop()
#         st.success("Data structured successfully!")
#         st.write("**Structured Dataset Preview**")
#         st.write(df_structured.head())

#     # Stage 5: Preprocessing by Type
#     st.header("5. Preprocessing by Type")
#     with st.spinner("Preprocessing data by type..."):
#         df_preprocessed, error = preprocess_by_type(df_structured.copy())
#         if error:
#             st.error(error)
#             st.stop()
#         st.success("Data preprocessed successfully!")
#         st.write("**Preprocessed Dataset Preview**")
#         st.write(df_preprocessed.head())

#     # Stage 6: Feature Selection
#     st.header("6. Feature Selection")
#     with st.spinner("Selecting features..."):
#         df_selected, error = select_features(df_preprocessed.copy())
#         if error:
#             st.error(error)
#             st.stop()
#         st.success("Features selected successfully!")
#         st.write("**Selected Features Dataset Preview**")
#         st.write(df_selected.head())

#     # Stage 7: Handle Zeros
#     st.header("7. Handle Invalid Zeros")
#     with st.spinner("Handling invalid zeros..."):
#         df_final, error = handle_zeros(df_selected.copy())
#         if error:
#             st.error(error)
#             st.stop()
#         st.success("Invalid zeros handled successfully!")
#         st.write("**Final Dataset Preview**")
#         st.write(df_final.head())

#     # Prepare data for modeling
#     X = df_final.iloc[:, :-1].to_numpy()
#     y = df_final.iloc[:, -1].to_numpy()

#     # Stage 8: Data Splitting
#     st.header("8. Data Splitting")
#     with st.spinner("Splitting data..."):
#         split_result = dynamic_split(X, y, strategy=split_strategy, task=task, n_splits=n_splits, random_state=random_state)
#         if split_strategy == "standard":
#             X_train, X_val, X_test, y_train, y_val, y_test = (
#                 split_result["X_train"], split_result["X_val"], split_result["X_test"],
#                 split_result["y_train"], split_result["y_val"], split_result["y_test"]
#             )
#             X_train_val = np.vstack([X_train, X_val])
#             y_train_val = np.concatenate([y_train, y_val])
#             st.write(f"Train Shape: {X_train.shape}, Validation Shape: {X_val.shape}, Test Shape: {X_test.shape}")
#         else:
#             folds = split_result["folds"]
#             X_train_val, y_train_val = X, y
#             X_test, y_test = None, None
#             st.write(f"Cross-Validation Folds: {len(folds)}")

#     # Stage 9: Model Configuration
#     st.header("9. Model Training and Selection")
#     model_configs = [
#         {
#             "name": "random_forest",
#             "model": train_random_forest_classifier(X_train_val, y_train_val) if task == "classification" else train_random_forest_regressor(X_train_val, y_train_val),
#             "task": task,
#             "model_type": "sklearn",
#             "params": {"n_estimators": 100, "max_depth": 10}
#         },
#         {
#             "name": "xgboost",
#             "model": train_xgboost_classifier(X_train_val, y_train_val) if task == "classification" else train_xgboost_regressor(X_train_val, y_train_val),
#             "task": task,
#             "model_type": "sklearn",
#             "params": {"n_estimators": 100, "max_depth": 3}
#         },
#         {
#             "name": "neural_network",
#             "model": train_neural_network_classifier(X_train_val, y_train_val, device="cpu") if task == "classification" else train_neural_network_regressor(X_train_val, y_train_val, device="cpu"),
#             "task": task,
#             "model_type": "pytorch",
#             "params": {"lr": 0.001, "batch_size": 32},
#             "kwargs": {"epochs": 10}
#         }
#     ]

#     # Stage 10: Hyperparameter Tuning
#     st.header("10. Hyperparameter Tuning")
#     param_grids = {
#         "random_forest": {
#             "n_estimators": [50, 100, 200],
#             "max_depth": [None, 10, 20]
#         },
#         "xgboost": {
#             "n_estimators": [50, 100, 200],
#             "max_depth": [3, 5, 7]
#         },
#         "neural_network": {
#             "lr": (0.0001, 0.01),
#             "batch_size": [16, 32, 64]
#         }
#     }

#     tuned_configs = []
#     for config in model_configs:
#         model_name = config["name"]
#         with st.spinner(f"Tuning hyperparameters for {model_name}..."):
#             tuned_params = tune_hyperparameters(
#                 model=config["model"],
#                 X=X_train_val,
#                 y=y_train_val,
#                 param_grid=param_grids[model_name],
#                 method="bayesian",
#                 task=task,
#                 cv_method=cv_method,
#                 n_splits=n_splits,
#                 random_state=random_state,
#                 model_type=config["model_type"],
#                 device="cpu",
#                 n_iter=5,
#                 output_file=os.path.join(OUTPUT_DIR, f"{model_name}_best_params.json")
#             )
#             config["params"] = tuned_params["best_params"]
#             st.write(f"{model_name} Best Params: {tuned_params['best_params']}, Best Score: {tuned_params['best_score']:.4f}")
#             tuned_configs.append(config)

#     # Stage 11: Model Selection and Final Training
#     st.header("11. Model Selection and Final Training")
#     with st.spinner("Selecting and training the best model..."):
#         results = select_and_train_best_model(
#             model_configs=tuned_configs,
#             X_train_val=X_train_val,
#             y_train_val=y_train_val,
#             X_test=X_test,
#             y_test=y_test,
#             task=task,
#             cv_method=cv_method,
#             n_splits=n_splits,
#             random_state=random_state,
#             device="cpu",
#             output_dir=PLOT_DIR
#         )
#         st.success(f"Best Model: {results['best_model_name']}")
#         st.write("**Final Metrics**")
#         st.write(results["final_results"])

#     # Display evaluation plots
#     st.subheader("Evaluation Visualizations")
#     for file in os.listdir(PLOT_DIR):
#         if file.startswith(("roc_curve", "pr_curve", "confusion_matrix", "pred_vs_actual")):
#             st.image(os.path.join(PLOT_DIR, file), caption=file, use_column_width=True)

#     # Save results
#     st.write("**Results Saved**")
#     st.write(f"Best model parameters: {os.path.join(OUTPUT_DIR, f'{results['best_model_name']}_best_params.json')}")
#     st.write(f"Evaluation plots: {PLOT_DIR}")
# else:
#     st.info("Please upload a CSV file to start the data science pipeline.")

import streamlit as st
import pandas as pd
import numpy as np
import os
from data_cleaningPreprocessing import load_data, understand_data, clean_data, transform_data, structure_data, preprocess_by_type, select_features, handle_zeros, generate_visualizations
from data_selection import dynamic_split
from data_modeling import (
    # Regression
    train_linear_regression, train_ridge_regression, train_lasso_regression,
    train_decision_tree_regressor, train_random_forest_regressor, train_xgboost_regressor,
    train_lightgbm_regressor, train_svr, train_neural_network_regressor,
    # Classification
    train_logistic_regression, train_knn_classifier, train_decision_tree_classifier,
    train_random_forest_classifier, train_xgboost_classifier, train_lightgbm_classifier,
    train_catboost_classifier, train_svm_classifier, train_naive_bayes_classifier,
    train_neural_network_classifier,
    # Clustering
    train_kmeans, train_hierarchical_clustering, train_dbscan,
    # Dimensionality Reduction
    train_pca, train_tsne, train_umap, train_autoencoder,
    # Semi-Supervised
    train_self_training, train_label_propagation,
    # Reinforcement Learning
    train_q_learning, train_multi_armed_bandit
)
from model_training import train_model
from hyperparameter_tuning import tune_hyperparameters
from final_model_selection import select_and_train_best_model
from model_evaluation import evaluate_model
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Set up directories
OUTPUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Streamlit app layout
st.set_page_config(page_title="Data Science Management Service", layout="wide")
st.title("Data Science Management Service")
st.markdown("Upload a CSV dataset to perform automated data preprocessing, model training, hyperparameter tuning, and evaluation.")

# Initialize session state
if 'split_confirmed' not in st.session_state:
    st.session_state['split_confirmed'] = False
if 'models_trained' not in st.session_state:
    st.session_state['models_trained'] = {}

# Sidebar for configuration
st.sidebar.header("Configuration")

# Task type selection
task_options = [
    "Regression (Supervised)",
    "Classification (Supervised)",
    "Clustering (Unsupervised)",
    "Dimensionality Reduction (Unsupervised)",
    "Semi-Supervised",
    "Reinforcement Learning"
]
task = st.sidebar.selectbox("Task Type", task_options, help="Select the machine learning task")
task = task.lower().split()[0]  # Extract 'regression', 'classification', etc.

# Train-test-validation split
st.sidebar.subheader("Train-Test-Validation Split")
train_ratio = st.sidebar.slider("Training Set Ratio", 0.5, 0.9, 0.7, 0.05, help="Proportion of data for training")
val_ratio = st.sidebar.slider("Validation Set Ratio", 0.0, 0.3, 0.15, 0.05, help="Proportion of data for validation")
test_ratio = 1.0 - train_ratio - val_ratio
if test_ratio < 0:
    st.sidebar.error("Invalid split: Train + Validation ratios must leave room for the test set.")
    st.stop()
st.sidebar.write(f"Test Set Ratio: {test_ratio:.2f}")
if st.sidebar.button("Confirm Split", help="Apply the selected split ratios"):
    st.session_state['split_confirmed'] = True

# Other configurations
split_strategy = st.sidebar.selectbox("Data Split Strategy", ["standard", "kfold", "stratified_kfold", "time_series"], help="Choose how to split the dataset")
cv_method = st.sidebar.selectbox("Cross-Validation Method", ["kfold", "stratified_kfold", "time_series"], help="Choose cross-validation method for model evaluation")
n_splits = st.sidebar.slider("Number of CV Folds", 2, 10, 5, help="Number of folds for cross-validation")
random_state = st.sidebar.number_input("Random Seed", value=42, help="Random seed for reproducibility")

# File upload
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], help="Upload a CSV file with features and a target column")
if uploaded_file is not None:
    # Load and validate data
    with st.spinner("Loading dataset..."):
        file_content = uploaded_file.read()
        df, error = load_data(file_content)
        if error:
            st.error(error)
            st.stop()
        st.success("Dataset loaded successfully!")

    # Target variable selection
    st.sidebar.subheader("Target Variable")
    if "columns" not in st.session_state:
        st.session_state["columns"] = df.columns.tolist()
    target_column = st.sidebar.selectbox(
        "Select Target Column",
        ["None"] + st.session_state["columns"],
        key="target_column",
        help="Choose the target variable column (required for regression, classification, semi-supervised tasks)"
    )

    # Validate target column
    if task in ["regression", "classification", "semi-supervised"] and target_column == "None":
        st.error("Please select a target column for the selected task.")
        st.stop()
    if target_column != "None" and target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in dataset.")
        st.stop()

    # Move target column to the last position
    if target_column != "None" and task in ["regression", "classification", "semi-supervised"]:
        cols = [col for col in df.columns if col != target_column] + [target_column]
        df = df[cols]

    # Stage 1: Data Understanding
    st.header("1. Data Understanding")
    info, error = understand_data(df)
    if error:
        st.error(error)
        st.stop()
    st.write("**Dataset Info**")
    st.write(f"Shape: {info['shape']}")
    st.write(f"Null Counts: {info['null_counts']}")
    st.write(f"Duplicates: {info['duplicates']}")
    st.write("**Data Types**")
    st.write(info['dtypes'])
    st.write("**Summary Statistics**")
    st.write(pd.DataFrame(info['stats']))

    # Generate and display visualizations
    with st.spinner("Generating visualizations..."):
        vis_files, error = generate_visualizations(df, output_dir=PLOT_DIR)
        if error:
            st.error(error)
        else:
            st.subheader("Visualizations")
            for file in vis_files:
                st.image(file, caption=os.path.basename(file), use_column_width=True)

    # Stage 2: Data Cleaning
    st.header("2. Data Cleaning")
    with st.spinner("Cleaning data..."):
        df_cleaned, error = clean_data(df.copy())
        if error:
            st.error(error)
            st.stop()
        st.success("Data cleaned successfully!")
        st.write("**Cleaned Dataset Preview**")
        st.write(df_cleaned.head())

    # Stage 3: Data Transformation
    st.header("3. Data Transformation")
    with st.spinner("Transforming data..."):
        df_transformed, error = transform_data(df_cleaned.copy())
        if error:
            st.error(error)
            st.stop()
        st.success("Data transformed successfully!")
        st.write("**Transformed Dataset Preview**")
        st.write(df_transformed.head())

    # Stage 4: Data Structuring
    st.header("4. Data Structuring")
    with st.spinner("Structuring data..."):
        df_structured, error = structure_data(df_transformed.copy())
        if error:
            st.error(error)
            st.stop()
        st.success("Data structured successfully!")
        st.write("**Structured Dataset Preview**")
        st.write(df_structured.head())

    # Stage 5: Preprocessing by Type
    st.header("5. Preprocessing by Type")
    with st.spinner("Preprocessing data by type..."):
        df_preprocessed, error = preprocess_by_type(df_structured.copy(), task=task)
        if error:
            st.error(error)
            st.stop()
        st.success("Data preprocessed successfully!")
        st.write("**Preprocessed Dataset Preview**")
        st.write(df_preprocessed.head())

    # Stage 6: Feature Selection
    st.header("6. Feature Selection")
    with st.spinner("Selecting features..."):
        df_selected, error = select_features(df_preprocessed.copy())
        if error:
            st.error(error)
            st.stop()
        st.success("Features selected successfully!")
        st.write("**Selected Features Dataset Preview**")
        st.write(df_selected.head())

    # Stage 7: Handle Zeros
    st.header("7. Handle Invalid Zeros")
    with st.spinner("Handling invalid zeros..."):
        df_final, error = handle_zeros(df_selected.copy())
        if error:
            st.error(error)
            st.stop()
        st.success("Invalid zeros handled successfully!")
        st.write("**Final Dataset Preview**")
        st.write(df_final.head())

    # Prepare data for modeling
    if task in ["regression", "classification", "semi-supervised"]:
        if target_column == "None":
            st.error("Target column must be selected for this task.")
            st.stop()
        X = df_final.iloc[:, :-1].to_numpy()
        y = df_final.iloc[:, -1].to_numpy()
    else:
        X = df_final.to_numpy()
        y = None

    # Stage 8: Data Splitting
    st.header("8. Data Splitting")
    if not st.session_state['split_confirmed']:
        st.warning("Please confirm the split configuration in the sidebar.")
        st.stop()
    else:
        with st.spinner("Splitting data..."):
            if task in ["regression", "classification", "semi-supervised"] and split_strategy == "standard":
                split_result = dynamic_split(
                    X, y,
                    strategy=split_strategy,
                    task=task,
                    train_size=train_ratio,
                    val_size=val_ratio,
                    test_size=test_ratio,
                    random_state=random_state
                )
                X_train, X_val, X_test, y_train, y_val, y_test = (
                    split_result["X_train"], split_result["X_val"], split_result["X_test"],
                    split_result["y_train"], split_result["y_val"], split_result["y_test"]
                )
                X_train_val = np.vstack([X_train, X_val]) if val_ratio > 0 else X_train
                y_train_val = np.concatenate([y_train, y_val]) if val_ratio > 0 else y_train
                st.write(f"Train Shape: {X_train.shape}, Validation Shape: {X_val.shape if val_ratio > 0 else 'None'}, Test Shape: {X_test.shape}")
            elif task in ["clustering", "dimensionality reduction"]:
                X_train_val = X
                X_test = None
                y_train_val = None
                y_test = None
                st.write(f"Full Data Shape for {task}: {X.shape}")
            elif task == "reinforcement":
                st.info("Reinforcement learning does not require data splitting. Environment setup will be handled during training.")
                X_train_val, y_train_val, X_test, y_test = None, None, None, None
            else:
                split_result = dynamic_split(
                    X, y,
                    strategy=split_strategy,
                    task=task,
                    n_splits=n_splits,
                    random_state=random_state
                )
                folds = split_result["folds"]
                X_train_val, y_train_val = X, y
                X_test, y_test = None, None
                st.write(f"Cross-Validation Folds: {len(folds)}")

    # Stage 9: Model Training
    st.header("9. Model Training")
    model_configs = []
    if task == "regression":
        model_configs = [
            {"name": "Linear Regression", "model_func": train_linear_regression, "model_type": "sklearn", "params": {}},
            {"name": "Ridge Regression", "model_func": train_ridge_regression, "model_type": "sklearn", "params": {"alpha": 1.0}},
            {"name": "Lasso Regression", "model_func": train_lasso_regression, "model_type": "sklearn", "params": {"alpha": 1.0}},
            {"name": "Decision Tree Regressor", "model_func": train_decision_tree_regressor, "model_type": "sklearn", "params": {"max_depth": None}},
            {"name": "Random Forest Regressor", "model_func": train_random_forest_regressor, "model_type": "sklearn", "params": {"n_estimators": 100}},
            {"name": "XGBoost Regressor", "model_func": train_xgboost_regressor, "model_type": "sklearn", "params": {"n_estimators": 100}},
            {"name": "LightGBM Regressor", "model_func": train_lightgbm_regressor, "model_type": "sklearn", "params": {"n_estimators": 100}},
            {"name": "SVR", "model_func": train_svr, "model_type": "sklearn", "params": {"kernel": "rbf"}},
            {"name": "Neural Network Regressor", "model_func": train_neural_network_regressor, "model_type": "pytorch", "params": {"hidden_layers": (64, 32), "epochs": 10}}
        ]
    elif task == "classification":
        model_configs = [
            {"name": "Logistic Regression", "model_func": train_logistic_regression, "model_type": "sklearn", "params": {}},
            {"name": "KNN Classifier", "model_func": train_knn_classifier, "model_type": "sklearn", "params": {"n_neighbors": 5}},
            {"name": "Decision Tree Classifier", "model_func": train_decision_tree_classifier, "model_type": "sklearn", "params": {"max_depth": None}},
            {"name": "Random Forest Classifier", "model_func": train_random_forest_classifier, "model_type": "sklearn", "params": {"n_estimators": 100}},
            {"name": "XGBoost Classifier", "model_func": train_xgboost_classifier, "model_type": "sklearn", "params": {"n_estimators": 100}},
            {"name": "LightGBM Classifier", "model_func": train_lightgbm_classifier, "model_type": "sklearn", "params": {"n_estimators": 100}},
            {"name": "CatBoost Classifier", "model_func": train_catboost_classifier, "model_type": "sklearn", "params": {"n_estimators": 100}},
            {"name": "SVM Classifier", "model_func": train_svm_classifier, "model_type": "sklearn", "params": {"kernel": "rbf"}},
            {"name": "Naive Bayes", "model_func": train_naive_bayes_classifier, "model_type": "sklearn", "params": {}},
            {"name": "Neural Network Classifier", "model_func": train_neural_network_classifier, "model_type": "pytorch", "params": {"hidden_layers": (64, 32), "epochs": 10}}
        ]
    elif task == "clustering":
        model_configs = [
            {"name": "K-Means", "model_func": train_kmeans, "model_type": "sklearn", "params": {"n_clusters": 3}},
            {"name": "Hierarchical Clustering", "model_func": train_hierarchical_clustering, "model_type": "sklearn", "params": {"n_clusters": 3}},
            {"name": "DBSCAN", "model_func": train_dbscan, "model_type": "sklearn", "params": {"eps": 0.5, "min_samples": 5}}
        ]
    elif task == "dimensionality reduction":
        model_configs = [
            {"name": "PCA", "model_func": train_pca, "model_type": "sklearn", "params": {"n_components": 2}},
            {"name": "t-SNE", "model_func": train_tsne, "model_type": "sklearn", "params": {"n_components": 2}},
            {"name": "UMAP", "model_func": train_umap, "model_type": "sklearn", "params": {"n_components": 2}},
            {"name": "Autoencoder", "model_func": train_autoencoder, "model_type": "pytorch", "params": {"encoding_dim": 32, "epochs": 10}}
        ]
    elif task == "semi-supervised":
        model_configs = [
            {"name": "Self-Training (RF)", "model_func": train_self_training, "model_type": "sklearn", "params": {"base_estimator": "random_forest", "threshold": 0.9}},
            {"name": "Label Propagation", "model_func": train_label_propagation, "model_type": "sklearn", "params": {"kernel": "rbf"}}
        ]
    elif task == "reinforcement":
        model_configs = [
            {"name": "Q-Learning", "model_func": train_q_learning, "model_type": "custom", "params": {"n_episodes": 1000, "alpha": 0.1, "gamma": 0.99, "epsilon": 0.1}},
            {"name": "Multi-Armed Bandit", "model_func": train_multi_armed_bandit, "model_type": "custom", "params": {"n_arms": 10, "n_rounds": 1000, "epsilon": 0.1}}
        ]

    st.subheader("Select Models to Train")
    selected_models = []
    for config in model_configs:
        if st.checkbox(config["name"], value=False, key=config["name"]):
            selected_models.append(config)
    if not selected_models:
        st.warning("Please select at least one model to train.")
        st.stop()

    # Train selected models with progress bar and accuracy
    trained_configs = {}
    for config in selected_models:
        model_name = config["name"]
        with st.spinner(f"Training {model_name}..."):
            if task in ["regression", "classification"]:
                model = config["model_func"](X_train_val, y_train_val, **config["params"])
                # Evaluate accuracy on validation set
                model.fit(X_train_val, y_train_val)
                accuracy = model.score(X_val, y_val) if val_ratio > 0 else model.score(X_test, y_test)
                st.success(f"{model_name} Training Complete! Accuracy: {accuracy:.4f}")
            elif task == "clustering" or task == "dimensionality reduction":
                model = config["model_func"](X_train_val, **config["params"])
                st.success(f"{model_name} Training Complete!")
            elif task == "semi-supervised":
                n_labeled = len(X_train_val) // 2
                X_labeled, y_labeled = X_train_val[:n_labeled], y_train_val[:n_labeled]
                X_unlabeled = X_train_val[n_labeled:]
                model = config["model_func"](X_labeled, y_labeled, X_unlabeled, **config["params"])
                accuracy = model.score(X_val, y_val) if val_ratio > 0 else model.score(X_test, y_test)
                st.success(f"{model_name} Training Complete! Accuracy: {accuracy:.4f}")
            elif task == "reinforcement":
                st.warning(f"{model_name} requires a custom environment. Skipping training for now.")
                continue
        trained_configs[model_name] = {
            "name": model_name,  # Added 'name' key
            "model": model,
            "model_type": config["model_type"],
            "params": config["params"],
            "task": task
        }
    st.session_state['models_trained'] = trained_configs

    # Stage 10: Hyperparameter Tuning (Optional)
    st.header("10. Hyperparameter Tuning")
    if st.button("Tune Hyperparameters for Selected Models?"):
        param_grids = {
            "Linear Regression": {},
            "Ridge Regression": {"alpha": [0.1, 1.0, 10.0]},
            "Lasso Regression": {"alpha": [0.1, 1.0, 10.0]},
            "Decision Tree Regressor": {"max_depth": [None, 10, 20]},
            "Random Forest Regressor": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
            "XGBoost Regressor": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]},
            "LightGBM Regressor": {"n_estimators": [50, 100, 200], "num_leaves": [31, 50, 100]},
            "SVR": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
            "Neural Network Regressor": {"lr": [0.0001, 0.01], "batch_size": [16, 32, 64]},
            "Logistic Regression": {"C": [0.1, 1.0, 10.0]},
            "KNN Classifier": {"n_neighbors": [3, 5, 7]},
            "Decision Tree Classifier": {"max_depth": [None, 10, 20]},
            "Random Forest Classifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
            "XGBoost Classifier": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]},
            "LightGBM Classifier": {"n_estimators": [50, 100, 200], "num_leaves": [31, 50, 100]},
            "CatBoost Classifier": {"iterations": [50, 100, 200], "depth": [4, 6, 8]},
            "SVM Classifier": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
            "Naive Bayes": {},
            "Neural Network Classifier": {"lr": [0.0001, 0.01], "batch_size": [16, 32, 64]},
            "K-Means": {"n_clusters": [2, 3, 4, 5]},
            "Hierarchical Clustering": {"n_clusters": [2, 3, 4, 5]},
            "DBSCAN": {"eps": [0.3, 0.5, 0.7], "min_samples": [3, 5, 7]},
            "PCA": {"n_components": [2, 3, 5]},
            "t-SNE": {"n_components": [2, 3]},
            "UMAP": {"n_components": [2, 3, 5]},
            "Autoencoder": {"encoding_dim": [16, 32, 64]},
            "Self-Training (RF)": {"threshold": [0.7, 0.9, 0.95]},
            "Label Propagation": {"gamma": [0.1, 1.0, 10.0]},
            "Q-Learning": {"alpha": [0.05, 0.1, 0.2], "epsilon": [0.05, 0.1, 0.2]},
            "Multi-Armed Bandit": {"epsilon": [0.05, 0.1, 0.2]}
        }

        tuned_configs = {}
        for model_name, config in st.session_state['models_trained'].items():
            with st.spinner(f"Tuning hyperparameters for {model_name}..."):
                if task == "reinforcement" or not param_grids.get(model_name):
                    tuned_configs[model_name] = config
                    continue
                tuned_params = tune_hyperparameters(
                    model=config["model"],
                    X=X_train_val if task in ["regression", "classification", "semi-supervised"] else X,
                    y=y_train_val if task in ["regression", "classification", "semi-supervised"] else None,
                    param_grid=param_grids[model_name],
                    method="bayesian",
                    task=task,
                    cv_method=cv_method,
                    n_splits=n_splits,
                    random_state=random_state,
                    model_type=config["model_type"],
                    device="cpu",
                    n_iter=5,
                    output_file=os.path.join(OUTPUT_DIR, f"{model_name}_best_params.json")
                )
                config["params"] = tuned_params["best_params"]
                st.write(f"{model_name} Best Params: {tuned_params['best_params']}, Best Score: {tuned_params['best_score']:.4f}")
                tuned_configs[model_name] = config
        st.session_state['models_trained'] = tuned_configs

    # Stage 11: Model Selection and Final Training
    st.header("11. Model Selection and Final Training")
    with st.spinner("Selecting and training the best model..."):
        results = select_and_train_best_model(
            model_configs=list(st.session_state['models_trained'].values()),
            X_train_val=X_train_val if task in ["regression", "classification", "semi-supervised"] else X,
            y_train_val=y_train_val if task in ["regression", "classification", "semi-supervised"] else None,
            X_test=X_test if task in ["regression", "classification", "semi-supervised"] else None,
            y_test=y_test if task in ["regression", "classification", "semi-supervised"] else None,
            task=task,
            cv_method=cv_method,
            n_splits=n_splits,
            random_state=random_state,
            device="cpu",
            output_dir=PLOT_DIR
        )
        st.success(f"Best Model: {results['best_model_name']}")
        st.write("**Final Metrics**")
        st.write(results["final_results"])

    # Display evaluation plots
    st.subheader("Evaluation Visualizations")
    for file in os.listdir(PLOT_DIR):
        if file.startswith(("roc_curve", "pr_curve", "confusion_matrix", "pred_vs_actual")):
            st.image(os.path.join(PLOT_DIR, file), caption=file, use_column_width=True)

    # Save results
    st.write("**Results Saved**")
    st.write(f"Best model parameters: {os.path.join(OUTPUT_DIR, f'{results['best_model_name']}_best_params.json')}")
    st.write(f"Evaluation plots: {PLOT_DIR}")
else:
    st.info("Please upload a CSV file to start the data science pipeline.")