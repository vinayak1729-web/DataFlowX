import numpy as np
import json
from typing import Dict, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error
import optuna
from model_training import train_model
from data_selection import kfold_split, stratified_kfold_split
import warnings
warnings.filterwarnings('ignore')

def tune_hyperparameters(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, Any],
    method: str = 'grid',
    task: str = 'classification',
    cv_method: str = 'kfold',
    n_splits: int = 5,
    random_state: int = 42,
    n_iter: int = 10,
    model_type: str = 'sklearn',
    device: str = 'cpu',
    **kwargs
) -> Dict[str, Any]:
    """
    Tune hyperparameters using Grid Search, Random Search, or Bayesian Optimization.
    
    Args:
        model: Model object (scikit-learn or PyTorch).
        X (np.array): Features.
        y (np.array): Target.
        param_grid (dict): Hyperparameter grid or distributions for tuning.
        method (str): 'grid', 'random', or 'bayesian' for tuning method.
        task (str): 'classification' or 'regression'.
        cv_method (str): 'kfold' or 'stratified_kfold' for cross-validation.
        n_splits (int): Number of folds for cross-validation.
        random_state (int): Random seed.
        n_iter (int): Number of iterations for Random Search or Bayesian Optimization.
        model_type (str): 'sklearn' or 'pytorch'.
        device (str): Device for PyTorch models ('cpu' or 'cuda').
        **kwargs: Additional arguments for training or optimization.
    
    Returns:
        dict: Best parameters and best score.
    """
    # Define scoring metric
    if task == 'classification':
        scoring = make_scorer(accuracy_score)
    elif task == 'regression':
        scoring = make_scorer(mean_squared_error, greater_is_better=False, squared=False)  # RMSE
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Select cross-validation method
    if cv_method == 'kfold':
        cv_func = kfold_split
    elif cv_method == 'stratified_kfold' and task == 'classification':
        cv_func = stratified_kfold_split
    else:
        raise ValueError(f"Invalid cv_method: {cv_method} for task {task}")

    # Hyperparameter tuning
    if method in ['grid', 'random'] and model_type == 'sklearn':
        # Scikit-learn Grid or Random Search
        SearchClass = GridSearchCV if method == 'grid' else RandomizedSearchCV
        # Define common parameters for both GridSearchCV and RandomizedSearchCV
        search_params = {
            'estimator': model,
            'param_grid': param_grid,
            'cv': n_splits,
            'scoring': scoring,
            'n_jobs': -1
        }
        # Add random_state and n_iter only for RandomizedSearchCV
        if method == 'random':
            search_params['random_state'] = random_state
            search_params['n_iter'] = n_iter
        # Initialize search with dynamic parameters
        search = SearchClass(**search_params)
        search.fit(X, y)
        best_params = search.best_params_
        best_score = search.best_score_ if task == 'classification' else -search.best_score_  # Negate RMSE

    elif method == 'bayesian' or model_type == 'pytorch':
        # Bayesian Optimization with Optuna (for both sklearn and pytorch)
        def objective(trial):
            # Sample hyperparameters
            trial_params = {}
            for param, values in param_grid.items():
                if isinstance(values, list):
                    trial_params[param] = trial.suggest_categorical(param, values)
                elif isinstance(values, tuple) and len(values) == 2:
                    if isinstance(values[0], int):
                        trial_params[param] = trial.suggest_int(param, values[0], values[1])
                    else:
                        trial_params[param] = trial.suggest_float(param, values[0], values[1])
            
            # Create model instance with trial parameters and required arguments
            model_init_params = {k: v for k, v in trial_params.items() if k in model.__init__.__code__.co_varnames}
            # For PyTorch models, ensure input_dim and hidden_layers are passed
            if model_type == 'pytorch':
                model_init_params['input_dim'] = model.input_dim  # Use input_dim from original model
                model_init_params['hidden_layers'] = model.hidden_layers  # Use hidden_layers from original model
            
            model_instance = model.__class__(**model_init_params)
            
            # Evaluate with cross-validation
            metrics = cv_func(
                model=model_instance,
                X=X,
                y=y,
                n_splits=n_splits,
                random_state=random_state,
                task=task,
                is_pytorch_model=(model_type == 'pytorch'),
                device=device
            )
            return metrics['mean'] if task == 'classification' else -metrics['mean']  # Maximize accuracy, minimize RMSE

        study = optuna.create_study(direction='maximize' if task == 'classification' else 'minimize')
        study.optimize(objective, n_trials=n_iter)
        best_params = study.best_params
        best_score = study.best_value if task == 'classification' else -study.best_value

    else:
        raise ValueError(f"Method {method} not supported for model_type {model_type}")

    # Save best parameters to JSON
    output_file = kwargs.get('output_file', 'best_params.json')
    with open(output_file, 'w') as f:
        json.dump({'best_params': best_params, 'best_score': float(best_score), 'task': task, 'model_type': model_type}, f)
    
    return {'best_params': best_params, 'best_score': best_score}

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from data_modeling import NeuralNetworkClassifier
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Grid Search for Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf_results = tune_hyperparameters(
        model=rf_model,
        X=X,
        y=y,
        param_grid=rf_param_grid,
        method='grid',
        task='classification',
        cv_method='stratified_kfold',
        output_file='rf_best_params.json'
    )
    print("Random Forest Best Params:", rf_results['best_params'], "Best Score:", rf_results['best_score'])
    
    # Bayesian Optimization for Neural Network
    nn_model = NeuralNetworkClassifier(input_dim=X.shape[1], hidden_layers=(64, 32))
    nn_param_grid = {
        'lr': (0.0001, 0.01),
        'batch_size': [16, 32, 64]
    }
    nn_results = tune_hyperparameters(
        model=nn_model,
        X=X,
        y=y,
        param_grid=nn_param_grid,
        method='bayesian',
        task='classification',
        cv_method='stratified_kfold',
        model_type='pytorch',
        n_iter=5,
        output_file='nn_best_params.json'
    )
    print("Neural Network Best Params:", nn_results['best_params'], "Best Score:", nn_results['best_score'])