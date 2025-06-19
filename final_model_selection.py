
import numpy as np
import json
from model_training import train_model
from model_evaluation import evaluate_model
import logging

# Set up logging
logging.basicConfig(filename='model_selection.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def select_and_train_best_model(model_configs, X_train_val, y_train_val, X_test=None, y_test=None, task='classification', cv_method='kfold', n_splits=5, random_state=42, device='cpu', output_dir='plots'):
    """
    Select the best model based on cross-validation and retrain on full training data.
    
    Args:
        model_configs (list): List of dicts with 'model', 'name', 'task', 'model_type', 'params', and optional 'kwargs'.
        X_train_val (np.array): Combined training and validation features.
        y_train_val (np.array): Combined training and validation target.
        X_test (np.array): Test features (optional).
        y_test (np.array): Test target (optional).
        task (str): 'classification' or 'regression'.
        cv_method (str): 'kfold', 'stratified_kfold', or 'time_series'.
        n_splits (int): Number of folds.
        random_state (int): Random seed.
        device (str): Device for PyTorch models.
        output_dir (str): Directory for evaluation plots.
    
    Returns:
        dict: Best model, its performance, and final metrics.
    """
    # Evaluate all models
    model_results = {}
    for config in model_configs:
        model_name = config['name']
        model = config['model']
        model_type = config.get('model_type', 'sklearn')
        params = config.get('params', {})
        
        # Create model instance with best parameters
        if model_type == 'sklearn':
            model_instance = model.__class__(**params)
        elif model_type == 'pytorch':
            model_instance = model  # Assume model is already instantiated with params
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        # Evaluate with cross-validation
        results = evaluate_model(
            model=model_instance,
            X=X_train_val,
            y=y_train_val,
            task=task,
            cv_method=cv_method,
            n_splits=n_splits,
            random_state=random_state,
            model_type=model_type,
            device=device,
            visualize=False  # Avoid plotting during selection
        )
        
        cv_score = results['cv_metrics']['mean']
        model_results[model_name] = {
            'model': model_instance,
            'cv_score': cv_score,
            'results': results,
            'params': params,
            'model_type': model_type
        }
        
        logging.info(f"Model {model_name}: CV Score = {cv_score:.4f}, Params = {params}")
    
    # Select best model
    best_model_name = max(model_results, key=lambda k: model_results[k]['cv_score'] if task == 'classification' else -model_results[k]['cv_score'])
    best_config = model_results[best_model_name]
    best_model = best_config['model']
    best_params = best_config['params']
    best_model_type = best_config['model_type']
    
    logging.info(f"Selected Model: {best_model_name}, CV Score = {best_config['cv_score']:.4f}")
    
    # Retrain on full training data
    best_model = train_model(
        model=best_model,
        X_train=X_train_val,
        y_train=y_train_val,
        task=task,
        model_type=best_model_type,
        device=device,
        **best_config.get('kwargs', {})
    )
    
    # Final evaluation (including test set if provided)
    final_results = evaluate_model(
        model=best_model,
        X=X_train_val,
        y=y_train_val,
        X_test=X_test,
        y_test=y_test,
        task=task,
        cv_method=cv_method,
        n_splits=n_splits,
        random_state=random_state,
        model_type=best_model_type,
        device=device,
        visualize=True,
        output_dir=output_dir
    )
    
    # Save selection results
    selection_results = {
        'best_model_name': best_model_name,
        'best_params': best_params,
        'cv_score': best_config['cv_score'],
        'final_metrics': final_results,
        'all_results': {k: v['results'] for k, v in model_results.items()}
    }
    with open('model_selection_results.json', 'w') as f:
        json.dump(selection_results, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    logging.info(f"Final Model Trained: {best_model_name}, Final Metrics = {final_results}")
    
    return {'best_model': best_model, 'best_model_name': best_model_name, 'final_results': final_results}

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from data_modeling import NeuralNetworkClassifier
    from data_selection import train_val_test_split
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, task='classification')
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    model_configs = [
        {
            'name': 'random_forest',
            'model': RandomForestClassifier(random_state=42),
            'task': 'classification',
            'model_type': 'sklearn',
            'params': {'n_estimators': 100, 'max_depth': 10}
        },
        {
            'name': 'neural_network',
            'model': NeuralNetworkClassifier(input_dim=X.shape[1], hidden_layers=(64, 32)),
            'task': 'classification',
            'model_type': 'pytorch',
            'params': {'lr': 0.001, 'batch_size': 32},
            'kwargs': {'epochs': 10}
        }
    ]
    
    results = select_and_train_best_model(
        model_configs=model_configs,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test,
        y_test=y_test,
        task='classification',
        cv_method='stratified_kfold',
        output_dir='final_plots'
    )
    print("Best Model:", results['best_model_name'])
    print("Final Metrics:", results['final_results'])