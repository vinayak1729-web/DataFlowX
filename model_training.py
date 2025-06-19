import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
from torch.utils.data import TensorDataset, DataLoader
import warnings

def train_model(model, X_train, y_train, task='classification', model_type='sklearn', device='cpu', batch_size=32, epochs=50, **kwargs):
    """
    Generic function to train a machine learning model dynamically based on model type and task.
    
    Args:
        model: Model object (scikit-learn, PyTorch, or custom reinforcement learning).
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        task (str): 'classification', 'regression', 'clustering', 'reinforcement'.
        model_type (str): 'sklearn' for scikit-learn models, 'pytorch' for PyTorch models, 'reinforcement' for RL.
        device (str): Device for PyTorch models ('cpu' or 'cuda').
        batch_size (int): Batch size for PyTorch training.
        epochs (int): Number of epochs for PyTorch training.
        **kwargs: Additional arguments (e.g., env for reinforcement learning, hyperparameters).
    
    Returns:
        Trained model.
    """
    if model_type not in ['sklearn', 'pytorch', 'reinforcement']:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    if task not in ['classification', 'regression', 'clustering', 'reinforcement']:
        raise ValueError(f"Unsupported task: {task}")
    
    # Convert inputs to appropriate format
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train) if y_train is not None else None
    
    if model_type == 'sklearn':
        # Scikit-learn models
        if not isinstance(model, BaseEstimator):
            raise ValueError("Model must be a scikit-learn estimator for model_type='sklearn'")
        
        if task in ['classification', 'regression']:
            if y_train is None:
                raise ValueError("y_train must be provided for supervised learning")
            model.fit(X_train, y_train)
        elif task == 'clustering':
            model.fit(X_train)  # Clustering doesn't require y_train
        else:
            raise ValueError(f"Task {task} not supported for scikit-learn models")
    
    elif model_type == 'pytorch':
        # PyTorch models
        if not isinstance(model, nn.Module):
            raise ValueError("Model must be a PyTorch nn.Module for model_type='pytorch'")
        
        if task in ['classification', 'regression']:
            if y_train is None:
                raise ValueError("y_train must be provided for supervised learning")
            
            # Convert data to PyTorch tensors
            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            
            # Set up loss function based on task
            criterion = nn.BCELoss() if task == 'classification' else nn.MSELoss()
            
            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=kwargs.get('lr', 0.001))
            
            # DataLoader for batch training
            dataset = TensorDataset(X_train_t, y_train_t)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
        
        elif task == 'clustering':
            raise ValueError("Clustering not supported for PyTorch models in this implementation")
        
        else:
            raise ValueError(f"Task {task} not supported for PyTorch models")
    
    elif model_type == 'reinforcement':
        # Reinforcement learning (assumes model is a callable function like Q-Learning)
        if task != 'reinforcement':
            raise ValueError("Reinforcement learning models only support task='reinforcement'")
        
        env = kwargs.get('env')
        if env is None:
            raise ValueError("Environment (env) must be provided for reinforcement learning")
        
        # Call the model as a training function (e.g., train_q_learning)
        try:
            model = model(env=env, **kwargs)  # Pass environment and additional RL parameters
        except Exception as e:
            raise ValueError(f"Error training reinforcement learning model: {e}")
    
    return model

def train_multiple_models(model_configs, X_train, y_train, device='cpu'):
    """
    Train multiple models using their configurations.
    
    Args:
        model_configs (list): List of dicts, each containing 'model', 'task', 'model_type', and optional kwargs.
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        device (str): Device for PyTorch models ('cpu' or 'cuda').
    
    Returns:
        dict: Dictionary of trained models with their names as keys.
    """
    trained_models = {}
    
    for config in model_configs:
        model_name = config.get('name', f"model_{len(trained_models)}")
        model = config['model']
        task = config.get('task', 'classification')
        model_type = config.get('model_type', 'sklearn')
        kwargs = config.get('kwargs', {})
        
        try:
            trained_model = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                task=task,
                model_type=model_type,
                device=device,
                **kwargs
            )
            trained_models[model_name] = trained_model
        except Exception as e:
            warnings.warn(f"Failed to train {model_name}: {e}")
    
    return trained_models

if __name__ == "__main__":
    # Example usage with sample data
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from data_modeling import NeuralNetworkClassifier, train_q_learning  # From your ml_models.py
    import gym
    import numpy as np
    
    # Generate sample classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Example model configurations
    model_configs = [
        {
            'name': 'random_forest',
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'task': 'classification',
            'model_type': 'sklearn'
        },
        {
            'name': 'neural_network',
            'model': NeuralNetworkClassifier(input_dim=X.shape[1], hidden_layers=(64, 32)),
            'task': 'classification',
            'model_type': 'pytorch',
            'kwargs': {'batch_size': 32, 'epochs': 10, 'lr': 0.001}
        },
        {
            'name': 'q_learning',
            'model': train_q_learning,
            'task': 'reinforcement',
            'model_type': 'reinforcement',
            'kwargs': {
                'env': gym.make('CartPole-v1'),
                'n_episodes': 100,
                'alpha': 0.1,
                'gamma': 0.99,
                'epsilon': 0.1
            }
        }
    ]
    
    # Train multiple models
    trained_models = train_multiple_models(model_configs, X, y, device='cpu')
    print("Trained models:", list(trained_models.keys()))