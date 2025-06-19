import numpy as np
import torch
from sklearn.preprocessing import StandardScaler  # Example preprocessor, replace as needed

def test_model_with_query(model, query, task='classification', model_type='sklearn', preprocessor=None, device='cpu'):
    """
    Test a trained model with a user input query.
    
    Args:
        model: Trained model (scikit-learn or PyTorch).
        query (np.array or list): User input query as a feature vector.
        task (str): 'classification' or 'regression'.
        model_type (str): 'sklearn' or 'pytorch'.
        preprocessor: Preprocessing object (e.g., StandardScaler) fitted on training data.
        device (str): Device for PyTorch models ('cpu' or 'cuda').
    
    Returns:
        dict: Prediction and additional information (e.g., probabilities for classification).
    """
    # Convert query to numpy array and ensure correct shape (1, n_features)
    query = np.asarray(query).reshape(1, -1)
    
    # Apply preprocessing if provided
    if preprocessor is not None:
        try:
            query = preprocessor.transform(query)
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")
    
    # Make prediction based on model type
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
    
    # Prepare results
    results = {'prediction': prediction}
    if probabilities is not None:
        results['probabilities'] = probabilities.tolist()
    
    return results