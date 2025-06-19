import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, roc_curve, precision_recall_curve
import seaborn as sns
from data_selection import kfold_split, stratified_kfold_split, time_series_split
import os
import torch
def evaluate_model(model, X, y, X_test=None, y_test=None, task='classification', cv_method='kfold', n_splits=5, random_state=42, model_type='sklearn', device='cpu', visualize=True, output_dir='plots'):
    """
    Evaluate model performance on validation and test sets with multiple metrics and visualizations.
    
    Args:
        model: Trained model (scikit-learn or PyTorch).
        X (np.array): Features for cross-validation (train+val).
        y (np.array): Target for cross-validation.
        X_test (np.array): Test features (optional).
        y_test (np.array): Test target (optional).
        task (str): 'classification' or 'regression'.
        cv_method (str): 'kfold', 'stratified_kfold', or 'time_series'.
        n_splits (int): Number of folds.
        random_state (int): Random seed.
        model_type (str): 'sklearn' or 'pytorch'.
        device (str): Device for PyTorch models.
        visualize (bool): Whether to generate plots.
        output_dir (str): Directory to save plots.
    
    Returns:
        dict: Metrics for cross-validation and test set (if provided).
    """
    # Create output directory for plots
    if visualize and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Select cross-validation method
    if cv_method == 'kfold':
        cv_func = kfold_split
    elif cv_method == 'stratified_kfold' and task == 'classification':
        cv_func = stratified_kfold_split
    elif cv_method == 'time_series':
        cv_func = time_series_split
    else:
        raise ValueError(f"Invalid cv_method: {cv_method} for task {task}")
    
    # Perform cross-validation
    cv_metrics = cv_func(
        model=model,
        X=X,
        y=y,
        n_splits=n_splits,
        random_state=random_state,
        task=task,
        is_pytorch_model=(model_type == 'pytorch'),
        device=device
    )
    
    results = {'cv_metrics': cv_metrics}
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        if model_type == 'pytorch':
            model.eval()
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            with torch.no_grad():
                y_pred = model(X_test_t).cpu().numpy().flatten()
                if task == 'classification':
                    y_pred_proba = y_pred
                    y_pred = (y_pred > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if task == 'classification' and hasattr(model, 'predict_proba') else None
        
        if task == 'classification':
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            if visualize and y_pred_proba is not None:
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.figure()
                plt.plot(fpr, tpr, label='ROC Curve')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
                plt.close()
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                plt.figure()
                plt.plot(recall, precision, label='PR Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
                plt.close()
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
                plt.close()
        
        else:  # regression
            test_metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            if visualize:
                # Prediction vs Actual
                plt.figure()
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Predicted vs Actual')
                plt.savefig(os.path.join(output_dir, 'pred_vs_actual.png'))
                plt.close()
        
        results['test_metrics'] = test_metrics
    
    return results

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from data_selection import train_val_test_split
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, task='classification')
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_val, y_train_val)
    
    results = evaluate_model(
        model=model,
        X=X_train_val,
        y=y_train_val,
        X_test=X_test,
        y_test=y_test,
        task='classification',
        cv_method='stratified_kfold',
        output_dir='eval_plots'
    )
    print("CV Metrics:", results['cv_metrics'])
    print("Test Metrics:", results['test_metrics'])