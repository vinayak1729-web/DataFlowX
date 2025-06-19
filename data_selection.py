import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold

def train_val_test_split(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42, stratify=None):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X (np.array): Features.
        y (np.array): Target.
        train_size (float): Proportion for training set.
        val_size (float): Proportion for validation set.
        test_size (float): Proportion for test set.
        random_state (int): Random seed for reproducibility.
        stratify (np.array): Target array for stratified splitting (None for non-stratified).
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size, val_size, and test_size must sum to 1.0")
    
    # First split: train + val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # Adjust proportions for train vs val split
    train_proportion = train_size / (train_size + val_size)
    stratify_train_val = y_train_val if stratify is not None else None
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, train_size=train_proportion, random_state=random_state, stratify=stratify_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def kfold_split(X, y, n_splits=5, random_state=42):
    """
    Generate indices for k-Fold Cross-Validation.
    
    Args:
        X (np.array): Features.
        y (np.array): Target.
        n_splits (int): Number of folds.
        random_state (int): Random seed.
    
    Returns:
        list: List of (train_idx, test_idx) tuples for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [(train_idx, test_idx) for train_idx, test_idx in kf.split(X)]

def stratified_kfold_split(X, y, n_splits=5, random_state=42):
    """
    Generate indices for Stratified k-Fold Cross-Validation.
    
    Args:
        X (np.array): Features.
        y (np.array): Target (class labels for stratification).
        n_splits (int): Number of folds.
        random_state (int): Random seed.
    
    Returns:
        list: List of (train_idx, test_idx) tuples for each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [(train_idx, test_idx) for train_idx, test_idx in skf.split(X, y)]

def time_series_split(X, y, n_splits=5):
    """
    Generate indices for Time Series Cross-Validation (Forward Chaining).
    
    Args:
        X (np.array): Features (in temporal order).
        y (np.array): Target (in temporal order).
        n_splits (int): Number of splits.
    
    Returns:
        list: List of (train_idx, test_idx) tuples for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [(train_idx, test_idx) for train_idx, test_idx in tscv.split(X)]

def group_kfold_split(X, y, groups, n_splits=5):
    """
    Generate indices for Group k-Fold Cross-Validation.
    
    Args:
        X (np.array): Features.
        y (np.array): Target.
        groups (np.array): Group labels for the samples.
        n_splits (int): Number of folds.
    
    Returns:
        list: List of (train_idx, test_idx) tuples for each fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    return [(train_idx, test_idx) for train_idx, test_idx in gkf.split(X, y, groups)]

def dynamic_split(X, y, strategy='standard', task='classification', groups=None, n_splits=5, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Dynamically select and apply a splitting strategy based on data type and task.
    
    Args:
        X (np.array): Features.
        y (np.array): Target.
        strategy (str): Splitting strategy ('standard', 'kfold', 'stratified_kfold', 'time_series', 'group_kfold').
        task (str): 'classification' or 'regression' (affects stratification).
        groups (np.array): Group labels (required for group_kfold).
        n_splits (int): Number of folds for cross-validation.
        train_size (float): Proportion for training set (standard split).
        val_size (float): Proportion for validation set (standard split).
        test_size (float): Proportion for test set (standard split).
        random_state (int): Random seed.
    
    Returns:
        dict: Split data or fold indices based on strategy.
    """
    # Determine if stratification is needed for classification
    stratify = y if task == 'classification' else None
    
    # Apply selected strategy
    if strategy == 'standard':
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, train_size, val_size, test_size, random_state, stratify
        )
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
    
    elif strategy == 'kfold':
        folds = kfold_split(X, y, n_splits, random_state)
        return {'folds': folds}
    
    elif strategy == 'stratified_kfold':
        if task != 'classification':
            raise ValueError("Stratified k-Fold is only for classification tasks")
        folds = stratified_kfold_split(X, y, n_splits, random_state)
        return {'folds': folds}
    
    elif strategy == 'time_series':
        folds = time_series_split(X, y, n_splits)
        return {'folds': folds}
    
    elif strategy == 'group_kfold':
        if groups is None:
            raise ValueError("Groups must be provided for group_kfold")
        folds = group_kfold_split(X, y, groups, n_splits)
        return {'folds': folds}
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

if __name__ == "__main__":
    # Example usage with sample data
    from sklearn.datasets import make_classification
    import numpy as np
    
    # Generate sample classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Standard split
    result = dynamic_split(X, y, strategy='standard', task='classification')
    print("Standard Split Shapes:")
    print(f"X_train: {result['X_train'].shape}, y_train: {result['y_train'].shape}")
    print(f"X_val: {result['X_val'].shape}, y_val: {result['y_val'].shape}")
    print(f"X_test: {result['X_test'].shape}, y_test: {result['y_test'].shape}")
    
    # Stratified k-Fold
    result = dynamic_split(X, y, strategy='stratified_kfold', task='classification', n_splits=5)
    print("\nStratified k-Fold Folds:", len(result['folds']))
    
    # Group k-Fold
    groups = np.random.randint(0, 10, size=X.shape[0])
    result = dynamic_split(X, y, strategy='group_kfold', task='classification', groups=groups, n_splits=5)
    print("\nGroup k-Fold Folds:", len(result['folds']))