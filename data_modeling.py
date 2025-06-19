import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import umap as UMAP
import joblib
import os
import random

# PyTorch Neural Network Definitions

class NeuralNetworkRegressor(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(NeuralNetworkRegressor, self).__init__()
        layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class NeuralNetworkClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(NeuralNetworkClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Supervised Learning: Regression Models

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
    
    Returns:
        LinearRegression: Trained model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train, alpha=1.0):
    """
    Train a Ridge Regression model.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        alpha (float): Regularization strength.
    
    Returns:
        Ridge: Trained model.
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_lasso_regression(X_train, y_train, alpha=1.0):
    """
    Train a Lasso Regression model.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        alpha (float): Regularization strength.
    
    Returns:
        Lasso: Trained model.
    """
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_decision_tree_regressor(X_train, y_train, max_depth=None, random_state=42):
    """
    Train a Decision Tree Regressor.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        max_depth (int): Maximum depth of the tree.
        random_state (int): Random seed.
    
    Returns:
        DecisionTreeRegressor: Trained model.
    """
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest_regressor(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Regressor.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        n_estimators (int): Number of trees.
        random_state (int): Random seed.
    
    Returns:
        RandomForestRegressor: Trained model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_xgboost_regressor(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train an XGBoost Regressor.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        n_estimators (int): Number of boosting rounds.
        random_state (int): Random seed.
    
    Returns:
        XGBRegressor: Trained model.
    """
    model = XGBRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_lightgbm_regressor(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a LightGBM Regressor.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        n_estimators (int): Number of boosting rounds.
        random_state (int): Random seed.
    
    Returns:
        LGBMRegressor: Trained model.
    """
    model = LGBMRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_svr(X_train, y_train, kernel='rbf'):
    """
    Train a Support Vector Regressor.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        kernel (str): Kernel type ('linear', 'rbf', etc.).
    
    Returns:
        SVR: Trained model.
    """
    model = SVR(kernel=kernel)
    model.fit(X_train, y_train)
    return model

def train_neural_network_regressor(X_train, y_train, hidden_layers=(64, 32), epochs=50, batch_size=32, device='cpu'):
    """
    Train a Neural Network for regression using PyTorch.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        hidden_layers (tuple): Neurons in each hidden layer.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        device (str): Device to train on ('cpu' or 'cuda').
    
    Returns:
        NeuralNetworkRegressor: Trained model.
    """
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    
    model = NeuralNetworkRegressor(X_train.shape[1], hidden_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    return model

# Supervised Learning: Classification Models

def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train a Logistic Regression Classifier.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        random_state (int): Random seed.
    
    Returns:
        LogisticRegression: Trained model.
    """
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_knn_classifier(X_train, y_train, n_neighbors=5):
    """
    Train a k-Nearest Neighbors Classifier.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        n_neighbors (int): Number of neighbors.
    
    Returns:
        KNeighborsClassifier: Trained model.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_decision_tree_classifier(X_train, y_train, max_depth=None, random_state=42):
    """
    Train a Decision Tree Classifier.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        max_depth (int): Maximum depth of the tree.
        random_state (int): Random seed.
    
    Returns:
        DecisionTreeClassifier: Trained model.
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest_classifier(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Classifier.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        n_estimators (int): Number of trees.
        random_state (int): Random seed.
    
    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_xgboost_classifier(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train an XGBoost Classifier.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        n_estimators (int): Number of boosting rounds.
        random_state (int): Random seed.
    
    Returns:
        XGBClassifier: Trained model.
    """
    model = XGBClassifier(n_estimators=n_estimators, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def train_lightgbm_classifier(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a LightGBM Classifier.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        n_estimators (int): Number of boosting rounds.
        random_state (int): Random seed.
    
    Returns:
        LGBMClassifier: Trained model.
    """
    model = LGBMClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_catboost_classifier(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a CatBoost Classifier.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        n_estimators (int): Number of boosting rounds.
        random_state (int): Random seed.
    
    Returns:
        CatBoostClassifier: Trained model.
    """
    model = CatBoostClassifier(n_estimators=n_estimators, random_state=random_state, verbose=0)
    model.fit(X_train, y_train)
    return model

def train_svm_classifier(X_train, y_train, kernel='rbf', random_state=42):
    """
    Train a Support Vector Machine Classifier.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        kernel (str): Kernel type ('linear', 'rbf', etc.).
        random_state (int): Random seed.
    
    Returns:
        SVC: Trained model.
    """
    model = SVC(kernel=kernel, random_state=random_state, probability=True)
    model.fit(X_train, y_train)
    return model

def train_naive_bayes_classifier(X_train, y_train):
    """
    Train a Gaussian Naive Bayes Classifier.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
    
    Returns:
        GaussianNB: Trained model.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_neural_network_classifier(X_train, y_train, hidden_layers=(64, 32), epochs=50, batch_size=32, device='cpu'):
    """
    Train a Neural Network for binary classification using PyTorch.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        hidden_layers (tuple): Neurons in each hidden layer.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        device (str): Device to train on ('cpu' or 'cuda').
    
    Returns:
        NeuralNetworkClassifier: Trained model.
    """
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    
    model = NeuralNetworkClassifier(X_train.shape[1], hidden_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    return model

# Unsupervised Learning: Clustering

def train_kmeans(X, n_clusters=3, random_state=42):
    """
    Train a K-Means Clustering model.
    
    Args:
        X (np.array): Input features.
        n_clusters (int): Number of clusters.
        random_state (int): Random seed.
    
    Returns:
        KMeans: Trained model.
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X)
    return model

def train_hierarchical_clustering(X, n_clusters=3):
    """
    Train a Hierarchical Clustering model.
    
    Args:
        X (np.array): Input features.
        n_clusters (int): Number of clusters.
    
    Returns:
        AgglomerativeClustering: Trained model.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters)
    model.fit(X)
    return model

def train_dbscan(X, eps=0.5, min_samples=5):
    """
    Train a DBSCAN Clustering model.
    
    Args:
        X (np.array): Input features.
        eps (float): Maximum distance between two samples.
        min_samples (int): Number of samples in a neighborhood.
    
    Returns:
        DBSCAN: Trained model.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model

# Unsupervised Learning: Dimensionality Reduction

def train_pca(X, n_components=2):
    """
    Train a PCA model for dimensionality reduction.
    
    Args:
        X (np.array): Input features.
        n_components (int): Number of components to keep.
    
    Returns:
        PCA: Trained model.
    """
    model = PCA(n_components=n_components)
    model.fit(X)
    return model

def train_tsne(X, n_components=2, random_state=42):
    """
    Train a t-SNE model for dimensionality reduction.
    
    Args:
        X (np.array): Input features.
        n_components (int): Number of components.
        random_state (int): Random seed.
    
    Returns:
        TSNE: Trained model.
    """
    model = TSNE(n_components=n_components, random_state=random_state)
    model.fit(X)
    return model

def train_umap(X, n_components=2, random_state=42):
    """
    Train a UMAP model for dimensionality reduction.
    
    Args:
        X (np.array): Input features.
        n_components (int): Number of components.
        random_state (int): Random seed.
    
    Returns:
        UMAP: Trained model.
    """
    model = UMAP(n_components=n_components, random_state=random_state)
    model.fit(X)
    return model

def train_autoencoder(X, encoding_dim=32, epochs=50, batch_size=32, device='cpu'):
    """
    Train an Autoencoder for dimensionality reduction using PyTorch.
    
    Args:
        X (np.array): Input features.
        encoding_dim (int): Size of the encoded representation.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        device (str): Device to train on ('cpu' or 'cuda').
    
    Returns:
        Autoencoder: Trained model.
    """
    X = torch.tensor(X, dtype=torch.float32).to(device)
    
    model = Autoencoder(X.shape[1], encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    dataset = torch.utils.data.TensorDataset(X, X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for X_batch, _ in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, X_batch)
            loss.backward()
            optimizer.step()
    
    return model

# Semi-Supervised Learning

def train_self_training(X_labeled, y_labeled, X_unlabeled, base_estimator='random_forest', threshold=0.9):
    """
    Train a Self-Training Classifier.
    
    Args:
        X_labeled (np.array): Labeled features.
        y_labeled (np.array): Labeled target.
        X_unlabeled (np.array): Unlabeled features.
        base_estimator (str): Base estimator ('random_forest', 'svm').
        threshold (float): Confidence threshold for pseudo-labeling.
    
    Returns:
        SelfTrainingClassifier: Trained model.
    """
    if base_estimator == 'random_forest':
        estimator = RandomForestClassifier(random_state=42)
    elif base_estimator == 'svm':
        estimator = SVC(probability=True, random_state=42)
    else:
        raise ValueError("Unsupported base estimator")
    
    model = SelfTrainingClassifier(estimator, threshold=threshold)
    X_combined = np.vstack([X_labeled, X_unlabeled])
    y_combined = np.concatenate([y_labeled, np.full(X_unlabeled.shape[0], -1)])
    model.fit(X_combined, y_combined)
    return model

def train_label_propagation(X_labeled, y_labeled, X_unlabeled, kernel='rbf'):
    """
    Train a Label Propagation model.
    
    Args:
        X_labeled (np.array): Labeled features.
        y_labeled (np.array): Labeled target.
        X_unlabeled (np.array): Unlabeled features.
        kernel (str): Kernel type ('rbf', 'knn').
    
    Returns:
        LabelPropagation: Trained model.
    """
    model = LabelPropagation(kernel=kernel)
    X_combined = np.vstack([X_labeled, X_unlabeled])
    y_combined = np.concatenate([y_labeled, np.full(X_unlabeled.shape[0], -1)])
    model.fit(X_combined, y_combined)
    return model

# Reinforcement Learning

def train_q_learning(env, n_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Train a Q-Learning agent.
    
    Args:
        env: Environment (e.g., OpenAI Gym).
        n_episodes (int): Number of episodes.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
    
    Returns:
        dict: Q-table (state-action values).
    """
    q_table = {}
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            state_key = str(state)
            if state_key not in q_table:
                q_table[state_key] = np.zeros(env.action_space.n)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_key])
            next_state, reward, done, _ = env.step(action)
            next_state_key = str(next_state)
            if next_state_key not in q_table:
                q_table[next_state_key] = np.zeros(env.action_space.n)
            q_table[state_key][action] += alpha * (
                reward + gamma * np.max(q_table[next_state_key]) - q_table[state_key][action]
            )
            state = next_state
    return q_table

def train_multi_armed_bandit(n_arms=10, n_rounds=1000, epsilon=0.1):
    """
    Train a Multi-Armed Bandit with epsilon-greedy strategy.
    
    Args:
        n_arms (int): Number of arms.
        n_rounds (int): Number of rounds.
        epsilon (float): Exploration rate.
    
    Returns:
        tuple: (Q-values, counts) for each arm.
    """
    q_values = np.zeros(n_arms)
    counts = np.zeros(n_arms)
    for _ in range(n_rounds):
        if random.random() < epsilon:
            arm = random.randint(0, n_arms-1)
        else:
            arm = np.argmax(q_values)
        reward = np.random.normal(loc=arm, scale=1.0)  # Simulated reward
        counts[arm] += 1
        q_values[arm] += (reward - q_values[arm]) / counts[arm]
    return q_values, counts

# Utility Functions

def evaluate_regression_model(model, X_test, y_test, is_pytorch_model=False, device='cpu'):
    """
    Evaluate a regression model using RMSE.
    
    Args:
        model: Trained model.
        X_test (np.array): Test features.
        y_test (np.array): Test target.
        is_pytorch_model (bool): Whether the model is a PyTorch model.
        device (str): Device for PyTorch model ('cpu' or 'cuda').
    
    Returns:
        float: Root Mean Squared Error.
    """
    if is_pytorch_model:
        model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = model(X_test).cpu().numpy().flatten()
    else:
        y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

def evaluate_classification_model(model, X_test, y_test, is_pytorch_model=False, device='cpu'):
    """
    Evaluate a classification model using accuracy, classification report, and confusion matrix.
    
    Args:
        model: Trained model.
        X_test (np.array): Test features.
        y_test (np.array): Test target.
        is_pytorch_model (bool): Whether the model is a PyTorch model.
        device (str): Device for PyTorch model ('cpu' or 'cuda').
    
    Returns:
        dict: Evaluation metrics (accuracy, classification report, confusion matrix).
    """
    if is_pytorch_model:
        model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = (model(X_test) > 0.5).cpu().numpy().astype(int).flatten()
    else:
        y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {'accuracy': accuracy, 'classification_report': report, 'confusion_matrix': cm}

def predict_model(model, X_new, is_pytorch_model=False, device='cpu'):
    """
    Predict using a trained model.
    
    Args:
        model: Trained model.
        X_new (np.array): New data for prediction.
        is_pytorch_model (bool): Whether the model is a PyTorch model.
        device (str): Device for PyTorch model ('cpu' or 'cuda').
    
    Returns:
        np.array: Predictions.
    """
    if is_pytorch_model:
        model.eval()
        X_new = torch.tensor(X_new, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = model(X_new).cpu().numpy().flatten()
        if isinstance(model, NeuralNetworkClassifier):
            predictions = (predictions > 0.5).astype(int)
    else:
        predictions = model.predict(X_new)
    return predictions

def save_model(model, model_path, is_pytorch_model=False):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model.
        model_path (str): Path to save the model.
        is_pytorch_model (bool): Whether the model is a PyTorch model.
    """
    if is_pytorch_model:
        torch.save(model.state_dict(), model_path)
    else:
        joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path, model_class=None, input_dim=None, hidden_layers=None, encoding_dim=None, is_pytorch_model=False, device='cpu'):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model.
        model_class (str): PyTorch model class ('regressor', 'classifier', 'autoencoder') if is_pytorch_model.
        input_dim (int): Input dimension for PyTorch model.
        hidden_layers (tuple): Hidden layers for PyTorch regressor/classifier.
        encoding_dim (int): Encoding dimension for PyTorch autoencoder.
        is_pytorch_model (bool): Whether the model is a PyTorch model.
        device (str): Device for PyTorch model ('cpu' or 'cuda').
    
    Returns:
        Trained model.
    """
    if is_pytorch_model:
        if not model_class or not input_dim:
            raise ValueError("model_class and input_dim required for PyTorch models")
        if model_class == 'regressor':
            model = NeuralNetworkRegressor(input_dim, hidden_layers).to(device)
        elif model_class == 'classifier':
            model = NeuralNetworkClassifier(input_dim, hidden_layers).to(device)
        elif model_class == 'autoencoder':
            if not encoding_dim:
                raise ValueError("encoding_dim required for autoencoder")
            model = Autoencoder(input_dim, encoding_dim).to(device)
        else:
            raise ValueError("Unsupported model_class")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model
        return None
    else:
        return joblib.load(model_path) if os.path.exists(model_path) else None

if __name__ == "__main__":
    # Example usage with sample data
    from sklearn.datasets import make_regression, make_classification
    
    # Regression example
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
    reg_model = train_linear_regression(X_reg, y_reg)
    rmse = evaluate_regression_model(reg_model, X_reg, y_reg)
    print(f"Linear Regression RMSE: {rmse}")
    
    # Classification example
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    clf_model = train_random_forest_classifier(X_clf, y_clf)
    metrics = evaluate_classification_model(clf_model, X_clf, y_clf)
    print(f"Random Forest Accuracy: {metrics['accuracy']}")
    
    # PyTorch example
    nn_model = train_neural_network_classifier(X_clf, y_clf, device='cpu')
    metrics = evaluate_classification_model(nn_model, X_clf, y_clf, is_pytorch_model=True)
    print(f"Neural Network Accuracy: {metrics['accuracy']}")
    
    # Clustering example
    cluster_model = train_kmeans(X_clf, n_clusters=3)
    labels = predict_model(cluster_model, X_clf)
    print(f"K-Means Labels (first 5): {labels[:5]}")