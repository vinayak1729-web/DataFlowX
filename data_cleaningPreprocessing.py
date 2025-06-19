import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def load_data(file_content):
    """Load and validate CSV file content into a DataFrame."""
    try:
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        if df.empty or df.columns.empty:
            return None, "ERROR: Empty CSV file or no columns detected."
        return df, None
    except Exception as e:
        return None, f"ERROR: Failed to load CSV: {str(e)}"

def understand_data(df):
    """Stage 1: Raw Data Understanding."""
    try:
        info = {
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
            "null_counts": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "stats": df.describe().to_dict(),
            "info": str(df.info())
        }
        # Check for target leakage (assuming last column is target, if present)
        if df.shape[1] > 1:
            correlations = df.corr(numeric_only=True).iloc[:-1, -1] if df.select_dtypes(include=['int64', 'float64']).shape[1] > 1 else None
            info["target_leakage"] = correlations.to_dict() if correlations is not None else "No numeric target for correlation"
        return info, None
    except Exception as e:
        return None, f"ERROR: Failed to understand data: {str(e)}"

def clean_data(df):
    """Stage 2: Data Cleaning."""
    try:
        # 1. Missing Values
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns

        # Low % missing (<5%): SimpleImputer
        for col in numeric_cols:
            if df[col].isnull().mean() < 0.05:
                imp = SimpleImputer(strategy='median')
                df[col] = imp.fit_transform(df[[col]]).ravel()
        for col in categorical_cols:
            if df[col].isnull().mean() < 0.05:
                imp = SimpleImputer(strategy='most_frequent')
                df[col] = imp.fit_transform(df[[col]]).ravel()

        # Medium % missing (5-30%): KNNImputer
        for col in numeric_cols:
            if 0.05 <= df[col].isnull().mean() <= 0.30:
                knn_imp = KNNImputer(n_neighbors=5)
                df[col] = knn_imp.fit_transform(df[[col]]).ravel()

        # High % missing (>30%): Drop column
        cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > 0.30]
        df.drop(columns=cols_to_drop, inplace=True)

        # Datetime: Forward fill
        for col in datetime_cols:
            df[col].fillna(method='ffill', inplace=True)

        # Handle invalid negative values (e.g., age, salary)
        non_negative_cols = [col for col in numeric_cols if any(x in col.lower() for x in ['age', 'salary', 'price', 'quantity', 'count']) or df[col].min() >= 0]
        for col in non_negative_cols:
            df.loc[df[col] < 0, col] = np.nan
            if df[col].isnull().any():
                imp = SimpleImputer(strategy='median')
                df[col] = imp.fit_transform(df[[col]]).ravel()
            df[col] = df[col].clip(lower=0)

        # 2. Duplicates
        df.drop_duplicates(inplace=True)

        # 3. Fix Data Types
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    if df[col].str.match(r'\d{4}-\d{2}-\d{2}').all():
                        df[col] = pd.to_datetime(df[col])
                except AttributeError:
                    pass

        # 4. Outlier Detection (Isolation Forest)
        if len(numeric_cols) > 0:
            iso = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso.fit_predict(df[numeric_cols])
            df = df[outliers == 1]  # Keep non-outliers

        return df, None
    except Exception as e:
        return None, f"ERROR: Failed to clean data: {str(e)}"

def transform_data(df):
    """Stage 3: Data Transformation."""
    try:
        # 1. Feature Encoding
        categorical_cols = df.select_dtypes(include=['category']).columns
        if len(categorical_cols) > 0:
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df[categorical_cols])
            encoded_cols = [f"{col}_{val}" for col, vals in zip(categorical_cols, encoder.categories_) for val in vals[1:]]
            df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            df = pd.concat([df, df_encoded], axis=1)  # Append encoded columns

        # 2. Scaling
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            df[numeric_cols] = df[numeric_cols].clip(lower=0)

        # 3. Feature Engineering
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_weekday'] = df[col].dt.weekday
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            df[f'{col}_word_count'] = df[col].str.split().str.len().fillna(0).clip(lower=0)
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2  # Non-negative since input is [0, 1]

        return df, None
    except Exception as e:
        return None, f"ERROR: Failed to transform data: {str(e)}"

def structure_data(df):
    """Stage 4: Data Structuring."""
    try:
        # 1. Handle Imbalanced Data (only for classification)
        if df.shape[1] > 1:
            target = df.iloc[:, -1]
            is_classification = target.dtype in ['int64', 'category', 'object'] or (target.dtype == 'float64' and target.nunique() < 10 and target.apply(lambda x: x.is_integer()).all())
            if is_classification:
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                try:
                    smote = SMOTE(random_state=42)
                    X_res, y_res = smote.fit_resample(X, y)
                    df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=df.columns[-1])], axis=1)
                except ValueError:
                    pass  # Skip SMOTE if insufficient data

        # 2. Dimensionality Reduction (PCA, append)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 10:
            pca = PCA(n_components=0.95, random_state=42)
            reduced = pca.fit_transform(df[numeric_cols])
            reduced_cols = [f'PCA_{i+1}' for i in range(reduced.shape[1])]
            df_reduced = pd.DataFrame(reduced, columns=reduced_cols, index=df.index)
            df = pd.concat([df, df_reduced], axis=1)

        return df, None
    except Exception as e:
        return None, f"ERROR: Failed to structure data: {str(e)}"

def preprocess_by_type(df, task=None):
    """
    Stage 5: Preprocessing by Data Type.

    Args:
        df (pd.DataFrame): Input dataframe.
        task (str, optional): Type of machine learning task ('regression', 'classification',
                             'clustering', 'dimensionality reduction', 'semi-supervised', 'reinforcement').

    Returns:
        tuple: (preprocessed dataframe, error message if any)
    """
    try:
        # Initialize NLTK resources
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        if task in ['regression', 'classification', 'semi-supervised']:
            # Ensure there's a target column (assumed to be the last column)
            if len(df.columns) < 2:
                return None, "Dataset must have at least one feature and one target column."
            target_col = df.columns[-1]
            feature_cols = [col for col in df.columns if col != target_col]
            # Separate features and target
            X = df[feature_cols]
            y = df[target_col]
        else:
            # No target column for unsupervised tasks
            X = df
            y = None
            target_col = None

        # Numeric Data
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            imp = SimpleImputer(strategy='median')
            X[numeric_cols] = imp.fit_transform(X[numeric_cols])
            X[numeric_cols] = X[numeric_cols].clip(lower=0)

        # Time-Series Data
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            X[f'{col}_lag1'] = X[col].shift(1)
            X[f'{col}_lag1'].fillna(method='bfill', inplace=True)

        # Text Data
        text_cols = X.select_dtypes(include=['object']).columns
        for col in text_cols:
            X[f'{col}_processed'] = X[col].str.lower().apply(
                lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(str(x)) if word not in stop_words])
            ).fillna('')
            tfidf = TfidfVectorizer(max_features=10)
            tfidf_features = tfidf.fit_transform(X[f'{col}_processed'])
            tfidf_cols = [f'{col}_tfidf_{i+1}' for i in range(tfidf_features.shape[1])]
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_cols, index=X.index)
            X = pd.concat([X, tfidf_df], axis=1)

        # Reconstruct the dataframe
        if task in ['regression', 'classification', 'semi-supervised']:
            # Reattach target column
            df_preprocessed = X.copy()
            df_preprocessed[target_col] = y.reset_index(drop=True)
        else:
            df_preprocessed = X.copy()

        return df_preprocessed, None

    except Exception as e:
        return None, f"ERROR: Failed to preprocess by type: {str(e)}"
def select_features(df):
    """Stage 6: Feature Selection."""
    try:
        # 1. Filter Method: Variance Threshold
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(df[numeric_cols])
            selected_cols = numeric_cols[selector.get_support()]
            df = df[list(selected_cols) + [col for col in df.columns if col not in numeric_cols]]

        # 2. Embedded Method: Random Forest Feature Importance
        if df.shape[1] > 1:
            X = df.iloc[:, :-1].select_dtypes(include=['int64', 'float64'])
            y = df.iloc[:, -1]
            is_classification = y.dtype in ['int64', 'category', 'object'] or (y.dtype == 'float64' and y.nunique() < 10 and y.apply(lambda x: x.is_integer()).all())
            model = RandomForestClassifier(random_state=42) if is_classification else RandomForestRegressor(random_state=42)
            model.fit(X, y)
            selector = SelectFromModel(model, prefit=True)
            selected_cols = X.columns[selector.get_support()]
            df = pd.concat([df[selected_cols], df.iloc[:, [-1]]], axis=1)

        return df, None
    except Exception as e:
        return None, f"ERROR: Failed to select features: {str(e)}"

def handle_zeros(df):
    """Handle invalid zero values after feature selection."""
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        non_zero_cols = [col for col in numeric_cols if any(x in col.lower() for x in ['age', 'salary', 'price', 'quantity', 'count'])]

        for col in non_zero_cols:
            df.loc[df[col] == 0, col] = np.nan
            if df[col].isnull().any():
                imp = SimpleImputer(strategy='median')
                df[col] = imp.fit_transform(df[[col]]).ravel()
            df[col] = df[col].clip(lower=0.001)  # Ensure strictly positive

        return df, None
    except Exception as e:
        return None, f"ERROR: Failed to handle zeros: {str(e)}"
def generate_visualizations(df, output_dir="visualizations"):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        visualization_files = []
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        # Histograms
        for column in numeric_cols:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[column], bins=30)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            file_path = os.path.join(output_dir, f"{column}_histogram.png")
            plt.savefig(file_path)
            plt.close()
            visualization_files.append(file_path)

        # Correlation Heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Heatmap')
            file_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(file_path)
            plt.close()
            visualization_files.append(file_path)

        # Feature Importance
        if df.shape[1] > 1:
            X = df.iloc[:, :-1].select_dtypes(include=['int64', 'float64'])
            y = df.iloc[:, -1]
            is_classification = y.dtype in ['int64', 'category', 'object'] or (y.dtype == 'float64' and y.nunique() < 10 and y.apply(lambda x: x.is_integer()).all())
            model = RandomForestClassifier(random_state=42) if is_classification else RandomForestRegressor(random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances, y=X.columns)
            plt.title('Feature Importance')
            file_path = os.path.join(output_dir, "feature_importance.png")
            plt.savefig(file_path)
            plt.close()
            visualization_files.append(file_path)

        return visualization_files, None
    except Exception as e:
        return None, f"ERROR: Failed to generate visualizations: {str(e)}"