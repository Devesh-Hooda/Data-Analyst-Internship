"""
Bank Marketing Dataset Analysis and Modeling

This script performs comprehensive analysis and modeling on the bank marketing dataset,
including data cleaning, preprocessing, feature engineering, and machine learning model training
and evaluation for predicting customer subscription to a term deposit.

Models implemented: Logistic Regression, Random Forest Classifier, K-Nearest Neighbors
with hyperparameter tuning for KNN.

Author: Devesh Hooda (Adapted for local execution)
"""

# IMPORTS
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data and validate required columns.

    Args:
        filepath: Path to the CSV file

    Returns:
        Loaded and validated DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {filepath}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Validate expected columns
        expected_columns = [
            'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_wk', 'duration', 'campaign', 'pdays',
            'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
            'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'
        ]

        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)

        if missing_cols:
            print(f"Warning: Missing columns: {list(missing_cols)}")
        if extra_cols:
            print(f"Warning: Extra columns: {list(extra_cols)}")

        print("Data inspection complete.\n")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform comprehensive data cleaning.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df

    print("Performing data cleaning...")
    df_clean = df.copy()

    print("Missing values before cleaning:")
    print(df_clean.isnull().sum())

    # Special handling for pdays (999 = no previous contact)
    if 'pdays' in df_clean.columns:
        df_clean['previous_contact'] = df_clean['pdays'].apply(lambda x: 0 if x == 999 else 1)
        df_clean['pdays'] = df_clean['pdays'].replace(999, np.nan)
        print("Created 'previous_contact' flag and handled pdays=999")

    # Fill missing values
    num_cols = df_clean.select_dtypes(include=np.number).columns
    cat_cols = df_clean.select_dtypes(include='object').columns

    for col in num_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median: {median_val:.2f}")

    for col in cat_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"Filled missing values in {col} with mode: '{mode_val}'")

    # Remove duplicates
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dup_count = initial_count - len(df_clean)
    print(f"Removed {dup_count} duplicate rows")

    # Standardize categorical data
    for col in cat_cols:
        df_clean[col] = df_clean[col].str.lower().str.strip()
        print(f"Standardized {col}")

    print("Missing values after cleaning:")
    print(df_clean.isnull().sum())
    print(f"Final shape: {df_clean.shape}\n")
    return df_clean


def preprocess_and_feature_engineer(df: pd.DataFrame) -> tuple:
    """
    Perform preprocessing and feature engineering.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (X_processed_df, y, preprocessor)
    """
    print("Performing feature engineering and preprocessing...")

    # Encode target
    df['y_encoded'] = df['y'].map({'no': 0, 'yes': 1})

    # Prepare features and target
    X = df.drop(['y', 'y_encoded'], axis=1)
    y = df['y_encoded']

    # Identify feature types
    categorical_cols = X.select_dtypes(include='object').columns
    numerical_cols = X.select_dtypes(include=np.number).columns

    print(f"Feature types - Categorical: {len(categorical_cols)}, Numerical: {len(numerical_cols)}")

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ]), categorical_cols)
        ])

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Get feature names
    num_features = numerical_cols.tolist()
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_features = cat_encoder.get_feature_names_out(categorical_cols)
    all_features = num_features + cat_features.tolist()

    # Create processed DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=all_features)

    print(f"Processed dataset shape: {X_processed_df.shape}")
    print("Preprocessing complete.\n")
    return X_processed_df, y, preprocessor


def train_and_evaluate_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Train and evaluate multiple models.

    Returns:
        DataFrame with model results
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    results = []

    print("Training and evaluating models...\n")

    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")

    return pd.DataFrame(results)


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_model_comparison(results_df: pd.DataFrame):
    """
    Visualize model performance comparison as separate subplots.
    """
    metrics = results_df.columns.drop("Model")
    num_metrics = len(metrics)
    num_cols = 2  # Two per row
    num_rows = (num_metrics + num_cols - 1) // num_cols

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[m for m in metrics],
        specs=[[{}] * num_cols for _ in range(num_rows)]
    )

    for i, metric in enumerate(metrics):
        row = i // num_cols + 1
        col = i % num_cols + 1

        metric_data = results_df[['Model', metric]].copy()
        metric_data.columns = ['Model', 'Value']

        # Get best
        best_idx = metric_data['Value'].idxmax()
        best_model = metric_data.loc[best_idx, 'Model']
        best_value = metric_data.loc[best_idx, 'Value']

        # Add bars with text only on the best
        texts = [''] * len(metric_data)
        texts[best_idx] = f'{best_value:.3f}'

        fig.add_trace(
            go.Bar(
                x=metric_data['Model'],
                y=metric_data['Value'],
                text=texts,
                textposition='outside',
                name=metric,
                showlegend=False
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        title_text="Model Performance Comparison",
        height=800,
        paper_bgcolor="white",
        plot_bgcolor="lightgrey"
    )

    fig.update_xaxes(title_text="Models")
    fig.update_yaxes(title_text="Score", range=[0, 1.1])  # Add buffer for text above bars

    fig.show()


def tune_knn_hyperparameters(X_train, X_test, y_train, y_test, results_df: pd.DataFrame):
    """
    Perform hyperparameter tuning for KNN and update results.
    """
    print("Performing KNN hyperparameter tuning...")

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_knn = grid_search.best_estimator_

    print(f"Best parameters: {best_params}")

    y_pred_knn = best_knn.predict(X_test)
    f1_tuned = f1_score(y_test, y_pred_knn)
    f1_before = results_df.loc[results_df['Model'] == 'K-Nearest Neighbors', 'F1-Score'].values[0]

    print(f"F1 before tuning: {f1_before:.4f}")
    print(f"F1 after tuning: {f1_tuned:.4f}")

    results_df.loc[results_df['Model'] == 'K-Nearest Neighbors', 'F1-Score'] = f1_tuned
    return results_df


# MAIN EXECUTION
if __name__ == "__main__":
    print("Starting Bank Marketing Dataset Analysis...")

    # Load and validate data
    df = load_and_validate_data('bankmarketing.csv')

    # Clean data
    df_clean = clean_data(df)

    # Preprocess and feature engineer
    X_processed_df, y, preprocessor = preprocess_and_feature_engineer(df_clean)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed_df, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Train and evaluate models
    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Plot comparison
    plot_model_comparison(results_df)

    # Tune KNN hyperparameters
    results_df = tune_knn_hyperparameters(X_train, X_test, y_train, y_test, results_df)

    print("Analysis complete.")
    print("Final model comparison:")
    print(results_df)
