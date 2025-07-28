#!/usr/bin/env python3
"""
SmartAlert AI - Model Training Script

Command-line interface for training machine learning models to predict 
critical incidents from Splunk log data.

Usage:
    python scripts/train_model.py --data data/splunk_logs.csv --target target_column
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.feature_engineering import preprocess_log_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb


def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple ML models and return the best one.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
    
    Returns:
        tuple: (best_model, best_model_name, results)
    """
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
    }
    
    results = {}
    best_auc = 0
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate AUC
        if len(np.unique(y_train)) == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        
        results[name] = {
            'model': model,
            'auc': auc,
            'predictions': y_pred
        }
        
        print(f"{name} AUC: {auc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Track best model
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name
    
    return best_model, best_model_name, results


def hyperparameter_tuning(model, model_name, X_train, y_train):
    """
    Perform hyperparameter tuning for the best model.
    
    Args:
        model: The model to tune
        model_name: Name of the model
        X_train, y_train: Training data
    
    Returns:
        Best tuned model
    """
    print(f"\nPerforming hyperparameter tuning for {model_name}...")
    
    if model_name == 'RandomForest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    elif model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif model_name == 'LightGBM':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    else:
        return model
    
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def save_model_artifacts(model, model_name, preprocessing_objects, feature_names, 
                        target_column, model_metrics, output_dir='models'):
    """
    Save all model artifacts.
    
    Args:
        model: Trained model
        model_name: Name of the model
        preprocessing_objects: Preprocessing objects (scalers, encoders)
        feature_names: List of feature names
        target_column: Name of target column
        model_metrics: Performance metrics
        output_dir: Directory to save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f'best_model_{model_name.lower()}.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")
    
    # Save preprocessing objects
    if 'scaler' in preprocessing_objects:
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(preprocessing_objects['scaler'], scaler_path)
        print(f"Scaler saved: {scaler_path}")
    
    if 'encoders' in preprocessing_objects:
        encoders_path = os.path.join(output_dir, 'encoders.joblib')
        joblib.dump(preprocessing_objects['encoders'], encoders_path)
        print(f"Encoders saved: {encoders_path}")
    
    # Save feature names
    features_path = os.path.join(output_dir, 'feature_names.joblib')
    joblib.dump(feature_names, features_path)
    print(f"Feature names saved: {features_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'target_column': target_column,
        'auc_score': model_metrics['auc'],
        'feature_count': len(feature_names)
    }
    metadata_path = os.path.join(output_dir, 'model_metadata.joblib')
    joblib.dump(metadata, metadata_path)
    print(f"Metadata saved: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Train SmartAlert AI models')
    parser.add_argument('--data', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--target', required=True, help='Name of the target column')
    parser.add_argument('--timestamp', help='Name of the timestamp column')
    parser.add_argument('--output-dir', default='models', help='Output directory for model artifacts')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    print("SmartAlert AI - Model Training")
    print("=" * 40)
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Dataset shape: {df.shape}")
    
    if args.target not in df.columns:
        print(f"Error: Target column '{args.target}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Preprocessing
    print("\nPreprocessing data...")
    df_processed, preprocessing_objects = preprocess_log_data(
        df, 
        timestamp_col=args.timestamp,
        target_col=args.target
    )
    
    # Prepare features and target
    X = df_processed.drop(columns=[args.target])
    y = df_processed[args.target]
    
    # Ensure all features are numeric
    numeric_features = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_features]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    # Train models
    best_model, best_model_name, results = train_models(X_train, X_test, y_train, y_test)
    
    # Hyperparameter tuning
    if args.tune:
        best_model = hyperparameter_tuning(best_model, best_model_name, X_train, y_train)
    
    # Final evaluation
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    if len(np.unique(y)) == 2:
        final_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        final_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    print(f"\n=== Final Results ===")
    print(f"Best Model: {best_model_name}")
    print(f"Final AUC Score: {final_auc:.4f}")
    
    # Save artifacts
    model_metrics = {'auc': final_auc}
    save_model_artifacts(
        best_model, best_model_name, preprocessing_objects,
        list(X.columns), args.target, model_metrics, args.output_dir
    )
    
    print(f"\nTraining completed! All artifacts saved to '{args.output_dir}' directory.")


if __name__ == '__main__':
    main() 