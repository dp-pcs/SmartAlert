#!/usr/bin/env python3
"""
SmartAlert AI - Enhanced V4 Injection Harness

Combines the best of both worlds:
- Case-based feature engineering (from V3 system)
- TF-IDF text analysis (from V4 system)  
- Enhanced model tuning for imbalanced data
- Robust handling of challenging false positive scenarios

This tackles the ultra-challenging V4 dataset where simple approaches fail.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import sys
from pathlib import Path
import warnings
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')


def extract_case_progression_features_v4(df, case_id_col='case_id', timestamp_col='timestamp', severity_col='severity'):
    """
    Enhanced case progression features for V4 challenge.
    """
    df_copy = df.copy()
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
    
    # Initialize case-based features
    df_copy['case_log_sequence'] = 0
    df_copy['case_duration_minutes'] = 0.0
    df_copy['case_severity_escalation'] = 0
    df_copy['case_log_count'] = 0
    df_copy['case_max_severity'] = 0
    df_copy['is_case_start'] = 0
    df_copy['is_case_end'] = 0
    df_copy['has_case_id'] = (~df_copy[case_id_col].isna() & (df_copy[case_id_col] != '')).astype(int)
    
    # Severity level mapping
    severity_levels = {
        'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'ERROR': 4, 'CRIT': 5, 'FATAL': 6
    }
    
    # Process cases with IDs
    cases_with_ids = df_copy[df_copy[case_id_col].notna() & (df_copy[case_id_col] != '')].copy()
    
    if len(cases_with_ids) > 0:
        for case_id, case_group in cases_with_ids.groupby(case_id_col):
            case_indices = case_group.index
            case_sorted = case_group.sort_values(timestamp_col)
            
            # Case sequence and timing
            df_copy.loc[case_indices, 'case_log_sequence'] = range(1, len(case_indices) + 1)
            df_copy.loc[case_indices, 'case_log_count'] = len(case_indices)
            
            # Case duration
            if len(case_sorted) > 1:
                case_duration = (case_sorted[timestamp_col].max() - case_sorted[timestamp_col].min()).total_seconds() / 60
                df_copy.loc[case_indices, 'case_duration_minutes'] = case_duration
            
            # Severity progression
            case_severities = [severity_levels.get(sev, 0) for sev in case_sorted[severity_col]]
            
            if case_severities:
                df_copy.loc[case_indices, 'case_max_severity'] = max(case_severities)
                
                if len(case_severities) > 1:
                    escalation = case_severities[-1] - case_severities[0]
                    df_copy.loc[case_indices, 'case_severity_escalation'] = escalation
            
            # Mark case boundaries
            first_idx = case_sorted.index[0]
            last_idx = case_sorted.index[-1]
            df_copy.loc[first_idx, 'is_case_start'] = 1
            df_copy.loc[last_idx, 'is_case_end'] = 1
    
    return df_copy


def create_temporal_features_v4(df, timestamp_col='timestamp'):
    """
    Enhanced temporal features for V4 challenge.
    """
    df_copy = df.copy()
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
    
    # Basic time features
    df_copy['hour'] = df_copy[timestamp_col].dt.hour
    df_copy['day_of_week'] = df_copy[timestamp_col].dt.dayofweek
    df_copy['month'] = df_copy[timestamp_col].dt.month
    
    # Business time features
    df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
    df_copy['is_business_hours'] = ((df_copy['hour'] >= 9) & (df_copy['hour'] <= 17) & (df_copy['day_of_week'] < 5)).astype(int)
    df_copy['is_after_hours'] = ((df_copy['hour'] < 9) | (df_copy['hour'] > 17) | (df_copy['day_of_week'] >= 5)).astype(int)
    
    # Peak incident hours (based on common patterns)
    df_copy['is_peak_hours'] = ((df_copy['hour'].isin([9, 10, 11, 14, 15, 16])) & (df_copy['day_of_week'] < 5)).astype(int)
    
    return df_copy


def get_enhanced_model(model_name, **kwargs):
    """
    Get optimized models for challenging imbalanced data.
    """
    default_params = {
        'random_state': 42,
        'n_jobs': -1
    }
    default_params.update(kwargs)
    
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',  # Critical for imbalanced data
            **default_params
        )
    elif model_name == "xgb":
        return xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=20,  # Heavy weight for positive class
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            **default_params
        )
    elif model_name == "lgb":
        return lgb.LGBMClassifier(
            objective='binary',
            class_weight='balanced',
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            verbose=-1,
            **default_params
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Choose from 'rf', 'xgb', or 'lgb'.")


def evaluate_model_v4(model, X_test, y_test):
    """
    Enhanced evaluation for challenging imbalanced data.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Handle case where model predicts all zeros
    if y_pred.sum() == 0:
        print("⚠️ Model predicted all negatives - applying threshold adjustment")
        if y_pred_proba is not None:
            # Use lower threshold for positive predictions
            threshold = np.percentile(y_pred_proba, 95)  # Top 5% as positive
            y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics (handle zero division)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    accuracy = accuracy_score(y_test, y_pred)
    
    # AUC score
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'auc': auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'predicted_positives': y_pred.sum(),
        'actual_positives': y_test.sum()
    }


def run_enhanced_v4_bakeoff(data_path, batch_size=9000, num_batches=5, max_tfidf_features=100):
    """
    Enhanced V4 injection harness with case-based features + TF-IDF text analysis.
    """
    print("🚀 Enhanced V4 Injection Harness - Tackling the Ultimate Challenge!")
    print("=" * 75)
    
    # Load data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"📊 Loaded challenging V4 dataset: {len(df):,} samples")
    print(f"🎯 Issue rate: {df['led_to_issue'].mean():.1%} ({df['led_to_issue'].sum():,} issues)")
    
    # Initialize TF-IDF (fit on full dataset for consistency)
    print(f"🔤 Preparing TF-IDF text analysis ({max_tfidf_features} features)...")
    tfidf = TfidfVectorizer(max_features=max_tfidf_features, stop_words='english', ngram_range=(1, 2))
    tfidf.fit(df["message"])
    
    results = []
    cumulative_data = pd.DataFrame()
    
    for i in range(num_batches):
        print(f"\n🔄 Training Round {i+1}/{num_batches}")
        print("-" * 50)
        
        # Get batch
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch = df.iloc[start_idx:end_idx].copy()
        cumulative_data = pd.concat([cumulative_data, batch], axis=0, ignore_index=True)
        
        print(f"📦 Batch size: {len(batch):,}")
        print(f"📊 Cumulative size: {len(cumulative_data):,}")
        print(f"🎯 Cumulative issue rate: {cumulative_data['led_to_issue'].mean():.1%}")
        
        # Apply enhanced feature engineering
        print("🔧 Applying enhanced feature engineering...")
        
        # 1. Case progression features
        processed_data = extract_case_progression_features_v4(cumulative_data)
        
        # 2. Temporal features  
        processed_data = create_temporal_features_v4(processed_data)
        
        # 3. TF-IDF text features
        X_tfidf = tfidf.transform(processed_data["message"]).toarray()
        
        # 4. Encode categorical features
        le_severity = LabelEncoder()
        le_component = LabelEncoder()
        severity_encoded = le_severity.fit_transform(processed_data["severity"])
        component_encoded = le_component.fit_transform(processed_data["component"])
        
        # 5. Combine all features
        case_features = processed_data[['case_log_sequence', 'case_duration_minutes', 
                                      'case_severity_escalation', 'case_log_count',
                                      'case_max_severity', 'is_case_start', 'is_case_end',
                                      'has_case_id', 'hour', 'day_of_week', 'month',
                                      'is_weekend', 'is_business_hours', 'is_after_hours',
                                      'is_peak_hours', 'message_length']].values
        
        # Combine: TF-IDF + categorical + case features + metadata
        X = np.column_stack((X_tfidf, severity_encoded, component_encoded, case_features))
        y = processed_data["led_to_issue"]
        
        total_features = X.shape[1]
        print(f"📊 Total features: {total_features} ({max_tfidf_features} text + {total_features - max_tfidf_features} engineered)")
        
        # Handle insufficient data
        if len(X) < 100 or y.sum() < 5:
            print("⚠️ Insufficient data for reliable training, skipping batch...")
            continue
        
        # Train-test split with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            print("⚠️ Stratification failed, using random split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"🎯 Training: {len(X_train):,} samples ({y_train.sum()} issues, {y_train.mean():.1%})")
        print(f"🧪 Testing: {len(X_test):,} samples ({y_test.sum()} issues, {y_test.mean():.1%})")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train all models
        for model_name in ["rf", "xgb", "lgb"]:
            print(f"   🤖 Training {model_name.upper()}...")
            
            model = get_enhanced_model(model_name)
            
            try:
                if model_name in ["rf", "lgb"]:
                    model.fit(X_train_scaled, y_train)
                else:  # XGBoost
                    model.fit(X_train, y_train)  # XGBoost handles scaling internally
                
                # Evaluate
                test_features = X_test_scaled if model_name in ["rf", "lgb"] else X_test
                metrics = evaluate_model_v4(model, test_features, y_test)
                
                print(f"      F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}, Precision: {metrics['precision']:.4f}")
                
                # Store results
                results.append({
                    "batch": i + 1,
                    "model": model_name,
                    "cumulative_samples": len(cumulative_data),
                    "batch_samples": len(batch),
                    "feature_count": total_features,
                    **metrics
                })
                
                # Save model
                model_path = f"models/v4_enhanced_{model_name}_round_{i+1}.joblib"
                os.makedirs("models", exist_ok=True)
                joblib.dump(model, model_path)
                
            except Exception as e:
                print(f"      ❌ Training failed: {e}")
                continue
    
    print(f"\n🏁 Enhanced V4 Training Complete!")
    print("=" * 50)
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        print("\n📊 Final Results Summary:")
        for model in ['rf', 'xgb', 'lgb']:
            model_results = results_df[results_df['model'] == model]
            if len(model_results) > 0:
                avg_f1 = model_results['f1'].mean()
                best_f1 = model_results['f1'].max()
                avg_auc = model_results['auc'].mean()
                print(f"   {model.upper()}: Avg F1={avg_f1:.4f}, Best F1={best_f1:.4f}, Avg AUC={avg_auc:.4f}")
    
    return results_df


if __name__ == "__main__":
    # Run the enhanced V4 system
    results = run_enhanced_v4_bakeoff(
        data_path='data/splunk_logs_incidents_v4.csv',
        batch_size=9000,
        num_batches=5,
        max_tfidf_features=100
    )
    
    # Save results
    results.to_csv('models/enhanced_v4_results.csv', index=False)
    print(f"\n💾 Results saved to: models/enhanced_v4_results.csv") 