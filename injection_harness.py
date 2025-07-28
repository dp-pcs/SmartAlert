#!/usr/bin/env python3
"""
SmartAlert AI - Adaptive Training Injection Harness (Case-Based Version)

This module implements an adaptive system that simulates multiple rounds of log ingestion,
retrains models each time, and tracks how performance evolves over time.

Enhanced for case-based incident tracking with realistic low issue rates.

Features:
- Incremental learning with batch processing
- Model drift detection
- Performance tracking across multiple metrics
- Case-based feature engineering with incident progression
- Support for RandomForest, XGBoost, and LightGBM
- Model versioning and artifact management
- Realistic incident patterns (low issue rates)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
from pathlib import Path
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our case-based feature engineering
try:
    from utils.case_feature_engineering import preprocess_case_data
    CASE_FEATURES_AVAILABLE = True
except ImportError:
    from utils.feature_engineering import preprocess_log_data
    CASE_FEATURES_AVAILABLE = False
    print("⚠️ Case-based features not available, falling back to basic features")

warnings.filterwarnings('ignore')


class AdaptiveModelTracker:
    """
    Tracks model performance and detects drift across training rounds.
    Enhanced for case-based incident prediction with realistic metrics.
    """
    
    def __init__(self, drift_threshold=0.03):  # Lower threshold for realistic data
        self.history = []
        self.models = {}
        self.drift_threshold = drift_threshold  # 3% performance drop indicates drift
        
    def add_round(self, round_num, metrics, model, data_stats):
        """Add results from a training round."""
        self.history.append({
            'round': round_num,
            'timestamp': datetime.now(),
            'metrics': metrics,
            'data_stats': data_stats
        })
        self.models[round_num] = model
        
    def detect_drift(self, current_metrics, metric='f1'):
        """Detect if model performance has degraded."""
        if len(self.history) < 2:
            return False, 0.0
            
        previous_score = self.history[-2]['metrics'][metric]
        current_score = current_metrics[metric]
        drift_amount = previous_score - current_score
        
        return drift_amount > self.drift_threshold, drift_amount
    
    def get_best_model(self, metric='f1'):
        """Get the best performing model based on specified metric."""
        if not self.history:
            return None, None
            
        best_round = max(self.history, key=lambda x: x['metrics'][metric])
        return best_round['round'], self.models[best_round['round']]
    
    def plot_performance(self, save_path=None):
        """Plot performance trends over time."""
        if len(self.history) < 2:
            print("Need at least 2 rounds to plot trends")
            return
            
        df = pd.DataFrame([
            {
                'round': h['round'],
                'precision': h['metrics']['precision'],
                'recall': h['metrics']['recall'],
                'f1': h['metrics']['f1'],
                'auc': h['metrics']['auc'],
                'accuracy': h['metrics']['accuracy'],
                'cases_predicted': h['data_stats'].get('cases_with_predictions', 0),
                'issue_rate': h['data_stats']['issue_rate']
            }
            for h in self.history
        ])
        
        plt.figure(figsize=(15, 10))
        
        # Performance metrics
        plt.subplot(2, 3, 1)
        plt.plot(df['round'], df['precision'], 'o-', label='Precision', linewidth=2)
        plt.plot(df['round'], df['recall'], 's-', label='Recall', linewidth=2)
        plt.plot(df['round'], df['f1'], '^-', label='F1-Score', linewidth=2)
        plt.title('Performance Metrics Over Time')
        plt.xlabel('Training Round')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # AUC and Accuracy
        plt.subplot(2, 3, 2)
        plt.plot(df['round'], df['auc'], 'o-', label='AUC', color='purple', linewidth=2)
        plt.plot(df['round'], df['accuracy'], 's-', label='Accuracy', color='orange', linewidth=2)
        plt.title('AUC and Accuracy Over Time')
        plt.xlabel('Training Round')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Data volume
        plt.subplot(2, 3, 3)
        data_sizes = [h['data_stats']['total_samples'] for h in self.history]
        plt.plot(df['round'], data_sizes, 'o-', color='green', linewidth=2)
        plt.title('Cumulative Data Size')
        plt.xlabel('Training Round')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3)
        
        # Issue rate tracking
        plt.subplot(2, 3, 4)
        plt.plot(df['round'], df['issue_rate'] * 100, 'o-', color='red', linewidth=2)
        plt.title('Issue Rate Over Time')
        plt.xlabel('Training Round')
        plt.ylabel('Issue Rate (%)')
        plt.grid(True, alpha=0.3)
        
        # Cases predicted (if available)
        plt.subplot(2, 3, 5)
        if 'cases_predicted' in df.columns:
            plt.plot(df['round'], df['cases_predicted'], 'o-', color='blue', linewidth=2)
            plt.title('Cases with Predictions')
            plt.xlabel('Training Round')
            plt.ylabel('Number of Cases')
            plt.grid(True, alpha=0.3)
        
        # Model comparison (F1 trend with drift markers)
        plt.subplot(2, 3, 6)
        plt.plot(df['round'], df['f1'], 'b-o', linewidth=2, label='F1-Score')
        
        # Mark drift detection points
        for i, h in enumerate(self.history[1:], 1):  # Skip first round
            if len(self.history) > i:
                drift_detected, _ = self.detect_drift(h['metrics'])
                if drift_detected:
                    plt.axvline(x=h['round'], color='red', linestyle='--', alpha=0.7)
        
        plt.title('F1-Score Trend with Drift Detection')
        plt.xlabel('Training Round')
        plt.ylabel('F1-Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def get_model(model_name, **kwargs):
    """
    Get a model instance optimized for imbalanced data (low issue rates).
    
    Args:
        model_name (str): Model type ('rf', 'xgb', 'lgb')
        **kwargs: Additional model parameters
    
    Returns:
        Model instance
    """
    default_params = {
        'random_state': 42,
        'n_jobs': -1
    }
    default_params.update(kwargs)
    
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=200,  # More trees for better performance
            class_weight='balanced',  # Handle imbalanced data
            **default_params
        )
    elif model_name == "xgb":
        return xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=10,  # Handle imbalanced data
            n_estimators=200,
            **default_params
        )
    elif model_name == "lgb":
        return lgb.LGBMClassifier(
            objective='binary',
            class_weight='balanced',  # Handle imbalanced data
            n_estimators=200,
            verbose=-1,
            **default_params
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Choose from 'rf', 'xgb', or 'lgb'.")


def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation optimized for imbalanced data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        dict: Comprehensive metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Basic metrics (handle zero division)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    accuracy = accuracy_score(y_test, y_pred)
    
    # AUC score (important for imbalanced data)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Additional metrics for imbalanced data
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
    
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
        'classification_report': classification_report(y_test, y_pred)
    }


def run_training_injection_harness(
    data_path,
    model_name="rf",
    batch_size=10000,
    num_batches=5,
    target_column="led_to_issue",
    timestamp_column="timestamp",
    case_id_column="case_id",
    test_size=0.2,
    output_dir="models/adaptive",
    save_models=True,
    verbose=True,
    use_case_features=True
):
    """
    Run the adaptive training injection harness for case-based incident prediction.
    
    Args:
        data_path (str): Path to the dataset CSV file
        model_name (str): Model type ('rf', 'xgb', 'lgb')
        batch_size (int): Number of samples per batch
        num_batches (int): Number of training rounds
        target_column (str): Name of the target column
        timestamp_column (str): Name of the timestamp column
        case_id_column (str): Name of the case ID column
        test_size (float): Proportion of data for testing
        output_dir (str): Directory to save model artifacts
        save_models (bool): Whether to save model artifacts
        verbose (bool): Whether to print detailed progress
        use_case_features (bool): Whether to use case-based feature engineering
    
    Returns:
        tuple: (results_df, tracker, final_model)
    """
    
    if verbose:
        print("🧪 SmartAlert AI - Adaptive Training Injection Harness (Case-Based)")
        print("=" * 70)
    
    # Load and prepare data
    if verbose:
        print(f"📊 Loading case-based incident data from {data_path}...")
    
    df = pd.read_csv(data_path)
    
    # Validate columns
    required_cols = [target_column, timestamp_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
    
    # Sort by timestamp for realistic batch processing
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values(timestamp_column).reset_index(drop=True)
    
    if verbose:
        print(f"📈 Dataset loaded: {df.shape[0]:,} samples")
        print(f"📅 Time range: {df[timestamp_column].min()} to {df[timestamp_column].max()}")
        print(f"🎯 Issue rate: {df[target_column].mean():.1%} ({df[target_column].sum():,} issues)")
        
        # Case statistics
        if case_id_column in df.columns:
            cases_with_ids = df[df[case_id_column].notna() & (df[case_id_column] != '')]
            unique_cases = cases_with_ids[case_id_column].nunique()
            print(f"📋 Cases tracked: {unique_cases:,} unique cases")
    
    # Initialize tracker
    tracker = AdaptiveModelTracker()
    results = []
    cumulative_data = pd.DataFrame()
    
    # Create output directory
    if save_models:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run incremental training rounds
    for round_num in range(1, num_batches + 1):
        if verbose:
            print(f"\n🔄 Training Round {round_num}/{num_batches}")
            print("-" * 50)
        
        # Get batch data (chronological batches)
        start_idx = (round_num - 1) * batch_size
        end_idx = min(round_num * batch_size, len(df))
        batch_data = df.iloc[start_idx:end_idx].copy()
        
        # Add to cumulative data
        cumulative_data = pd.concat([cumulative_data, batch_data], axis=0, ignore_index=True)
        
        if verbose:
            print(f"📦 Batch size: {len(batch_data):,} samples")
            print(f"📊 Cumulative size: {len(cumulative_data):,} samples")
            print(f"🎯 Batch issue rate: {batch_data[target_column].mean():.1%}")
            print(f"📈 Cumulative issue rate: {cumulative_data[target_column].mean():.1%}")
        
        # Apply feature engineering
        try:
            if use_case_features and CASE_FEATURES_AVAILABLE:
                if verbose:
                    print("🔧 Applying case-based feature engineering...")
                processed_data, preprocessing_objects = preprocess_case_data(
                    cumulative_data,
                    case_id_col=case_id_column,
                    timestamp_col=timestamp_column,
                    target_col=target_column
                )
            else:
                if verbose:
                    print("🔧 Applying basic feature engineering...")
                processed_data, preprocessing_objects = preprocess_log_data(
                    cumulative_data,
                    timestamp_col=timestamp_column,
                    target_col=target_column
                )
        except Exception as e:
            if verbose:
                print(f"⚠️ Feature engineering failed: {e}")
                print("   Falling back to basic preprocessing...")
            
            # Minimal preprocessing fallback
            processed_data = cumulative_data.copy()
            for col in ['severity', 'component']:
                if col in processed_data.columns:
                    processed_data[col] = processed_data[col].astype('category').cat.codes
            preprocessing_objects = {}
        
        # Prepare features and target
        feature_columns = [col for col in processed_data.columns if col != target_column]
        X = processed_data[feature_columns].select_dtypes(include=[np.number])
        y = processed_data[target_column]
        
        if len(X.columns) == 0:
            if verbose:
                print("❌ No numeric features available, skipping this round")
            continue
        
        # Check for sufficient data and class balance
        if len(X) < 50 or y.sum() < 2:  # Need at least 2 positive examples
            if verbose:
                print(f"⚠️ Insufficient data or no positive examples (issues: {y.sum()}), skipping round")
            continue
            
        # Split data ensuring stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError as e:
            if verbose:
                print(f"⚠️ Stratification failed: {e}. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        if verbose:
            print(f"🎯 Training samples: {len(X_train):,} (issues: {y_train.sum()}, {y_train.mean():.1%})")
            print(f"🧪 Test samples: {len(X_test):,} (issues: {y_test.sum()}, {y_test.mean():.1%})")
            print(f"📊 Features: {len(X.columns)}")
        
        # Train model
        model = get_model(model_name)
        
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            if verbose:
                print(f"❌ Training failed: {e}")
            continue
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Calculate data statistics
        data_stats = {
            'total_samples': len(cumulative_data),
            'batch_samples': len(batch_data),
            'issue_rate': y.mean(),
            'batch_issue_rate': batch_data[target_column].mean(),
            'feature_count': len(X.columns),
            'issues_in_test': y_test.sum(),
            'cases_with_predictions': 0  # Could be enhanced to track case-level predictions
        }
        
        # Case-specific statistics
        if case_id_column in cumulative_data.columns:
            cases_with_ids = cumulative_data[cumulative_data[case_id_column].notna() & (cumulative_data[case_id_column] != '')]
            data_stats['unique_cases'] = cases_with_ids[case_id_column].nunique()
            data_stats['logs_with_cases'] = len(cases_with_ids)
        
        # Check for drift
        drift_detected, drift_amount = tracker.detect_drift(metrics)
        
        if verbose:
            print(f"📈 Performance Metrics:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1']:.4f}")
            print(f"   AUC: {metrics['auc']:.4f}")
            print(f"   Specificity: {metrics['specificity']:.4f}")
            
            if drift_detected:
                print(f"⚠️ Model drift detected! Performance dropped by {drift_amount:.4f}")
            else:
                print("✅ No significant model drift detected")
        
        # Save results
        round_result = {
            'round': round_num,
            'model_name': model_name,
            'cumulative_samples': len(cumulative_data),
            'batch_samples': len(batch_data),
            'feature_count': len(X.columns),
            'drift_detected': drift_detected,
            'drift_amount': drift_amount,
            **metrics,
            **data_stats
        }
        results.append(round_result)
        
        # Add to tracker
        tracker.add_round(round_num, metrics, model, data_stats)
        
        # Save model artifacts
        if save_models:
            model_path = os.path.join(output_dir, f"{model_name}_round_{round_num}.joblib")
            joblib.dump(model, model_path)
            
            # Save preprocessing objects
            if preprocessing_objects:
                preprocessing_path = os.path.join(output_dir, f"preprocessing_round_{round_num}.joblib")
                joblib.dump(preprocessing_objects, preprocessing_path)
            
            # Save feature names
            feature_path = os.path.join(output_dir, f"features_round_{round_num}.joblib")
            joblib.dump(list(X.columns), feature_path)
            
            if verbose:
                print(f"💾 Model artifacts saved to {output_dir}")
    
    # Final summary
    if verbose:
        print(f"\n🏁 Adaptive Training Complete!")
        print("=" * 70)
        print(f"🎯 Total rounds completed: {len(results)}")
        
        if results:
            best_round, best_model = tracker.get_best_model()
            best_f1 = max(r['f1'] for r in results)
            best_auc = max(r['auc'] for r in results)
            print(f"🏆 Best performing round: {best_round} (F1: {best_f1:.4f}, AUC: {best_auc:.4f})")
            
            drift_rounds = sum(1 for r in results if r.get('drift_detected', False))
            print(f"⚠️ Rounds with drift detected: {drift_rounds}")
            
            final_issue_rate = results[-1]['issue_rate'] if results else 0
            print(f"📊 Final issue rate: {final_issue_rate:.1%}")
    
    results_df = pd.DataFrame(results)
    final_model = tracker.models.get(len(results), None) if results else None
    
    return results_df, tracker, final_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SmartAlert AI Adaptive Training Harness (Case-Based)')
    parser.add_argument('--data', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--model', default='rf', choices=['rf', 'xgb', 'lgb'], 
                       help='Model type (default: rf)')
    parser.add_argument('--batch-size', type=int, default=8000, 
                       help='Number of samples per batch (default: 8000)')
    parser.add_argument('--num-batches', type=int, default=5, 
                       help='Number of training rounds (default: 5)')
    parser.add_argument('--target', default='led_to_issue', 
                       help='Target column name (default: led_to_issue)')
    parser.add_argument('--timestamp', default='timestamp', 
                       help='Timestamp column name (default: timestamp)')
    parser.add_argument('--case-id', default='case_id', 
                       help='Case ID column name (default: case_id)')
    parser.add_argument('--output-dir', default='models/adaptive', 
                       help='Output directory for models (default: models/adaptive)')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate performance plots')
    parser.add_argument('--no-case-features', action='store_true',
                       help='Disable case-based feature engineering')
    
    args = parser.parse_args()
    
    # Run the harness
    results_df, tracker, final_model = run_training_injection_harness(
        data_path=args.data,
        model_name=args.model,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        target_column=args.target,
        timestamp_column=args.timestamp,
        case_id_column=args.case_id,
        output_dir=args.output_dir,
        use_case_features=not args.no_case_features
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, f"adaptive_results_{args.model}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n📊 Results saved to: {results_path}")
    
    # Generate plots
    if args.plot:
        plot_path = os.path.join(args.output_dir, f"performance_trends_{args.model}.png")
        tracker.plot_performance(save_path=plot_path)
        print(f"📈 Performance plots saved to: {plot_path}")
