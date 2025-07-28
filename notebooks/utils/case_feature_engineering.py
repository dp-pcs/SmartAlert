"""
Case-Based Feature Engineering for SmartAlert AI

This module extends the basic feature engineering to handle case-based incident tracking,
including case progression analysis, temporal patterns within cases, and incident escalation features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def extract_case_progression_features(df, case_id_col='case_id', timestamp_col='timestamp', severity_col='severity'):
    """
    Extract case progression features from incident logs.
    
    Args:
        df (pd.DataFrame): Input dataframe with case tracking
        case_id_col (str): Name of case ID column
        timestamp_col (str): Name of timestamp column  
        severity_col (str): Name of severity column
    
    Returns:
        pd.DataFrame: Dataframe with case progression features
    """
    df_copy = df.copy()
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
    
    # Initialize case-based features
    df_copy['case_log_sequence'] = 0
    df_copy['case_duration_minutes'] = 0.0
    df_copy['case_severity_escalation'] = 0
    df_copy['case_log_count'] = 0
    df_copy['case_max_severity'] = 0
    df_copy['case_min_severity'] = 0
    df_copy['is_case_start'] = 0
    df_copy['is_case_end'] = 0
    df_copy['time_since_case_start'] = 0.0
    df_copy['severity_change_from_previous'] = 0
    
    # Severity level mapping for progression analysis
    severity_levels = {
        'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'ERROR': 4, 'CRIT': 5, 'FATAL': 6
    }
    
    # Process cases with IDs
    cases_with_ids = df_copy[df_copy[case_id_col].notna() & (df_copy[case_id_col] != '')].copy()
    
    if len(cases_with_ids) > 0:
        # Group by case and add progression features
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
                
                # Time since case start for each log
                for idx, row in case_sorted.iterrows():
                    time_since_start = (row[timestamp_col] - case_sorted[timestamp_col].min()).total_seconds() / 60
                    df_copy.loc[idx, 'time_since_case_start'] = time_since_start
            
            # Severity progression analysis
            case_severities = [severity_levels.get(sev, 0) for sev in case_sorted[severity_col]]
            
            if case_severities:
                df_copy.loc[case_indices, 'case_max_severity'] = max(case_severities)
                df_copy.loc[case_indices, 'case_min_severity'] = min(case_severities)
                
                # Calculate escalation (how much severity increased from start to end)
                if len(case_severities) > 1:
                    escalation = case_severities[-1] - case_severities[0]
                    df_copy.loc[case_indices, 'case_severity_escalation'] = escalation
                    
                    # Severity change from previous log
                    for i, idx in enumerate(case_sorted.index):
                        if i > 0:
                            change = case_severities[i] - case_severities[i-1]
                            df_copy.loc[idx, 'severity_change_from_previous'] = change
            
            # Mark case boundaries
            first_idx = case_sorted.index[0]
            last_idx = case_sorted.index[-1]
            df_copy.loc[first_idx, 'is_case_start'] = 1
            df_copy.loc[last_idx, 'is_case_end'] = 1
    
    return df_copy


def create_temporal_features_advanced(df, timestamp_col='timestamp'):
    """
    Create advanced temporal features including business hours, shift patterns, etc.
    
    Args:
        df (pd.DataFrame): Input dataframe
        timestamp_col (str): Name of timestamp column
    
    Returns:
        pd.DataFrame: Dataframe with advanced temporal features
    """
    df_copy = df.copy()
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
    
    # Basic time features
    df_copy['hour'] = df_copy[timestamp_col].dt.hour
    df_copy['day_of_week'] = df_copy[timestamp_col].dt.dayofweek
    df_copy['day_of_month'] = df_copy[timestamp_col].dt.day
    df_copy['month'] = df_copy[timestamp_col].dt.month
    df_copy['quarter'] = df_copy[timestamp_col].dt.quarter
    
    # Business time features
    df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
    df_copy['is_business_hours'] = ((df_copy['hour'] >= 9) & (df_copy['hour'] <= 17) & (df_copy['day_of_week'] < 5)).astype(int)
    df_copy['is_after_hours'] = ((df_copy['hour'] < 9) | (df_copy['hour'] > 17) | (df_copy['day_of_week'] >= 5)).astype(int)
    
    # Shift patterns (common in IT operations)
    def get_shift(hour):
        if 6 <= hour < 14:
            return 1  # Day shift
        elif 14 <= hour < 22:
            return 2  # Evening shift
        else:
            return 3  # Night shift
    
    df_copy['shift'] = df_copy['hour'].apply(get_shift)
    
    # Peak hours (when incidents are more likely)
    df_copy['is_peak_hours'] = ((df_copy['hour'].isin([9, 10, 11, 14, 15, 16])) & (df_copy['day_of_week'] < 5)).astype(int)
    
    return df_copy


def create_case_aggregation_features(df, case_id_col='case_id', target_col='led_to_issue'):
    """
    Create aggregation features based on case patterns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        case_id_col (str): Name of case ID column
        target_col (str): Name of target column
    
    Returns:
        pd.DataFrame: Dataframe with case aggregation features
    """
    df_copy = df.copy()
    
    # Global aggregations (for logs without case IDs)
    df_copy['total_logs_in_timeframe'] = len(df_copy)
    
    # For logs with case IDs, add case-level statistics
    cases_with_ids = df_copy[df_copy[case_id_col].notna() & (df_copy[case_id_col] != '')]
    
    if len(cases_with_ids) > 0:
        # Calculate case-level statistics
        case_stats = cases_with_ids.groupby(case_id_col).agg({
            target_col: ['mean', 'max', 'sum'],
            'severity': 'count',
            'message_length': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        case_stats.columns = ['_'.join(col).strip() for col in case_stats.columns]
        case_stats = case_stats.add_prefix('case_')
        
        # Merge back to main dataframe
        df_copy = df_copy.merge(case_stats, left_on=case_id_col, right_index=True, how='left')
        
        # Fill NaN values for logs without case IDs
        case_feature_cols = [col for col in df_copy.columns if col.startswith('case_')]
        for col in case_feature_cols:
            if col not in df_copy.columns:
                continue
            df_copy[col] = df_copy[col].fillna(0)
    
    return df_copy


def detect_anomalous_patterns(df, case_id_col='case_id', severity_col='severity', component_col='component'):
    """
    Detect anomalous patterns that might indicate incidents.
    
    Args:
        df (pd.DataFrame): Input dataframe
        case_id_col (str): Name of case ID column
        severity_col (str): Name of severity column
        component_col (str): Name of component column
    
    Returns:
        pd.DataFrame: Dataframe with anomaly detection features
    """
    df_copy = df.copy()
    
    # Initialize anomaly features
    df_copy['is_severity_anomaly'] = 0
    df_copy['is_component_anomaly'] = 0
    df_copy['is_rapid_escalation'] = 0
    df_copy['is_unusual_case_pattern'] = 0
    
    # Severity anomaly detection (unusual severity for component)
    severity_by_component = df_copy.groupby(component_col)[severity_col].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'INFO')
    
    for idx, row in df_copy.iterrows():
        normal_severity = severity_by_component.get(row[component_col], 'INFO')
        if row[severity_col] != normal_severity:
            severity_levels = {'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'ERROR': 4, 'CRIT': 5, 'FATAL': 6}
            current_level = severity_levels.get(row[severity_col], 2)
            normal_level = severity_levels.get(normal_severity, 2)
            
            if current_level > normal_level + 1:  # More than one level above normal
                df_copy.loc[idx, 'is_severity_anomaly'] = 1
    
    # Case-based anomaly detection
    cases_with_ids = df_copy[df_copy[case_id_col].notna() & (df_copy[case_id_col] != '')]
    
    if len(cases_with_ids) > 0:
        for case_id, case_group in cases_with_ids.groupby(case_id_col):
            case_indices = case_group.index
            
            # Rapid escalation detection (WARN to FATAL in same case)
            severities = case_group[severity_col].tolist()
            if 'WARN' in severities and 'FATAL' in severities:
                df_copy.loc[case_indices, 'is_rapid_escalation'] = 1
            
            # Unusual case patterns (too many logs, unusual components)
            if len(case_group) > 5:  # More than 5 logs per case is unusual
                df_copy.loc[case_indices, 'is_unusual_case_pattern'] = 1
    
    return df_copy


def preprocess_case_data(df, case_id_col='case_id', timestamp_col='timestamp', 
                        target_col='led_to_issue', severity_col='severity', 
                        component_col='component'):
    """
    Main preprocessing pipeline for case-based incident data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        case_id_col (str): Name of case ID column
        timestamp_col (str): Name of timestamp column
        target_col (str): Name of target column
        severity_col (str): Name of severity column
        component_col (str): Name of component column
    
    Returns:
        tuple: (processed_df, preprocessing_objects)
    """
    print("🔧 Starting case-based feature engineering...")
    
    df_processed = df.copy()
    preprocessing_objects = {}
    
    # 1. Extract case progression features
    print("   📊 Extracting case progression features...")
    df_processed = extract_case_progression_features(
        df_processed, case_id_col, timestamp_col, severity_col
    )
    
    # 2. Create advanced temporal features
    print("   ⏰ Creating temporal features...")
    df_processed = create_temporal_features_advanced(df_processed, timestamp_col)
    
    # 3. Create case aggregation features
    print("   📈 Creating case aggregation features...")
    df_processed = create_case_aggregation_features(df_processed, case_id_col, target_col)
    
    # 4. Detect anomalous patterns
    print("   🚨 Detecting anomalous patterns...")
    df_processed = detect_anomalous_patterns(df_processed, case_id_col, severity_col, component_col)
    
    # 5. Encode categorical features
    print("   🔤 Encoding categorical features...")
    categorical_cols = [severity_col, component_col]
    if case_id_col in df_processed.columns:
        # Don't encode case_id as it's an identifier, but create a flag
        df_processed['has_case_id'] = (~df_processed[case_id_col].isna() & (df_processed[case_id_col] != '')).astype(int)
    
    encoders = {}
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le
    
    preprocessing_objects['encoders'] = encoders
    
    # 6. Handle missing values
    print("   🔧 Handling missing values...")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    
    # 7. Feature scaling
    print("   📏 Scaling features...")
    scaler = StandardScaler()
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    preprocessing_objects['scaler'] = scaler
    preprocessing_objects['feature_columns'] = numeric_cols
    
    print(f"✅ Feature engineering complete! Created {len(numeric_cols)} numeric features")
    
    return df_processed, preprocessing_objects


def get_case_level_features(df, case_id_col='case_id', target_col='led_to_issue'):
    """
    Aggregate log-level data to case-level for case-based prediction.
    
    Args:
        df (pd.DataFrame): Input dataframe with log-level data
        case_id_col (str): Name of case ID column
        target_col (str): Name of target column
    
    Returns:
        pd.DataFrame: Case-level aggregated features
    """
    # Filter to logs with case IDs
    case_logs = df[df[case_id_col].notna() & (df[case_id_col] != '')].copy()
    
    if len(case_logs) == 0:
        return pd.DataFrame()
    
    # Aggregate features at case level
    case_features = case_logs.groupby(case_id_col).agg({
        # Target variable (max = did any log in case lead to issue)
        target_col: 'max',
        
        # Basic stats
        'message_length': ['mean', 'std', 'min', 'max'],
        'severity_encoded': ['mean', 'std', 'min', 'max'],
        'component_encoded': ['nunique'],
        
        # Case progression features
        'case_log_count': 'first',
        'case_duration_minutes': 'first', 
        'case_severity_escalation': 'first',
        'case_max_severity': 'first',
        'case_min_severity': 'first',
        
        # Temporal features
        'hour': ['mean', 'std'],
        'day_of_week': 'first',
        'is_business_hours': 'mean',
        'is_after_hours': 'mean',
        'shift': 'first',
        
        # Anomaly features
        'is_severity_anomaly': 'sum',
        'is_rapid_escalation': 'first',
        'is_unusual_case_pattern': 'first'
    }).round(3)
    
    # Flatten column names
    case_features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in case_features.columns]
    
    return case_features.reset_index()


if __name__ == "__main__":
    # Test with the new incident dataset
    df = pd.read_csv('data/splunk_logs_incidents.csv')
    
    print("🧪 Testing case-based feature engineering...")
    processed_df, preprocessing_objects = preprocess_case_data(df)
    
    print(f"\n📊 Results:")
    print(f"   Original shape: {df.shape}")
    print(f"   Processed shape: {processed_df.shape}")
    print(f"   New features created: {processed_df.shape[1] - df.shape[1]}")
    
    # Show case-level aggregation
    case_features = get_case_level_features(processed_df)
    if len(case_features) > 0:
        print(f"   Case-level features: {case_features.shape}")
        print(f"   Sample cases:\n{case_features.head()}") 