"""
Feature Engineering Utilities for SmartAlert AI

This module contains functions for preprocessing and feature engineering
of Splunk log data for incident prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime


def extract_timestamp_features(df, timestamp_col):
    """
    Extract time-based features from timestamp column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        timestamp_col (str): Name of timestamp column
    
    Returns:
        pd.DataFrame: Dataframe with new time features
    """
    df_copy = df.copy()
    
    # Convert to datetime
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col], errors='coerce')
    
    # Extract features
    df_copy['hour'] = df_copy[timestamp_col].dt.hour
    df_copy['day_of_week'] = df_copy[timestamp_col].dt.dayofweek
    df_copy['day_of_month'] = df_copy[timestamp_col].dt.day
    df_copy['month'] = df_copy[timestamp_col].dt.month
    df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
    df_copy['is_business_hours'] = ((df_copy['hour'] >= 9) & (df_copy['hour'] <= 17)).astype(int)
    
    return df_copy


def encode_categorical_features(df, categorical_cols=None):
    """
    Encode categorical features using label encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_cols (list): List of categorical columns to encode
    
    Returns:
        tuple: (encoded_df, encoder_dict)
    """
    df_copy = df.copy()
    
    if categorical_cols is None:
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
    
    encoder_dict = {}
    
    for col in categorical_cols:
        if col in df_copy.columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            encoder_dict[col] = le
    
    return df_copy, encoder_dict


def create_log_aggregations(df, group_by_cols, agg_cols):
    """
    Create aggregation features from log data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        group_by_cols (list): Columns to group by
        agg_cols (list): Columns to aggregate
    
    Returns:
        pd.DataFrame: Dataframe with aggregation features
    """
    df_copy = df.copy()
    
    # Create aggregations
    for agg_col in agg_cols:
        if agg_col in df_copy.columns:
            # Count aggregations
            count_feature = f"{agg_col}_count"
            df_copy[count_feature] = df_copy.groupby(group_by_cols)[agg_col].transform('count')
            
            # If numeric, add mean and std
            if df_copy[agg_col].dtype in ['int64', 'float64']:
                mean_feature = f"{agg_col}_mean"
                std_feature = f"{agg_col}_std"
                df_copy[mean_feature] = df_copy.groupby(group_by_cols)[agg_col].transform('mean')
                df_copy[std_feature] = df_copy.groupby(group_by_cols)[agg_col].transform('std')
    
    return df_copy


def detect_anomalies(df, numerical_cols, threshold=3):
    """
    Detect anomalies using z-score method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numerical_cols (list): List of numerical columns
        threshold (float): Z-score threshold for anomaly detection
    
    Returns:
        pd.DataFrame: Dataframe with anomaly flags
    """
    df_copy = df.copy()
    
    for col in numerical_cols:
        if col in df_copy.columns:
            mean = df_copy[col].mean()
            std = df_copy[col].std()
            z_scores = np.abs((df_copy[col] - mean) / std)
            df_copy[f"{col}_is_anomaly"] = (z_scores > threshold).astype(int)
    
    return df_copy


def preprocess_log_data(df, timestamp_col=None, categorical_cols=None, target_col=None):
    """
    Main preprocessing pipeline for log data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        timestamp_col (str): Name of timestamp column
        categorical_cols (list): List of categorical columns
        target_col (str): Name of target column
    
    Returns:
        tuple: (processed_df, preprocessing_objects)
    """
    df_processed = df.copy()
    preprocessing_objects = {}
    
    # Extract timestamp features
    if timestamp_col and timestamp_col in df_processed.columns:
        df_processed = extract_timestamp_features(df_processed, timestamp_col)
    
    # Encode categorical features
    if categorical_cols is None:
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)
    
    df_processed, encoders = encode_categorical_features(df_processed, categorical_cols)
    preprocessing_objects['encoders'] = encoders
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    
    # Scale numerical features
    if target_col:
        feature_cols = [col for col in numeric_cols if col != target_col]
    else:
        feature_cols = numeric_cols
    
    scaler = StandardScaler()
    df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
    preprocessing_objects['scaler'] = scaler
    preprocessing_objects['feature_columns'] = feature_cols
    
    return df_processed, preprocessing_objects 