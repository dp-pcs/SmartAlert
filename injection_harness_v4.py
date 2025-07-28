import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import lightgbm as lgb

def get_model(model_name):
    if model_name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "xgb":
        return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_name == "lgb":
        return lgb.LGBMClassifier(objective='binary', random_state=42)
    else:
        raise ValueError("Invalid model name: choose 'rf', 'xgb', or 'lgb'")

def run_bakeoff_with_tfidf(data_path, batch_size=10000, num_batches=5):
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    results = []

    tfidf = TfidfVectorizer(max_features=50)
    tfidf.fit(df["message"])

    all_data = pd.DataFrame()
    for i in range(num_batches):
        batch = df.iloc[i * batch_size:(i + 1) * batch_size]
        all_data = pd.concat([all_data, batch], axis=0)

        X_tfidf = tfidf.transform(all_data["message"]).toarray()
        severity_codes = all_data["severity"].astype("category").cat.codes
        component_codes = all_data["component"].astype("category").cat.codes
        X = np.column_stack((X_tfidf, severity_codes, component_codes, all_data["message_length"]))
        y = all_data["led_to_issue"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for model_name in ["rf", "xgb", "lgb"]:
            model = get_model(model_name)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")

            results.append({
                "batch": i + 1,
                "model": model_name,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

            model_dir = os.path.join("models", f"{model_name}_round_{i+1}.pkl")
            joblib.dump(model, model_dir)

    return pd.DataFrame(results)
